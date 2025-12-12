
import os
import random
import numpy as np
import soundfile as sf
import librosa
from scipy.signal.windows import get_window
from scipy.fftpack import fft
import tensorflow as tf
from tensorflow.keras import layers, models
import time
from sklearn.metrics import f1_score

# ---------------- constants ----------------
SR = 16000
WIN_MS = 25 # 25ms
HOP_MS = 10 # 10ms
N_MELS = 40
N_FFT = 512
NUM_BINS = N_FFT // 2 + 1


def build_tflm_dog_bark_micro_lite_32d():
    """
    始终输出原始 32D Embedding。L2 归一化在 GE2E Loss 内部处理。
    """
    inputs = layers.Input(shape=(18, N_MELS, 1), name="logmel_input")

    # Block 1
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.SeparableConv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Shape: (9, 20, 8)

    # Block 2
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Shape: (4, 10, 16)

    # Global Compression
    x = layers.AveragePooling2D(pool_size=(4, 10))(x)  # Shape: (1, 1, 16)
    x = layers.Reshape((16,))(x)  # Shape: (16,)

    # Dense Layer for 32D Embedding
    embedding_output = layers.Dense(32, name="embedding_output")(x)

    # 始终输出原始 Embedding
    return models.Model(inputs, embedding_output, name="feature_extractor")

# ---------------- feature extraction ----------------
def compute_logmel(
    wave,
    sr=SR,
    frame_size = int(SR * WIN_MS / 1000),
    hop = int(SR * HOP_MS / 1000),
    n_fft = N_FFT,
    num_bins = NUM_BINS,
    n_mels = N_MELS,
    fmin = 0.0,
    fmax = 8000.0,
):
    """Compute log-mel (matching MCU pipeline). Returns (n_mels, time_steps) and its transpose."""
    win_py = get_window("hann", frame_size, fftbins=True).astype(np.float32)
    frames = librosa.util.frame(wave, frame_length=frame_size, hop_length=hop).astype(np.float32)
    frames_win = frames * win_py[:, None]
    X = fft(frames_win, n=n_fft, axis=0)
    powspec = np.abs(X[:num_bins, :]) ** 2
    powspec = powspec.astype(np.float32)
    M_py = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False, norm="slaney").astype(np.float32)
    mel_power = np.dot(M_py, powspec).astype(np.float32)
    amin = 1e-8
    top_db = 100.0
    S_db = 10.0 * np.log10(np.maximum(mel_power.astype(np.float32), amin))
    S_db = np.maximum(S_db, S_db.max() - top_db)
    Sdb_py = S_db.astype(np.float32)
    # print("logmel shape:", Sdb_py.shape)
    return Sdb_py, Sdb_py.T  # (n_mels, time), (time, n_mels)    [40 * 18], [18 * 40]



def read_wave_mcu_style_float(wav_path, target_sr=SR):
    wave, sr = sf.read(wav_path, dtype='int16')
    wave = wave.astype(np.float32)
    if wave.ndim == 2:
        wave = np.mean(wave, axis=1).astype(np.float32)
    wave = wave / 32768.0
    if sr != target_sr:
        # try to resample automatically for convenience
        wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    assert sr == target_sr, f"采样率必须是 {target_sr}Hz，当前: {sr}"
    return wave.astype(np.float32)

def sliding_windows_200ms_50ms(wave, sr=SR, win_ms=200, hop_ms=50):
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    wav_len = len(wave)
    frames = []
    start = 0
    while True:
        end = start + win
        if end <= wav_len:
            frames.append(wave[start:end])
        else:
            seg = wave[start:wav_len]
            need = win - len(seg)
            pad = wave[:need] if need > 0 else np.array([], dtype=np.float32)
            frame = np.concatenate([seg, pad])
            frames.append(frame)
            break
        start += hop
        if start >= wav_len:
            break
    return frames


def build_audio_dataset(dataset_root, split=0.7,
                        min_extensions=('.WAV','.wav','.Wave','.wave')):
    """
    dataset_root/
        ├── dogA/
        ├── dogB/
        ├── dogC/
    返回 {"train": [...], "val": [...]}
    """

    train_items = []
    val_items = []

    class_dirs = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])

    for pet_id in class_dirs:
        full_dir = os.path.join(dataset_root, pet_id)
        wav_files = [
            f for f in os.listdir(full_dir)
            if f.endswith(min_extensions)
        ]

        # 收集此 pet_id 的所有滑窗样本
        pet_items = []
        for wav in wav_files:
            wav_path = os.path.join(full_dir, wav)
            wave = read_wave_mcu_style_float(wav_path)
            frames = sliding_windows_200ms_50ms(wave)

            for f in frames:
                pet_items.append({"wave": f, "label": pet_id})

        # --- 关键点：每只狗内部划分 7:3 ---------
        random.shuffle(pet_items)
        n = len(pet_items)
        n_train = int(n * split)

        train_items.extend(pet_items[:n_train])
        val_items.extend(pet_items[n_train:])

    # 最后全局 shuffle
    random.shuffle(train_items)
    random.shuffle(val_items)

    return {"train": train_items, "val": val_items}





# ---------------- ArcFace layer (corrected) ----------------
class ArcFaceLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes: int, s: float = 30.0, m: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.s = float(s)
        self.m = float(m)

    def build(self, input_shape):
        emb_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name="W",
            shape=(emb_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, embeddings, labels=None):
        x = tf.nn.l2_normalize(embeddings, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = tf.matmul(x, W)
        if labels is None:
            return logits * self.s
        labels = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels, depth=self.num_classes, dtype=logits.dtype)
        logits_m = logits - one_hot * self.m
        return logits_m * self.s

# ---------------- Dataset -> tf.data ----------------
def tf_dataset_builder(items, batch=32, shuffle=True):
    unique_labels = sorted({item["label"] for item in items})
    label2id = {lab: idx for idx, lab in enumerate(unique_labels)}

    def gen():
        for item in items:
            wave = item["wave"]
            label_str = item["label"]
            logmel, _ = compute_logmel(wave)
            logmel_T = logmel.T.astype(np.float32)  # shape (18,40)
            # print("=====", logmel_T.shape, "======")
            yield logmel_T, np.int32(label2id[label_str])

    output_signature = (
        tf.TensorSpec(shape=(18, N_MELS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    def _map_fn(x,y):
        x = tf.expand_dims(x, axis=-1)  # (18,40,1)
        return x, y
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2000)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, unique_labels

# ---------------- Training function ----------------
def train_embedding_model(
    dataset_root: str,
    epochs: int = 20,
    batch: int = 32,
    lr: float = 1e-3,
    save_dir: str = "dog_bark_embed32_export",
):
    print("==" * 10,  " Load dataset ", "==" * 10)
    data_all = build_audio_dataset(dataset_root)
    print("———— * ---" * 10)
    print(len(data_all['train']), len(data_all['val']))   # 193 24 25
    ds_train, label_names = tf_dataset_builder(data_all["train"], batch=batch, shuffle=True)
    ds_val, _ = tf_dataset_builder(data_all["val"], batch=batch, shuffle=False)
    # ds_test, _ = tf_dataset_builder(data_all["test"], batch=batch, shuffle=False)
    num_classes = len(label_names)
    print(f"Classes found: {num_classes}, labels: {label_names}")

    # model and arcface
    model = build_tflm_dog_bark_micro_lite_32d(mode="train")
    emb_dim = int(model.output_shape[-1])
    arcface_layer = ArcFaceLayer(num_classes=num_classes, s=30.0, m=0.35)
    # warm build
    _ = arcface_layer(tf.zeros((1, emb_dim)), labels=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            emb = model(x, training=True)
            logits = arcface_layer(emb, labels=y)
            loss = loss_fn(y, logits)
        vars_to_update = model.trainable_variables + arcface_layer.trainable_variables
        grads = tape.gradient(loss, vars_to_update)
        optimizer.apply_gradients(zip(grads, vars_to_update))
        train_loss.update_state(loss)
        train_acc.update_state(y, logits)

    @tf.function
    def val_step(x, y):
        emb = model(x, training=False)
        logits = arcface_layer(emb, labels=None)
        val_acc_metric.update_state(y, logits)
        return logits, emb

    # training loop
    best_val_f1 = -1.0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss.reset_state(); train_acc.reset_state()
        val_acc_metric.reset_state()

        for xb, yb in ds_train:
            train_step(xb, yb)

        # validation
        all_true, all_pred = [], []
        for xb, yb in ds_val:
            logits, _ = val_step(xb, yb)
            preds = tf.argmax(logits, axis=1).numpy().tolist()
            all_pred.extend(preds)
            all_true.extend(yb.numpy().tolist())

        val_f1 = f1_score(all_true, all_pred, average="macro") if len(all_true) > 0 else 0.0
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # save weights
            model.save_weights(os.path.join(save_dir, "best_model.weights.h5"))
            np.save(os.path.join(save_dir, "arcface_W.npy"), arcface_layer.W.numpy())

        print(f"Epoch {epoch}/{epochs} loss={train_loss.result():.4f} "
              f"train_acc={train_acc.result():.4f} val_acc={val_acc_metric.result():.4f} "
              f"val_f1={val_f1:.4f} time={time.time()-t0:.1f}s")

    # test evaluation
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    all_true, all_pred = [], []
    for xb, yb in ds_val:
        logits, _ = val_step(xb, yb)
        test_acc.update_state(yb, logits)
        all_pred.extend(tf.argmax(logits, axis=1).numpy().tolist())
        all_true.extend(yb.numpy().tolist())
    test_f1 = f1_score(all_true, all_pred, average="macro") if len(all_true) > 0 else 0.0
    print("== Test Results ==")
    print("Test Acc =", float(test_acc.result()))
    print("Test F1 (macro) =", test_f1)

    # restore best weights and arcface W if exist
    best_w = os.path.join(save_dir, "best_model.weights.h5")
    if os.path.exists(best_w):
        model.load_weights(best_w)
    arc_W_path = os.path.join(save_dir, "arcface_W.npy")
    if os.path.exists(arc_W_path):
        arcface_layer.W.assign(np.load(arc_W_path))

    return model, arcface_layer, label_names


def export_embedding_models(model_train_mode, export_dir="dog_bark_embed32_export"):
    """
    model_train_mode: the trained model in 'train' mode (normalized outputs)
    We will build an 'export' model (raw embedding), transfer weights, save .keras and tflite.
    """
    os.makedirs(export_dir, exist_ok=True)
    # build export model (no L2)
    export_model = build_tflm_dog_bark_micro_lite_32d(mode="export")
    # copy weights from train model (layer names match)
    export_model.set_weights(model_train_mode.get_weights())

    # Save as Keras native format (.keras)
    saved_path = os.path.join(export_dir, "embedding_model.keras")
    export_model.save(saved_path)  # Keras 3 will write .keras bundle

    # Convert to TFLite (float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = []
    tflite_model = converter.convert()
    tflite_path = os.path.join(export_dir, "embedding_model_fp32.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Optionally produce a size-optimized float16 or int8 (quant) model later
    print(f"SavedModel exported to: {saved_path}")
    print(f"TFLite exported to: {tflite_path}")
    return saved_path, tflite_path

def main():
    dataset_root = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog"
    epochs = 50
    batch = 32
    lr = 1e-5
    save_dir = "dog_bark_embed32_export"

    model, arcface, label_names = train_embedding_model(
        dataset_root=dataset_root,
        epochs=epochs,
        batch=batch,
        lr=lr,
        save_dir=save_dir
    )

    saved_model_path, tflite_path = export_embedding_models(model, export_dir=save_dir)
    print("Export done. Label names:", label_names)
    print("SavedModel:", saved_model_path)
    print("TFLite:", tflite_path)

if __name__ == "__main__":
    main()
