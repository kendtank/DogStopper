# part1_train_embedding.py  —  Part 1/3
# Imports + feature extraction + data pipeline + model definition

import os
import random
import math
from typing import List, Tuple, Dict

import numpy as np
import soundfile as sf
import librosa
from scipy.signal.windows import get_window
from scipy.fftpack import fft

import tensorflow as tf
from tensorflow.keras import layers, models

# --------- Config / Hyperparams (可按需修改) ----------
SR = 16000
WIN_MS = 200
HOP_MS = 50
FRAME_SIZE = int(SR * WIN_MS / 1000)   # 3200
HOP_SIZE = int(SR * HOP_MS / 1000)     # 800
N_FFT = 512
N_MELS = 40
NUM_BINS = N_FFT // 2 + 1
TOP_DB = 100.0
MIN_AMP = 1e-8

# Embedding dim
EMBED_DIM = 32

# --------------------------------------------------------

def compute_logmel(
    wave: np.ndarray,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
    hop: int = HOP_SIZE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = 0.0,
    fmax: float = None,
) -> np.ndarray:
    """
    Compute log-mel for an input waveform window.
    Returns: (time_frames, n_mels)  -> shape expected (18, 40)
    """
    if fmax is None:
        fmax = sr / 2.0
    # STFT frames aligned with MCU style: use hann window then FFT
    win = get_window("hann", frame_size, fftbins=True).astype(np.float32)
    # librosa.util.frame requires wave length >= frame_length; caller ensures that
    frames = librosa.util.frame(wave, frame_length=frame_size, hop_length=hop).astype(np.float32)  # shape (frame_size, n_frames)
    frames_win = frames * win[:, None]
    X = fft(frames_win, n=n_fft, axis=0)  # shape (n_fft, n_frames)
    powspec = (np.abs(X[:NUM_BINS, :]) ** 2).astype(np.float32)  # (num_bins, n_frames)

    M = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
                            htk=False, norm="slaney").astype(np.float32)  # (n_mels, num_bins)
    mel_power = np.dot(M, powspec).astype(np.float32)  # (n_mels, n_frames)
    S_db = 10.0 * np.log10(np.maximum(mel_power, MIN_AMP))
    # top_db clamp
    S_db = np.maximum(S_db, S_db.max() - TOP_DB)
    S_db = S_db.astype(np.float32)
    # return S_db.T  # (n_frames, n_mels) — expected (18,40)
    return S_db  # (n_frames, n_mels) — expected (18,40)


def read_wave_mcu_style_float(wav_path: str, target_sr: int = SR) -> np.ndarray:
    """
    Read WAV file and return float32 waveform scaled to [-1,1) per MCU behavior.
    Supports multi-channel (averages channels).
    Ensures target_sr (raises AssertionError if mismatch).
    """
    # 1. 以 int16 格式读取原始数据
    wave_int16, sr = sf.read(wav_path, dtype='int16')

    # 2. 多通道混合 (Mixdown)
    if wave_int16.ndim == 2:
        # MCU 风格：先在定点域内进行平均 (虽然在 float32 域更常见，但这里保持逻辑清晰)
        # ⚠️ 注意：这里使用 float32 进行平均，因为 int16 平均会截断
        wave_float = wave_int16.astype(np.float32)
        wave_float = np.mean(wave_float, axis=1)
    else:
        wave_float = wave_int16.astype(np.float32)

    # 3. MCU 风格归一化：int16 -> [-1, 1)
    # 使用 32768.0 确保精确的 int16 范围缩放
    wave_float = wave_float / 32768.0

    # 4. 重采样
    if sr != target_sr:
        print(f"Resampling {sr}Hz to {target_sr}Hz for {wav_path}")
        # ⚠️ 确保 librosa.resample 的输入是 float 类型
        wave_float = librosa.resample(wave_float, orig_sr=sr, target_sr=target_sr)

    # 5. 确保最终输出是 float32 类型
    return wave_float.astype(np.float32)


def sliding_windows_200ms_50ms(wave: np.ndarray, sr: int = SR,
                               win_ms: int = WIN_MS, hop_ms: int = HOP_MS) -> List[np.ndarray]:
    """
    Return list of windows (raw samples). Last window padded by wrapping from beginning.
    """
    win = int(sr * win_ms // 1000)
    hop = int(sr * hop_ms // 1000)
    L = len(wave)
    frames = []
    start = 0
    while True:
        end = start + win
        if end <= L:
            frames.append(wave[start:end].copy())
        else:
            seg = wave[start:L]
            need = win - len(seg)
            pad = wave[:need]
            frame = np.concatenate([seg, pad]).copy()
            frames.append(frame)
            break
        start += hop
        if start >= L:
            break
    return frames


def build_audio_dataset(dataset_root: str, split=(0.8, 0.1, 0.1), seed=1234) -> Dict[str, List[dict]]:
    """
    Walk dataset_root expecting per-class subfolders (dog01, dog02, ...).
    Returns dict with lists of {"wave": np.ndarray, "label": str}
    """
    random.seed(seed)
    all_items = []
    dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    for pet in dirs:
        pet_dir = os.path.join(dataset_root, pet)
        wavs = sorted([f for f in os.listdir(pet_dir) if f.lower().endswith('.wav')])
        for w in wavs:
            path = os.path.join(pet_dir, w)
            wave = read_wave_mcu_style_float(path)
            windows = sliding_windows_200ms_50ms(wave)
            for win in windows:
                all_items.append({"wave": win, "label": pet})
    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    train = all_items[:n_train]
    val = all_items[n_train:n_train + n_val]
    test = all_items[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def tf_dataset_builder(items: List[dict], batch: int = 32, shuffle: bool = True):
    """
    Build tf.data.Dataset from items list. Output: (batch_x, batch_y)
    x: float32 tensor (B, 18, 40, 1)
    y: int32 label ids
    Returns: ds, label_names
    """
    unique_labels = sorted({it["label"] for it in items})
    label2id = {lab: i for i, lab in enumerate(unique_labels)}

    def gen():
        for it in items:
            logmel = compute_logmel(it["wave"])  # (18, 40)
            # ensure shape
            if logmel.shape != (18, N_MELS):
                # defensive: resize/pad/trim second dim (shouldn't usually happen)
                logmel = librosa.util.fix_length(logmel, size=18, axis=0)
                logmel = librosa.util.fix_length(logmel, size=N_MELS, axis=1)
            yield logmel.astype(np.float32), np.int32(label2id[it["label"]])

    output_signature = (tf.TensorSpec(shape=(18, N_MELS), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.int32))
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.map(lambda x, y: (tf.expand_dims(x, -1), y), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2000)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, unique_labels


# ----------------- Model (train/export) ------------------
class L2NormLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


def build_tflm_dog_bark_micro_lite_32d(mode: str = "train") -> tf.keras.Model:
    """
    mode: "train" -> outputs L2-normalized embedding
          "export" -> outputs raw embedding (no L2) for tflite
    """
    inputs = tf.keras.layers.Input(shape=(18, N_MELS, 1), name="logmel_input")
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.SeparableConv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.SeparableConv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.AveragePooling2D(pool_size=(4, 10))(x)
    x = layers.Reshape((16,))(x)

    embedding = layers.Dense(EMBED_DIM, name="embedding_output")(x)

    if mode == "train":
        out = L2NormLayer()(embedding)
    else:
        out = embedding
    return tf.keras.Model(inputs, out, name=f"feature_extractor_{mode}")

# End of Part 1
# part2_train_embedding.py  —  Part 2/3
# ArcFace layer + training / validation steps

import time
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import losses

# Reuse model and dataset utilities from Part 1 (import if saved as module) if splitting files.
# If you pasted Part1 and Part2 into same file, no imports needed.

# ---------------- ArcFace as Keras Layer (produces scaled logits) ----------------
class ArcFaceLayer(tf.keras.layers.Layer):
    """
    ArcFace style layer:
    - holds trainable W (embedding_dim, num_classes)
    - call(embeddings, labels=None) -> if labels provided returns logits with angular margin applied
      else returns logits (x @ W) scaled by s
    Note: embeddings should be L2-normalized already in forward pass for training.
    """
    def __init__(self, num_classes: int, s: float = 30.0, m: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.s = float(s)
        self.m = float(m)

    def build(self, input_shape):
        emb_dim = int(input_shape[-1])
        # initialize W with shape (emb_dim, num_classes)
        # self.W = self.add_weight("W", shape=(emb_dim, self.num_classes),
        #                          initializer='glorot_uniform', trainable=True)
        self.W = self.add_weight(
            name="W",
            shape=(emb_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, embeddings, labels=None):
        # embeddings: (B, emb_dim)  - ideally already normalized
        x = tf.nn.l2_normalize(embeddings, axis=1)  # ensure robust
        W = tf.nn.l2_normalize(self.W, axis=0)      # normalize each class vector (emb_dim, C)
        logits = tf.matmul(x, W)                    # (B, C) -> cos(theta)
        if labels is None:
            return logits * self.s
        # create one-hot mask
        labels = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels, depth=self.num_classes, dtype=logits.dtype)
        # add margin by subtracting m on target cos
        logits_m = logits - one_hot * self.m
        return logits_m * self.s

# ---------------- Training function ----------------
def train_embedding_model(
    dataset_root: str,
    epochs: int = 20,
    batch: int = 64,
    lr: float = 1e-3,
    save_dir: str = "dog_bark_embed32_export",
):
    print("== Load dataset ==")
    data_all = build_audio_dataset(dataset_root)
    ds_train, label_names = tf_dataset_builder(data_all["train"], batch=batch, shuffle=True)
    ds_val, _ = tf_dataset_builder(data_all["val"], batch=batch, shuffle=False)
    ds_test, _ = tf_dataset_builder(data_all["test"], batch=batch, shuffle=False)
    num_classes = len(label_names)
    print(f"Classes found: {num_classes}, labels: {label_names}")

    # build model
    model = build_tflm_dog_bark_micro_lite_32d(mode="train")
    # arcface layer
    arcface_layer = ArcFaceLayer(num_classes=num_classes, s=30.0, m=0.35)
    # warm build arcface weights by calling once with dummy
    _ = arcface_layer(tf.zeros((1, EMBED_DIM)), labels=None)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="val_acc")

    # training/validation steps (tf.function for speed)
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            emb = model(x, training=True)  # normalized embeddings
            logits = arcface_layer(emb, labels=y)  # scaled logits with margin
            loss = loss_fn(y, logits)
        variables = model.trainable_variables + arcface_layer.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        train_loss.update_state(loss)
        train_acc.update_state(y, logits)

    @tf.function
    def val_step(x, y):
        emb = model(x, training=False)
        logits = arcface_layer(emb, labels=None)  # no margin during eval; returns scaled logits
        val_acc.update_state(y, logits)
        return logits, emb

    # Training loop
    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss.reset_state()
        train_acc.reset_state()
        val_acc.reset_state()

        # train
        for xb, yb in ds_train:
            train_step(xb, yb)

        # val: accumulate preds for F1
        all_true = []
        all_pred = []
        for xb, yb in ds_val:
            logits, _ = val_step(xb, yb)
            preds = tf.argmax(logits, axis=1).numpy().tolist()
            all_pred.extend(preds)
            all_true.extend(yb.numpy().tolist())

        # compute F1 macro
        f1 = f1_score(all_true, all_pred, average="macro") if len(all_true) > 0 else 0.0
        if f1 > best_val_f1:
            best_val_f1 = f1
            # save checkpointed weights (only model + arcface)
            os.makedirs(save_dir, exist_ok=True)
            # model.save_weights(os.path.join(save_dir, "best_model_weights.h5"))
            model.save_weights(os.path.join(save_dir, "best_model.weights.h5"))

            # also save arcface weights separately
            np.save(os.path.join(save_dir, "arcface_W.npy"), arcface_layer.W.numpy())

        print(f"Epoch {epoch}/{epochs}  loss={train_loss.result():.4f} "
              f"train_acc={train_acc.result():.4f} val_acc={val_acc.result():.4f} val_f1={f1:.4f}  time={time.time()-t0:.1f}s")

    # final test evaluation
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    all_true = []
    all_pred = []
    for xb, yb in ds_test:
        logits, _ = val_step(xb, yb)
        test_acc.update_state(yb, logits)
        all_pred.extend(tf.argmax(logits, axis=1).numpy().tolist())
        all_true.extend(yb.numpy().tolist())
    test_f1 = f1_score(all_true, all_pred, average="macro") if len(all_true) > 0 else 0.0
    print("== Test Results ==")
    print("Test Acc =", float(test_acc.result()))
    print("Test F1 (macro) =", test_f1)

    # load best weights back
    best_w = os.path.join(save_dir, "best_model_weights.h5")
    if os.path.exists(best_w):
        model.load_weights(best_w)
    # restore arcface W if saved
    arc_W_path = os.path.join(save_dir, "arcface_W.npy")
    if os.path.exists(arc_W_path):
        arcface_layer.W.assign(np.load(arc_W_path))

    return model, arcface_layer, label_names


# If parts combined into one file this will reuse definitions; otherwise import:
# from part1_train_embedding import build_tflm_dog_bark_micro_lite_32d, compute_logmel, ...
# from part2_train_embedding import train_embedding_model, ArcFaceLayer

def export_embedding_models(model, export_dir):
    os.makedirs(export_dir, exist_ok=True)

    # 1. 保存 SavedModel 格式，便于 MCU tflite 转换
    saved_model_dir = os.path.join(export_dir, "saved_model")
    # model.save(saved_model_dir)
    model.export(saved_model_dir)

    # 2. 转为 tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(export_dir, "embedding.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("SavedModel exported to:", saved_model_dir)
    print("TFLite model exported to:", tflite_path)

    return saved_model_dir, tflite_path



def main():
    # 修改成你数据集根目录
    dataset_root = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog"
    epochs = 10
    batch = 32
    lr = 1e-3
    save_dir = "dog_bark_embed32_export"

    model, arcface, label_names = train_embedding_model(
        dataset_root=dataset_root,
        epochs=epochs,
        batch=batch,
        lr=lr,
        save_dir=save_dir
    )

    # export
    saved_model_path, tflite_path = export_embedding_models(model, export_dir=save_dir)
    print("Export done. Label names:", label_names)
    print("SavedModel:", saved_model_path)
    print("TFLite:", tflite_path)


if __name__ == "__main__":
    main()

