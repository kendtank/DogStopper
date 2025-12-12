# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/9 13:43
@Author  : Kend
@FileName: train_embedding
@Software: PyCharm
@modifier:  One-Class Verification / Speaker Verification
注意： MCU 不会去判断是哪只狗（不是 closed-set classification）
MCU 只做：声音 → embedding → 与模板 embedding 做余弦距离 → 是否相似
"""


"""
目标：
    MCU 上的任务：设备里只有一个「模板狗」的 embedding，实时听到狗吠 → 判断是否为模板狗。
本质上是 声纹验证（verification），不是分类

训练embedding网络方法论：
模型 (build_tflm_dog_bark_micro_lite_32d) 是一个纯粹的 32D 特征提取器（Feature Extractor），专为声纹验证任务设计。采用**度量学习（Metric Learning）**的方法进行训练

训练一个 12 条狗的 metric-learning embedding 模型：
    核心思路：
        把 12 只狗都当成 类别（class）
        训练一个 embedding 网络
        Loss 要使用 ArcFace / CosFace / Triplet / Contrastive
        模型学到声音的 identity embedding
        然后 MCU 上：
        存模板狗 embedding（多条求平均）
        实时声纹 → embedding
        计算 余弦相似度 cosine similarity
        根据阈值判断是否是模板狗
        也就是说：
            训练时：12 分类（ArcFace loss）
            推理时：embedding + cosine，相当于二分类（是否模板狗）
        这是声音验证的标准做法。
不能做的事情是： 
    1：一个模型 1 vs 11，循环做 12 次 。 模型得到的是“模板 vs 其他”的决策边界， 学不到真正的 speaker identity 特征，无法泛化到新狗
    2：训练一个 12 分类模型， speaker-id 分类问题。结果： 训练出的 embedding 只适用于这 12 只狗
"""


import os
import random
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import librosa
from scipy.signal.windows import get_window
from scipy.fftpack import fft
import soundfile as sf


# 狗吠声纹重识别模型结构

# 自定义 L2Norm 层  ？ tf.math.l2_normalize 不能直接作用在 KerasTensor 上 ，必须封装成 Keras 层
class L2NormLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


def build_tflm_dog_bark_micro_lite_32d(mode="train"):
    """
    mode: "train" —— 输出 normalized embedding
          "export" —— 输出未归一化 embedding（更适合 tflite/mcu）
    """
    inputs = tf.keras.layers.Input(shape=(18, 40, 1), name="logmel_input")
    x = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.SeparableConv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.SeparableConv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # 移除 Conv2D(32)
    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 10))(x)
    x = tf.keras.layers.Reshape((16,))(x)

    embedding_output = tf.keras.layers.Dense(32, name="embedding_output")(x)

    if mode == "train":
        # 训练时输出 L2-normalized embedding
        output = L2NormLayer()(embedding_output)
        # output = tf.nn.l2_normalize(embedding_output, axis=1, name='normalized_embedding')
    else:
        # 导出 tflite 时不做 L2，落地 MCU推理后处理
        output = embedding_output

    return tf.keras.models.Model(inputs, output, name=f"feature_extractor_{mode}")


# logmel特征提取
def compute_logmel(
        wave,
        sr=16000,
        frame_size = 400,
        hop = 160,
        n_fft = 512,
        num_bins = 257,
        n_mels = 40,
        fmin = 0.0,
        fmax = 8000.0,
    ):
    # return Sdb_py, Sdb_py.T
    """计算音频的log-mel特征， 保持和mcu算法的一致性"""
    win_py = get_window("hann", frame_size, fftbins=True).astype(np.float32)
    frames = librosa.util.frame(wave, frame_length=frame_size, hop_length=hop).astype(np.float32)  # 分帧
    frames_win = frames * win_py[:, None]   # 矩阵乘矩阵， 加hann窗
    y = frames_win.astype(np.float32)
    # 功率谱 scipy.fftpack.fft → 对齐 ESP-DSP 的 dsps_fft2r_fc32
    X = fft(y, n=n_fft, axis=0)  # 默认 scipy.fftpack.fft(x) 会把整个二维数组当作 扁平数组 做 FFT（按行展开）。
    # 功率谱（对齐 MCU：只取前 N/2+1 个 bin）
    powspec = np.abs(X[:num_bins, :]) ** 2  # Power Spectrum
    powspec = powspec.astype(np.float32)
    M_py = librosa.filters.mel(sr=sr, n_fft=512, n_mels=n_mels,
                               fmin=fmin, fmax=fmax, htk=False, norm="slaney").astype(np.float32)
    # 计算mel
    mel_power = np.dot(M_py, powspec).astype(np.float32)
    amin = 1e-8  # 最小功率值，避免 log(0)
    top_db = 100.0  # 最大动态范围
    # 计算 dB
    S_db = 10.0 * np.log10(np.maximum(mel_power.astype(np.float32), amin))
    # top_db 限制
    S_db = np.maximum(S_db, S_db.max() - top_db)
    Sdb_py = S_db.astype(np.float32)
    return Sdb_py, Sdb_py.T


# 读取wav音频文件
def read_wave_mcu_style_float(wav_path, target_sr=16000):
    # 用 soundfile 直接读 int16 PCM， 前提是音频都已经预处理为 16KHz 单声道
    wave, sr = sf.read(wav_path, dtype='int16')  # 对齐pcm数据采样
    wave = wave.astype(np.float32)
    # --- 2. 多声道处理：取平均变单声道 ---
    if len(wave.shape) == 2:
        print(f"[WARN] 音频为多声道 {wave.shape} → 自动混为单声道")
        wave = np.mean(wave, axis=1).astype(np.float32)

    # --- 3. 除以 32768 → 与 MCU 端一致 ---
    wave = wave / 32768.0

    # --- 4. 采样率检查，不是 16k 打印出来，直接报错 ---
    assert sr == target_sr, f"采样率必须是 {target_sr}Hz，当前: {sr}"

    wave_float = wave.astype(np.float32)
    return wave_float


# 读取音频文件-> 训练的片段（部署也是这个方式）
def sliding_windows_200ms_50ms(wave, sr=16000, win_ms=200, hop_ms=50):
    win = int(sr * win_ms / 1000)     # 200ms = 3200 samples
    hop = int(sr * hop_ms / 1000)     # 50ms = 800 samples
    wav_len = len(wave)

    frames = []
    start = 0

    while True:
        end = start + win
        if end <= wav_len:
            # 正常窗口
            frames.append(wave[start:end])
        else:
            # ⭐ 末尾窗口不够 win → 用真实音频循环补齐
            seg = wave[start:wav_len]

            need = win - len(seg)
            # 用前面的真实音频循环补
            pad = wave[:need]

            frame = np.concatenate([seg, pad])
            frames.append(frame)
            break   # 最后一帧补齐后结束

        start += hop
        if start >= wav_len:
            break

    return frames

# 制作数据集  生成训练数据清单
def build_audio_dataset(dataset_root, split=(0.8, 0.1, 0.1)):
    """
    返回结构化数据：
    {
        "train": [{"wave": np.ndarray, "label": "dogA"}, ...],
        "val":   [...],
        "test":  [...]
    }
    """
    all_items = []  # 临时收集所有切好的 frame

    # 遍历每个宠物 ID 文件夹
    for pet_id in sorted(os.listdir(dataset_root)):
        full_dir = os.path.join(dataset_root, pet_id)
        if not os.path.isdir(full_dir):
            continue

        wav_files = [f for f in os.listdir(full_dir) if f.endswith(".wav")]

        for wav in wav_files:
            wav_path = os.path.join(full_dir, wav)

            # 1. 读音频
            wave = read_wave_mcu_style_float(wav_path)

            # 2. 滑窗切片
            frames = sliding_windows_200ms_50ms(wave)

            # 3. 每个 frame 都是一条训练样本
            for f in frames:
                all_items.append({"wave": f, "label": pet_id})

    # 打乱
    random.shuffle(all_items)

    # -------------------------
    # 划分 train / val / test
    # -------------------------
    n = len(all_items)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    train = all_items[:n_train]
    val = all_items[n_train:n_train + n_val]
    test = all_items[n_train + n_val:]

    return {
        "train": train,
        "val": val,
        "test": test
    }


# 构建 TensorFlow Dataset (自动计算 logmel)
def tf_dataset_builder(items, batch=32, shuffle=True):
    """
    items: list of {"wave": np.ndarray, "label": str}
    返回: ds, label_names (list)
    """
    # 1) 构建 label -> id 映射（在 python 层）
    unique_labels = sorted({item["label"] for item in items})
    label2id = {lab: idx for idx, lab in enumerate(unique_labels)}

    def gen():
        for item in items:
            wave = item["wave"]           # np.array, raw samples for the 200ms window
            label_str = item["label"]
            logmel, _ = compute_logmel(wave)   # shape (n_mels, time) == (40, 18) by your compute_logmel
            # 你的 compute_logmel 返回 (S_db, S_db.T) 之前代码里你用 Sdb_py, Sdb_py.T -> Sdb_py.T 是 (time, n_mels)
            # 这里保证我们输出 (18, 40)
            logmel_T = logmel.T.astype(np.float32)   # shape (18,40)
            label_id = np.int32(label2id[label_str])
            yield logmel_T, label_id

    # 2) 使用 output_signature（不再使用已弃用的 output_types/output_shapes）
    output_signature = (
        tf.TensorSpec(shape=(18, 40), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    # 3) map -> add channel dim and optional augmentation
    def _map_fn(x, y):
        x = tf.expand_dims(x, axis=-1)   # (18,40,1)
        return x, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=2000)

    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, unique_labels




class ArcFaceLoss(tf.keras.layers.Layer):
    def __init__(self, num_classes, s=30.0, m=0.5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.s = s
        self.m = m

    def build(self, input_shape):
        # input_shape: (batch, embedding_dim)
        embedding_dim = input_shape[-1]

        # 可训练分类权重
        self.W = self.add_weight(
            name='W',
            shape=(embedding_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, embeddings, labels):
        # Normalize
        x = tf.nn.l2_normalize(embeddings, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        # logits = x · W
        logits = tf.matmul(x, W)

        # 取出正类 logit
        labels = tf.cast(labels, tf.int32)
        mask = tf.one_hot(labels, depth=self.num_classes)

        # cos(theta)
        cos_theta = logits

        # 加 margin
        cos_theta_m = cos_theta - mask * self.m

        # scale
        scaled_logits = cos_theta_m * self.s
        return scaled_logits


def train_embedding_model(
    dataset_root,
    epochs=20,
    batch=64,
    lr=1e-3,
):
    print("== 加载数据 ==")
    data_all = build_audio_dataset(dataset_root)

    ds_train, label_names = tf_dataset_builder(data_all["train"], batch=batch, shuffle=True)
    ds_val, _           = tf_dataset_builder(data_all["val"],   batch=batch, shuffle=False)
    ds_test, _          = tf_dataset_builder(data_all["test"],  batch=batch, shuffle=False)

    num_classes = len(label_names)

    # model
    print("== 构建模型 ==")
    model = build_tflm_dog_bark_micro_lite_32d(mode="train")

    arcface = ArcFaceLoss(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(lr)

    train_loss_metric = tf.keras.metrics.Mean()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_f1_metric = tf.keras.metrics.Mean()  # 手动计算 F1

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            emb = model(x, training=True)
            loss = arcface(y, emb)

            # forward again for acc
            W = tf.math.l2_normalize(arcface.W, axis=0)
            logits = tf.matmul(emb, W)
            acc_logits = logits * arcface.s

        grads = tape.gradient(loss, model.trainable_variables + arcface.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables + arcface.trainable_variables))

        train_loss_metric.update_state(loss)
        train_acc_metric.update_state(y, acc_logits)

    @tf.function
    def val_step(x, y):
        emb = model(x, training=False)
        W = tf.math.l2_normalize(arcface.W, axis=0)
        logits = tf.matmul(emb, W) * arcface.s
        val_acc_metric.update_state(y, logits)
        return logits

    print("== 开始训练 ==")
    for epoch in range(1, epochs+1):
        train_loss_metric.reset_state()
        train_acc_metric.reset_state()

        for x, y in ds_train:
            train_step(x, y)

        # 验证
        val_acc_metric.reset_state()
        all_true = []
        all_pred = []

        for x, y in ds_val:
            logits = val_step(x, y)
            preds = tf.argmax(logits, axis=1)
            all_true.extend(y.numpy().tolist())
            all_pred.extend(preds.numpy().tolist())

        # F1
        from sklearn.metrics import f1_score
        f1 = f1_score(all_true, all_pred, average="macro")
        val_f1_metric.update_state(f1)

        print(f"Epoch {epoch}/{epochs}  "
              f"loss={train_loss_metric.result():.4f}  "
              f"acc={train_acc_metric.result():.4f}  "
              f"val_acc={val_acc_metric.result():.4f}  "
              f"val_f1={val_f1_metric.result():.4f}")

    print("== 训练结束 ==")

    # ---------------------
    # 测试性能
    # ---------------------
    print("== 测试集评估 ==")
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in ds_test:
        emb = model(x, training=False)
        W = tf.math.l2_normalize(arcface.W, axis=0)
        logits = tf.matmul(emb, W) * arcface.s
        test_acc.update_state(y, logits)
    print("Test Acc =", float(test_acc.result()))

    # ---------------------
    # 导出 embedding-only 模型
    # ---------------------
    print("== 导出 embedding 模型 ==")
    export_model = build_tflm_dog_bark_micro_lite_32d(mode="export")
    export_model.set_weights(model.get_weights())
    export_model.save("dog_bark_embed32_export")

    print("[OK] 已成功导出 32D embedding 模型 → dog_bark_embed32_export/")
    return model, arcface, label_names



if __name__ == "__main__":
    dataset_root = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog"
    train_embedding_model(dataset_root)

