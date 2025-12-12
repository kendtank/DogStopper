# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/12 15:56
@Author  : Kend
@FileName: comb_mcu_embed_model
@Software: PyCharm
@modifier:
"""
from typing import Tuple

"""
1. 读取指定的四组音频(两只相同的狗吠， 两只不同狗吠)，保存int16的原始音频
2. 对两只相同的狗吠分别做数据变换，用于测试变换后的相似度， 并保存int16音频数据
3. 对指定一个狗吠做embed推理，得到的32D数据， 保存在h头文件中，用于测试mcu测试误差

生成：
1. 六个 MCU int16 音频头文件（两只相同 + 两只不同 + 两个增强）
2. 一个 embed 输出 32D 头文件
3. 打印六组音频的余弦相似度

满足 MCU: 16kHz, 单通道, 3200 点（截断/补零）, int16
"""

import os
import numpy as np
import soundfile as sf
import tensorflow as tf

# ----- 全局参数 -----
SR = 16000
TARGET_SAMPLES = 3200     # 固定 MCU 输入长度

# ====== 1. 音频读取 + 3200点处理 ======
def load_audio_3200(path):
    """读取音频 → 单通道 → resample 16k → 截断/补零到 3200 → int16"""
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # 转成 float32
    wav = wav.astype(np.float32)

    # 重采样
    if sr != SR:
        wav = librosa.resample(wav, sr, SR)

    # ---- 截断/填充 ----
    if len(wav) >= TARGET_SAMPLES:
        wav = wav[:TARGET_SAMPLES]
    else:
        pad = TARGET_SAMPLES - len(wav)
        wav = np.pad(wav, (0, pad), mode="constant")

    # ---- float32 → int16 ----
    wav_int16 = np.clip(wav * 32767, -32768, 32767).astype(np.int16)
    return wav_int16


# ====== 2. 保存为 MCU 头文件 ======
def save_to_header_int16(arr, name, out_dir="headers"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.h")

    with open(path, "w") as f:
        f.write(f"// Auto-generated audio data ({len(arr)} samples)\n")
        f.write(f"const int16_t {name}[{len(arr)}] = {{\n")
        for i in range(0, len(arr), 16):
            line = ", ".join(str(x) for x in arr[i:i+16])
            f.write("    " + line + ",\n")
        f.write("};\n")

    print(f"[保存] {path}")


# ====== 3. 简单的数据增强：时间拉伸 + 变调 ======
def augment_time_stretch(wav_int16, rate=1.1):
    """int16 → float → stretch → float → int16 (截断3200)"""

    y = wav_int16.astype(np.float32) / 32767.0

    try:
        y2 = librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:
        y2 = librosa.effects.time_stretch(y, rate)

    y2 = np.clip(y2, -1.0, 1.0)
    y2_int16 = (y2 * 32767.0).astype(np.int16)

    return fix_len_3200(y2_int16)


def augment_pitch_shift(wav_int16, n_steps=3):
    """int16 → float → pitch shift → float → int16 (截断3200)"""

    # 1) int16 → float32 normalized to [-1,1]
    y = wav_int16.astype(np.float32) / 32767.0

    # 2) 兼容不同版本 librosa
    try:
        y2 = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=n_steps)
    except TypeError:
        y2 = librosa.effects.pitch_shift(y, SR, n_steps)

    # 3) [-1,1] → int16
    y2 = np.clip(y2, -1.0, 1.0)
    y2_int16 = (y2 * 32767.0).astype(np.int16)

    # 4) 保证长度 = 3200（截断或填零）
    return fix_len_3200(y2_int16)

def fix_len_3200(w):
    """确保音频长度为 3200 点"""
    TARGET = 3200
    L = len(w)

    if L > TARGET:
        return w[:TARGET]
    elif L < TARGET:
        pad = TARGET - L
        return np.concatenate([w, np.zeros(pad, dtype=np.int16)])
    return w


def load_audio_3200_from_float(wav):
    """内部使用：float 输入 → 3200点 → int16"""
    if len(wav) >= TARGET_SAMPLES:
        wav = wav[:TARGET_SAMPLES]
    else:
        wav = np.pad(wav, (0, TARGET_SAMPLES - len(wav)))
    return np.clip(wav, -32768, 32767).astype(np.int16)


# ====== 4. 计算 logmel + embed ======
from scipy.signal.windows import get_window
from scipy.fftpack import fft
import librosa

# logmel 参数
WIN_MS = 25      # 帧长 25ms (STFT window)
HOP_MS = 10      # 帧移 10ms
N_MELS = 40
N_FFT = 512
FRAME_SIZE = int(SR * WIN_MS / 1000)   # 400
HOP = int(SR * HOP_MS / 1000)         # 160

def compute_logmel(
    wav_int16: np.ndarray,
    sr: int = SR,
    frame_size: int = FRAME_SIZE,
    hop: int = HOP,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = 0.0,
    fmax: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if fmax is None:
        fmax = sr / 2
    wave = wav_int16.astype(np.float32) / 32768.0
    win = get_window("hann", frame_size, fftbins=True).astype(np.float32)
    frames = librosa.util.frame(wave, frame_length=frame_size, hop_length=hop).astype(np.float32)  # (frame_size, T)
    frames_win = frames * win[:, None]
    X = fft(frames_win, n=n_fft, axis=0)
    powspec = np.abs(X[:n_fft // 2 + 1, :]) ** 2  # (num_bins, T)
    M = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False, norm="slaney").astype(np.float32)
    mel_power = np.dot(M, powspec).astype(np.float32)  # (n_mels, T)
    amin = 1e-8
    S_db = 10.0 * np.log10(np.maximum(mel_power, amin))
    top_db = 100.0
    S_db = np.maximum(S_db, S_db.max() - top_db)
    return S_db.T.astype(np.float32)  #(T, n_mels)



# ====== 5. 推理 embed ======
def infer_embed(logmel, interpreter):
    # (T, 40) → (1, T, 40, 1)
    x = logmel[np.newaxis, ..., np.newaxis].astype(np.float32)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details['index'], x)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details['index'])
    return out.squeeze()



# ====== 6. 保存 embedding 头文件 ======
def save_embed_header(embed, name="embed_dogA", out_dir="headers"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.h")

    with open(path, "w") as f:
        f.write("// 32D embedding\n")
        f.write(f"const float {name}[32] = {{\n")
        for i in range(0, 32, 8):
            line = ", ".join(f"{x:.6f}" for x in embed[i:i+8])
            f.write("    " + line + ",\n")
        f.write("};\n")

    print(f"[保存] {path}")


# ====== 7. 余弦相似度 ======
def cosine(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ==============================================================
# ====================== 主流程（你需求的所有东西） ======================
# ==============================================================

def main():
    # 你自己替换文件路径
    dogA1 = "train/compare_dog/dog02/dog_bark_009.WAV"
    dogA2 = "train/compare_dog/dog02/dog_bark_017.WAV"
    dogB1 = "train/compare_dog/dog06/dog_bark_039.WAV"
    dogC2 = "train/compare_dog/dog04/dog_bark_027.WAV"

    # ---- 读取 ----
    A1 = load_audio_3200(dogA1)
    A2 = load_audio_3200(dogA2)
    B1 = load_audio_3200(dogB1)
    C1 = load_audio_3200(dogC2)

    # ---- two augmentations for A ----
    A_aug1 = augment_pitch_shift(A1, n_steps=3)
    A_aug2 = augment_time_stretch(A1, rate=1.15)

    # ---- 保存 6 个音频头文件 ----
    save_to_header_int16(A1, "dogA1_raw")
    save_to_header_int16(A_aug1, "dogA1_aug1")
    save_to_header_int16(A_aug2, "dogA1_aug2")

    save_to_header_int16(A2, "dogA2_raw")
    save_to_header_int16(B1, "dogB1_raw")
    save_to_header_int16(C1, "dogC1_raw")

    # ---- embed 模型推理 ----
    interpreter = tf.lite.Interpreter(model_path="train/dog_bark_embed32_export/embedding_model_fp32.tflite")
    interpreter.allocate_tensors()

    logmel_A1 = compute_logmel(A1)
    embed_A1 = infer_embed(logmel_A1, interpreter)

    save_embed_header(embed_A1, "embed_dogA")

    # ---- 余弦相似度 ----
    print("\n===== Cosine Similarity (Embed) =====")
    waves = {
        "A1": A1, "A2": A2,
        "A_aug1": A_aug1, "A_aug2": A_aug2,
        "B1": B1, "C1": C1
    }
    embeds = {}
    for key in waves:
        logmel = compute_logmel(waves[key])
        embeds[key] = infer_embed(logmel, interpreter)

    keys = list(embeds.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            s = cosine(embeds[keys[i]], embeds[keys[j]])
            print(f"{keys[i]} vs {keys[j]} : {s:.4f}")


if __name__ == "__main__":
    main()

"""
[保存] headers/dogA1_raw.h
[保存] headers/dogA1_aug1.h
[保存] headers/dogA1_aug2.h
[保存] headers/dogA2_raw.h
[保存] headers/dogB1_raw.h
[保存] headers/dogC1_raw.h
[保存] headers/embed_dogA.h

===== Cosine Similarity (Embed) =====
A1 vs A2 : 0.8372
A1 vs A_aug1 : 0.6204
A1 vs A_aug2 : 0.9937
A1 vs B1 : -0.5760
A1 vs C1 : 0.0289
A2 vs A_aug1 : 0.6421
A2 vs A_aug2 : 0.8425
A2 vs B1 : -0.4587
A2 vs C1 : 0.0234
A_aug1 vs A_aug2 : 0.6365
A_aug1 vs B1 : -0.4401
A_aug1 vs C1 : 0.3101
A_aug2 vs B1 : -0.6008
A_aug2 vs C1 : 0.0488
B1 vs C1 : -0.1488
"""