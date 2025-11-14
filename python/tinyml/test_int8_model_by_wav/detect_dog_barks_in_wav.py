# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/14 16:38
@Author  : Kend
@FileName: detect_dog_barks_in_wav
@Software: PyCharm
@modifier:
"""



"""
实时滑窗检测狗叫（基于 量化后的INT8 TFLite 模型）注意：这个模型是最终部署到mcu中的模型
输出所有被判定为“狗叫”的 200ms 音频片段，供人工验证  后续需要加后处理， 连续检测几组是才是一个“狗叫” # TODO
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # tinyml目录
import numpy as np
import tensorflow as tf
import soundfile as sf
from audio_features import compute_mfcc
from dataset.norm_mfcc import norm_mfcc


# ==============================
# 配置
# ==============================
WAV_PATH = "dog_in_home_001.wav"  # 测试音频
MODEL_INT8_PATH = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/tinyml/train/quantized_model_mfcc_int8.tflite"
MODEL_FLOAT_PATH = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/tinyml/train/float_model_mfcc.tflite"
OUTPUT_DIR_INT8 = "detected_barks_int8"
OUTPUT_DIR_FP = "detected_barks_float"
os.makedirs(OUTPUT_DIR_INT8, exist_ok=True)
os.makedirs(OUTPUT_DIR_FP, exist_ok=True)

# 音频参数
SAMPLE_RATE = 16000  # 假设你用 16kHz
FRAME_DURATION = 0.2  # 200ms
HOP_DURATION = 0.05  # 50ms
FRAME_SAMPLES = int(FRAME_DURATION * SAMPLE_RATE)  # 3200
HOP_SAMPLES = int(HOP_DURATION * SAMPLE_RATE)      # 800
# INT8 量化参数（模型量化得到）
INPUT_SCALE, INPUT_ZP = 0.00784314, -1
OUTPUT_SCALE, OUTPUT_ZP = 0.00390625, -128


# 先把整个音频加载进入内存
def read_wave_mcu_style_float(wav_path):
    # 用 soundfile 直接读 int16 PCM， 前提是音频都已经预处理为 16KHz 单声道
    wave, sr = sf.read(wav_path, dtype='int16')  # 对齐pcm数据采样
    assert sr == SAMPLE_RATE, f"采样率必须是 {SAMPLE_RATE}Hz，当前: {sr}"
    if wave.ndim > 1:
        wave = wave.mean(axis=1).astype(np.int16)
    # 转 float32，除以 32768.f
    wave_float = wave.astype(np.float32) / 32768.0   # 对齐mcu行为，转换为 float32， 归一化
    print(f"音频加载成功: {len(wave_float)/SAMPLE_RATE:.2f}s, shape={wave_float.shape}")
    return wave_float

def extract_and_norm_mfcc(audio_segment: np.ndarray) -> np.ndarray:
    """
    对 200ms 音频片段提取 MFCC 并归一化，返回 (18, 13, 1) 形状
    """
    mfcc = compute_mfcc(audio_segment)  # shape: (18, 13)
    mfcc = norm_mfcc(mfcc)  # 归一化 shape: (18, 13)
    mfcc = np.expand_dims(mfcc, axis=-1)  # (18, 13, 1)
    return mfcc


# 主函数
def detect_barks():
    # 1. 加载音频
    audio = read_wave_mcu_style_float(WAV_PATH)
    total_samples = len(audio)

    # 2. 加载模型
    # --- INT8 模型 ---
    interpreter_int8 = tf.lite.Interpreter(model_path=MODEL_INT8_PATH)
    interpreter_int8.allocate_tensors()
    input_detail_int8 = interpreter_int8.get_input_details()[0]
    output_detail_int8 = interpreter_int8.get_output_details()[0]

    # --- Float32 TFLite 模型（用于对比）---
    interpreter_fp = tf.lite.Interpreter(model_path=MODEL_FLOAT_PATH)
    interpreter_fp.allocate_tensors()
    input_detail_fp = interpreter_fp.get_input_details()[0]
    output_detail_fp = interpreter_fp.get_output_details()[0]

    # 3. 滑窗检测
    start = 0
    count_int8, count_fp = 0, 0

    while start + FRAME_SAMPLES <= total_samples:
        # 提取 200ms 音频片段
        segment = audio[start:start + FRAME_SAMPLES]  # (3200,)
        time_sec = start / SAMPLE_RATE

        # 提取并归一化 MFCC
        try:
            mfcc = extract_and_norm_mfcc(segment)  # (18, 13, 1)
        except Exception as e:
            print(f"MFCC 提取失败 @ {time_sec:.2f}s: {e}")
            start += HOP_SAMPLES
            continue

        # =============== INT8 推理 ===============
        input_int8 = np.round(mfcc / INPUT_SCALE + INPUT_ZP).astype(np.int8)
        input_int8 = np.expand_dims(input_int8, axis=0)  # (1, 18, 13, 1)

        interpreter_int8.set_tensor(input_detail_int8['index'], input_int8)
        interpreter_int8.invoke()
        out_i8 = interpreter_int8.get_tensor(output_detail_int8['index'])[0][0]
        prob_int8 = (out_i8 - OUTPUT_ZP) * OUTPUT_SCALE

        # =============== Float32 推理 ===============
        input_fp = np.expand_dims(mfcc, axis=0).astype(np.float32)
        interpreter_fp.set_tensor(input_detail_fp['index'], input_fp)
        interpreter_fp.invoke()
        prob_fp = interpreter_fp.get_tensor(output_detail_fp['index'])[0][0]

        # =============== 保存狗叫片段 ===============
        THRESHOLD = 0.5
        print(f"t={time_sec:.2f}s, p_int8={prob_int8:.3f}, p_fp={prob_fp:.3f}")

        if prob_int8 > THRESHOLD:
            filename = f"bark_int8_{count_int8:03d}_{time_sec:.2f}s.wav"
            sf.write(os.path.join(OUTPUT_DIR_INT8, filename), segment, SAMPLE_RATE)
            print(f"[INT8] 狗叫! t={time_sec:.2f}s, p={prob_int8:.3f} → {filename}")
            count_int8 += 1

        if prob_fp > THRESHOLD:
            filename = f"bark_fp_{count_fp:03d}_{time_sec:.2f}s.wav"
            sf.write(os.path.join(OUTPUT_DIR_FP, filename), segment, SAMPLE_RATE)
            print(f"[FP32] 狗叫! t={time_sec:.2f}s, p={prob_fp:.3f} → {filename}")
            count_fp += 1

        start += HOP_SAMPLES

    # =============== 总结 ===============
    print("\n" + "="*60)
    print(f"检测完成!")
    print(f"INT8 检出: {count_int8} 段 → {OUTPUT_DIR_INT8}/")
    print(f"FP32 检出: {count_fp} 段 → {OUTPUT_DIR_FP}/")
    print(f"\n请人工播放这些 .wav 文件，验证是否为真实狗叫！")
    print("="*60)


if __name__ == "__main__":
    detect_barks()