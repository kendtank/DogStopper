# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/13 16:08
@Author  : Kend
@FileName: compute_robust_norm_params
@Software: PyCharm
@modifier:
"""


"""
Robust Min-Max 映射到 [-1,1]，其实是两步合一：
    基于数据集统计：用训练集的 1%-99% 分位数确定“有效范围”
    线性映射到 [-1,1]：让这个有效范围刚好占满目标区间
    所以它既是“基于数据集的”，又是“映射到 [-1,1] 的”，两者不矛盾！
TensorFlow Lite 的 int8 量化默认假设输入范围是 [-1, 1] 或 [0, 1]
    如果你输入实际范围是 [-0.2, 0.3]，TFLite 会浪费 80% 的 int8 表示能力
    而你的方法让 98% 的数据占满 [-1,1] → 量化误差最小
"""


import os
import sys
# 将上级目录（即包含 audio_features.py 的目录）加入 Python 模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import argparse
from tqdm import tqdm
import soundfile as sf
from audio_features import compute_mfcc


SAMPLE_RATE = 16000
WINDOW_LENGTH = 0.2   # 秒
STEP_LENGTH = 0.02    # 秒（滑窗步长）
SEED = 42
MIN_AUDIO_LEN = 0.1   # 秒


# ==============================
# 这两个函数需要和mcu保持一致
# ==============================

def read_wave_mcu_style_float(file_path):
    """
    int16读取音频文件，返回 float32 的一维数组（采样率需与训练一致）
    """
    # 用 soundfile 直接读 int16 PCM， 前提是音频都已经预处理为 16KHz 单声道
    wave, sr = sf.read(file_path, dtype='int16')  # 对齐pcm数据采样
    if wave.ndim > 1:
        wave = wave.mean(axis=1).astype(np.int16)
    # 转 float32，除以 32768.f
    wave_float = wave.astype(np.float32) / 32768.0   # 对齐mcu行为，转换为 float32， 归一化
    if len(wave_float) / sr < MIN_AUDIO_LEN:
        return None
    return wave_float


def load_audio_file(wav_path):
    """
    读取音频文件，返回 float32 的一维数组（采样率需与训练一致）
    """
    audio = read_wave_mcu_style_float(wav_path)
    if audio is None:
        return None
    return audio


def extract_mfcc(audio_segment):
    """
    从 3200 点音频片段提取 MFCC (18, 13)
    （必须和训练时完全一致！）
    """
    mfcc = compute_mfcc(audio_segment)  # 必须返回 (18, 13)
    assert mfcc.shape == (18, 13), f"Expected (18,13), got {mfcc.shape}"
    return mfcc.astype(np.float32)


def add_random_noise_like(length, noise_level):
    return np.random.normal(0, noise_level, length)



def sliding_windows(y, window_len=WINDOW_LENGTH, step_len=STEP_LENGTH, sr=SAMPLE_RATE):
    """
    输入音频，输出滑窗后的音频 固定200ms
    """
    target_len = int(window_len * sr)
    step = int(step_len * sr)
    segments = []

    if len(y) < target_len:
        pad_total = target_len - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        rms = np.sqrt(np.mean(y ** 2)) if np.any(y) else 1e-6
        noise_level = rms * 0.1

        left_noise = add_random_noise_like(pad_left, noise_level)
        right_noise = add_random_noise_like(pad_right, noise_level)
        seg = np.concatenate([left_noise, y, right_noise])
        segments.append(seg)
        return segments

    for start in range(0, len(y) - target_len + 1, step):
        segments.append(y[start:start + target_len])

    return segments


# ==============================
# 主函数
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Compute robust Min-Max normalization params from training data.")
    parser.add_argument("--train_list", required=True, help="训练文件列表，每行格式: /path/to/audio.wav label(0/1)")
    parser.add_argument("--target_min", type=float, default=-1.0, help="目标归一化最小值 (default: -1.0)")
    parser.add_argument("--target_max", type=float, default=1.0, help="目标归一化最大值 (default: 1.0)")
    parser.add_argument("--percentile_low", type=float, default=1.0, help="低百分位 (default: 1.0)")
    parser.add_argument("--percentile_high", type=float, default=99.0, help="高百分位 (default: 99.0)")
    parser.add_argument("--output_npy", default="norm_params.npy", help="输出 .npy 参数文件")
    parser.add_argument("--output_c_header", default="norm_params.h", help="输出 C 头文件")
    args = parser.parse_args()

    print("正在加载训练数据并提取 MFCC...")

    all_mfcc_flat = []  # 扁平化存储所有 MFCC 系数

    with open(args.train_list, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="Processing"):
        parts = line.split()
        if len(parts) < 2:
            continue
        wav_path, label = parts[0], int(parts[1])
        if not os.path.exists(wav_path):
            print(f"跳过不存在的文件: {wav_path}")
            continue

        try:
            audio = load_audio_file(wav_path)
            if audio is None:
                continue

            segments = sliding_windows(audio)
            for seg in segments:
                # 再次确保长度为 3200（防御性编程）
                if len(seg) != 3200:
                    print("滑窗处理后还是没有对齐3200点")
                    continue
                mfcc = extract_mfcc(seg)  # (18, 13)
                all_mfcc_flat.append(mfcc.flatten())

        except Exception as e:
            print(f"处理 {wav_path} 出错: {e}")

    if not all_mfcc_flat:
        raise RuntimeError("没有成功加载任何 MFCC 数据！")

    # 合并成一维数组
    all_vals = np.concatenate(all_mfcc_flat)
    print(f"共收集 {len(all_vals)} 个 MFCC 系数值")

    # 计算 robust min/max
    x_low = np.percentile(all_vals, args.percentile_low)
    x_high = np.percentile(all_vals, args.percentile_high)

    print(f"Robust range: [{x_low:.3f}, {x_high:.3f}] (percentile {args.percentile_low}-{args.percentile_high})")

    # 计算线性映射参数: x_norm = x_raw * scale + offset
    scale = (args.target_max - args.target_min) / (x_high - x_low)
    offset = args.target_min - scale * x_low

    # 保存参数到 .npy 文件
    params = {
        "method": "robust_minmax",
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
        "original_low": float(x_low),
        "original_high": float(x_high),
        "target_min": args.target_min,
        "target_max": args.target_max,
        "scale": float(scale),
        "offset": float(offset)
    }

    np.save(args.output_npy, params)
    print(f"归一化参数已保存至: {args.output_npy}")

    # 生成 C 头文件。 用于部署在mcu中，归一化mfcc特征参数
    with open(args.output_c_header, 'w') as f:
        f.write("// Auto-generated by compute_robust_norm_params.py\n")
        f.write("#ifndef NORM_PARAMS_H\n")
        f.write("#define NORM_PARAMS_H\n\n")
        f.write(f"#define NORM_SCALE  ({scale:.8f}f)\n")
        f.write(f"#define NORM_OFFSET ({offset:.8f}f)\n")
        f.write(f"#define NORM_TARGET_MIN ({args.target_min}f)\n")
        f.write(f"#define NORM_TARGET_MAX ({args.target_max}f)\n\n")
        f.write("#endif // NORM_PARAMS_H\n")
    print(f"C 头文件已生成: {args.output_c_header}")

    # 验证：随机选一个值看看是否在范围内
    test_raw = np.random.choice(all_vals)
    test_norm = test_raw * scale + offset
    print(f"\n示例验证:")
    print(f"   Raw value: {test_raw:.3f}")
    print(f"   Norm value: {test_norm:.3f} (应在 [{args.target_min}, {args.target_max}] 内)")

if __name__ == "__main__":
    main()



"""
python compute_robust_norm_params.py --train_list train.txt


(yolo) kend@kend-Guanxin:~/文档/PlatformIO/Projects/DogStopper/python/tinyml/dataset$ python compute_robust_norm_params.py --train_list train.txt
正在加载训练数据并提取 MFCC...
共收集 1650636 个 MFCC 系数值   # 由16w个数据得出的，不会出错
Robust range: [-456.162, 90.466] (percentile 1.0-99.0)
归一化参数已保存至: norm_params.npy
C 头文件已生成: norm_params.h

归一化mfcc特征
# 加载参数
params = np.load("norm_params.npy", allow_pickle=True).item()
SCALE = params["scale"]
OFFSET = params["offset"]

def normalize_mfcc(raw_mfcc):
    return raw_mfcc * SCALE + OFFSET


#include "norm_params.h"

// raw_mfcc 是 float 数组，长度 18*13
for (int i = 0; i < 18 * 13; i++) {
    float norm_val = raw_mfcc[i] * NORM_SCALE + NORM_OFFSET;
    // 可选：裁剪到 [-1, 1]
    if (norm_val > 1.0f) norm_val = 1.0f;
    else if (norm_val < -1.0f) norm_val = -1.0f;
    
    // 接着量化到 int8: int8_val = (int8_t)(norm_val * 127.0f);
}

"""