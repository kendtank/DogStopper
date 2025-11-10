"""
py端实现的MFCC特征提取和logmel特征提取，用于验证MCU端实现的结果
"""


import numpy as np
from scipy.signal import get_window
from scipy.fftpack import dct


def extract_logmel_and_mfcc(
    y,
    sr,
    n_fft=512,
    hop_length=256,
    n_mels=64,
    n_mfcc=13,
    fmin=20.0,
    fmax=None,
    window='hann',
    amin=1e-10,
    top_db=80.0
):
    """
    手动实现 Log-Mel 和 MFCC 特征提取（无 librosa）
    
    参数:
        y: 1D 音频信号 (float32)
        sr: 采样率
        n_fft: FFT 点数
        hop_length: 帧移
        n_mels: Mel 滤波器数量
        n_mfcc: MFCC 系数数量
        fmin/fmax: Mel 滤波器频率范围
        window: 窗函数类型
        amin: 避免除零的最小值
        top_db: 动态范围上限（dB）
    
    返回:
        log_mel: (n_mels, T) 相对 dB 谱
        mfcc: (n_mfcc, T) MFCC 特征
    """
    # 1. 降噪


    # 2. 分帧
    frame_len = n_fft
    num_frames = 1 + (len(y) - frame_len) // hop_length
    if num_frames <= 0:
        raise ValueError("Audio too short for given n_fft and hop_length")
    
    frames = np.array([
        y[i * hop_length : i * hop_length + frame_len]
        for i in range(num_frames)
    ])  # (T, n_fft)

    # 3. 加窗
    win = get_window(window, frame_len)
    frames = frames * win  # 广播

    # 4. FFT
    spectrum = np.fft.rfft(frames, n=n_fft)  # (T, n_fft//2 + 1)
    power = np.abs(spectrum) ** 2  # 功率谱

    # 5. 构建 Mel 滤波器组
    if fmax is None:
        fmax = sr / 2.0
    mel_filters = _create_mel_filterbanks(sr, n_fft, n_mels, fmin, fmax)

    # 6. 应用 Mel 滤波器 → Mel 功率谱
    mel_power = np.dot(power, mel_filters.T)  # (T, n_mels)

    # 7. 转换为 log-Mel (dB)，并相对化
    log_mel = 10.0 * np.log10(np.maximum(mel_power, amin))  # (T, n_mels)
    log_mel = log_mel - np.max(log_mel)  # 相对 dB：最大值为 0
    log_mel = np.clip(log_mel, -top_db, 0.0)  # 限制动态范围

    # 转置为 (n_mels, T) 以匹配常见格式
    log_mel = log_mel.T  # (n_mels, T)

    # 8. 计算 MFCC via DCT-II (ortho)
    # 对每一帧（列）做 DCT，即沿 axis=0
    mfcc = dct(log_mel.T, type=2, norm='ortho', axis=1)[:, :n_mfcc].T  # (n_mfcc, T)

    return log_mel.astype(np.float32), mfcc.astype(np.float32)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _create_mel_filterbanks(sr, n_fft, n_mels, fmin, fmax):
    """创建 Mel 滤波器组 (n_mels, n_fft//2 + 1)"""
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2.0, n_freqs)

    # Mel 频率点
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = _mel_to_hz(mels)

    # 映射到 FFT bins
    bin_idx = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_freqs - 1)

    # 构建三角滤波器
    filters = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        left = bin_idx[m - 1]
        center = bin_idx[m]
        right = bin_idx[m + 1]

        # 左斜坡
        if center > left:
            filters[m - 1, left:center] = (freqs[left:center] - hz_points[m - 1]) / (hz_points[m] - hz_points[m - 1])
        # 右斜坡
        if right > center:
            filters[m - 1, center:right] = (hz_points[m + 1] - freqs[center:right]) / (hz_points[m + 1] - hz_points[m])

    # 归一化（可选，但推荐）
    enorm = 2.0 / (hz_points[2:n_mels+2] - hz_points[:n_mels])
    filters *= enorm[:, np.newaxis]

    return filters



if __name__ == '__main__':
    