# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/5 16:44
@Author  : Kend
@FileName: compute_logmel
@Software: PyCharm
@modifier:
"""

import numpy as np
import librosa
from scipy.signal.windows import get_window
from scipy.fftpack import dct
from scipy.fftpack import fft
import soundfile as sf


def compute_mfcc(
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


def read_wave_mcu_style_float(wav_path):
    # 用 soundfile 直接读 int16 PCM， 前提是音频都已经预处理为 16KHz 单声道
    wave, _ = sf.read(wav_path, dtype='int16')  # 对齐pcm数据采样
    # 取前 3200 点（对应 200ms @16kHz）
    wave_window = wave[:3200]
    # 转 float32，除以 32768.f
    wave_float = wave_window.astype(np.float32) / 32768.0   # 对齐mcu行为，转换为 float32， 归一化
    print(f"波形数组读取成功: shape={wave_float.shape}, dtype={wave_float.dtype}")
    return wave_float