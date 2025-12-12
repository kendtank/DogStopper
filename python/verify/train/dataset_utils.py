# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/11 16:08
@Author  : Kend
@FileName: dataset_utils.py
@Software: PyCharm
@modifier:
"""


# 负责：读取 WAV -> 滑窗 -> compute_logmel -> 按 speaker 构建样本库 -> N x M batch 生成器


import os
import random
from typing import List, Dict, Tuple
import numpy as np
import soundfile as sf
import librosa
from scipy.signal.windows import get_window
from scipy.fftpack import fft

# ----- 全局参数（可调整） -----
SR = 16000
WIN_MS = 25      # 帧长 25ms (STFT window)
HOP_MS = 10      # 帧移 10ms
FRAME_SIZE = int(SR * WIN_MS / 1000)   # 400
HOP = int(SR * HOP_MS / 1000)         # 160
N_FFT = 512
NUM_BINS = N_FFT // 2 + 1
N_MELS = 40

SEG_MS = 200     # 用于后续滑窗切片的片段长度（200ms）
SEG_HOP_MS = 50  # 片段滑动步长（50ms）
SEG_SAMPLES = int(SR * SEG_MS / 1000)  # 3200
SEG_HOP = int(SR * SEG_HOP_MS / 1000)  # 800

# ---------------- I/O helpers ----------------
def read_wave_mcu_style_float(wav_path: str, target_sr=SR) -> np.ndarray:
    wave, sr = sf.read(wav_path, dtype='int16')
    wave = wave.astype(np.float32)
    if wave.ndim == 2:
        wave = np.mean(wave, axis=1).astype(np.float32)
    wave = wave / 32768.0
    if sr != target_sr:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    assert sr == target_sr, f"sampling rate must be {target_sr}"
    return wave.astype(np.float32)

# ---------------- sliding windows for 200ms windows with wrap padding ----------------
def sliding_windows_200ms_50ms(wave: np.ndarray,
                               win_samples: int = SEG_SAMPLES,
                               hop_samples: int = SEG_HOP) -> List[np.ndarray]:
    L = len(wave)
    frames = []
    start = 0
    while True:
        end = start + win_samples
        if end <= L:
            frames.append(wave[start:end])
        else:
            seg = wave[start:L]
            need = win_samples - len(seg)
            pad = wave[:need] if need > 0 else np.array([], dtype=wave.dtype)
            frames.append(np.concatenate([seg, pad]))
            break
        start += hop_samples
        if start >= L:
            break
    return frames  # list of 1D np arrays length win_samples

# ---------------- log-mel (matching MCU pipeline) ----------------
def compute_logmel(
    wave: np.ndarray,
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
    return S_db.astype(np.float32), S_db.T.astype(np.float32)  # (n_mels, T), (T, n_mels)

# ---------------- build per-speaker sample bank ----------------
def build_speaker_bank(dataset_root: str,
                       allow_ext=('.WAV', '.wav', '.Wave', '.wave')) -> Dict[str, List[np.ndarray]]:
    """
    Returns dict: speaker_id -> list of segments (raw waveform, each length SEG_SAMPLES)
    """
    bank = {}
    class_dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
    for pid in class_dirs:
        full = os.path.join(dataset_root, pid)
        all_wavs = [f for f in os.listdir(full) if f.endswith(allow_ext)]
        segments = []
        for w in all_wavs:
            path = os.path.join(full, w)
            wave = read_wave_mcu_style_float(path)
            segs = sliding_windows_200ms_50ms(wave)
            segments.extend(segs)
        if len(segments) > 0:
            bank[pid] = segments
    return bank

# ---------------- N x M batch sampler (balanced by speakers) ----------------
def n_m_batch_generator(speaker_bank: Dict[str, List[np.ndarray]],
                        N: int = 8, M: int = 4,
                        infinite: bool = True):
    """
    Yields batches: (batch_wave_array, speaker_ids)
    - batch_wave_array: shape (N*M, SEG_SAMPLES)
    - speaker_ids: list length N (the selected speaker ids in order)
    NOTE: each speaker must have >= M segments
    """
    speakers = [k for k, v in speaker_bank.items() if len(v) >= M]
    if len(speakers) < N:
        raise ValueError(f"Not enough speakers with >=M samples. have {len(speakers)}, need {N}")
    while True:
        chosen = random.sample(speakers, N)
        batch_waves = []
        for s in chosen:
            segs = random.sample(speaker_bank[s], M)
            batch_waves.extend(segs)
        # stack
        batch_arr = np.stack(batch_waves, axis=0).astype(np.float32)  # (N*M, SEG_SAMPLES)
        yield batch_arr, chosen
        if not infinite:
            break

# ---------------- helper to convert wave batch -> logmel tensors ----------------
def batch_wave_to_logmel(batch_wave: np.ndarray) -> np.ndarray:
    """
    batch_wave: (B, SEG_SAMPLES)
    return: (B, time_steps, n_mels, 1)
    """
    B = batch_wave.shape[0]
    out = []
    for i in range(B):
        s_db, t = compute_logmel(batch_wave[i])
        # compute_logmel returns (n_mels, time), we want (time, n_mels)
        out.append(t)  # (time, n_mels)
    arr = np.stack(out, axis=0)  # (B, time, n_mels)
    arr = arr.astype(np.float32)
    # ensure shape (B, time, n_mels, 1)
    return np.expand_dims(arr, axis=-1)
