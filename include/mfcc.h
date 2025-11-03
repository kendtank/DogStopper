#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==================================================
// 参数宏定义（保持与 Python 一致）
// ==================================================
#define FRAME_SIZE   400
#define FRAME_SHIFT  160
#define N_MEL        128
#define N_MFCC       40
#define SAMPLE_RATE  16000
#define NFFT_BINS    (FRAME_SIZE / 2 + 1)

#ifndef NUM_MFCC
#define NUM_MFCC N_MFCC
#endif
#ifndef NUM_MEL_FILTERS
#define NUM_MEL_FILTERS N_MEL
#endif

// ==================================================
// 函数声明
// ==================================================

/**
 * @brief 计算 MFCC 特征
 * @param audio 输入 PCM（float，16kHz 单声道）
 * @param num_samples 输入采样点数
 * @param mfcc_out 输出缓存，大小至少为 max_frames * N_MFCC
 * @param max_frames 最大帧数
 * @return 实际帧数，<0 表示错误
 */
int compute_mfcc(const float *audio, int num_samples, float *mfcc_out, int max_frames);

// ==================================================
// 可选：暴露中间步骤（用于 MCU 上实时打印或验证）
// ==================================================
void make_hann(float *win);  // Hann 窗
void build_mel_filters(float W[N_MEL][NFFT_BINS]);  // Mel 滤波器
void rfft_power_400(const float *x, float *P);  // 实数 FFT 功率谱
void dct_ortho_1x_double(const double *x, float *c); // DCT-II(ortho)

// 可选辅助函数（仅用于验证）
float hz_to_mel(float f);
float mel_to_hz(float m);

#ifdef __cplusplus
}
#endif

#endif  // MFCC_H
