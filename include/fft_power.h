#ifndef FFT_POWER_H
#define FFT_POWER_H

#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif


/**
 * 初始化 FFT 功能（分配 twiddles、复数缓冲）
 * @param nfft FFT 长度（必须为 2 的幂）
 */
void fft_power_init(int nfft);



// 修改：12-12： 新增int字段， 控制是mfcc计算使用，还是logmel计算使用

/**
 * 计算帧信号的功率谱（使用已初始化的缓冲/twiddles）
 * @param frames 输入窗后帧信号，按帧顺序排列，长度 = num_frames * frame_size  18 * 400
 * @param num_frames 帧数  分了多少帧，200ms就是18帧， 因为需要对每一帧做fft
 * @param frame_size 帧大小  400点一个帧
 * @param nfft FFT长度（初始化时指定， 保持一致）  512 多的补0
 * @param logmel_flag 0: mfcc 1: logmel
 * @param power_out 输出功率谱缓存，长度 = num_frames * (nfft/2 + 1)  18 * 257
 * @return 0 成功， 非0 失败
 */
int fft_power_compute(const float *frames_in, int num_frames, int frame_size, int nfft, int logmel_flag, float *power_out);


// 释放资源
void fft_power_free(void);


#ifdef __cplusplus
}
#endif

#endif // FFT_POWER_H
