/*
     2025 © Kend.tank
    *  Log-Mel 音频特征提取 固定200ms (16k * 0.2s = 3200 samples)
    *  输入: 3200 个采样点
    *  输出: 18 x 40 的 log-mel 特征矩阵
    *  适用于 ESP32S3，Arduino 框架
    *  使用DSP库进行FFT加速计算
    *  DSP库使用参考：https://github.com/espressif/esp-dsp/blob/master/examples/fft/main/dsps_fft_main.c
    *  算法流程：
    *  1. 输入 3200 个采样点
    *  2. 声音分帧：每帧 400 点，帧移 160 点，共 18 帧（25ms窗口， 10ms步长）
    *  3. 每帧乘汉宁窗：对每帧乘以一个窗函数，减少边缘频谱泄露。w[n]=0.54−0.46cos(2πn/(N−1))
    *  4. 每帧做快速傅里叶变换 (FFT)，将时域转为频域， 计算功率谱
    *  5. 构建 Mel 滤波器组（中心频率在 Mel 频率尺度上均匀分布），乘功率谱求和得到每帧能量谱（mel特征）模拟人耳感知
    *  6. 对每帧能量取对数 → log-mel 特征
    *  7. 输出 18 x 40 的 log-mel 特征矩阵
    *  8. 按照tinyml设计，判断在端侧是否需要进行矩阵展开
    *  注意：所有大数组均分配在 PSRAM，避免栈溢出
*/

#ifndef COMPUTE_LOGMEL_H
#define COMPUTE_LOGMEL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// ====================== 配置参数 ======================
#define SAMPLE_RATE     16000      // 音频采样率 16kHz
#define FRAME_LENGTH    400       // 分帧窗口长度25ms
#define FRAME_SHIFT     160       // 步长10ms
#define N_FFT           512       // FFT 窗口长度 (必须是2的幂次方，且 >= FRAME_LENGTH)
#define N_MEL_BINS      40        // Mel 滤波器数量
#define INPUT_SAMPLES   3200      // 输入样本数
#define NUM_FRAMES      18        // 输出帧数


// ====================== 函数接口 ======================
/**
 * @brief  计算固定 200ms 的 log-mel 特征
 * @param  input     输入波形 (长度 = 3200)  mcu音频数据点原生就是 int16_t
 * @param  output    输出 log-mel 特征 (shape = 18 x 40)  float
 */
void compute_logmel_200ms(const int16_t *input, float *output);


#ifdef __cplusplus
}
#endif


#endif // COMPUTE_LOGMEL_H
