/*
     2025 © Kend.tank
    *  Audio Feature Extraction for ESP32-S3 (Arduino Framework)
    *  支持两种特征提取：
    *    ① Log-Mel 特征提取
    *    ② MFCC 特征提取（基于 DCT-II）
    *  输入: 3200 个采样点 (固定 200ms @ 16kHz；16k * 0.2s = 3200 samples)
    *  输出:
    *    - Log-Mel: 18 x 40
    *    - MFCC: 18 x 13  (默认取前13维)
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
    *  7. DCT-II -> MFCC 特征
    *  优化说明：
    *    - 使用 ESP-DSP 库加速 FFT
    *    - 所有大数组建议放置在 PSRAM
    *    - 针对不同的特征提取需求，设计独立函数，低耦合高内聚
    *  * #TODO: 待功能一致性测试通过后重新封装功能
*/


#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>


// ====================== 配置参数 ======================
#define SAMPLE_RATE     16000      // 音频采样率 16kHz
#define INPUT_SAMPLES   3200      // 输入样本数
#define FRAME_LENGTH    400       // 分帧窗口长度25ms
#define FRAME_SHIFT     160       // 步长10ms
#define N_FFT           512       // FFT 窗口长度 (必须是2的幂次方，且 >= FRAME_LENGTH)
#define N_MEL_BINS      40        // Mel 滤波器数量
#define NUM_FRAMES      18        // 输出帧数
#define N_MFCC          40         // MFCC系数数量 与 Python 训练时一致（根据资源情况来调整）

// -------------------- 可选宏--------------------
// 若启用，尽量将大数组分配到 PSRAM（默认启用）
#ifndef USE_PSRAM_BUFFERS
#define USE_PSRAM_BUFFERS 1
#endif

// 是否预计算 DCT 矩阵（减少资源开销）
#ifndef PRECOMPUTE_DCT
#define PRECOMPUTE_DCT 1
#endif

// 调试功能开关
#ifndef DEBUG_FEATURES
#define DEBUG_FEATURES 1
#endif

// -------------------- 状态/结果尺寸 --------------------
#define LOGMEL_SIZE     (NUM_FRAMES * N_MEL_BINS)
#define MFCC_SIZE       (NUM_FRAMES * N_MFCC)




// ====================== 函数接口 ======================

/**
 * @brief 初始化特征提取模块
 *
 * 初始化特征提取所需的各种资源，包括FFT模块、窗口函数、滤波器组等
 *
 * @return 0表示成功，负数表示失败
 */
int feature_extractor_init(void);


/**
 * @brief 释放特征提取模块资源
 *
 * 释放初始化时分配的所有资源
 */
void feature_extractor_free(void);


/**
 * @brief 计算200ms音频的Log-Mel特征
 *
 * 对输入的200ms音频计算Log-Mel特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点）
 * @param output: 输出Log-Mel特征
 * @return 成功返回帧数，失败返回负数
 */
int compute_logmel_200ms(const int16_t *input, float *output);


/**
 * @brief 计算200ms音频的MFCC特征
 *
 * 对输入的200ms音频计算MFCC特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点）
 * @param output: 输出MFCC特征
 * @return 成功返回帧数，失败返回负数
 */
int compute_mfcc_200ms(const int16_t *input, float *output);


/**
 * @brief 计算DCT-II变换
 *
 * 计算输入信号的DCT-II变换，用于MFCC计算
 *
 * @param input: 输入信号
 * @param output: 输出DCT系数
 * @param n_input: 输入信号长度
 * @param n_output: 输出系数数量
 */
void compute_dct_ii(const float *input, float *output, int n_input, int n_output);


/**
 * @brief 打印内存使用信息
 *
 * 显示堆内存和PSRAM的使用情况
 */
void print_memory_info(void);


#ifdef __cplusplus
}
#endif


#endif // AUDIO_FEATURES_H