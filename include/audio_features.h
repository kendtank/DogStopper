/*
     2025 © Kend.tank szgxzl
    *  Audio Feature Extraction for ESP32-S3 (Arduino Framework)
    *  支持两种特征提取：
    *    ① Log-Mel 特征提取
    *    ② MFCC 特征提取（基于 DCT-II）
    *  输入: 3200 个采样点 (固定 200ms @ 16kHz；16k * 0.2s = 3200 samples)
    *  输出:
    *    - Log-Mel: 18 x 64 （后续视情况可以调整为40）
    *    - MFCC: 18 x 13  (默认取前13维)
    *  适用于 ESP32S3，Arduino 框架
    *  使用DSP库进行FFT和矩阵乘法(向量点积)加速计算
    *  DSP fft库使用参考：https://github.com/espressif/esp-dsp/blob/master/examples/fft/main/dsps_fft_main.c | 
    *  算法流程：
    *  1. 输入 3200 个采样点
    *  2. 声音分帧：每帧 400 点，帧移 160 点，共 18 帧（25ms窗口， 10ms步长）
    *  3. 每帧乘汉宁窗：对每帧乘以一个窗函数，减少边缘频谱泄露。w[n]=0.54−0.46cos(2πn/(N−1))
    *  4. 每帧做快速傅里叶变换 (FFT)，将时域转为频域， 计算功率谱
    *  5. 构建 Mel 滤波器组（中心频率在 Mel 频率尺度上均匀分布），（）使用py端预生成的滤波矩阵， 程序开始预存入PSRAM中，加速）乘功率谱求和得到每帧能量谱（mel特征）模拟人耳感知
    *  6. 对每帧能量取对数 → log-mel 特征， （log-mel特征转db特征， 需要主要下限值与py保持一致， 否则低音的bin数值会不一致， 导致tinyml模型推理和训练不一致）
    *  7. DCT-II（py预生成矩阵） -> MFCC 特征
    *  优化说明：
    *    - 使用 ESP-DSP 库加速 FFT
    *    - 所有大数组建议放置在 PSRAM
    *    - 针对不同的特征提取需求，设计独立函数，低耦合高内聚
    * # TODO: 待测试封装后的功能一致性与py新封装的函数
    * # TODO：功率计算， 性能还可以使用 DSP库的dsps_bit_rev_lookup_fc32进行更高性能的计算
    * # TODO: 滤波的计算未使用DSP库的点积加速
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
#define INPUT_SAMPLES   3200      // 输入样本数  200ms的窗口对应3200个采样点
#define FRAME_LENGTH    400       // 分帧窗口长度25ms
#define FRAME_SHIFT     160       // 步长10ms
#define N_FFT           512       // FFT 窗口长度 (必须是2的幂次方，且 >= FRAME_LENGTH)
#define N_MEL_BINS      64        // Mel 滤波器数量
#define NUM_FRAMES      18        // 输出帧数（3200点音频被分了多少帧）
#define N_MFCC          13         // MFCC系数数量 与 Python 训练时一致（根据资源情况来调整）



// 调试功能开关
#ifndef DEBUG_FEATURES
#define DEBUG_FEATURES 1
#endif


// -------------------- 状态/结果尺寸 --------------------
#define LOGMEL_SIZE     (NUM_FRAMES * N_MEL_BINS)
#define MFCC_SIZE       (NUM_FRAMES * N_MFCC)



// ====================== 功能接口 ======================


/**
 * @brief 提取200ms音频的Log-Mel特征
 *
 * 对输入的200ms音频计算Log-Mel特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点） 麦克风 / ADC 输出	int16_t	原始PCM
 * @param output: 输出Log-Mel特征矩阵
 * @return 成功返回帧数，失败返回负数
 */
int compute_logmel_200ms(const int16_t *input, float *output);



/**
 * @brief 提取200ms音频的MFCC特征
 *
 * 对输入的200ms音频计算MFCC特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点）
 * @param output: 输出MFCC特征矩阵
 * @return 成功返回帧数，失败返回负数
 */
int compute_mfcc_200ms(const int16_t *input, float *output);




#ifdef __cplusplus
}
#endif


#endif // AUDIO_FEATURES_H