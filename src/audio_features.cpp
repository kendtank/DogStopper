/*
 * audio_features.cpp
 *
 * 2025 © Kend.tank
 *
 * 功能：Log-Mel + MFCC 特征提取
 * 平台：ESP32-S3 + Arduino
 */

#include "audio_features.h"
#include <Arduino.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_dsp.h"
#include "hann_frame.h"
#include "fft_power.h"
#include "mel_filterbank.h"
#include "mfcc.h"


// 顶部只定义指针，不占用内存
static float *fp_pcm_src_mfcc = NULL;   // 存放原始PCM数据   
static float *frames_src_mfcc = NULL;   // 存放窗后的帧信号
static float *power_out_mfcc = NULL;    // 存放功率谱
static float *log_mel_out_all_mfcc = NULL;  // 存放log-mel特征

static float *fp_pcm_src_logmel = NULL;   // 存放原始PCM数据   
static float *frames_src_logmel = NULL;   // 存放窗后的帧信号
static float *power_out_logmel = NULL;    // 存放功率谱



// 初始化中间的buffer函数
bool init_feature_buffers(void) {

    fft_power_init(N_FFT);  // 放到初始化函数中

    // mfcc需要使用到的buffer
    fp_pcm_src_mfcc = (float*) heap_caps_malloc(INPUT_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);  // 实时音频输入缓存 (放PSRAM) 3200点 ≈ 12.8KB
    frames_src_mfcc = (float*) heap_caps_malloc(NUM_FRAMES * FRAME_LENGTH * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);   // 存放全量窗后信号 7200点  30KB  PSRAM
    power_out_mfcc  = (float*) heap_caps_malloc(NUM_FRAMES * (N_FFT/2 + 1) * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);  // 18 * 257 点  PSRAM
    log_mel_out_all_mfcc  = (float*) heap_caps_malloc(NUM_FRAMES * N_MEL_BINS * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);  // 18 * 64 点  PSRAM
    // mfcc_out_all  = (float*) heap_caps_malloc(NUM_FRAMES * N_MFCC * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);  // 18 * 13 点  PSRAM
    
    // logmel需要使用到的buffer
    fp_pcm_src_logmel = (float*) heap_caps_malloc(INPUT_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);  // psram
    frames_src_logmel = (float*) heap_caps_malloc(NUM_FRAMES * FRAME_LENGTH * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);  // psram
    power_out_logmel  = (float*) heap_caps_malloc(NUM_FRAMES * (N_FFT/2 + 1) * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);  // psram


    if (!fp_pcm_src_mfcc || !frames_src_mfcc || !power_out_mfcc || !log_mel_out_all_mfcc || !fp_pcm_src_logmel || !frames_src_logmel || !power_out_logmel ) {

        return false;  // 分配失败
    }

    //清理分配的空间， 避免内存碎片化
    memset(fp_pcm_src_mfcc, 0, INPUT_SAMPLES * sizeof(float));
    memset(frames_src_mfcc, 0, NUM_FRAMES * FRAME_LENGTH * sizeof(float));
    memset(power_out_mfcc, 0, NUM_FRAMES * (N_FFT/2 + 1) * sizeof(float));
    memset(log_mel_out_all_mfcc, 0, NUM_FRAMES * N_MEL_BINS * sizeof(float));
    memset(fp_pcm_src_logmel, 0, INPUT_SAMPLES * sizeof(float));
    memset(frames_src_logmel, 0, NUM_FRAMES * FRAME_LENGTH * sizeof(float));
    memset(power_out_logmel, 0, NUM_FRAMES * (N_FFT/2 + 1) * sizeof(float));

    return true;
}


// 释放中间的buffer函数， 实时分析不需要释放
void free_feature_buffers(void) {
    if (fp_pcm_src_mfcc) { heap_caps_free(fp_pcm_src_mfcc); fp_pcm_src_mfcc = NULL; }
    if (frames_src_mfcc) { heap_caps_free(frames_src_mfcc); frames_src_mfcc = NULL; }
    if (power_out_mfcc)  { heap_caps_free(power_out_mfcc);  power_out_mfcc  = NULL; }
    if (log_mel_out_all_mfcc) { heap_caps_free(log_mel_out_all_mfcc); log_mel_out_all_mfcc = NULL; }
    if (fp_pcm_src_logmel) { heap_caps_free(fp_pcm_src_logmel); fp_pcm_src_logmel = NULL; }
    if (frames_src_logmel) { heap_caps_free(frames_src_logmel); frames_src_logmel = NULL; }
    if (power_out_logmel) { heap_caps_free(power_out_logmel); power_out_logmel = NULL; }
    fft_power_free();
}



// 函数接口的实现


// -------------------- Log-Mel main --------------------
/**
 * @brief 计算200ms音频的Log-Mel特征
 *
 * 对输入的200ms音频计算Log-Mel特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点）
 * @param output: 输出Log-Mel特征
 * 
 * @return 成功返回帧数，失败返回负数
 */
int compute_logmel_200ms(const int16_t *input, float *output, int logmel_flag) {

    // 计算mfcc使用的logmel提取
    if (logmel_flag == 0)
    {
        // 1.PCM 数据转浮点数  避免异常数据，虽然int16_t 数据范围 [-32768, 32767]，但可能存在异常数据
        for (int i = 0; i < INPUT_SAMPLES; i++) {
            float sample = input[i];
            // 注意：直接转 float 不会溢出，但可能有非法大值
            if (sample > 32767.0f) sample = 32767.0f;
            else if (sample < -32768.0f) sample = -32768.0f;
            fp_pcm_src_mfcc[i] = sample / 32768.0f;
        }

        // 2.对输入帧加汉宁窗，注意：汉宁窗的初始化已经在hann_frame.h中定义调用就会自动初始化
        (void)frames_win(fp_pcm_src_mfcc, frames_src_mfcc, INPUT_SAMPLES);

        // 3.FFT变换得到功率谱
        int ret = fft_power_compute(frames_src_mfcc,   // 输入窗后帧信号（psRAM）
                            NUM_FRAMES,          // 一共多少帧
                            FRAME_LENGTH,          // 每帧点数
                            N_FFT,                // FFT长度（初始化时相同）
                            logmel_flag,
                            power_out_mfcc);          // 输出功率谱（PSRAM）

        if (ret != 0) {
            fft_power_free();
        return -1;
        }

        // 4.功率谱乘以滤波器矩阵， 获得logmel 特征
        // 18帧遍历计算
        for (int i = 0; i < NUM_FRAMES; i++) {
            // const float* p_frame = power_out + i * int(N_FFT/2 + 1);  // 拿出遍历索引的这一帧的所有功率谱数据
            // float* mel_frame_out = log_mel_out_all + i * N_MEL_BINS;  // 存一帧对应的log_mel特征, 一帧64点

            // // 单帧 log_Mel 特征计算
            // apply_log_mel(p_frame, mel_frame_out);   // 输入logmel特征矩阵在 log_mel_out_all数组中
            apply_log_mel(power_out_mfcc + i*(N_FFT/2+1), output + i*N_MEL_BINS);
        }

    }
    else if (logmel_flag == 1)
    {
        for (int i = 0; i < INPUT_SAMPLES; i++) {
            float sample = input[i];
            if (sample > 32767.0f) sample = 32767.0f;
            else if (sample < -32768.0f) sample = -32768.0f;
            fp_pcm_src_logmel[i] = sample / 32768.0f;
        }
        (void)frames_win(fp_pcm_src_logmel, frames_src_logmel, INPUT_SAMPLES);

        int ret = fft_power_compute(frames_src_logmel,   // 输入窗后帧信号（psRAM）
                            NUM_FRAMES,          // 一共多少帧
                            FRAME_LENGTH,          // 每帧点数
                            N_FFT,                // FFT长度（初始化时相同）
                            logmel_flag,
                            power_out_logmel);          // 输出功率谱（PSRAM）

        if (ret != 0) {
            fft_power_free();
        return -1;
        }
        // 18帧遍历计算
        for (int i = 0; i < NUM_FRAMES; i++) {
            // // 单帧 log_Mel 特征计算
            apply_log_mel(power_out_logmel + i*(N_FFT/2+1), output + i*N_MEL_BINS);
        }
    }
    // 传入其他值报错
    else
    {
        return -1;
    }
    return NUM_FRAMES;

}



// -------------------- MFCC main --------------------
/**
 * @brief 计算200ms音频的MFCC特征
 *
 * 对输入的200ms音频计算MFCC特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点）
 * @param output: 输出MFCC特征
 * @return 成功返回帧数，失败返回负数
 */
int compute_mfcc_200ms(const int16_t *input, float *output){
    
    // 调试：内存与计时
    // print_memory_info();
    // uint32_t start_mfcc_time1 = start_timer();
    // 对得到的logmel特征做一次DCT-II变换，取前13维
    int num_frames =  compute_logmel_200ms(input, log_mel_out_all_mfcc, 0);
    if (num_frames != NUM_FRAMES){
        return -1;
    }
    // 13维MFCC特征提取
    for (int i = 0; i < NUM_FRAMES; i++) {
        const float *logmel_frame = log_mel_out_all_mfcc + i * N_MEL_BINS; // 输入：log-Mel
        float *mfcc_frame = output + i * N_MFCC;                      // 直接写到用户输出， 取消中间的缓存
        compute_mfcc(logmel_frame, mfcc_frame);
    }
    return num_frames;
}
