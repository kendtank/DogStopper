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
static float *fp_pcm_src = NULL;   // 存放原始PCM数据   
static float *frames_src = NULL;   // 存放窗后的帧信号
static float *power_out = NULL;    // 存放功率谱
static float *log_mel_out_all = NULL;  // 存放log-mel特征
static float *mfcc_out_all = NULL;    // 存放MFCC特征



#if DEBUG_FEATURES

/**
 * @brief 打印堆内存和PSRAM使用情况
 */
static void print_memory_info(void) {
    Serial.printf("[MEM] 堆内存剩余: %d bytes\n", ESP.getFreeHeap());

    if (psramFound()) {
        Serial.printf("[MEM] PSRAM总大小: %d bytes\n", ESP.getPsramSize());
        Serial.printf("[MEM] PSRAM可用: %d bytes\n", ESP.getFreePsram());
    }
}


/**
 * @brief 开始计时
 * @return 返回 micros() 时间戳
 */
static inline uint32_t start_timer() {
    return micros();
}


/**
 * @brief 打印函数执行耗时
 * @param func_name 函数名称
 * @param start_time 调用 start_timer() 的返回值
 */
static void print_time_cost(const char* func_name, uint32_t start_time) {
    uint32_t elapsed_us = micros() - start_time;
    Serial.printf("[TIME] %s 执行耗时: %lu us (%.3f ms)\n",
                  func_name, elapsed_us, elapsed_us / 1000.0f);
}


// 由于算法在静音端，一直有3db的误差，使用一样的信号，执行算法，看看误差如何
int compute_logmel_from_float(const float *input_normalized, float *output) {
    print_memory_info();
    uint32_t start_time = start_timer();
    // 因为 frames_win 应该只读输入
    #if DEBUG_FEATURES
        int num_frames = frames_win(input_normalized, frames_src, INPUT_SAMPLES);
        Serial.printf("frames_win: %d frames\n", num_frames);
    #else
        (void)frames_win(input_normalized, frames_src, INPUT_SAMPLES);
    #endif

    // ... 后续 FFT、Mel 不变 ...
    fft_power_init(N_FFT);
    int ret = fft_power_compute(frames_src, NUM_FRAMES, FRAME_LENGTH, N_FFT, power_out);
    if (ret != 0) {
        #if DEBUG_FEATURES
            Serial.printf("fft_power_compute failed, ret=%d\n", ret);
        #endif
        fft_power_free();
        return -1;
    }

    for (int i = 0; i < NUM_FRAMES; i++) {
        apply_log_mel(power_out + i*(N_FFT/2+1), output + i*N_MEL_BINS);
    }

    print_memory_info();
    print_time_cost("compute_logmel_from_float", start_time);
    return NUM_FRAMES;
}


#else  // 调试关闭为0，完全不生成任何函数或代码

// 定义空宏，调用处完全不生成代码
#define print_memory_info()        do {} while(0)
#define start_timer()              (0)
#define print_time_cost(name, t)   do {} while(0)

#endif



// 初始化中间的buffer函数
bool init_feature_buffers(void) {
    fp_pcm_src = (float*) heap_caps_malloc(INPUT_SAMPLES * sizeof(float), MALLOC_CAP_INTERNAL);  // 实时音频输入缓存 (放RAM) 3200点 ≈ 12.8KB
    frames_src = (float*) heap_caps_malloc(NUM_FRAMES * FRAME_LENGTH * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);   // 存放全量窗后信号 7200点  30KB  PSRAM
    power_out  = (float*) heap_caps_malloc(NUM_FRAMES * (N_FFT/2 + 1) * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);  // 18 * 257 点  PSRAM
    log_mel_out_all  = (float*) heap_caps_malloc(NUM_FRAMES * N_MEL_BINS * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);  // 18 * 64 点  PSRAM
    mfcc_out_all  = (float*) heap_caps_malloc(NUM_FRAMES * N_MFCC * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);  // 18 * 13 点  PSRAM

    if (!fp_pcm_src || !frames_src || !power_out || !log_mel_out_all || !mfcc_out_all) {
        #if DEBUG_FEATURES
            Serial.println("Error: Failed to allocate feature buffers!");
        #endif
        
        return false;  // 分配失败
    }

    //清理分配的空间， 避免内存碎片化
    memset(fp_pcm_src, 0, INPUT_SAMPLES * sizeof(float));
    memset(frames_src, 0, NUM_FRAMES * FRAME_LENGTH * sizeof(float));
    memset(power_out, 0, NUM_FRAMES * (N_FFT/2 + 1) * sizeof(float));
    memset(log_mel_out_all, 0, NUM_FRAMES * N_MEL_BINS * sizeof(float));
    memset(mfcc_out_all, 0, NUM_FRAMES * N_MFCC * sizeof(float));

    return true;
}

// 释放中间的buffer函数， 实时分析不需要释放
void free_feature_buffers(void) {
    if (fp_pcm_src) { heap_caps_free(fp_pcm_src); fp_pcm_src = NULL; }
    if (frames_src) { heap_caps_free(frames_src); frames_src = NULL; }
    if (power_out)  { heap_caps_free(power_out);  power_out  = NULL; }
    if (log_mel_out_all) { heap_caps_free(log_mel_out_all); log_mel_out_all = NULL; }
    if (mfcc_out_all) { heap_caps_free(mfcc_out_all); mfcc_out_all = NULL; }
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
 * @return 成功返回帧数，失败返回负数
 */
int compute_logmel_200ms(const int16_t *input, float *output) {

    // 调试内存使用情况和时间
    // #ifdef DEBUG_FEATURES  只判断是不是定义了，不管值, 调试宏为0，这里默认是0， 不会报错
    print_memory_info();
    uint32_t start_time = start_timer();

    // 1.PCM 数据转浮点数
    for (int i = 0; i < INPUT_SAMPLES; i++) {
        fp_pcm_src[i] = (float)input[i] / 32768.0f;  // 归一化到 [-1, 1]
    }

    // 2.对输入帧加汉宁窗，注意：汉宁窗的初始化已经在hann_frame.h中定义调用就会自动初始化
    #if DEBUG_FEATURES
        int num_frames = frames_win(fp_pcm_src, frames_src, INPUT_SAMPLES);
        Serial.printf("frames_win: %d frames\n", num_frames);
    #else
        (void)frames_win(fp_pcm_src, frames_src, INPUT_SAMPLES);
    #endif

    // 3.FFT变换得到功率谱
    fft_power_init(N_FFT);  // TODO 考虑是否放到初始化函数中
    int ret = fft_power_compute(frames_src,   // 输入窗后帧信号（RAM）
                        NUM_FRAMES,          // 一共多少帧
                        FRAME_LENGTH,          // 每帧点数
                        N_FFT,                // FFT长度（初始化时相同）
                        power_out);          // 输出功率谱（RAM）

    if (ret != 0) {
        #if DEBUG_FEATURES
        Serial.printf("fft_power_compute failed, ret=%d\n", ret);
        #endif
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
        apply_log_mel(power_out + i*(N_FFT/2+1), output + i*N_MEL_BINS);
    }

    // 打印内存和时间开销
    print_memory_info();
    print_time_cost("compute_logmel_200ms", start_time);

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
    // 对得到的logmel特征做一次DCT-II变换，取前13维
    // int num_frame =  compute_logmel_200ms(input, log_mel_out_all);
    // if (num_frame != NUM_FRAMES){
    //     #if DEBUG_FEATURES
    //     Serial.printf("compute_logmel_200ms failed, ret=%d\n", num_frame);
    //     #endif
    //     return -1;
    // }
    // print_memory_info();
    // uint32_t start_mfcc_time = start_timer();
    // // 13维MFCC特征提取
    // for (int i = 0; i < NUM_FRAMES; i++) {
    //     const float* logmel_frame = logmel_frame + i * N_MEL_BINS;  // 指向第i帧的log-Mel
    //     float* mfcc_frame = mfcc_out_all + i * N_MFCC;    // 指向第i帧的MFCC输出
        
    //     compute_mfcc(logmel_frame, mfcc_frame);  // 正确调用
    // }

    return 0;
}
