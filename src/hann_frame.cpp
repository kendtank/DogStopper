#include "hann_frame.h"
#include <math.h>
#include <string.h>
#include "esp_dsp.h"
// #include "Arduino.h"


// ================== 静态数据 不放在栈中 ==================
static float hann_win[FRAME_SIZE];   // 只读汉宁窗权重
static int hann_initialized = 0;


// ================== 汉宁窗初始化 只需要计算一次，后续一直存在静态数据中 ==================
static void hann_init(void){
    if(!hann_initialized){
        for(int n=0; n<FRAME_SIZE; n++)
            hann_win[n] = 0.5f - 0.5f * cosf(2.0f * M_PI * n / FRAME_SIZE);   // 测试与py端无误差
        hann_initialized = 1;
    }
}


// ================== 分帧加窗 ==================
// 耗时：3200 sample 耗时 952 us
// int frames_win(const float *audio_in, float *audio_out, int num_samples){
//     if(!hann_initialized){
//         hann_init();
//     }

//     if(num_samples < FRAME_SIZE) {
//         return 0;
//     }
//     uint32_t start_us = micros();

//     int num_frames = (num_samples - FRAME_SIZE) / FRAME_SHIFT + 1;

//     for(int f=0; f<num_frames; f++){
//         const float *frame_in = audio_in + f * FRAME_SHIFT;
//         // 输出分帧后的数据到新的缓冲区
//         float *frame_out = audio_out + f * FRAME_SIZE; // 这里需要帧长步进

//         for(int n=0; n<FRAME_SIZE; n++){
//             frame_out[n] = frame_in[n] * hann_win[n];
//         }
//     }
//     uint32_t cost_us = micros() - start_us;

//     Serial.print("compute hann cost time (us): ");
//     Serial.println(cost_us);

//     return num_frames;
// }


// 优化：1204， 使用dsp库加速       3200 sample 耗时 627 us
int frames_win(const float *audio_in, float *audio_out, int num_samples){
    if(!hann_initialized){
        hann_init();
    }
    // printf("dsps_mul_f32_ae32_enabled = %d\n", dsps_mul_f32_ae32_enabled);  true

    if(num_samples < FRAME_SIZE) {
        return 0;
    }
    int num_frames = (num_samples - FRAME_SIZE) / FRAME_SHIFT + 1;

    for(int f=0; f<num_frames; f++){
        const float *frame_in = audio_in + f * FRAME_SHIFT;
        float *frame_out = audio_out + f * FRAME_SIZE;
        // 使用 DSP 加速
        dsps_mul_f32(frame_in, hann_win, frame_out, FRAME_SIZE, 1, 1, 1);
    }
    return num_frames;
}

/*
dsps_mul_f32(x, y, result, length, x_step, y_step, res_step);
x: 输入数组 1（frame_in）
y: 输入数组 2（hann_win）
result: 输出数组
length: 数据长度（FRAME_SIZE）
x_step / y_step / res_step: 步长 = 1（连续内存）
*/