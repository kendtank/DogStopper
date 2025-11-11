#include "hann_frame.h"
#include <math.h>
#include <string.h>


// ================== 静态数据 不放在栈中 ==================
static float hann_win[FRAME_SIZE];
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
int frames_win(const float *audio_in, float *audio_out, int num_samples){
    if(!hann_initialized){
        hann_init();
    }

    if(num_samples < FRAME_SIZE) {
        return 0;
    }

    int num_frames = (num_samples - FRAME_SIZE) / FRAME_SHIFT + 1;

    for(int f=0; f<num_frames; f++){
        const float *frame_in = audio_in + f * FRAME_SHIFT;
        // 输出分帧后的数据到新的缓冲区
        float *frame_out = audio_out + f * FRAME_SIZE; // 这里需要帧长步进

        for(int n=0; n<FRAME_SIZE; n++){
            frame_out[n] = frame_in[n] * hann_win[n];
        }
    }
    return num_frames;
}
