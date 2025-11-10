#ifndef HANN_FRAME_H
#define HANN_FRAME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================== 参数宏 ==================
#define FRAME_SIZE   400   // 帧长
#define FRAME_SHIFT  160   // 帧移


// ================== 接口 ==================

// 初始化汉宁窗（只需调用一次）后续复用，节省计算, 不对外暴露
// void hann_init(void);


// 对音频分帧加窗
// audio_in: 输入 PCM float
// audio_out: 输出缓冲区（可以与 audio_in 相同）， 由上层决定存放在哪里
// num_samples: 输入采样点数
// 返回帧数
int frames_win(const float *audio_in, float *audio_out, int num_samples);


#ifdef __cplusplus
}
#endif

#endif // HANN_FRAME_H
