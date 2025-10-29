#ifndef LOGMEL_EXTRACTOR_H
#define LOGMEL_EXTRACTOR_H

#ifdef __cplusplus
extern "C" {
#endif

// 配置参数 
#define SR 16000
#define NFFT 512  // ESP32S3 FFT 库对长度 NFFT 有要求，2^N，400 不是 2^N，官方库有时候会断言失败， 改为512
#define HOP 128   // NFFT/4，保持帧重叠
#define NFFT_BINS (NFFT/2 +1)
#define N_MEL 40  // Mel 滤波器数量

// 控制是否循环打印
// #define LOOP_PRINT_LOGMEL 1

// 分段计算 Log-Mel
// audio: 输入 PCM float 单声道数组
// num_samples: 输入样本数
// logmel_out: 输出缓冲 (size >= max_frames * N_MEL)
// max_frames: 输出缓冲可存储最大帧数
// 返回实际帧数或负数错误码
int compute_logmel(const float *audio, int num_samples, float *logmel_out, int max_frames);


#ifdef __cplusplus
}
#endif

#endif // LOGMEL_EXTRACTOR_H
