#ifndef AUDIO_INPUT_H
#define AUDIO_INPUT_H

/**
 * 音频输入生产者模块（ESP32-S3）
 * 作者: Kend.tank
 * 日期: 2025.11.19
 * 更新：
 *  取消filter_mutex互斥锁，因为只有生产者会使用到dc滤波器，这里取消减少资源开销
 *  取消模块内部创建队列，又外部统一创建，初始化，并做参数的传递
 *  注意：模块详细说明请查看audio_input.cpp
 */


#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
// #include "freertos/queue.h"

#ifdef __cplusplus
extern "C" {
#endif


// =================== 配置宏（麦克风 + I2S驱动配置） ===================
// 麦克风相关配置
#define MIC_I2S_PORT I2S_NUM_0         // I2S端口号
#define MIC_SAMPLE_RATE 16000          // 采样率 (Hz)
#define MIC_BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_16BIT  // 每个样本位数
#define MIC_CHANNEL_FORMAT I2S_CHANNEL_FMT_ONLY_LEFT   // 通道格式（单声道左）
#define MIC_COMM_FORMAT I2S_COMM_FORMAT_I2S           // 通信格式
#define MIC_DMA_BUF_COUNT 8            // DMA缓冲数量
#define MIC_DMA_BUF_LEN 64             // DMA缓冲长度（样本）
#define MIC_USE_APLL true              // 使用APLL时钟（精确采样）
#define MIC_PIN_BCK 4                  // BCK引脚
#define MIC_PIN_WS 5                   // WS引脚
#define MIC_PIN_DATA_IN 18             // 数据输入引脚
#define MIC_PIN_DATA_OUT -1            // 数据输出引脚（RX模式，无需）


// =================== 内部配置 ===================
#define TEMP_BUFFER_SAMPLES 256        // 批量大小（PCM块）每次从DMA读取的样本数，i2s_read(), 意思就是说dma  buffer 64 * 4 读满了一半，就取走。设置512，就全部满了一次取走，延迟大点，cpu轮询次数降低.  256点音频 就是   256 / 16000  = 16ms 的音频数据
#define PREHEAT_SAMPLES 1024           // 预热样本数 开机时用来让DC滤波器稳定
#define FILTER_A_FLOAT 0.995f          // IIR系数 一阶DC滤波系数
#define USE_FIXED_POINT 1              // 0: float计算DC滤波，1: Q15定点计算
#define DROP_LOG_INTERVAL 10          // 队列满丢帧日志节流，每10次打印一次

// 队列配置
#define AUDIO_QUEUE_DEPTH 20           // 队列深度（批次）  TEMP_BUFFER_SAMPLES * AUDIO_QUEUE_DEPTH = 5120 个样本，320ms音频， 256 * 2 bytes * 20 = 10240 bytes ≈ 10KB


// 声明队列（不分配内存，不初始化， 模块外部创建）
// extern QueueHandle_t audio_queue;  // PCM批次队列（item: int16_t[TEMP_BUFFER_SAMPLES]）在main函数中初始化，模块中只做句柄获取


// 公共API
bool audio_input_init(void);  // 初始化I2S + 滤波


void audio_input_task(void* param);  // 生产者任务 // param = QueueHandle_t


// 其他API
// 
uint64_t audio_get_dropped_samples(void);
void audio_reset_dropped_counter(void);
void audio_set_dc_filter_enabled(bool enabled);
void audio_force_repreheat(void);



#ifdef __cplusplus
}
#endif


#endif