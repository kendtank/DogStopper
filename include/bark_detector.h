/*
* bark_detector — 事件级精剪方案
*
* 作者：Kend
* 时间：2025.11.26
*
* 模块目标：
* - 从 tinyml_mfcc_queue 队列获取可能的狗吠事件 PCM 数据
* - 对事件进行滑窗分段推理，识别每声狗吠
* - 输出干净的 200ms PCM 给验证模块（可重复，但保证不漏）
*
 核心架构说明：
* 1. 队列输入：
*    - 队列中每条事件 PCM 已是可能狗吠片段，长度不一致
* 2. 事件长度分类：
*    - 150ms <= len < 200ms：
*       → 前填充到 200ms → 推理 → 高概率直接输出 → 组装 BarkEvent
*    - len >= 200ms：滑窗推理
*       a) 窗长 200ms
*       b) 步长 50ms
*       c) 高概率窗：直接生成 BarkEvent → push → 清空缓冲区
*       d) 中概率窗：存入缓冲区，连续中概率窗合并 → 找中心窗 → 精剪 200ms → push
*           → 清空第一个已处理窗，保留第二个窗继续聚合
*       e) 低概率窗：丢弃，并清空缓冲区
*       f) 尾窗：不足 200ms → 前填充到 200ms → 推理 → 按高/中/低概率处理
* 3. 输出：
*    - 每条 BarkEvent PCM 长度固定 200ms
*    - 可重复、可能有重叠，但保证不漏狗吠
*
* 特点：
* - 高概率窗即时输出 → 快速响应制止
* - 中概率连续窗聚合 → 减少误判、保证连续狗吠捕获
* - 尾窗前填充 → 保证末尾吠声不漏
*
*/

#ifndef BARK_DETECTOR_H
#define BARK_DETECTOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "audio_consumer.h"  // tinyml_mfcc_queue队列生产者
#include "tiny_model.h"     // tinyml的mfcc模型

#ifdef __cplusplus
extern "C" {
#endif


#define BARK_WIN_LEN        3200    // 每窗 PCM 样本数（200ms @16kHz）
// #define BARK_STRIDE         800    // 滑窗步长（50ms）  可以改为100ms，减少计算量。
#define BARK_STRIDE       1600    // 滑窗步长（100ms）
#define BARK_LOW            2400   // 150ms @16kHz 0.15 * 16kHz = 2400
#define BARK_HIGH_THRESHOLD 0.8f    // 高概率阈值   大于 0.8 的窗直接输出
#define BARK_LOW_THRESHOLD  0.65f    // 中概率阈值，用于聚合   大于 0.6 的窗加入缓冲区， 低于0.6直接丢弃
#define GAIN_THRESHOLD      0.3f    // 增益补偿，前期模型训练的效果差， 需补偿，后续可以设置为0， 就是在推理出来的概率加这个值
#define BARK_QUEUE_DEPTH    20      // bark 事件队列深度    BARK_QUEUE_DEPTH * BarkEvent  大小约 = 20 *  (3200 * 2B) = 128000B = 125KB
#define BARK_MAX_K     2      // 一个vad事件最多输出 K 个窗（后续可以调整）



// ==========================================
// 生产者队列事件结构（bark事件）， 用于push到验证模块队列中（bark_queue）
// ==========================================
typedef struct {
    int16_t samples[BARK_WIN_LEN];              // PCM 数据 (对齐验证模型长度)
    int     length;                             // 实际有效样本数应该等于 BARK_WIN_LEN
    uint32_t timestamp_ms;                      // vad算法检测到的时间，单位 ms， 用于打印事件延时
} BarkEvent;                                    // BarkEvent拷贝入队，不使用指针


// bark_detector 模块初始化， 包含传递bark_queue队列句柄,  队列在主函数中创建
bool bark_detector_init(QueueHandle_t bark_queue);


// bark_detector 模块处理事件, 队列tinyml_mfcc_queue中获取的结构体，传结构体地址
void bark_detector_process_event(TinyMLEvent* event);




#ifdef __cplusplus
}
#endif



#endif // BARK_DETECTOR_H