#ifndef VERIFY_EMBEDDING_H
#define VERIFY_EMBEDDING_H

/*
 * ============================================================
 * verify_embedding — Bark 事件消费 & Embedding 推理入口模块
 * ============================================================
 *
 * 模块定位：
 *   - bark_event 的【消费者入口】
 *   - 负责从 BarkEvent PCM 中提取特征并推理 embedding
 *   - 根据系统运行状态，将 embedding 分流给：
 *       a) 验证模块（声纹验证）
 *       b) 自学习模块（自动聚类 / 模版更新）
 * 
 *
 * 核心流程：
 *   1. 从 bark_queue 中阻塞式获取 BarkEvent（200ms PCM）
 *   2. 对 PCM 进行 logmel 特征提取
 *   3. 调用 embedding 模型进行推理，得到 embedding 向量
 *   4. 根据系统状态：
 *        - 是否已存在稳定模版？
 *        - 是否处于学习阶段？
 *      决定：
 *        - 是否执行声纹验证
 *        - 是否将 embedding 送入自学习模块
 *
 * 解耦原则（非常重要）：
 *   本模块【只负责推理与调度】，不关心以下细节：
 *     - 聚类 / 自学习算法如何实现
 *     - 模版如何更新（mean / EMA / 权重）
 *     - Flash 如何持久化、是否成功
 *
 * 这些能力必须由独立模块实现：
 *   - learning_core   （自学习 / 聚类 / 模版统计）
 *   - template_storage（Flash 持久化 / 上电恢复）
 *
 * 设计目标：
 *   - 实时、稳定、低耦合
 *   - 作为 Bark → 声纹系统的唯一入口
 *   - 后续可无侵入地扩展验证策略或学习策略
 *
 * ============================================================
 */

#include <stdbool.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "bark_detector.h"   // BarkEvent队列消费者
#include "tiny_model.h"      // logmel & embedding 推理接口

#ifdef __cplusplus
extern "C" {
#endif


// 初始化声纹验证模块，由于这是最后一层消费者，不需要再push到任何队列，不需要绑定外部队列句柄
bool verify_embedding_init();

// 推理embedding
bool tinyml_embedding_inference(int16_t* pcm_samples, int16_t len, float* embed_out);


// 声纹验证模块处理队列中的狗吠事件, 上层从队列bark_queue中获取BarkEvent结构体，传结构体地址
void verify_embedding_process(BarkEvent *event);


#ifdef __cplusplus
}
#endif

#endif // VERIFY_EMBEDDING_H
