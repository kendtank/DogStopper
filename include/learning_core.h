#ifndef LEARNING_CORE_H
#define LEARNING_CORE_H

/* 
 * 自学习算法： 上层模块从生产者队列 bark_event_queue 读取 PCM 事件数据(每次一个event)，执行声纹验证，改模块自动进行声纹模板的自我学习与建立。
 * 作者: Kend.tank
 * 日期：2025-12-12
 * 
 * 声纹验证模块
    * 基于 embedding 模型进行声纹模板的自我学习与建立
    * 通过聚类筛选核心 embedding，剔除噪声
    * 使用指数移动平均（EMA）更新声纹模板
    * 使用余弦相似度进行声纹验证
    * 提供接口获取当前模板及进行验证
    * 
 * 模版建立流程：
    在 MCU 端对狗吠 embedding 进行事件级自学习：通过批量一致性聚类自动生成声纹模板，并使用指数滑动平均逐步稳定模板，最终在样本充分后冻结，实现无需人工标注的个体声纹建模。
    1. 收集 embedding：每次 bark 事件触发时，收集对应的 32 维 embedding。
    2. 批量聚类筛选核心样本：将收集到的 embedding 按批次（默认 10 个）进行聚类，计算批次中心，并筛选出与中心相似度高于阈值（0.6）的核心样本，剔除噪声， 并使用TOP K 约束簇的质量，保证模板质量。
    3. 模板初始化：首次生成模板时，直接使用第一个批次的中心（mean）作为初始模板。
    4. 指数移动平均更新模板：后续每个批次的中心通过指数移动平均（EMA）与当前模板融合，逐步稳定模板。学习率根据已完成批次数动态调整。
    5. 冻结模板：当累计处理的批次数达到最大值（默认20 个批次）后，冻结模板，不再继续学习。
 * 优化点：TODO: 连续N次相似度 < 0.5 认为环境发生变化清空模板，重新学习。（目前先不考虑）
 */

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================== 参数配置 ================== */

// embedding 维度（训练保持一致）
#define EMBED_DIM               32
// 每 batch 的样本数（默认 10）
#define EMBED_BATCH_SIZE        10
// 最大 batch 数（20 batch → 200 次后冻结）
#define MAX_BATCH_NUM         10
// 相似度阈值（用于核心簇筛选, 保证簇的聚类质量）
#define EMBED_CORE_SIM_TH       0.5f  // 根据测试调整
// 至少需要的核心样本数（保证每个batch本身的质量）
#define EMBED_CORE_TOPK         3
// 模板更新相似度阈值防止漂移
#define TEMPLATE_UPDATE_SIM_TH  0.5f

#define EMA_ALPHA_START  0.5f    // 前 1～2 次
#define EMA_ALPHA_MIN    0.1f   // 后期最小更新率
#define EMA_DECAY_STEP   0.05f   // 每次 batch 衰减
// #define MAX_BATCH_NUM    20      // 20 次后冻结



typedef enum {
    LEARN_SKIP,
    LEARN_FAIL,
    LEARN_SUCCESS,
    LEARN_FROZEN
} LearnResult;



/* ================== 生命周期管理 ================== */

// 初始化（上电调用）（flash_storage_init 之后调用）
void learning_core_init(void);

// 重置（清空模板，重新学习）， 注意不要轻易使用。另外：TODO: 需要在flash备份一个模版，后续考虑
// void learning_core_reset(void);

// 实时获取的狗吠特征，返回与模版的相似度分数  没有模版返回 -2.0f
float learning_core_calc_similarity(const float* embed);

/* 自学习 */
LearnResult learning_core_try_learn(uint32_t batch_size);


#ifdef __cplusplus
}
#endif

#endif // LEARNING_CORE_H