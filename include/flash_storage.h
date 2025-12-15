#ifndef FLASH_STORAGE_H
#define FLASH_STORAGE_H

/*
 * ============================================================
 * flash_storage — MCU Embedding 持久化与自学习触发模块
 * ============================================================
 *
 * 模块定位：
 *   - 负责存储每次推理得到的 embedding 到 Flash
 *   - 维护 batch 计数和总计数
 *   - 根据 batch 触发自学习模块 (learning_core)
 *   - 控制模版建立完成标记
 *
 * 核心功能：
 *   1. flash_save_embedding()：存储 embedding 并管理状态机
 *   2. template_storage_is_ready()：查询模版是否建立完成
 *   3. reset_storage()：重置计数和模版状态，用于调试或重新学习
 *
 * 数据结构：
 *   - MAX_BATCH：每 batch 的 embedding 数量
 *   - MAX_TOTAL：总计数，达到后冻结学习
 *   - template_ready：模版是否建立完成
 *
 * ============================================================
 * 
 * 来一个 embedding，就存一个

不关心 batch / 学习 / 成功失败

只提供：存、取、状态快照
 */
#include <Arduino.h>


#ifdef __cplusplus
extern "C" {
#endif


#define EMBED_OUTPUT_SIZE 32      // 每个 embedding 长度


// 状态机结构体，这里做存储flash，读取flash到内存，修改在聚类端。
typedef struct {
    uint32_t batch_embed_counter;       // 当前 batch 已存 embedding 数
    uint32_t total_embed_counter;    // 总 embedding 数
    bool template_ready;     // 第一次聚类ok就建立好了模版，可以在上层做声纹验证了，模版是否建立完成
    bool close_learning;      // 冻结学习， 达到 MAX_TOTAL 后， 就不再学习了， 也不存 embedding 了， 不更新模版。
} FlashState;

// 状态机
extern FlashState flash_state;

/* 生命周期 */
bool flash_storage_init(void);
bool flash_save_state(void);

/* embedding 日志 */
bool flash_save_embedding(const float* embed);
bool flash_read_embedding(uint32_t index, float* out_embed);

/* template */
bool flash_storage_save_template(const void* data, size_t size);
bool flash_storage_load_template(void* data, size_t size);

// 提供一个重置的接口，用于调试和重新学习
bool reset_storage(void);


#ifdef __cplusplus
}
#endif

#endif // FLASH_STORAGE_H