/**
 * 
 * VAD 消费端：从生产者队列 audio_queue 读取 PCM 流数据(每次一个block)，进行狗吠粗筛。
 * 
 * 作者: Kend.tank
 * 日期：2025-11-22
 * 
 * 算法核心原理：
 * ---------------------------------------------------------
 * 1. MCU 上无法一直做tinyml推理，因此此处做“高召回”粗筛：
 *    - 短时能量（狗吠会在 0.12～0.5s 内能量快速上升）
 *    - 零交叉率（狗吠是富含高频的爆破声，ZCR 明显变高）
 * 所以：高能量 + 高 ZCR ≈ 可能是狗吠（或其他爆破音）
 *
 * 2. 采用 EMA 平滑能量，减少误触发
 *    ema = alpha * ema + (1-alpha) * energy
 *    适用于 MCU，计算量小、响应快
 *
 * 3. 设计 VAD 内部事件缓存，合并碎片：
 *    - 狗吠常常只有 120~500ms
 *    - 滑窗的流设计会碎片化狗吠，因此必须设计缓存和逻辑，保证事件的狗吠完整。
 *
 * 4. 事件结束时，前后自动扩展 padding，确保狗吠尽可能的不会被剪掉
 *
 * 数据流与数据结构：
 *  - 从生产者队列（每次一个 block）读取 PCM 批次（每 block = 256 samples）
 *  - 使用滑窗（10 blocks = 160ms）+ 步长（3 blocks = 48ms）做粗筛（短时能量 + ZCR）
 *  - 事件缓存（最多 32 blocks ≈ 512ms），合并跨窗的碎片，保证传给 TinyML 的是连续/完整的片段
 *  - 前/后 padding（各 3 blocks）用于补齐边缘
 *  - 输出队列项为结构体 TinyMLEvent（避免 malloc/指针、无内存泄漏）
 * 
 * 设计要点：
 *  - 单个接口 vad_consumer_process_block 每次接收一个 block；内部做滑窗填充、滑动、判定；
 *  - 当滑窗判定为 "命中（可能狗吠）" 时：
 *      如果不是 in_event -> 拼接 prev_stride + 整个窗口（事件开始）
 *      如果已经 in_event -> 只拼接窗口中新出现的尾部 STRIDE 数据（避免重复）
 *  - 当 in_event 且连续 EVENT_END_NOHIT 次滑窗未命中时，认为事件结束 -> 拼接前/后 padding 后送队列
 *  - 事件超长（超过 EVENT_MAX_SAMPLES + 2*PAD_SAMPLES）时强制切断并推送
 * 
 * 使用：
 *  - 在系统初始化时调用 vad_consumer_init(&ctx)
 *  - 主循环或消费者任务中从 audio_queue 的队列取 block 后调用 vad_consumer_process_block(&ctx, block)
 *  - TinyML 消费者通过 xQueueReceive(tinyml_mfcc_queue, &evt, ...) 获取事件（evt.samples / evt.length）
 * 
 * -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 * 核心设计要点与规则（总结）：
 * 生产者（audio_input）每次传入一个 block（固定 BLOCK_SAMPLES=256）。
 * VAD 使用滑窗：WINDOW_BLOCKS=10（窗 = 10 * 256 = 2560 samples ≈160ms），步长 STEP_BLOCKS=3（步 = 3 * 256 = 768 samples ≈48ms）。窗口采用重叠，处理流程为：填满窗口 → 判定 → 滑动（把窗口向左移动 STRIDE_SAMPLES）→ 继续接收块。
 * 前/后 padding（pre_pad / post_pad）：各 PAD_SAMPLES = PAD_BLOCKS * BLOCK_SAMPLES，用于在事件检测时把前后各 3 block 合并到事件头尾保证狗吠尽可能完整。
 * pre_pad 更新规则：仅在 当前不在事件中 且 当前 window 判定为 hit（is_bark） 时，用上一次输出的 stride（即上次滑出队列的那 3 block）更新 pre_pad。这样保证 pre_pad 是紧邻事件前方、且不会重复。
 * post_pad 更新规则：仅在 当前在事件中 且 当前 window 判定为 non-hit 并且是第一次 non-hit（nohit_count==1） 时，用当前滑窗即将出队的 stride 更新 post_pad（因为该 stride 就是事件后的第一个非狗吠块），最终在下一个窗口是狗吠时，拼接两个狗吠窗，不是狗吠，直接作为post_padding。
 * event_buf 用于累计当前事件音频样本（只追加新增部分，避免重复）。当 event_buf 达到 EVENT_OUTPUT_MAX_SAMPLES 时会强制 push；当事件中连续 EVENT_END_NOHIT（=2）个窗口都非 hit 时认为事件结束并 push。
 * 队列使用 TinyMLEvent 结构体（包含 samples[] 及 length），没有 malloc指针；队列深度由宏 EVENT_QUEUE_DEPTH 控制。队列满时覆盖最旧事件（保证新事件优先）。
 * -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 *
 * 更新：
 *  事件拼接不在push时候完成，在event缓冲的时候就做好了pre_padding和第一次狗吠的全窗拼接
 *  取消post_padding, 因为第一次非吠叫，直接把cur_stride_buf拼接到事件中，后续只有两种可能：1.下次继续非吠叫，直接不处理， push。2.下次是吠叫，直接继续使用cur_stride_buf拼接到event_buff.(节省了2kb)
 *  FreeRTOS 支持把队列数据放在 PSRAM， 消费者内部不在管理tinyml_mfcc_queue队列, (放在psram)， 事件结构体还是放在RAM中
 */

#ifndef AUDIO_CONSUMER_H
#define AUDIO_CONSUMER_H

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"


#ifdef __cplusplus
extern "C" {
#endif


// 注意：生产者队列长度是20， 每个block块是256个采样点，一共大小是20*256=5120点，大小10kb


// =================== 配置宏 ===================

// VAD 滑窗长度和步长（样本数），16kHz 采样
#define BLOCK_SAMPLES        256     // 每个block块是256个采样点(与生产者保持一致)
#define WINDOW_BLOCKS        10      // 10 blocks = 160ms
#define STEP_BLOCKS          3       // 每次滑动 3 blocks    48ms
#define MIC_RATE             16000   // 16kHz

#define WIN_SAMPLES          (WINDOW_BLOCKS * BLOCK_SAMPLES)  
#define STRIDE_SAMPLES       (STEP_BLOCKS * BLOCK_SAMPLES)
#define OVERLAP_SAMPLES       (WIN_SAMPLES - STRIDE_SAMPLES)  // 7个block块的滑动

// padding(前后各 PAD_SAMPLES），用来在事件前后补偿
#define PAD_BLOCKS           3        // 填充的大小，也是步长，对齐这一块
#define PAD_SAMPLES          (PAD_BLOCKS * BLOCK_SAMPLES)   // 256  * 3 = 768点, 用于填充事件前后的样本数，也是用于步长

// 最大事件长度：32 blocks = 32 * 256 = 8192 512ms （狗吠一般不会超过这个时间）
#define EVENT_MAX_BLOCKS     26    // 32-6 = 26
#define EVENT_MAX_SAMPLES    (EVENT_MAX_BLOCKS * BLOCK_SAMPLES)   // 缓存狗吠事件的最大长度    26个block
/* 事件输出缓冲的最大总样本 = pre_pad + event_buf + post_pad */
#define EVENT_OUTPUT_MAX_SAMPLES (PAD_SAMPLES + EVENT_MAX_SAMPLES + PAD_SAMPLES)


#define EVENT_END_NOHIT      2   // 连续两次未命中就结束事件  也就是说滑动的6个block 基本上确认无狗吠， 间隔96ms

// tinyML 队列深度（队列指针）
#define QUEUE_DEPTH          10   // tinyml 队列深度, 设置为 10，即最多10个候选片段， 尽可能牺牲内存换取数据不丢(26*256*int16 * 10) 130kb  （TODO: RAM?PSRAM?）

// 默认判定阈值（运行时调整）
#define DEFAULT_ENERGY_THRESHOLD  150000.0f
#define DEFAULT_ZCR_THRESHOLD     260.0f
#define DEFAULT_EMA_ALPHA          0.8f


// =================== TinyML 事件结构（队列项） ===================
// 事件以结构体完整拷贝到队列中（取消使用指针，无 malloc， 使用FreeRTOS中的队列）
typedef struct {
    int16_t samples[EVENT_OUTPUT_MAX_SAMPLES]; // PCM 数据 (整合 pre+event+post)
    int length;                         // 实际有效样本数
    uint32_t timestamp_ms;   // 放入队列的时间，单位 ms
} TinyMLEvent;


// =======================================
// VAD 上下文结构体
// =======================================
typedef struct {
    QueueHandle_t tinyml_queue;  // 外部创建的给mfcc推理的消费队列句柄，初始化时绑定

    // ---------------- 临时事件缓冲 ----------------
    TinyMLEvent tmp_event;  // 放在RAM中，避免栈溢出

    // ---------------- 滑窗 ----------------
    int16_t window_buf[WIN_SAMPLES];   // 滑动窗口， 直接对接生产队列， 保证不阻塞生产队列， 用于计算短时能量和 ZCR
    int win_fill;        // 当前窗口已填充多少样本数


    // 这在窗口判定为命中且非 in_event 时，被作为 pre_pad 用来补前部
    int16_t prev_stride_buf[STRIDE_SAMPLES];  // 上一次窗口出队的3 block， 用于前padding
    bool prev_stride_valid;  // 窗口判定为命中且非 in_event 时，才更新pre_padding, 用于event_buf的实时拼接判断

    // 当前滑窗将要滑出的 stride（上一个窗和当前窗的差异部分，用于 post_pad 填充）
    int16_t cur_stride_buf[STRIDE_SAMPLES];

    // ---------------- padding ----------------
    int16_t pre_pad[PAD_SAMPLES];   // 前填充缓冲区
    // int16_t post_pad[PAD_SAMPLES];  // 后填充缓冲区  取消
    
    // ---------------- 事件缓冲区 ----------------
    int16_t event_buf[EVENT_OUTPUT_MAX_SAMPLES];       // 候选片段缓存，用于发送 tinyml 队列， 最大500ms， 狗吠不会超过500ms，也是为了确保狗吠尽可能的连续
    int event_pos;       // 当前事件的长度

    // 状态机
    bool in_event;                 // 是否在事件中（true: 事件正在累积中）
    int nohit_count;               // 事件中的连续未命中的次数，超过指定次数，则事件结束，push
    
    // 全局阈值/EMA（可在运行时调）
    float ema_energy;
    float ema_alpha;
    float energy_threshold;
    float zcr_threshold;
} VADContext;


// 暴露 tinyml_queue 队列（消费者用xQueueReceive消费）main函数中初始化，模块中只获取句柄
// extern QueueHandle_t tinyml_mfcc_queue;  // vad检测出来的候选片段， 初始化函数绑定



// 初始化 VAD 消费端， tinyml_queue由模块外部管理，定义，这里只传递句柄
bool vad_consumer_init(VADContext* ctx, QueueHandle_t tinyml_queue);

// 处理流式音频的入口： MCU 生产者队列是 单 block 出队，每次轮询拿到一个 block后调用。
void vad_consumer_process_block(VADContext* ctx, const int16_t* new_block);



/*
 * 动态调阈（可在运行时调参）
 */
void vad_set_energy_threshold(VADContext* ctx, float thr);

void vad_set_zcr_threshold(VADContext* ctx, float thr);

void vad_set_ema_alpha(VADContext* ctx, float alpha);




#ifdef __cplusplus
}
#endif



#endif