#include "audio_consumer.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <esp_log.h>
#include "esp_heap_caps.h"
#include <Arduino.h>


static const char* TAG = "VAD_MODULE";

static TinyMLEvent discard_event_buf; // 专门用来接收被丢弃的队列元素

// 定义全局生产到mfcc的队列
// QueueHandle_t tinyml_mfcc_queue = NULL;



// 采样float已经不会溢出
static inline float compute_energy(const int16_t *buf, int len) {
    // 短时平方能量的均值，避免做 log，计算量小
    float s = 0.0f;
    for (int i = 0; i < len; ++i) {
        float v = (float)buf[i];
        s += v * v;
    }
    return (len > 0) ? (s / (float)len) : 0.0f;
}

static inline int compute_zcr(const int16_t *buf, int len) {
    int z = 0;
    for (int i = 1; i < len; ++i) {
        if ((buf[i-1] >= 0 && buf[i] < 0) || (buf[i-1] < 0 && buf[i] >= 0)) ++z;
    }
    return z;
}


// 动态阈值设置， 后续如果提供api，远程修改参数。这里不做进一步处理，只提供api
void vad_set_energy_threshold(VADContext* ctx, float thr) { if (ctx) ctx->energy_threshold = thr; }

void vad_set_zcr_threshold(VADContext* ctx, float thr) { if (ctx) ctx->zcr_threshold = thr; }

void vad_set_ema_alpha(VADContext* ctx, float alpha) { if (ctx) ctx->ema_alpha = alpha; }




// 把当前 event_buf 推送(已经组装好，这里只做推送)
static void assemble_and_push_event(VADContext *ctx) {
    if (ctx->event_pos <= 0) {
        // nothing
        ctx->in_event = false;
        ctx->nohit_count = 0;
        ctx->event_pos = 0;
        // ctx->prev_stride_valid = false;
        return;
    }

    // push进入队列的临时buffer放在RAM（注意：嵌入MCU要保证栈足够）
    // 使用 VADContext 中的 tmp_event，避免栈溢出
    // 解释这行代码：TinyMLEvent *ev; 声明一个指针变量，告诉编译器：“我有一个指向 TinyMLEvent 类型的指针 ev”, 只是声明
    // ev = &ctx->tmp_event; 这里做了 指针赋值，让 ev 指向 VADContext 中已经分配好的 tmp_event， 这里在初始化中已经开辟了
    TinyMLEvent *ev = &ctx->tmp_event;
    // 所以这里只是让这个指针*ev指向了ctx->tmp_event的地址：&ctx->tmp_event


    // 二次防呆，裁剪长度
    int len = (ctx->event_pos > EVENT_OUTPUT_MAX_SAMPLES) ? EVENT_OUTPUT_MAX_SAMPLES : ctx->event_pos;

    // ram->ram
    memcpy(ev->samples, ctx->event_buf, len * sizeof(int16_t));
    ev->length = len;
    ev->timestamp_ms = millis();

    // 准备push到队列
    if (!ctx->tinyml_queue) return;

    BaseType_t ret = xQueueSend(ctx->tinyml_queue, ev, 0);   // 非阻塞发送
    if (ret != pdTRUE) {
        // 队列满了， 直接覆盖旧的数据 注意：是丢弃，不会造成内存泄漏
        // 修改：1125
        // 队列已满，丢弃最旧事件再发送
        // TinyMLEvent dummy;
        // xQueueReceive(ctx->tinyml_queue, &dummy, 0);  // 非阻塞接收
        Serial.println("[Consumer] tinyml_queue full, dropping sample...");

        // 更新：1126 不在栈上做取出
        xQueueReceive(ctx->tinyml_queue, &discard_event_buf, 0); // 丢弃最旧事件
        // xQueueSend(ctx->tinyml_queue, ev, portMAX_DELAY);  // 再发送 阻塞的，需要等队列腾出空间
        if (xQueueSend(ctx->tinyml_queue, ev, 0) != pdTRUE) {
            ESP_LOGE(TAG, "tinyml_queue still full after discard!"); // 再次发送失败，打印错误
        }
        // xQueueOverwrite(ctx->tinyml_queue, ev);
        ESP_LOGW(TAG, "tinyml_queue full -> overwrite newest event (len=%d)", ev->length);
    }

    // 清除状态机
    ctx->event_pos = 0;
    ctx->in_event = false;
    ctx->nohit_count = 0;
    // prev_stride_valid再每次窗口满，就会更新状态机，这里不需要重复处理
    // ctx->prev_stride_valid = false;
}



// 推送当前 event
void vad_force_push_event(VADContext* ctx) {
    if (!ctx) return;
    if (ctx->event_pos > 0) {
        assemble_and_push_event(ctx);
    }
}


// ------------------ VAD 窗口判定 ------------------
static bool vad_check_window(VADContext *ctx, const int16_t *window) {
    // 计算能量和过零率
    float energy = compute_energy(window, WIN_SAMPLES);
    int zcr = compute_zcr(window, WIN_SAMPLES);

    // EMA 平滑能量
    // ctx->ema_energy = ctx->ema_alpha * ctx->ema_energy + (1.0f - ctx->ema_alpha) * energy;
    // 冷启动保护：首次使用真实能量初始化 EMA
    if (ctx->ema_energy <= 1e-3f) {
        ctx->ema_energy = energy;
    } else {
        ctx->ema_energy = ctx->ema_alpha * ctx->ema_energy + (1.0f - ctx->ema_alpha) * energy;
    }

    // 打印一下，调试使用  test: 数据一直在流
    // Serial.println("ema_energy:");
    // Serial.println(ctx->ema_energy);
    // Serial.println("zcr:");
    // Serial.println(zcr);

    bool is_bark = (ctx->ema_energy >= ctx->energy_threshold) && (zcr >= (int)ctx->zcr_threshold);

    // hit和miss作为真假值直接return
    return is_bark;
}


// ==================== vad结构体初始化 ====================
bool vad_consumer_init(VADContext* ctx, QueueHandle_t tinyml_queue) {

    if (!ctx || !tinyml_queue) return false;

    // zero 清理
    memset(ctx, 0, sizeof(VADContext));

    // 11-25 修改：直接使用传入的队列，不再初始化
    ctx->tinyml_queue = tinyml_queue;

    // // 创建队列（队列元素为 TinyMLEvent 结构体），放在 PSRAM
    // if (tinyml_mfcc_queue == NULL) {
    //     // 使用 heap_caps_malloc 在 PSRAM 分配队列内存  
    //     uint8_t *queue_buf = (uint8_t *)heap_caps_malloc(QUEUE_DEPTH * sizeof(TinyMLEvent), MALLOC_CAP_SPIRAM);
    //     if (!queue_buf) {
    //         ESP_LOGE(TAG, "PSRAM malloc for tinyml_queue failed");
    //         return false;
    //     }

    //     // 静态队列结构体放在内部 RAM
    //     static StaticQueue_t tinyml_queue_struct;

    //     // 静态创建队列，队列内存由 PSRAM 提供
    //     tinyml_mfcc_queue = xQueueCreateStatic(
    //         QUEUE_DEPTH,
    //         sizeof(TinyMLEvent),
    //         queue_buf,
    //         &tinyml_queue_struct
    //     );

    //     if (!tinyml_mfcc_queue) {
    //         ESP_LOGE(TAG, "create tinyml_queue failed");
    //         return false;
    //     }
    // }
    // // 绑定结构体中的指针队列就是这个对外暴露的生产队列
    // ctx->tinyml_queue = tinyml_mfcc_queue;

    // 初始阈值 & EMA
    ctx->ema_alpha = DEFAULT_EMA_ALPHA;                  // 平滑因子（可调整）
    ctx->ema_energy = 1.0f;
    // 初始经验阈值
    ctx->energy_threshold = DEFAULT_ENERGY_THRESHOLD;
    ctx->zcr_threshold = DEFAULT_ZCR_THRESHOLD;

    // 初始化状态
    ctx->win_fill = 0;
    ctx->event_pos = 0;
    ctx->in_event = false;
    ctx->nohit_count = 0;
    // 标记 prev_stride 无效（首次数据，没有前padding）
    ctx->prev_stride_valid = false;

    // 初始化 事件缓存 和 padding 缓冲
    memset(ctx->pre_pad, 0, sizeof(ctx->pre_pad));  // 前填充
    // memset(ctx->post_pad, 0, sizeof(ctx->post_pad));  // 后填充
    memset(ctx->window_buf, 0, sizeof(ctx->window_buf)); // 窗口缓冲区
    memset(ctx->cur_stride_buf, 0, sizeof(ctx->cur_stride_buf));   // 滑窗每次出队的3个block
    memset(ctx->prev_stride_buf, 0, sizeof(ctx->prev_stride_buf));   // 上一次窗口的最后的3 block
    memset(ctx->event_buf, 0, sizeof(ctx->event_buf)); // 事件缓存区， 直接对接push函数，发送到tinyml_queue

    // tmp_event 已在 RAM 中，不需要额外初始化，结构体清零已经覆盖
    // ctx->tmp_event.samples 全为 0, ctx->tmp_event.length = 0

    ESP_LOGI(TAG, "VAD init ok: WIN=%d samples, STRIDE=%d samples, PAD=%d samples, EVENT_BUF=%d samples",
            WIN_SAMPLES, STRIDE_SAMPLES, PAD_SAMPLES, EVENT_MAX_SAMPLES);
            
    return true;
}




/*
 * 主入口：生产队列每次出队一个block（长度 BLOCK_SAMPLES）
 * 设计：严格按照流式处理，一个 block 一个 block的出生产队列。
 * 设计流程：滑窗处理的八个步骤（容易思维紊乱， 先确定好流程再写实现函数）
 */
void vad_consumer_process_block(VADContext* ctx, const int16_t* new_block) {

    if (!ctx || !new_block) return;


    // 1. 追加完整 block 到 window_buf（保证每次都是整 block）
    if ((ctx->win_fill + BLOCK_SAMPLES) > WIN_SAMPLES) {
        // 保护性检查：正常情况下应该不会发生（因为我们在窗口满时会滑动， 且每次追加的数据都是block的整数倍）
        // 如果发生，截断拷贝到满为止
        int left = WIN_SAMPLES - ctx->win_fill;
        if (left > 0) {
            memcpy(ctx->window_buf + ctx->win_fill, new_block, left * sizeof(int16_t));
            ctx->win_fill += left;    //  刚刚好截断到10个block块
        }
    } 
    else   // 正常追加数据  定义数组：左边是旧的数据，右边是新的数据， 出队需要从左边出，流数据：从右边开始入队
    {
        memcpy(ctx->window_buf + ctx->win_fill, new_block, BLOCK_SAMPLES * sizeof(int16_t));
        ctx->win_fill += BLOCK_SAMPLES;
    }

    // 追加完还没有满的话，直接返回
    if (ctx->win_fill < WIN_SAMPLES) return;


    // 2. 窗口满：更新cur_stride 缓冲区  prev_stride的更新放到滑动窗口前，保证这次窗口处理的prev_stride是上次窗口出队的3个block
     // NOTE: 不马上覆盖 prev_stride_buf。prev_stride_buf 是上一次出队的数据（保留直到本次处理完后更新）。

    // 2.1 更新 cur_stride_buf：新入窗口的3个block数据（取窗口尾部）， 后续根据状态机和命中情况用于追加事件使用
    // 从 window_buf 尾部截取 STRIDE_SAMPLES，放入 cur_stride_buf
    memcpy(ctx->cur_stride_buf, ctx->window_buf + (ctx->win_fill - STRIDE_SAMPLES), STRIDE_SAMPLES * sizeof(int16_t));


    // 3. 检查窗口是否命中事件
    bool is_bark = vad_check_window(ctx, ctx->window_buf);


    // 4. 更新状态机（只是更新状态，不更新数据）数据由状态机状态判断
    if (!ctx->in_event) {
        // 事件的首次命中
        if (is_bark) {
            // 首次命中 → 事件开始
            ctx->in_event = true;
            ctx->nohit_count = 0;
            // prev_stride_valid 表示上次出队的 stride 可用作 pre_pad， 也是更新pre_pad的标识
            ctx->prev_stride_valid = true;  // 更新pre_pad标识
        }
        // 不在事件中，且未命中
        else {
            // 持续未命中
            ctx->nohit_count = 0;
            // 这个窗口依旧不在事件中，保持 prev_stride_valid false（表示不需要更新 pre_pad 来做事件的前padding）
            ctx->prev_stride_valid = false;  
        }
    }
    // 在事件中
    else {
        ctx->prev_stride_valid = false;   // 在事件中，不需要更新pre_pad

        // 持续命中
        if (is_bark) {
            ctx->nohit_count = 0;  // 事件持续 重置 nohit 计数
        }
        // 事件中未命中  
        else {
            ctx->nohit_count++;    // 累加 nohit（第一次 miss=1，第二次 miss=2 -> 结束）
        }
    }


    // 5. 检查是否需要更新pre_pad : 首次命中 更新 pre_pad（前拼接数据）注意：首次启动算法就狗吠，这里会填充三个0的block块，因为都没有前填充的数据
    if (ctx->prev_stride_valid && ctx->in_event && ctx->nohit_count == 0 && is_bark) {
        memcpy(ctx->pre_pad,ctx->prev_stride_buf,PAD_SAMPLES * sizeof(int16_t));
        // 注意：使用完 pre_pad 之后不需要清理 prev_stride_valid（push event 后清理， 会跟着in_event变量的赋假，进入首次命中分支，这里又会更新的）
    }

    // NOTE: 取消使用post_pad
    // // 6. 检查是否需要更新post_pad：  事件中的第一次 miss：需要保存cur_stride_buf到post_pad 候选
    // if (ctx->in_event && !is_bark && ctx->nohit_count == 1) {
    //     //  cur_stride_buf 保存到 post_pad， 用于下一次不是狗吠事件时，拼接事件， 或者是丢一次命中，又出现了命中，需要连接两个事件。
    //     memcpy(ctx->post_pad, ctx->cur_stride_buf, PAD_SAMPLES * sizeof(int16_t));
    // }


    //  7.  事件数据拼接(核心逻辑)（按状态机判断执行）

    // 7.1 事件首次命中：拼 pre_pad + 整个窗口 追加到 event_buf
    if (ctx->in_event && is_bark && ctx->prev_stride_valid && ctx->nohit_count == 0) {

        // 清空事件缓存区  memcpy会直接覆盖，没有必要
        // memset(ctx->event_buf, 0, EVENT_MAX_SAMPLES * sizeof(int16_t));

        // pre_pad
        memcpy(ctx->event_buf, ctx->pre_pad, PAD_SAMPLES * sizeof(int16_t));

        // 追加整个窗口
        memcpy(ctx->event_buf + PAD_SAMPLES, ctx->window_buf, WIN_SAMPLES * sizeof(int16_t));

        ctx->event_pos = PAD_SAMPLES + WIN_SAMPLES;
    }


    // 7.2 事件中的连续命中：追加 cur_stride_buff 到事件缓存区
    if (ctx->in_event && is_bark && !ctx->prev_stride_valid && ctx->nohit_count == 0) {
        // 每次追加需要检查，不能越界
        int remain = EVENT_OUTPUT_MAX_SAMPLES - ctx->event_pos;
        int to_copy = (remain >= STRIDE_SAMPLES) ? STRIDE_SAMPLES : remain;
        if (to_copy > 0) {
            // 追加 cur_stride_buf到事件缓存区
            memcpy(ctx->event_buf + ctx->event_pos, ctx->cur_stride_buf, to_copy * sizeof(int16_t));
            ctx->event_pos += to_copy;
        }
    
        // 检查缓存区是否已满， 否则强制推送
        if (ctx->event_pos >= EVENT_OUTPUT_MAX_SAMPLES) {
            // 强制推送（内部函数会做好推送， 边界检测， 状态机清理）
            vad_force_push_event(ctx);
        }
    }

    // 7.3 事件中 第一次未命中 miss=1 -> 追加cur_stride_buf到event_buf， 后续再次未中就push不做后padding了， 命中了就使用 那个时候的窗的cur_stride_buf拼接事件
    if (ctx->in_event && !is_bark && ctx->nohit_count == 1) {
         // 每次追加需要检查，不能越界
        int remain = EVENT_OUTPUT_MAX_SAMPLES - ctx->event_pos;
        int to_copy = (remain >= STRIDE_SAMPLES) ? STRIDE_SAMPLES : remain;
        if (to_copy > 0) {
            // 追加 cur_stride_buf到事件缓存区
            memcpy(ctx->event_buf + ctx->event_pos, ctx->cur_stride_buf, to_copy * sizeof(int16_t));
            ctx->event_pos += to_copy;
        }
        // // 直接补入 cur_stride_buf 充当 post_pad
        // memcpy(ctx->event_buf + ctx->event_pos, ctx->cur_stride_buf, PAD_SAMPLES * sizeof(int16_t));
        // ctx->event_pos += PAD_SAMPLES;
        
        // 判断是否越界
        if (ctx->event_pos >= EVENT_OUTPUT_MAX_SAMPLES) {
            // 边界检测， 推送
            vad_force_push_event(ctx);
        }
    }


    // 7.4 事件中 但是（第二次 miss）：直接push，因为前面的一次miss已经拼接过了post_pad
    if (ctx->in_event && !is_bark && ctx->nohit_count >= EVENT_END_NOHIT) {
        vad_force_push_event(ctx);
    }


    // 二次保护，虽然在push函数和拼接函数内部已经做了保护，但是为了保险，这里再做一次保护
    // 前面push了的，状态机会清0，这里就不会进来了的
    if (ctx->event_pos >= EVENT_OUTPUT_MAX_SAMPLES) {
        vad_force_push_event(ctx);
    }


    // 更新prev_stride_buf：前一次出队的三个block数据， 因为后续根据状态机判断是否需要更新到 pre_pad， 用于事件的pre_padding
    memcpy(ctx->prev_stride_buf, ctx->window_buf, STRIDE_SAMPLES * sizeof(int16_t));


    // 8. 滑窗（FIFO）：向左滑动 STRIDE（右侧为新入队）
    // 滑动：把窗口向左移动 STRIDE_SAMPLES（重叠 OVERLAP_SAMPLES 保留）
    // window_buf 的左侧是出队口，右侧是入队口， 数组左侧是头部， 右侧是尾部
    memmove(ctx->window_buf, ctx->window_buf + STRIDE_SAMPLES, OVERLAP_SAMPLES * sizeof(int16_t));
    ctx->win_fill = OVERLAP_SAMPLES;


}  // end vad_consumer_process_block