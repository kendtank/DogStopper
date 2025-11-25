/*
================================================================================
DMA生产者模块 (audio_input.cpp)

作者：Kend

模块作用：
    本模块负责从麦克风通过I2S接口和DMA连续采集PCM音频数据，
    并将数据安全、高效地推送到FreeRTOS队列供后续消费者模块使用。
    消费者是VAD音频分析任务。

核心特性：
1. 高可用数据流：
    - 使用FreeRTOS队列（audio_queue）实现生产者-消费者解耦。
    - 队列深度为AUDIO_QUEUE_DEPTH，每个元素为TEMP_BUFFER_SAMPLES大小的PCM块。
    - 当队列满时，采用覆盖老数据策略（xQueueOverwrite），保证最新音频不丢失。
    - 丢帧计数器g_drop_count用于监控丢帧情况，可用于日志或报警。

2. 数据质量保障：
    - 提供DC滤波器，可去除直流偏置，减少低频噪声影响。
    - DC滤波器支持预热（预读取若干样本稳定状态），避免开机时滤波器输出异常。
    - 可动态启用/禁用DC滤波，并支持重新预热。

3. 异常处理：
    - DMA读取失败或队列满时，任务不会阻塞整个系统，而是延迟或覆盖处理。
    - 使用互斥锁保护DC滤波器状态，防止多任务同时访问引发数据竞争。
    - 日志节流机制防止频繁打印丢帧信息影响性能。

4. 性能优化：
    - 热路径函数（如remove_dc）使用IRAM_ATTR放入内部RAM，加快每个样本的滤波计算。
    - 队列和滤波器操作尽量非阻塞，保证实时性。

模块依赖：
    - FreeRTOS (队列、任务、互斥锁)
    - ESP-IDF I2S 驱动
    - esp_log 用于日志打印

使用方式：
    1. 调用 audio_input_init() 初始化I2S、队列和滤波器。
    2. 创建任务 audio_input_task()，作为DMA生产者持续采集数据。
    3. 消费者任务通过 xQueueReceive(audio_queue, ...) 获取PCM数据块进行处理。
================================================================================
*/


#include "audio_input.h"
#include <driver/i2s.h>
#include <esp_log.h>
#include <esp_err.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// =================== 内部配置 ===================
#define TEMP_BUFFER_SAMPLES 256        // 批量大小（PCM块）每次从DMA读取的样本数，i2s_read(), 意思就是说dma  buffer 64 * 4 读满了一半，就取走。设置512，就全部满了一次取走，延迟大点，cpu轮询次数降低.  256点音频 就是   256 / 16000  = 16ms 的音频数据
#define PREHEAT_SAMPLES 1024           // 预热样本数 开机时用来让DC滤波器稳定
#define FILTER_A_FLOAT 0.995f          // IIR系数 一阶DC滤波系数
#define USE_FIXED_POINT 1              // 0: float计算DC滤波，1: Q15定点计算
#define DROP_LOG_INTERVAL 10          // 队列满丢帧日志节流，每10次打印一次

static const char* TAG = "AudioInput";

// =================== 队列 ===================
// FreeRTOS 队列句柄，用于生产者（DMA）和消费者（VAD/TinyML）解耦
// 生产者将音频块发送到队列，消费者从队列获取音频块
QueueHandle_t audio_queue = NULL;

// 丢帧计数
static volatile uint64_t g_drop_count = 0;  // 当队列满或覆盖数据时统计丢帧数
static uint32_t drop_log_counter = 0;       // 丢帧日志节流


// DC滤波器结构-去直流
typedef struct {
#if USE_FIXED_POINT
    int16_t x_prev;
    int64_t y_prev_q15;
    int32_t a_q15;
#else
    int16_t x_prev;
    float y_prev;
    float a;
#endif
    bool enabled;  // 是否启用DC滤波
    bool preheated;  // 是否已预热
} DCFilter;
static DCFilter dc_filter;


// 滤波互斥锁（保护滤波器状态）
static SemaphoreHandle_t filter_mutex = NULL;




// =================== 热路径宏 ===================
// 热路径函数（调用频率高，因为每个样本都调用DC滤波）
// 使用IRAM_ATTR将函数放在IRAM中，提高速度
#ifdef IRAM_ATTR
#define HOT_ATTR IRAM_ATTR
#else
#define HOT_ATTR
#endif


// 限幅函数， 防止16位PCM溢出
static inline int16_t clamp_int16_from_int64(int64_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}


// DC滤波实现
#if USE_FIXED_POINT
// Q15定点DC滤波
static inline HOT_ATTR int16_t remove_dc_q15(DCFilter* f, int16_t x) {
    if (!f->enabled) return x;
    int64_t diff = (int64_t)x - (int64_t)f->x_prev;
    int64_t left = diff * 32768LL;
    int64_t mul = ((int64_t)f->a_q15 * f->y_prev_q15) >> 15;
    int64_t y_q15 = left + mul;
    f->x_prev = x;
    f->y_prev_q15 = y_q15;
    return clamp_int16_from_int64(y_q15 >> 15);
}
#else
// float DC滤波
static inline HOT_ATTR int16_t remove_dc_float(DCFilter* f, int16_t x) {
    if (!f->enabled) return x;
    float y = (float)x - (float)f->x_prev + f->a * f->y_prev;
    f->x_prev = x;
    f->y_prev = y;
    if (y > 32767.0f) y = 32767.0f;
    if (y < -32768.0f) y = -32768.0f;
    return (int16_t)lroundf(y);
}
#endif

// 根据宏选择DC滤波实现
static inline HOT_ATTR int16_t remove_dc(DCFilter* f, int16_t x) {
#if USE_FIXED_POINT
    return remove_dc_q15(f, x);
#else
    return remove_dc_float(f, x);
#endif
}

// =================== 预热滤波器 ===================
// 开机时滤波器初始值不稳定，需要读取一定数量的样本稳定状态
// 避免DC滤波在刚开始输出异常值
static void preheat_filter_safe(void) {
    int remaining = PREHEAT_SAMPLES;   // 还需要读取的样本数
    int timeout_cycles = 0;
    const int MAX_TIMEOUT = 50;
    int16_t pre_buf[TEMP_BUFFER_SAMPLES];
    while (remaining > 0 && timeout_cycles < MAX_TIMEOUT) {
        int to_read = (remaining > TEMP_BUFFER_SAMPLES) ? TEMP_BUFFER_SAMPLES : remaining;
        size_t bytes_read = 0;
        esp_err_t r = i2s_read(MIC_I2S_PORT, pre_buf, to_read * sizeof(int16_t), &bytes_read, 200 / portTICK_PERIOD_MS);
        if (r != ESP_OK || bytes_read == 0) { timeout_cycles++; vTaskDelay(10 / portTICK_PERIOD_MS); continue; }
        int got = (int)(bytes_read / sizeof(int16_t));
        for (int i = 0; i < got; i++) {
            remove_dc(&dc_filter, pre_buf[i]);  // 更新滤波器状态
        }
        remaining -= got;
    }
}



// 初始化生产者
bool audio_input_init(void) {
    // 创建音频队列（FreeRTOS队列）：
    // 队列深度 = AUDIO_QUEUE_DEPTH，即队列最多可缓存多少个音频批次（批次大小TEMP_BUFFER_SAMPLES）
    // 使用队列保证生产者/消费者线程安全
    audio_queue = xQueueCreate(AUDIO_QUEUE_DEPTH, TEMP_BUFFER_SAMPLES * sizeof(int16_t));
    if (audio_queue == NULL) {
        ESP_LOGE(TAG, "Queue create failed");
        return false;
    }

    // 创建滤波mutex，保护DC滤波器状态
    filter_mutex = xSemaphoreCreateMutex();
    if (filter_mutex == NULL) {
        ESP_LOGE(TAG, "Mutex create failed");
        return false;
    }

    // 初始化滤波器
#if USE_FIXED_POINT
    dc_filter.x_prev = 0;
    dc_filter.y_prev_q15 = 0;
    int32_t a_q15 = (int32_t)lroundf(FILTER_A_FLOAT * 32768.0f);
    if (a_q15 < 0) a_q15 = 0;
    if (a_q15 > 32767) a_q15 = 32767;
    dc_filter.a_q15 = a_q15;
#else
    dc_filter.x_prev = 0;
    dc_filter.y_prev = 0.0f;
    dc_filter.a = FILTER_A_FLOAT;
#endif
    dc_filter.enabled = true;
    dc_filter.preheated = false;

    // I2S初始化（使用宏配置）
    i2s_config_t i2s_cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = MIC_SAMPLE_RATE,
        .bits_per_sample = MIC_BITS_PER_SAMPLE,
        .channel_format = MIC_CHANNEL_FORMAT,
        .communication_format = MIC_COMM_FORMAT,
        .intr_alloc_flags = 0,
        .dma_buf_count = MIC_DMA_BUF_COUNT,
        .dma_buf_len = MIC_DMA_BUF_LEN,
        .use_apll = MIC_USE_APLL,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    if (i2s_driver_install(MIC_I2S_PORT, &i2s_cfg, 0, NULL) != ESP_OK) {
        ESP_LOGE(TAG, "I2S install failed");
        return false;
    }

    // 设置I2S引脚
    i2s_pin_config_t pin_cfg = {
        .bck_io_num = MIC_PIN_BCK,
        .ws_io_num = MIC_PIN_WS,
        .data_out_num = MIC_PIN_DATA_OUT,
        .data_in_num = MIC_PIN_DATA_IN
    };
    if (i2s_set_pin(MIC_I2S_PORT, &pin_cfg) != ESP_OK) {
        ESP_LOGE(TAG, "I2S set pin failed");
        return false;
    }

    i2s_zero_dma_buffer(MIC_I2S_PORT);  // 清空DMA缓冲

    // 预热滤波器
    if (dc_filter.enabled) {
        preheat_filter_safe();
        dc_filter.preheated = true;
    }

    return true;
}


// 生产者任务
void audio_input_task(void* param) {
    int16_t temp_buffer[TEMP_BUFFER_SAMPLES];
    size_t bytes_read = 0;
    while (true) {
        // 1. 从DMA读取音频数据
        esp_err_t r = i2s_read(MIC_I2S_PORT, temp_buffer, sizeof(temp_buffer), &bytes_read, portMAX_DELAY);
        if (r != ESP_OK) {
            vTaskDelay(5 / portTICK_PERIOD_MS);  // 读取失败，稍后重试
            continue;
        }

        int samples = (int)(bytes_read / sizeof(int16_t));

        // 2. DC滤波（互斥保护）
        // 每次处理 PCM 样本时，都会更新dc_filter的变量，使用互斥锁，保证线程安全，实际上，逻辑中只有读取pcm数据，应用滤波才会使用，但是还是加上了锁
        if (xSemaphoreTake(filter_mutex, portMAX_DELAY) == pdTRUE) {
            for (int i = 0; i < samples; i++) {
                temp_buffer[i] = remove_dc(&dc_filter, temp_buffer[i]);
            }
            xSemaphoreGive(filter_mutex);
        }

        // 3.发送到队列（非阻塞，高可用）
        BaseType_t res = xQueueSend(audio_queue, temp_buffer, 0);
        if (res != pdTRUE) {
            // 队列满，覆盖老数据（优先新，高召回）
            xQueueOverwrite(audio_queue, temp_buffer);
            g_drop_count += TEMP_BUFFER_SAMPLES;
            // 每满10次队列打印一次日志
            if (++drop_log_counter % DROP_LOG_INTERVAL == 0) {
                ESP_LOGW(TAG, "Queue full, overwrote old. Total dropped: %llu", g_drop_count);
            }
        }
    }
}


// 其他API


// 返回累计丢帧数
uint64_t audio_get_dropped_samples(void) {
    return g_drop_count;
}

// 重置丢帧计数器
void audio_reset_dropped_counter(void) {
    g_drop_count = 0;
    drop_log_counter = 0;
}


// DC滤波开关, 如果麦克风自带DC滤波可以动态关闭
void audio_set_dc_filter_enabled(bool enabled) {
    if (xSemaphoreTake(filter_mutex, portMAX_DELAY) == pdTRUE) {
#if USE_FIXED_POINT
        dc_filter.x_prev = 0;
        dc_filter.y_prev_q15 = 0;
#else
        dc_filter.x_prev = 0;
        dc_filter.y_prev = 0.0f;
#endif
        dc_filter.enabled = enabled;
        dc_filter.preheated = false;
        xSemaphoreGive(filter_mutex);
    }
}


// 强制重新预热滤波器 在 DMA 任务启动前或滤波器开关后调用
void audio_force_repreheat(void) {
    if (dc_filter.enabled) {
        preheat_filter_safe();
        dc_filter.preheated = true;
    }
}