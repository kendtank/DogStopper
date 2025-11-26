#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "audio_input.h"    // 生产者模块
#include "audio_consumer.h" // VAD 消费端


// ==================== audio_queue 队列 ====================
static QueueHandle_t audio_queue = nullptr;     // 句柄
static StaticQueue_t audio_queue_buffer;    // audio_queue管理器
static int16_t audio_queue_storage[AUDIO_QUEUE_DEPTH][TEMP_BUFFER_SAMPLES];



// ====================  TinyML 队列 ====================
static QueueHandle_t tinyml_mfcc_queue = nullptr;       // 句柄
static StaticQueue_t tinyml_queue_buffer;               // tinyml_queue管理器
// static TinyMLEvent tinyml_queue_storage[QUEUE_DEPTH];   // TinyMLEvent是一个结构体，里面有数组了，这里只需要一维数组存结构体
static TinyMLEvent* tinyml_queue_storage = nullptr;; // xQueueCreateStatic 要求 队列缓冲区是连续内存


// VAD 上下文
static VADContext vad_ctx;

// tinyml接受结构体，用来打印
static TinyMLEvent evt;



// ==================== 生产者任务 ====================
void AudioProducerTask(void* param)
{
    QueueHandle_t q = (QueueHandle_t)param;
    if (!q) {
        Serial.println("[Producer] queue NULL!");
        vTaskDelete(NULL);
        return;
    }

    if (!audio_input_init()) {
        Serial.println("[Producer] audio_input_init FAILED");
        vTaskDelete(NULL);
        return;
    }

    Serial.println("[Producer] init done, start streaming...");

    audio_input_task((void*)q); // 永久循环采集音频
}


// ==================== 消费者任务 ====================
void AudioConsumerTask(void* param)
{
    QueueHandle_t q = (QueueHandle_t)param;   // 这里是拿到audio_queue队列句柄
    if (!q) {
        Serial.println("[VAD Consumer] queue NULL!");
        vTaskDelete(NULL);
        return;
    }
    // 初始化 VAD 上下文

    Serial.println("[VAD Consumer] start consuming audio...");

    // 使用静态数组，减少栈压力
    static int16_t pcm_buffer[TEMP_BUFFER_SAMPLES];


    while (true) {
        if (xQueueReceive(q, pcm_buffer, portMAX_DELAY) == pdTRUE) {
            // 调用 VAD 消费函数
            vad_consumer_process_block(&vad_ctx, pcm_buffer);
        }
        else {
            Serial.println("[VAD Consumer] xQueueReceive FAILED");
        }
    }
}


// ==================== TinyML 消费者, 这里只做打印 ====================
void TinyMLConsumerTask(void* param)
{
    QueueHandle_t q = (QueueHandle_t)param;   // 这里是拿到tinyml_queue队列句柄
    if (!q) {
        Serial.println("[TinyML Consumer] tinyml_queue NULL!");
        vTaskDelete(NULL);
        return;
    }

    Serial.println("[TinyML Consumer] start consuming MFCC events...");

    
    while (true) {
        if (xQueueReceive(q, &evt, portMAX_DELAY) == pdTRUE) {
            // 调用 TinyML 消费函数，这里只做打印时间的长度（音频点数和毫秒数）
            Serial.printf("[TinyML Consumer] Event length: %d samples, %d ms\n", evt.length, evt.length * 1000 / MIC_SAMPLE_RATE);
        } else {
            Serial.println("[TinyML Consumer] Queue receive FAILED");
        }
    }
}


// ==================== setup ====================
void setup()
{
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== Audio Stream Test Start ===");

    // ===== audio_queue =====
    audio_queue = xQueueCreateStatic(
        AUDIO_QUEUE_DEPTH,
        sizeof(int16_t) * TEMP_BUFFER_SAMPLES,
        (uint8_t*)audio_queue_storage,
        &audio_queue_buffer
    );

    if (!audio_queue) {
        Serial.println("Queue create FAILED");
        while (1) vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    Serial.println("Queue create OK");

    // PSRAM创建tinyml队列

    // ===== tinyml_queue ===== 每个事件在 PSRAM =====
    tinyml_queue_storage = (TinyMLEvent*)heap_caps_malloc(QUEUE_DEPTH * sizeof(TinyMLEvent), MALLOC_CAP_SPIRAM);
    if (!tinyml_queue_storage) {
        Serial.println("PSRAM malloc failed for tinyml_queue_storage");
        while (1) vTaskDelay(1000 / portTICK_PERIOD_MS);
    }

    tinyml_mfcc_queue = xQueueCreateStatic(
    QUEUE_DEPTH,
    sizeof(TinyMLEvent),
    (uint8_t*)tinyml_queue_storage,
    &tinyml_queue_buffer
);

    delay(100);

    // 初始化 VAD 上下文结构体，绑定tinyml队列
    vad_consumer_init(&vad_ctx, tinyml_mfcc_queue);

    // 等待所有异步操作完成
    delay(200);


    // ===== 启动任务 优先消费者，再生产者 =====
    xTaskCreatePinnedToCore(TinyMLConsumerTask, "TinyMLConsumer", 8192, (void*)tinyml_mfcc_queue, 8, NULL, 1);

    xTaskCreatePinnedToCore(AudioConsumerTask, "VADConsumer", 8192, (void*)audio_queue, 9, NULL, 1);

    xTaskCreatePinnedToCore(AudioProducerTask, "AudioProducer", 8192, (void*)audio_queue, 10, NULL, 0);
    
    
}

// ==================== loop ====================
void loop()
{
    // FreeRTOS 已接管，不需要操作
}
