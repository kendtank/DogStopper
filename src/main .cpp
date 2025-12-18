#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "audio_input.h"    // 生产者模块
#include "audio_consumer.h" // VAD 消费端
#include "bark_detector.h"  // 狗吠检测消费者
#include "verify_embedding.h"  // 声纹验证消费者
#include "led_control.h"
#include "system_state.h"



// ==================== audio_queue 队列 ====================
static QueueHandle_t audio_queue = nullptr;     // 句柄
static StaticQueue_t audio_queue_buffer;    // audio_queue管理器
static int16_t audio_queue_storage[AUDIO_QUEUE_DEPTH][TEMP_BUFFER_SAMPLES];



// ====================  TinyML 队列 ====================
static QueueHandle_t tinyml_mfcc_queue = nullptr;       // 句柄
static StaticQueue_t tinyml_queue_buffer;               // tinyml_queue管理器
// static TinyMLEvent tinyml_queue_storage[QUEUE_DEPTH];   // TinyMLEvent是一个结构体，里面有数组了，这里只需要一维数组存结构体
static TinyMLEvent* tinyml_queue_storage = nullptr; // xQueueCreateStatic 要求 队列缓冲区是连续内存


// ====================  TinyML 队列 ====================
static QueueHandle_t bark_queue = nullptr;       // 句柄
static StaticQueue_t bark_queue_buffer;               // tinyml_queue管理器
static BarkEvent* bark_queue_storage = nullptr; // xQueueCreateStatic 要求 队列缓冲区是连续内存




// VAD 上下文
static VADContext vad_ctx;

// tinyml接收结构体，用来传递到brak检测模块
static TinyMLEvent evt;

// bark_event接收结构体
static BarkEvent bark_evt;

volatile bool g_system_ready = false;


// ==================== 音频生产者任务 ====================
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


// ==================== 音频消费者任务(tinyml生产者) ====================
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


// ==================== TinyML 消费者（bark_event消费者） ====================
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
            // 调用 TinyML 消费函数
            bark_detector_process_event(&evt);   // 传Tinyml事件地址
        } else {
            Serial.println("[TinyML Consumer] Queue receive FAILED");
        }
    }
}


// ==================== bark_event 消费者, 这里只做打印 ====================
void BarkEventConsumerTask(void* param)
{
    QueueHandle_t q = (QueueHandle_t)param;   // 这里是拿到bark_event队列句柄
    if (!q) {
        Serial.println("[bark_event Consumer] bark_event NULL!");
        vTaskDelete(NULL);
        return;
    }

    Serial.println("[bark_event Consumer] start consuming brak events...");

    
    while (true) {
        if (xQueueReceive(q, &bark_evt, portMAX_DELAY) == pdTRUE) {
            // 拿到真实的狗吠事件， 这里只做打印长度和需要的时间， 大概就是延时
            verify_embedding_process(&bark_evt);
            
        } else {
            Serial.println("[TinyML Consumer] Queue receive FAILED");
        }
    }
}



// ==================== 内存打印任务 ====================
void MonitorMemoryTask(void* param) {
    const uint32_t intervalMs = 60000; // 1分钟
    while(true) {
        vTaskDelay(pdMS_TO_TICKS(intervalMs));

        Serial.printf("[MEM] PSRAM总: %d KB, 可用: %d KB\n", ESP.getPsramSize()/1024, ESP.getFreePsram()/1024);
        Serial.printf("[MEM] 内部堆总: %d KB, 可用: %d KB\n", ESP.getHeapSize()/1024, ESP.getFreeHeap()/1024);
    }
}



// ==================== setup ====================
void setup()
{
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== Audio Stream Test Start ===");
    Serial.printf("[MEM] PSRAM总大小: %d KB\n", ESP.getPsramSize() / 1024);
    Serial.printf("[MEM] PSRAM可用: %d KB\n", ESP.getFreePsram() / 1024);
    Serial.printf("[MEM] 内部RAM总堆: %d KB\n", ESP.getHeapSize() / 1024);
    Serial.printf("[MEM] 内部RAM可用: %d KB\n", ESP.getFreeHeap() / 1024);
    



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
    tinyml_queue_storage = (TinyMLEvent*)heap_caps_malloc(TINYML_QUEUE_DEPTH * sizeof(TinyMLEvent), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!tinyml_queue_storage) {
        Serial.println("PSRAM malloc failed for tinyml_queue_storage");
        while (1) vTaskDelay(1000 / portTICK_PERIOD_MS);
    }

    tinyml_mfcc_queue = xQueueCreateStatic(
    TINYML_QUEUE_DEPTH,
    sizeof(TinyMLEvent),
    (uint8_t*)tinyml_queue_storage,
    &tinyml_queue_buffer
);

    // PSRAM创建 bark_queue 队列
    bark_queue_storage = (BarkEvent*)heap_caps_malloc(BARK_QUEUE_DEPTH * sizeof(BarkEvent), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!bark_queue_storage) {
        Serial.println("PSRAM malloc failed for bark_queue_storage");
        while (1) vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    // 拿到句柄
    bark_queue = xQueueCreateStatic(
    BARK_QUEUE_DEPTH,
    sizeof(BarkEvent),
    (uint8_t*)bark_queue_storage,
    &bark_queue_buffer
);


    delay(100);

    // 初始化 VAD 上下文结构体，绑定tinyml队列
    vad_consumer_init(&vad_ctx, tinyml_mfcc_queue);

    // 初始化 BarkDetector
    bark_detector_init(bark_queue);


    // 初始化验证模块
    verify_embedding_init();

    led_init();

    g_system_ready = true;

    // 等待所有异步操作完成
    delay(500);




    // ===== 启动任务 优先消费者，再生产者 =====

    xTaskCreatePinnedToCore(BarkEventConsumerTask, "BarkDetector", 8192, (void*)bark_queue, 7, NULL, 0);

    xTaskCreatePinnedToCore(TinyMLConsumerTask, "TinyMLConsumer", 8192, (void*)tinyml_mfcc_queue, 8, NULL, 1);

    xTaskCreatePinnedToCore(AudioConsumerTask, "VADConsumer", 8192, (void*)audio_queue, 9, NULL, 1);

    xTaskCreatePinnedToCore(AudioProducerTask, "AudioProducer", 8192, (void*)audio_queue, 10, NULL, 1);

    xTaskCreatePinnedToCore(LedTask, "LED", 2048, NULL, 6, NULL, 0);

    // 创建监控任务
    xTaskCreatePinnedToCore(MonitorMemoryTask, "Monitor", 2048, NULL, 5, NULL, 0);

    Serial.println("All tasks started");


}

// ==================== loop ====================
void loop()
{
    // FreeRTOS 已接管，不需要操作
}







// #include "Arduino.h"
// #include "flash_storage.h"

// void setup() {
//     Serial.begin(115200);
//     delay(100);                 // 给 USB CDC 一点时间
//     Serial.println("booting...");

//     reset_storage();
//     Serial.println("flash erased");
// }

// void loop() {
//     delay(1000);
// }
