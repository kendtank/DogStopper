#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "audio_input.h"    // 生产者模块


/*
说明： 
    1.audio_queue 不是 AUDIO_QUEUE_DEPTH 指针， 是队列句柄，类型是 QueueHandle_t， xQueueCreateStatic() 返回的就是一个 QueueHandle_t
    2.FreeRTOS 静态队列接口要求队列存储区是 uint8_t* 类型（字节指针），它会按字节管理队列内存，  所以这里强制类型转换，把二维 int16_t 数组转成 uint8_t*
    3. FreeRTOS 内部会用 element_size（这里是 sizeof(int16_t) * TEMP_BUFFER_SAMPLES）去分块存储，每次队列操作就是一块 PCM 数据。 重点：转换只是为了接口匹配，
    内存布局没有改变，队列仍然每次读写一整块 PCM 数据
    4. static StaticQueue_t audio_queue_buffer; FreeRTOS 静态队列控制块,  它保存队列的管理信息（头尾索引、计数器、元素大小、等待任务列表等）作用：队列不分配堆内存，
    所有控制信息存在静态变量里内存大小固定，不大，通常几十字节到一百多字节，不用担心占用 RAM,当你用 xQueueCreateStatic() 时，FreeRTOS 就不需要在堆上开辟控制块，整个队列完全静态分配
    例子：
    比喻：
    audio_queue → 队列的门牌号
    audio_queue_storage → 队列里的房间（存放 PCM 数据）
    audio_queue_buffer → 房东的账本（管理房间使用情况）
    */


// ==================== 队列 ====================
static QueueHandle_t audio_queue = nullptr;  // 队列句柄，任务间通过它传递音频数据（PCM块）注意：不是指针，而是一个队列句柄，类型是 QueueHandle_t
static StaticQueue_t audio_queue_buffer;    // 静态队列控制块，FreeRTOS 内部用来管理队列状态
static int16_t audio_queue_storage[AUDIO_QUEUE_DEPTH][TEMP_BUFFER_SAMPLES];   // 队列存储区，本例是二维数组，每个元素就是一个 PCM block



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
    QueueHandle_t q = (QueueHandle_t)param;
    if (!q) {
        Serial.println("[Consumer] queue NULL!");
        vTaskDelete(NULL);
        return;
    }

    Serial.println("[Consumer] start consuming audio...");

    int16_t pcm_buffer[TEMP_BUFFER_SAMPLES];
    while (true) {
        if (xQueueReceive(q, pcm_buffer, portMAX_DELAY) == pdTRUE) {
            // 简单计算能量（平方和 / 样本数）
            uint64_t sum_sq = 0;
            for (int i = 0; i < TEMP_BUFFER_SAMPLES; i++) {
                sum_sq += (int32_t)pcm_buffer[i] * pcm_buffer[i];
            }
            float rms = sqrtf(sum_sq / (float)TEMP_BUFFER_SAMPLES);
            Serial.printf("[Consumer] PCM block RMS: %.1f\n", rms);
        }
    }
}

// ==================== setup ====================
void setup()
{
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== Audio Stream Test Start ===");

    // 创建静态队列
    audio_queue = xQueueCreateStatic(
        AUDIO_QUEUE_DEPTH,   // 队列长度
        sizeof(int16_t) * TEMP_BUFFER_SAMPLES,   // 队列每个元素大小
        (uint8_t*)audio_queue_storage,     // 队列存储区
        &audio_queue_buffer             // 队列控制块
    );

    if (!audio_queue) {
        Serial.println("Queue create FAILED");
        while (1) vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    Serial.println("Queue create OK");

    // 启动生产者任务
    xTaskCreatePinnedToCore(
        AudioProducerTask,   // 任务函数
        "AudioProducerTask",   // 任务名
        8192,                   // 任务栈大小
        (void*)audio_queue,         // 任务参数
        10,                       // 任务优先级
        NULL,                     // 任务句柄
        0                         // 任务运行核心
    );

    // 启动消费者任务
    xTaskCreatePinnedToCore(
        AudioConsumerTask,
        "AudioConsumerTask",
        4096,
        (void*)audio_queue,
        9,
        NULL,
        0
    );
}

// ==================== loop ====================
void loop()
{
    // FreeRTOS 已接管，不需要操作
}
