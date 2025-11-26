#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "audio_input.h"    // 生产者模块


// ==================== 队列 ====================
static QueueHandle_t audio_queue = nullptr;
static StaticQueue_t audio_queue_buffer;
static int16_t audio_queue_storage[AUDIO_QUEUE_DEPTH][TEMP_BUFFER_SAMPLES];


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

    // 启动生产者任务
    xTaskCreatePinnedToCore(
        AudioProducerTask,
        "AudioProducerTask",
        8192,
        (void*)audio_queue,
        10,
        NULL,
        0
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
