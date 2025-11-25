#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "audio_input.h"
#include "audio_consumer.h"

// ====================== 任务栈大小 ======================
#define AUDIO_PRODUCER_STACK 8192
#define AUDIO_CONSUMER_STACK 8192
#define TINYML_CONSUMER_STACK 4096

// ====================== 音频生产者任务 ======================
void AudioProducerTask(void* param)
{
    Serial.println("[Producer] init...");

    // 等待队列初始化完成（audio_input_init里创建）
    while (audio_queue == NULL) {
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }

    Serial.println("[Producer] start");

    // 使用封装的生产者循环
    audio_input_task(param);

    vTaskDelete(NULL);
}

// ====================== 音频消费者任务 ======================
void AudioConsumerTask(void* param)
{
    Serial.println("[Consumer] init...");

    static VADContext vad_ctx;

    if (!vad_consumer_init(&vad_ctx)) {
        Serial.println("[Consumer] vad_consumer_init FAILED");
        vTaskDelete(NULL);
        return;
    }

    Serial.println("[Consumer] start");

    // 每次取一个 block
    int16_t pcm_buf[BLOCK_SAMPLES];

    while (true)
    {
        // 阻塞等待 audio_queue 出队
        if (xQueueReceive(audio_queue, pcm_buf, portMAX_DELAY) == pdTRUE)
        {
            // 调用封装函数处理
            vad_consumer_process_block(&vad_ctx, pcm_buf);
        }
    }
}

// ====================== TinyML 队列消费者 ======================
void TinymlMfccQueueConsumerTask(void* param)
{
    Serial.println("[TinyML] start...");

    TinyMLEvent ev;

    while (true)
    {
        if (xQueueReceive(tinyml_mfcc_queue, &ev, portMAX_DELAY) == pdTRUE)
        {
            // Serial.printf("[TinyML] recv event");
        }
    }
}

// ====================== setup ======================
void setup()
{
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== TinyML Queue Test Start ===");

    // 初始化音频输入（包括队列 + DC滤波器预热）
    if (!audio_input_init()) {
        Serial.println("[setup] audio_input_init FAILED");
        while (1) { vTaskDelay(1000 / portTICK_PERIOD_MS); }
    }

    // 创建任务
    xTaskCreatePinnedToCore(AudioProducerTask, "AudioProducerTask",
                            AUDIO_PRODUCER_STACK, NULL, 10, NULL, 0);
    delay(100);

    xTaskCreatePinnedToCore(AudioConsumerTask, "AudioConsumerTask",
                            AUDIO_CONSUMER_STACK, NULL, 9, NULL, 0);
    delay(100);

    xTaskCreatePinnedToCore(TinymlMfccQueueConsumerTask, "TinymlMfccTask",
                            TINYML_CONSUMER_STACK, NULL, 8, NULL, 1);
}

// ====================== loop ======================
void loop()
{
    // FreeRTOS 已接管任务，不需要循环
}
