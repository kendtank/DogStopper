// #include "unity.h"
// #include <Arduino.h>
// #include "audio_input.h"
// #include <math.h>

// #define BLOCK_SIZE 256  // 每次读取样本数

// // ----------------- 测试函数 -----------------
// void test_int16_ring_buffer(void) {
//     int available = (rb_write_idx - rb_read_idx + RING_BUFFER_SIZE) % RING_BUFFER_SIZE;

//     if (available >= BLOCK_SIZE) {
//         int16_t block[BLOCK_SIZE];

//         // 从 ring buffer 中读取 BLOCK_SIZE 个样本
//         for (int i = 0; i < BLOCK_SIZE; i++) {
//             block[i] = ring_buffer[rb_read_idx];
//             rb_read_idx = (rb_read_idx + 1) % RING_BUFFER_SIZE;
//         }

//         // 打印前16个样本
//         Serial.printf("RingBuffer block: ");
//         for (int i = 0; i < 16; i++) {
//             Serial.printf("%d, ", block[i]);
//         }
//         Serial.println();

//         // RMS 计算
//         float rms = 0.0;
//         for (int i = 0; i < BLOCK_SIZE; i++) {
//             rms += block[i] * block[i];
//         }
//         rms = sqrt(rms / BLOCK_SIZE);
//         Serial.printf("RMS: %.2f\n", rms);
//     }
// }

// // ----------------- FreeRTOS 任务 -----------------
// void ring_buffer_task(void* param) {
//     while (true) {
//         test_int16_ring_buffer();
//         vTaskDelay(pdMS_TO_TICKS(200));  // 每200ms打印一次
//     }
// }

// // ----------------- Unity 框架 -----------------
// void setUp(void) {}
// void tearDown(void) {}

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_int16_ring_buffer);
//     return UNITY_END();
// }

// // ----------------- Arduino setup/loop -----------------
// void setup() {
//     Serial.begin(115200);
//     delay(2000);

//     // 初始化音频采集
//     audio_input_init();

//     // 创建音频采集任务
//     xTaskCreatePinnedToCore(
//         audio_input_task,
//         "AudioInputTask",
//         4096,
//         NULL,
//         1,
//         NULL,
//         0
//     );
//     Serial.println("Audio input task started.");

//     // 创建 ring buffer 打印测试任务
//     xTaskCreatePinnedToCore(
//         ring_buffer_task,
//         "RingBufferTask",
//         4096,
//         NULL,
//         1,
//         NULL,
//         1
//     );

//     // 运行 Unity 测试（只做一次断言）
//     runUnityTests();
// }

// void loop() {
//     // 空，所有工作交给任务处理
// }
