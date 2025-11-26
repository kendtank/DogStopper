// #include <Arduino.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "freertos/queue.h"
// #include "esp_heap_caps.h"

// #define QUEUE_DEPTH 5
// #define EVENT_SAMPLES 16

// typedef struct {
//     int16_t samples[EVENT_SAMPLES];
//     int length;
// } TinyMLEvent;

// // =================== 内部 RAM 队列 ===================
// static TinyMLEvent ram_queue_buf[QUEUE_DEPTH];
// static StaticQueue_t ram_queue_struct;
// QueueHandle_t ram_queue;

// // =================== PSRAM 队列 ===================
// TinyMLEvent *psram_queue_buf = nullptr;
// static StaticQueue_t psram_queue_struct;
// QueueHandle_t psram_queue;

// // =================== 发送任务 ===================
// void SenderTask(void* param) {
//     QueueHandle_t q = (QueueHandle_t)param;
//     TinyMLEvent ev;

//     int counter = 0;
//     while (true) {
//         for (int i = 0; i < EVENT_SAMPLES; i++) ev.samples[i] = counter + i;
//         ev.length = EVENT_SAMPLES;

//         if (xQueueSend(q, &ev, portMAX_DELAY) == pdTRUE) {
//             Serial.printf("[Sender] Sent event starting with %d\n", ev.samples[0]);
//         } else {
//             Serial.println("[Sender] Send FAILED");
//         }

//         counter += 10;
//         vTaskDelay(1000 / portTICK_PERIOD_MS);
//     }
// }

// // =================== 接收任务 ===================
// void ReceiverTask(void* param) {
//     QueueHandle_t q = (QueueHandle_t)param;
//     TinyMLEvent ev;

//     while (true) {
//         if (xQueueReceive(q, &ev, portMAX_DELAY) == pdTRUE) {
//             Serial.print("[Receiver] Recv: ");
//             for (int i = 0; i < ev.length; i++) {
//                 Serial.print(ev.samples[i]);
//                 Serial.print(" ");
//             }
//             Serial.println();
//         }
//     }
// }

// // =================== setup ===================
// void setup() {
//     Serial.begin(115200);
//     delay(1000);
//     Serial.println("\n=== Queue & Task Test Start ===");

//     // --------- 内部 RAM 队列 ---------
//     ram_queue = xQueueCreateStatic(
//         QUEUE_DEPTH,
//         sizeof(TinyMLEvent),
//         (uint8_t*)ram_queue_buf,
//         &ram_queue_struct
//     );

//     if (!ram_queue) {
//         Serial.println("RAM queue create FAILED!");
//     } else {
//         Serial.println("RAM queue create OK!");
//         xTaskCreate(SenderTask, "RAM_Sender", 4096, ram_queue, 5, NULL);
//         xTaskCreate(ReceiverTask, "RAM_Receiver", 4096, ram_queue, 5, NULL);
//     }

//     // --------- PSRAM 队列 ---------
//     psram_queue_buf = (TinyMLEvent*)heap_caps_malloc(QUEUE_DEPTH * sizeof(TinyMLEvent), MALLOC_CAP_SPIRAM);
//     if (!psram_queue_buf) {
//         Serial.println("PSRAM malloc FAILED!");
//     } else {
//         uintptr_t addr = (uintptr_t)psram_queue_buf;
//         if (addr % 4 != 0) Serial.println("PSRAM buffer NOT 4-byte aligned!");
//         psram_queue = xQueueCreateStatic(
//             QUEUE_DEPTH,
//             sizeof(TinyMLEvent),
//             (uint8_t*)psram_queue_buf,
//             &psram_queue_struct
//         );
//         if (!psram_queue) {
//             Serial.println("PSRAM queue create FAILED!");
//         } else {
//             Serial.println("PSRAM queue create OK!");
//             xTaskCreate(SenderTask, "PSRAM_Sender", 4096, psram_queue, 4, NULL);
//             xTaskCreate(ReceiverTask, "PSRAM_Receiver", 4096, psram_queue, 4, NULL);
//         }
//     }
// }

// void loop() {
//     // FreeRTOS 任务已经接管
// }



// /*
// [Sender] Sent event starting with 110
// [Receiver] Recv: 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 
// [Sender] Sent event starting with 110
// [Receiver] Recv: 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 
// [Sender] Sent event starting with 120
// [Receiver] Recv: 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 
// [Sender] Sent event starting with 120
// [Receiver] Recv: 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 
// [Sender] Sent event starting with 130
// [Receiver] Recv: 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 
// [Sender] Sent event starting with 130
// [Receiver] Recv: 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 
// [Sender] Sent event starting with 140
// [Receiver] Recv: 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 
// [Sender] Sent event starting with 140
// [Receiver] Recv: 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 
// [Sender] Sent event starting with 150
// [Receiver] Recv: 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 
// [Sender] Sent event starting with 150
// [Receiver] Recv: 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 
// [Sender] Sent event starting with 160
// [Receiver] Recv: 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 
// [Sender] Sent event starting with 160
// [Receiver] Recv: 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 
// [Sender] Sent event starting with 170
// [Receiver] Recv: 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 
// [Sender] Sent event starting with 170
// [Receiver] Recv: 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 
// 测试结果：
// 队列在内部 RAM 和 PSRAM 都能正常创建
// 队列元素正确进出队列，没有发生 assert failed。
// PSRAM 分配的队列也能正常读写。
// 任务发送/接收机制正常
// 发送者和接收者并行工作，事件完整传递。
// 没有出现乱码（在你原先疯狂打印的问题里可能是因为指针或者未初始化的内存）。
// 队列大小和数据对齐没问题
// 每个事件完整传输 16 个样本，和定义的 EVENT_SAMPLES 一致。
// 说明 FreeRTOS 队列的 sizeof(TinyMLEvent) 使用在内部 RAM 或 PSRAM 都是安全的。
// 测试说明： 需要把测试代码copy到main.cpp, 然后进行开发板中的测试
// */