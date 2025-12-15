// #include <Arduino.h>
// #include <unity.h>
// #include "verify_embedding.h"
// #include "bark_detector.h"  // BarkEvent结构体
// #include "audio_features.h"

// // 测试用音频 PCM
// #include "../python/tinyml/aug_3.h"

// // ----------------- 打印 embedding -----------------
// void print_embedding(const float* embed, size_t size) {
//     Serial.println("Embedding (first 8 values):");
//     for (size_t i = 0; i < 8 && i < size; i++) {
//         Serial.printf("%.6f ", embed[i]);
//     }
//     Serial.println();
// }

// // ---------------- Unity Test ----------------
// void test_embedding_inference(void) {

//     init_feature_buffers(); // 初始化特征提取所需buffer
    
//     Serial.println("=== Embedding Inference Test Start ===");

//     // 初始化推理模块
//     TEST_ASSERT_TRUE(verify_embedding_init());

//     // 构造 BarkEvent
//     BarkEvent evt;
//     memcpy(evt.samples, aug_3, 3200);  // PCM 数据
//     evt.length = 3200;
//     evt.timestamp_ms = millis();

//     // 推理 embedding
//     float embedding[EMBED_OUTPUT_SIZE] = {0};
//     TEST_ASSERT_TRUE(tinyml_embedding_inference(evt.samples, evt.length, embedding));

//     // 打印前几个 embedding 值
//     print_embedding(embedding, EMBED_OUTPUT_SIZE);
//     Serial.println("Embedding inference done");
// }

// void setup() {
//     Serial.begin(115200);
//     delay(2000); // 等待串口稳定

//     UNITY_BEGIN();
//     RUN_TEST(test_embedding_inference);
//     UNITY_END();
// }

// void loop() {
//     // Nothing to do
// }
