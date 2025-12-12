// /*
// */

// #include "unity.h"
// #include "audio_features.h"
// #include <string.h>
// #include <math.h>
// #include <Arduino.h>
// #include "../python/tinyml/aug_3.h"
// #include "../python/tinyml/aug_10.h"
// #include "../python/tinyml/aug_20.h"
// #include "tiny_model.h"



// static float test_input[INPUT_SAMPLES];
// static float test_output[LOGMEL_SIZE];
// static float test_embed_input[32];




// // 测试embed模型推理

// void test_embed_model_aug_3(void) { 

//     bool is_init = init_feature_buffers();
//     TEST_ASSERT_TRUE(is_init);

//     // 直接使用python生成的数据头文件
//     int result = compute_logmel_200ms(aug_3, test_output, 1);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);


//     // 调用推理接口
//     // 初始化模型
//     if (logmel_model_init() != 0) {
//     Serial.println("EMBED model init failed!");
//     while (1);
//     }
//     Serial.println("EMBED model init success!");

//     // 记录开始时间
//     unsigned long start_time = millis();

//     // 推理
//     int re = embed_model_infer(test_output, test_embed_input);
//     // 计算耗时
//     unsigned long end_time = millis();
//     unsigned long inference_time = end_time - start_time;
//     Serial.printf("Inference time: %lu ms\n", inference_time);
//     TEST_ASSERT_EQUAL_INT(0, re);

//     // 打印32维度的embedding结果
//     Serial.println("Embedding output:");
//     for (int i = 0; i < 32; i++) {
//         Serial.print(test_embed_input[i], 6);
//         Serial.print(", ");
//     } 
// }


// void setUp(void) {
// }

// void tearDown(void) {
// }

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_embed_model_aug_3);
//     return UNITY_END();
// }

// void setup() {
//     Serial.begin(115200);
//     delay(2000);
//     runUnityTests();
// }

// void loop() {
//     delay(1000);
// }



// /* 测试结果

// */