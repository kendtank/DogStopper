
// #include "unity.h"
// #include "audio_features.h"
// #include <string.h>
// #include <math.h>
// #include <Arduino.h>
// #include "tiny_model.h"


// // py端生成的测试数据，保持一致性
// #include "../python/verify/headers/dogA1_aug1.h"
// #include "../python/verify/headers/dogA1_aug2.h"
// #include "../python/verify/headers/dogA1_raw.h"
// #include "../python/verify/headers/dogA2_raw.h"
// #include "../python/verify/headers/dogB1_raw.h"
// #include "../python/verify/headers/dogC1_raw.h"
// #include "../python/verify/headers/embed_dogA.h"


// /*
// 测试用例，准备四组音频数据，分别是相同狗吠两组，和两组不同狗吠, 其中相同的狗吠做了简单的数据增强处理，一共六组数据
// 然后计算logmel特征，调用embed模型推理，观察输出的32维embedding
// 使用余弦相似度计算不同音频之间embedding的相似度
// 预期结果：
// 1. 相同狗吠的两组数据，embedding余弦相似度应比较高
// 2. 不同狗吠的两组数据，embedding余弦相似度应明显不相关
// */


// static float test_output[LOGMEL_SIZE];
// static float test_embed_output[EMBED_OUTPUT_SIZE];


// // 最大绝对误差
// float max_abs_diff(float *a, const float *b, int len) {
//     float max_diff = 0;
//     for (int i = 0; i < len; i++) {
//         float diff = fabsf(a[i] - b[i]);
//         if (diff > max_diff) {
//             max_diff = diff;
//         }
//     }
//     Serial.printf("Max abs diff: %f\n", max_diff);
//     return max_diff;
// }


// // 推理某段音频并得到 embedding
// bool compute_embedding(const int16_t *wav, float *embed_out) {
//     // 计算耗时
//     u_int32_t start_time = millis();

//     // 生成 logmel
//     int result = compute_logmel_200ms(wav, test_output, 1);
//     if (result != NUM_FRAMES) {
//         Serial.println("Logmel compute failed");
//         return false;
//     }

//     // 推理 embedding
//     int re = embed_model_infer(test_output, embed_out);

//     Serial.printf("Inference time: %d ms\n", millis() - start_time);
//     return (re == 0);
// }

// // 打印 NxN 相似度矩阵
// void print_similarity_matrix(const char **names, float **embeds, int count) {

//     Serial.println("\n=== Cosine Similarity Matrix ===");

//     // ---- 打印列标题 ----
//     Serial.printf("%12s", "");  // 左上角空位置
//     for (int i = 0; i < count; i++) {
//         Serial.printf("%12s", names[i]);
//     }
//     Serial.println();

//     // ---- 打印矩阵 ----
//     for (int i = 0; i < count; i++) {
//         Serial.printf("%12s", names[i]);

//         for (int j = 0; j < count; j++) {
//             float sim = cosine_similarity(embeds[i], embeds[j], EMBED_OUTPUT_SIZE);
//             Serial.printf("%12.3f", sim);
//         }
//         Serial.println();
//     }
// }



// // ===================== 主测试 =====================

// void test_embed_model_acc(void) {

//     // ----- 1. logmel buffer init -----
//     bool is_init = init_feature_buffers();
//     TEST_ASSERT_TRUE(is_init);

//     // ----- 2. 模型只初始化一次 -----
//     if (logmel_model_init() != 0) {
//         Serial.println("EMBED model init failed!");
//         TEST_FAIL();
//     }
//     Serial.println("EMBED model init success!");


//     // ============================================================
//     // (A) 量化误差验证 ———— dogA1_raw vs embed_dogA（Python）
//     // ============================================================
//     int result = compute_logmel_200ms(dogA1_raw, test_output, 1);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);

//     int re = embed_model_infer(test_output, test_embed_output);
//     TEST_ASSERT_EQUAL_INT(0, re);

//     Serial.println("Embedding output:");
//     for (int i = 0; i < EMBED_OUTPUT_SIZE; i++) {
//         Serial.print(test_embed_output[i], 6);
//         Serial.print(", ");
//     }
//     Serial.println();

//     float max_diff = max_abs_diff(test_embed_output, embed_dogA, EMBED_OUTPUT_SIZE);
//     TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, max_diff);

//     float cos_sim = cosine_similarity(test_embed_output, embed_dogA, EMBED_OUTPUT_SIZE);
//     Serial.printf("Cosine similarity with reference embedding: %f\n", cos_sim);
//     TEST_ASSERT_TRUE(cos_sim > 0.99f);



//     // ============================================================
//     // (B) 批量 6 段音频 ———— 计算 embedding & 打印相似度矩阵
//     // ============================================================

//     static const char *names[6] = {
//         "A1_raw",
//         "A1_aug1",
//         "A1_aug2",
//         "A2_raw",
//         "B1_raw",
//         "C1_raw"
//     };

//     static const int16_t *audios[6] = {
//         dogA1_raw,
//         dogA1_aug1,
//         dogA1_aug2,
//         dogA2_raw,
//         dogB1_raw,
//         dogC1_raw
//     };

//     static float embeds[6][EMBED_OUTPUT_SIZE];
//     static float* embed_ptrs[6];

//     for (int i = 0; i < 6; i++) embed_ptrs[i] = embeds[i];

//     for (int i = 0; i < 6; i++) {
//         // Serial.printf("\n[Embedding] Processing %s ...\n", names[i]);

//         bool ok = compute_embedding(audios[i], embeds[i]);
//         TEST_ASSERT_TRUE(ok);

//         // for (int k = 0; k < EMBED_OUTPUT_SIZE; k++) {
//         //     Serial.print(embeds[i][k], 4);
//         //     Serial.print(", ");
//         // }
//         // Serial.println();
//     }

//     print_similarity_matrix(names, embed_ptrs, 6);
// }


// // ============================================================
// // Unity framework
// // ============================================================
// void setUp() {}
// void tearDown() {}

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_embed_model_acc);
//     return UNITY_END();
// }

// void setup() {
//     Serial.begin(115200);
//     delay(2000);
//     runUnityTests();
// }

// void loop() { delay(1000); }



// /*
// ESP-ROM:esp32s3-20210327
// Build:Mar 27 2021
// rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)
// SPIWP:0xee
// mode:DIO, clock div:1
// load:0x3fce3808,len:0x4bc
// load:0x403c9700,len:0xbd8
// load:0x403cc700,len:0x2a0c
// entry 0x403c98d0
// E (200) esp_core_dump_flash: No core dump partition found!
// E (200) esp_core_dump_flash: No core dump partition found!
// EMBED model init success!
// Embedding output:
// 0.025663, 0.002632, -0.017109, -0.011845, -0.020399, -0.011845, -0.023031, -0.001316, -0.034218, -0.011186, 0.022373, -0.005922, -0.042772, -0.022373, 0.008554, -0.009212, -0.025005, 0.044088, -0.003290, 0.026321, 0.003290, 0.013819, -0.003290, 0.001974, -0.026321, -0.003948, -0.031585, -0.010528, -0.007238, 0.017109, 0.011845, -0.011186, 
// Max abs diff: 0.004780
// Cosine similarity with reference embedding: 0.993260
// Inference time: 304 ms
// Inference time: 305 ms
// Inference time: 304 ms
// Inference time: 305 ms
// Inference time: 304 ms
// Inference time: 304 ms

// === Cosine Similarity Matrix ===
//                   A1_raw     A1_aug1     A1_aug2      A2_raw      B1_raw      C1_raw
//       A1_raw       1.000       0.664       0.990       0.833      -0.620       0.138
//      A1_aug1       0.664       1.000       0.658       0.690      -0.453       0.318
//      A1_aug2       0.990       0.658       1.000       0.837      -0.650       0.082
//       A2_raw       0.833       0.690       0.837       1.000      -0.473       0.108
//       B1_raw      -0.620      -0.453      -0.650      -0.473       1.000      -0.249
//       C1_raw       0.138       0.318       0.082       0.108      -0.249       1.000
// test/test_embed_model_accuracy.cpp:186:test_embed_model_acc:PASS
// */