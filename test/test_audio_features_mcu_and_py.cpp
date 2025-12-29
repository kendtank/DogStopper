// /*
//     py端封装的提取函数，提取三段真实的音频，同时提取了音频pcm的int16信号作为输入
//     测试误差精度
// */

// #include "unity.h"
// #include "audio_features.h"
// #include <string.h>
// #include <math.h>
// #include <Arduino.h>
// #include "../python/tinyml/aug_3.h"
// #include "../python/tinyml/mfcc_py_aug_3.h"
// #include "../python/tinyml/aug_10.h"
// #include "../python/tinyml/mfcc_py_aug_10.h"
// #include "../python/tinyml/aug_20.h"
// #include "../python/tinyml/mfcc_py_aug_20.h"





// static float test_input[INPUT_SAMPLES];
// static float test_output[LOGMEL_SIZE];
// static float test_output_mfcc[MFCC_SIZE];




// // 计算最大绝对误差
// float calculate_max_absolute_error(const float* ref, const float* actual, int size) {
//     float max_error = 0.0f;
//     for (int i = 0; i < size; i++) {
//         float error = fabsf(ref[i] - actual[i]);
//         if (error > max_error) {
//             max_error = error;
//         }
//     }
//     return max_error;
// }


// // 计算均方根误差
// float calculate_rmse(const float* ref, const float* actual, int size) {
//     float sum_squared_error = 0.0f;
//     for (int i = 0; i < size; i++) {
//         float error = ref[i] - actual[i];
//         sum_squared_error += error * error;
//     }
//     return sqrtf(sum_squared_error / size);
// }


// // 测试MFCC

// void test_mfcc_accuracy_aug_3(void) { 

//     bool is_init = init_feature_buffers();
//     TEST_ASSERT_TRUE(is_init);

//     // 直接使用python生成的数据头文件
//     // 添加一个计时器
//     u32_t start_time = micros();
//     int result = compute_mfcc_200ms(aug_3, test_output_mfcc);
//     Serial.printf("[MFCC精度测试] 耗时: %d ms\n", (micros() - start_time)/1000);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);

//     // 打印logmel特征的前10bin数据和后十个bin的数据
//     Serial.printf("[精度测试] mcu-mfcc-前10bin数据:\n");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("%.6f ", test_output_mfcc[i]);
//     }
//     // python mfcc特征的前10bin数据和后十个bin数据
//     Serial.printf("\n[精度测试] python-mfcc bin前10个数据:\n");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("%.6f ", mfcc_py_aug_3[i]);
//     }
    
//     // 验证所有帧的所有特征
//     const int total_features = NUM_FRAMES * N_MFCC;  // 18 * 13
//     float max_error = calculate_max_absolute_error(mfcc_py_aug_3, test_output_mfcc, total_features);
//     float rmse = calculate_rmse(mfcc_py_aug_3, test_output_mfcc, total_features);

//     Serial.printf("[MFCC精度测试] 最大绝对误差=%.6f, 均方根误差=%.6f\n", max_error, rmse);   //   500Hz正弦波: 最大绝对误差=0.002594, 均方根差=0.000350
    
//     // 设置严格的误差阈值
//     const float max_error_threshold = 0.1f;   // 最大绝对误差不超过0.1  db值在0.1-10db之间，已经非常优秀
//     const float rmse_threshold = 0.05f;      // 均方根误差不超过0.05
    
//     char error_msg[100];
//     sprintf(error_msg, "Max absolute error %.6f exceeds threshold %.6f", max_error, max_error_threshold);
//     TEST_ASSERT_TRUE_MESSAGE(max_error < max_error_threshold, error_msg);
    
//     sprintf(error_msg, "Max absolute error %.6f exceeds threshold %.6f", rmse, rmse_threshold);
//     TEST_ASSERT_TRUE_MESSAGE(rmse < rmse_threshold, error_msg);
// }

// void test_mfcc_accuracy_aug_10(void) { 

//     bool is_init = init_feature_buffers();
//     TEST_ASSERT_TRUE(is_init);

//     // 直接使用python生成的数据头文件
//     u32_t start_time = micros();
//     int result = compute_mfcc_200ms(aug_10, test_output_mfcc);
//     Serial.printf("[MFCC精度测试] 耗时: %d ms\n", (micros() - start_time)/1000);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);

//     // 打印logmel特征的前10bin数据和后十个bin的数据
//     Serial.printf("[精度测试] mcu-mfcc-前10bin数据:\n");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("%.6f ", test_output_mfcc[i]);
//     }
//     // python mfcc特征的前10bin数据和后十个bin数据
//     Serial.printf("\n[精度测试] python-mfcc bin前10个数据:\n");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("%.6f ", mfcc_py_aug_10[i]);
//     }
    
//     // 验证所有帧的所有特征
//     const int total_features = NUM_FRAMES * N_MFCC;  // 18 * 13
//     float max_error = calculate_max_absolute_error(mfcc_py_aug_10, test_output_mfcc, total_features);
//     float rmse = calculate_rmse(mfcc_py_aug_10, test_output_mfcc, total_features);

//     Serial.printf("[MFCC精度测试] 最大绝对误差=%.6f, 均方根误差=%.6f\n", max_error, rmse);   //   500Hz正弦波: 最大绝对误差=0.002594, 均方根差=0.000350
    
//     // 设置严格的误差阈值
//     const float max_error_threshold = 0.1f;   // 最大绝对误差不超过0.1  db值在0.1-10db之间，已经非常优秀
//     const float rmse_threshold = 0.05f;      // 均方根误差不超过0.05
    
//     char error_msg[100];
//     sprintf(error_msg, "Max absolute error %.6f exceeds threshold %.6f", max_error, max_error_threshold);
//     TEST_ASSERT_TRUE_MESSAGE(max_error < max_error_threshold, error_msg);
    
//     sprintf(error_msg, "Max absolute error %.6f exceeds threshold %.6f", rmse, rmse_threshold);
//     TEST_ASSERT_TRUE_MESSAGE(rmse < rmse_threshold, error_msg);
// }


// void test_mfcc_accuracy_aug_20(void) { 

//     bool is_init = init_feature_buffers();
//     TEST_ASSERT_TRUE(is_init);

//     // 直接使用python生成的数据头文件
//     u32_t start_time = micros();
//     int result = compute_mfcc_200ms(aug_20, test_output_mfcc);
//     Serial.printf("[MFCC精度测试] 耗时: %d ms\n", (micros() - start_time)/1000);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);

//     // 打印logmel特征的前10bin数据和后十个bin的数据
//     Serial.printf("[精度测试] mcu-mfcc-前10bin数据:\n");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("%.6f ", test_output_mfcc[i]);
//     }
//     // python mfcc特征的前10bin数据和后十个bin数据
//     Serial.printf("\n[精度测试] python-mfcc bin前10个数据:\n");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("%.6f ", mfcc_py_aug_20[i]);
//     }
    
//     // 验证所有帧的所有特征
//     const int total_features = NUM_FRAMES * N_MFCC;  // 18 * 13
//     float max_error = calculate_max_absolute_error(mfcc_py_aug_20, test_output_mfcc, total_features);
//     float rmse = calculate_rmse(mfcc_py_aug_20, test_output_mfcc, total_features);

//     Serial.printf("[MFCC精度测试] 最大绝对误差=%.6f, 均方根误差=%.6f\n", max_error, rmse);   //   500Hz正弦波: 最大绝对误差=0.002594, 均方根差=0.000350
    
//     // 设置严格的误差阈值
//     const float max_error_threshold = 0.1f;   // 最大绝对误差不超过0.1  db值在0.1-10db之间，已经非常优秀
//     const float rmse_threshold = 0.05f;      // 均方根误差不超过0.05
    
//     char error_msg[100];
//     sprintf(error_msg, "Max absolute error %.6f exceeds threshold %.6f", max_error, max_error_threshold);
//     TEST_ASSERT_TRUE_MESSAGE(max_error < max_error_threshold, error_msg);
    
//     sprintf(error_msg, "Max absolute error %.6f exceeds threshold %.6f", rmse, rmse_threshold);
//     TEST_ASSERT_TRUE_MESSAGE(rmse < rmse_threshold, error_msg);
// }



// void setUp(void) {
// }

// void tearDown(void) {
// }

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_mfcc_accuracy_aug_3);
//     RUN_TEST(test_mfcc_accuracy_aug_10);
//     RUN_TEST(test_mfcc_accuracy_aug_20);
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



// /*
// Testing...
// If you don't see any output for the first 10 secs, please reset board (press reset button)

// TimeoutError: Could not automatically find serial port for the `Espressif ESP32-S3-DevKitC-1-N8 (8 MB QD, No PSRAM)` board based on the declared HWIDs=['303A:1001']
// ,boot:0x8 (SPI_FAST_FLASH_BOOT)
// SPIWP:0xee
// mode:DIO, clock div:1
// load:0x3fce3808,len:0x4bc
// load:0x403c9700,len:0xbd8
// load:0x403cc700,len:0x2a0c
// entry 0x403c98d0
// E (191) esp_core_dump_flash: No core dump partition found!
// E (191) esp_core_[MFCC] : 14 ms
// [] mcu-mfcc-10bin:
// -230.906738 84.329803 20.346903 -1.755843 8.647002 -16.128767 0.919875 -9.791492 1.856562 -3.267820 
// [] python-mfcc bin10:
// -230.901917 84.326942 20.345694 -1.756472 8.646265 -16.131538 0.924685 -9.796757 1.863426 -3.272874 [MFCC] =0.038795, =0.010895
// test/test_audio_features_mcu_and_py.cpp:190:test_mfcc_accuracy_aug_3:PASS
// [MFCC] : 12 ms
// [] mcu-mfcc-10bin:
// -175.993607 50.592129 -30.930349 20.422026 -8.830701 -6.271747 -25.413607 2.704144 21.443384 30.489834 
// [] python-mfcc bin10:
// -175.990265 50.587410 -30.927769 20.419739 -8.828226 -6.278226 -25.410358 2.712247 21.447262 30.492245 [MFCC] =0.042492, =0.013332
// test/test_audio_features_mcu_and_py.cpp:191:test_mfcc_accuracy_aug_10:PASS
// [MFCC] : 13 ms
// [] mcu-mfcc-10bin:
// -149.933609 64.756668 -12.778255 -1.470561 -13.787678 8.354923 -7.064749 -0.840027 -7.642793 -14.379223 
// [] python-mfcc bin10:
// -149.925491 64.750473 -12.779612 -1.470924 -13.787410 8.356293 -7.065049 -0.839793 -7.650611 -14.368452 [MFCC] =0.039062, =0.010461
// test/test_audio_features_mcu_and_py.cpp:192:test_mfcc_accuracy_aug_20:PASS

// -----------------------
// 3 Tests 0 Failures 0 Ignored 
// */