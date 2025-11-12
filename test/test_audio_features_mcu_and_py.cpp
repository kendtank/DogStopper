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
//     int result = compute_mfcc_200ms(aug_3, test_output_mfcc);
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
//     int result = compute_mfcc_200ms(aug_10, test_output_mfcc);
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
//     int result = compute_mfcc_200ms(aug_20, test_output_mfcc);
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
// [MEM] 堆内存剩余: 317.691406 KB
// [MEM] PSRAM总大小: 8189.663086 KB
// [MEM] PSRAM可用: 8137.815430 KB
// [MEM] 堆内存剩: 317.433594 KB
// [MEM] PSRAM总大小: 8189.663086 KB
// [MEM] PSRAM可用: 8137.815430 KB
// frames_win: 18 frames
// fft_power_init: 始化完成, FFT大小 = 512
// fft_power_compute: 功率谱计算完成
// [MEM] 堆内存剩余: 310.449219 KB
// [MEM] PSRAM总大小: 8189.663086 KB
// [MEM] PSRAM可用: 8137.815430 KB
// [TIME] compute_logmel_200ms 执行耗时: 58179 us (58.179 ms)
// [MEM] 堆内存剩余: 310.449219 KB
// [MEM] PSRAM总大小: 8189.663086 KB
// [MEM] PSRAM用: 8137.815430 KB
// [MEM] 堆内存剩余: 310.449219 KB
// [MEM] PSRAM总大小: 8189.663086 KB
// [MEM] PSRAM可用: 8137.815430 KB
// [TIME] 单独计算mfcc时间开销 执行耗时: 10849 us (10.849 ms)
// [TIME] 总计算mfcc时间开销 执行耗时: 100287 us (100.287 ms)
// [精度测试] mcu-mfcc-前10bin数据:
// -295.590973 105.727707 24.521528 -3.784542 9.139164 -21.907455 0.532185 -13.506745 0.757442 -4.937358 
// [精度测试] python-mfcc bin前10个数据:
// -295.588745 105.732544 24.513809 -3.780352 9.136531 -21.911320 0.537862 -13.513548 0.765523 -4.944677 [MFCC精度测试] 最大绝对误差=0.031578, 均方根误差=0.009195
// test/test_audio_features_mcu_and_py.cpp:183:test_mfcc_accuracy_aug_3:PASS
// [MEM] 堆内存剩余: 297.933594 KB
// [MEM] PSRAM总大小: 8189.600586 KB
// [MEM] PSRAM可用: 8086.143555 KB
// [MEM] 堆内存剩: 297.933594 KB
// [MEM] PSRAM总大小: 8189.600586 KB
// [MEM] PSRAM可用: 8086.143555 KB
// frames_win: 18 frames
// fft_power_init: 初始化，无需重复
// fft_power_compute: 功率谱计算完成
// [MEM] 堆内存剩余: 297.933594 KB
// [MEM] PSRAM总大小: 8189.600586 KB
// [MEM] PSRAM可用: 8086.143555 KB
// [TIME] compute_logmel_200ms 执行耗时: 66274 us (66.274 ms)
// [MEM] 堆内存剩余: 297.933594 KB
// [MEM] PSRAM总大小: 8189.600586 KB
// [MEM] PSRAM用: 8086.143555 KB
// [MEM] 堆内存剩余: 297.933594 KB
// [MEM] PSRAM总大小: 8189.600586 KB
// [MEM] PSRAM可用: 8086.143555 KB
// [TIME] 单独计算mfcc时间开销 执行耗时: 10849 us (10.849 ms)
// [TIME] 总计算mfcc时间开销 执行耗时: 108287 us (108.287 ms)
// [精度测试] mcu-mfcc-前10bin数据:
// -233.684830 63.682377 -43.375397 25.557131 -12.887185 -11.875598 -32.436863 5.031828 29.087748 39.692867 
// [精度测试] python-mfcc bin前10个数据:
// -233.679733 63.678844 -43.371521 25.554783 -12.883421 -11.883199 -32.434013 5.040243 29.093575 39.695473 [MFCC精度测试] 最大绝对误差=0.031784, 均方根误差=0.010109
// test/test_audio_features_mcu_and_py.cpp:184:test_mfcc_accuracy_aug_10:PASS
// [MEM] 堆内存剩余: 285.417969 KB
// [MEM] PSRAM总大小: 8189.538086 KB
// [MEM] PSRAM可用: 8034.471680 KB
// [MEM] 堆内存剩: 285.417969 KB
// [MEM] PSRAM总大小: 8189.538086 KB
// [MEM] PSRAM可用: 8034.471680 KB
// frames_win: 18 frames
// fft_power_init: 初始化，无需重复
// fft_power_compute: 功率谱计算完成
// [MEM] 堆内存剩余: 285.417969 KB
// [MEM] PSRAM总大小: 8189.538086 KB
// [MEM] PSRAM可用: 8034.471680 KB
// [TIME] compute_logmel_200ms 执行耗时: 66153 us (66.153 ms)
// [MEM] 堆内存剩余: 285.417969 KB
// [MEM] PSRAM总大小: 8189.538086 KB
// [MEM] PSRAM用: 8034.471680 KB
// [MEM] 堆内存剩余: 285.417969 KB
// [MEM] PSRAM总大小: 8189.538086 KB
// [MEM] PSRAM可用: 8034.471680 KB
// [TIME] 单独计算mfcc时间开销 执行耗时: 10850 us (10.850 ms)
// [TIME] 总计算mfcc时间开销 执行耗时: 108181 us (108.181 ms)
// [精度测试] mcu-mfcc-前10bin数据:
// -197.903061 81.145950 -17.537302 -2.247563 -17.779985 8.426172 -10.868982 -2.416545 -9.965975 -20.399273 
// [精度测试] python-mfcc bin前10个数据:
// -197.897583 81.149292 -17.542450 -2.255974 -17.775555 8.429510 -10.871561 -2.409623 -9.986266 -20.384758 [MFCC精度测试] 最大绝对误差=0.032432, 均方根误差=0.009623
// test/test_audio_features_mcu_and_py.cpp:185:test_mfcc_accuracy_aug_20:PASS

// -----------------------
// 3 Tests 0 Failures 0 Ignored 
// OK
// */