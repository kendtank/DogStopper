// /*
//  * 测试音频特征提取功能
//     * 使用Unity测试框架
//     * 覆盖封装函数的所有执行路径
//     * 确保每个路径至少有一个测试用例
// */

// #include "unity.h"
// #include "audio_features.h"
// #include <string.h>
// #include <math.h>
// #include <Arduino.h>

// // 测试用的全局变量
// static int16_t test_input[INPUT_SAMPLES];
// static float test_output[LOGMEL_SIZE > MFCC_SIZE ? LOGMEL_SIZE : MFCC_SIZE];

// // 测试前的设置函数
// void setUp(void) {
//     // 每个测试前初始化输入为静音
//     memset(test_input, 0, sizeof(test_input));
//     memset(test_output, 0, sizeof(test_output));
    
//     // 确保每次测试都从干净的状态开始
//     feature_extractor_free();
// }

// // 测试后的清理函数
// void tearDown(void) {
//     // 每个测试后释放资源
//     feature_extractor_free();
// }

// // 测试 feature_extractor_init 路径1: 正常初始化
// void test_feature_extractor_init_path_normal(void) {
//     int result = feature_extractor_init();
//     TEST_ASSERT_EQUAL_INT(0, result);
// }

// // 测试 feature_extractor_init 路径2: 重复初始化
// void test_feature_extractor_init_path_reinit(void) {
//     // 第一次初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 第二次初始化应该也成功
//     int result = feature_extractor_init();
//     TEST_ASSERT_EQUAL_INT(0, result);
// }

// // 测试 feature_extractor_free 路径1: 正常释放
// void test_feature_extractor_free_path_normal(void) {
//     // 初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 释放资源
//     feature_extractor_free();
// }

// // 测试 feature_extractor_free 路径2: 释放未初始化状态
// void test_feature_extractor_free_path_when_not_inited(void) {
//     // 确保未初始化
//     feature_extractor_free();
//     // 释放未初始化状态不应出错
//     feature_extractor_free();
// }

// // 测试 compute_logmel_200ms 路径1: 未初始化->初始化成功->正常处理
// void test_compute_logmel_200ms_path_uninit_init_success_process(void) {
//     // 确保未初始化
//     feature_extractor_free();
//     // 输入输出都有效
//     int result = compute_logmel_200ms(test_input, test_output);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);
// }

// // 测试 compute_logmel_200ms 路径2: 未初始化->初始化失败
// // 这个路径较难模拟，因为我们无法轻易让初始化失败

// // 测试 compute_logmel_200ms 路径3: 已初始化->输入为NULL
// void test_compute_logmel_200ms_path_inited_input_null(void) {
//     // 先初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 输入为NULL
//     int result = compute_logmel_200ms(NULL, test_output);
//     TEST_ASSERT_EQUAL_INT(-2, result);
// }

// // 测试 compute_logmel_200ms 路径4: 已初始化->输出为NULL
// void test_compute_logmel_200ms_path_inited_output_null(void) {
//     // 先初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 输出为NULL
//     int result = compute_logmel_200ms(test_input, NULL);
//     TEST_ASSERT_EQUAL_INT(-2, result);
// }

// // 测试 compute_logmel_200ms 路径5: 已初始化->输入输出都有效
// void test_compute_logmel_200ms_path_inited_normal(void) {
//     // 先初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 输入输出都有效
//     int result = compute_logmel_200ms(test_input, test_output);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);
// }

// // 测试 compute_mfcc_200ms 路径1: 未初始化->初始化成功->正常处理
// void test_compute_mfcc_200ms_path_uninit_init_success_process(void) {
//     // 确保未初始化
//     feature_extractor_free();
//     // 输入输出都有效
//     int result = compute_mfcc_200ms(test_input, test_output);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);
// }

// // 测试 compute_mfcc_200ms 路径2: 未初始化->初始化失败
// // 这个路径较难模拟，因为我们无法轻易让初始化失败

// // 测试 compute_mfcc_200ms 路径3: 已初始化->输入为NULL
// void test_compute_mfcc_200ms_path_inited_input_null(void) {
//     // 先初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 输入为NULL
//     int result = compute_mfcc_200ms(NULL, test_output);
//     TEST_ASSERT_EQUAL_INT(-2, result);
// }

// // 测试 compute_mfcc_200ms 路径4: 已初始化->输出为NULL
// void test_compute_mfcc_200ms_path_inited_output_null(void) {
//     // 先初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 输出为NULL
//     int result = compute_mfcc_200ms(test_input, NULL);
//     TEST_ASSERT_EQUAL_INT(-2, result);
// }

// // 测试 compute_mfcc_200ms 路径5: 已初始化->输入输出都有效->logmel计算失败
// // 这个路径较难模拟，因为正常情况下logmel计算不会失败

// // 测试 compute_mfcc_200ms 路径6: 已初始化->输入输出都有效->logmel计算成功
// void test_compute_mfcc_200ms_path_inited_normal(void) {
//     // 先初始化
//     TEST_ASSERT_EQUAL_INT(0, feature_extractor_init());
//     // 输入输出都有效
//     int result = compute_mfcc_200ms(test_input, test_output);
//     TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);
// }

// // 测试 compute_dct_ii 路径1: 正常计算
// void test_compute_dct_ii_path_normal(void) {
//     float input[N_MEL_BINS];
//     float output[N_MFCC];
    
//     // 初始化输入为固定值
//     for (int i = 0; i < N_MEL_BINS; i++) {
//         input[i] = 1.0f;
//     }
    
//     compute_dct_ii(input, output, N_MEL_BINS, N_MFCC);
    
//     // 验证输出不为NaN或无穷大
//     for (int i = 0; i < N_MFCC; i++) {
//         TEST_ASSERT_TRUE(isfinite(output[i]));
//     }
// }

// // 测试 compute_dct_ii 路径2: 输入为NULL
// void test_compute_dct_ii_path_input_null(void) {
//     float output[N_MFCC];
//     compute_dct_ii(NULL, output, N_MEL_BINS, N_MFCC);
//     // 函数没有返回值，只要不崩溃就算通过
// }

// // 测试 compute_dct_ii 路径3: 输出为NULL
// void test_compute_dct_ii_path_output_null(void) {
//     float input[N_MEL_BINS];
//     // 初始化输入为固定值
//     for (int i = 0; i < N_MEL_BINS; i++) {
//         input[i] = 1.0f;
//     }
    
//     compute_dct_ii(input, NULL, N_MEL_BINS, N_MFCC);
//     // 函数没有返回值，只要不崩溃就算通过
// }

// // 测试 print_memory_info 路径1: 正常执行
// void test_print_memory_info_path_normal(void) {
//     // 只需验证函数能正常执行不崩溃
//     print_memory_info();
// }

// // 主函数
// int runUnityTests(void) {
//     UNITY_BEGIN();
    
//     // feature_extractor_init 路径测试
//     RUN_TEST(test_feature_extractor_init_path_normal);
//     RUN_TEST(test_feature_extractor_init_path_reinit);
    
//     // feature_extractor_free 路径测试
//     RUN_TEST(test_feature_extractor_free_path_normal);
//     RUN_TEST(test_feature_extractor_free_path_when_not_inited);
    
//     // compute_logmel_200ms 路径测试
//     RUN_TEST(test_compute_logmel_200ms_path_uninit_init_success_process);
//     RUN_TEST(test_compute_logmel_200ms_path_inited_input_null);
//     RUN_TEST(test_compute_logmel_200ms_path_inited_output_null);
//     RUN_TEST(test_compute_logmel_200ms_path_inited_normal);
    
//     // compute_mfcc_200ms 路径测试
//     RUN_TEST(test_compute_mfcc_200ms_path_uninit_init_success_process);
//     RUN_TEST(test_compute_mfcc_200ms_path_inited_input_null);
//     RUN_TEST(test_compute_mfcc_200ms_path_inited_output_null);
//     RUN_TEST(test_compute_mfcc_200ms_path_inited_normal);
    
//     // compute_dct_ii 路径测试
//     RUN_TEST(test_compute_dct_ii_path_normal);
//     RUN_TEST(test_compute_dct_ii_path_input_null);
//     RUN_TEST(test_compute_dct_ii_path_output_null);
    
//     // print_memory_info 路径测试
//     RUN_TEST(test_print_memory_info_path_normal);
    
//     return UNITY_END();
// }

// // Arduino框架需要的入口函数
// void setup() {
//     // 初始化串口用于测试输出
//     Serial.begin(115200);
//     delay(2000); // 等待串口稳定
    
//     runUnityTests();
// }

// void loop() {
//     // 测试运行完成后不需要循环执行
//     delay(1000);
// }