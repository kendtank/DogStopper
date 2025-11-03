// /*
//  * 逐步调试测试
//  * 用于生成与Python端相同的中间结果，以便进行对比
//  * 
//  * 测试结果：
//  * 1. Hann窗函数生成测试：PASS
//  * 2. 窗后信号全量测试：PASS
//  */

// #include "unity.h"
// #include "mfcc.h"
// #include <string.h>
// #include <Arduino.h>
// #include <math.h>
// #include "dsps_fft2r.h"

// // ==================== 参数定义 ====================
// #define SR              16000
// #define FRAME_SIZE      400
// #define FRAME_SHIFT     160
// #define N_MEL           128
// #define N_MFCC          40
// #define NFFT_BINS       (FRAME_SIZE / 2 + 1)
// #define NFFT FRAME_SIZE
// #define NBIN (NFFT/2 + 1)

// // 容差（根据经验可调整）
// #define TOL_WIN   1e-6f

// // ==================== 导入Python生成的中间结果 ====================
// #include "../python/test_data.h"
// #include "../python/out/hann_window.h"
// #include "../python/out/frame0_windowed.h"
// #include "../python/out/frame_windowed_all.h"

// // ==================== 工具函数 ====================

// static float max_abs_diff(const float *a, const float *b, int n)
// {
//     // 计算两个数组的绝对差， 返回最大差值
//     float maxd = 0;
//     for (int i = 0; i < n; i++) {
//         float d = fabsf(a[i] - b[i]);
//         if (d > maxd)
//             maxd = d;
//     }
//     Serial.print("max abs diff: ");
//     Serial.println(maxd, 6);

//     return maxd;
// }

// // ==================== 阶段1：Hann窗测试 [PASS]====================
// void test_hann_window(void)
// {
//     // 先在栈上分配一个数组
//     float win_c[FRAME_SIZE];
//     // 直接使用mfcc.cpp中的窗函数生成逻辑
//     make_hann(win_c);
//     float diff = max_abs_diff(win_c, hann_window, FRAME_SIZE);
//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff);
// }


// // // ==================== 阶段2：窗后信号测试[pass] ====================

// // ==================== 测试第一帧 ====================
// void test_frame0_windowed(void)
// {
//     float win[FRAME_SIZE];
//     make_hann(win);

//     float frame[FRAME_SIZE];   // 存放窗后的第一帧
//     const float *frame_src = test_input_signal; // 拿到3200点输入数据的第一帧起点

//     for (int i = 0; i < FRAME_SIZE; i++)
//         frame[i] = frame_src[i] * win[i];  // 第一帧的400点都加权

//     // 对比Python导出的窗后数据
//     float diff1 = max_abs_diff(frame, frame0_windowed, FRAME_SIZE); // 直接对比第一帧
//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff1);

//     // 需要注意：：：NumPy 默认是 列优先保存（Fortran-like access）
//     // // 对比全量导出数据的第一帧
//     float diff2 = max_abs_diff(frame, frame_windowed_all, FRAME_SIZE);
//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff2);
// }


// void test_frame_windowed(void)
// {
//     float win[FRAME_SIZE];
//     // 生成Hann窗
//     make_hann(win);
//     // 实际帧数
//     int num_frames = (TEST_SIGNAL_LENGTH - FRAME_SIZE) / FRAME_SHIFT + 1;  // 18帧
//     Serial.print("num frames: ");
//     Serial.println(num_frames);
//     const int total_points = num_frames * FRAME_SIZE;  // 7200点
//     Serial.print("total points: ");
//     Serial.print(total_points);

//     // 生成全量窗后信号
//     float frames_windowed_all_c[total_points];  // 存放全量窗后信号
//     // 按帧滑动处理
//     for (int f = 0; f < num_frames; f++) {
//         const float *frame_src = test_input_signal + f * FRAME_SHIFT;  // 取出输入信号的第 f 帧起点。滑动是160点
//         float *frame_dst = frames_windowed_all_c + f * FRAME_SIZE;  // 处理好的窗后帧存放位置。每帧400点
//         // 对这个帧中的每个样本加权， 也就是400点
//         for (int i = 0; i < FRAME_SIZE; i++)
//             frame_dst[i] = frame_src[i] * win[i];
//     }
//     // 打印前十个值看看
//     Serial.println("C端前10个窗后值：");
//     for (int i = 0; i < 10; i++) {
//         Serial.println(frames_windowed_all_c[i], 8);
//     }
//     Serial.println("Python端前10个窗后值：");
//     for (int i = 0; i < 10; i++) {
//         Serial.println(frame_windowed_all[i], 8);
//     }

//     // 一次性与 Python 导出的窗后数组（已转置保存）进行对比
//     float diff = max_abs_diff(frames_windowed_all_c, frame_windowed_all, total_points);
//     Serial.print("全量窗口后 diff = ");
//     Serial.println(diff, 6);
//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff);
// }



// // ==================== 主入口 ====================
// void setUp(void) {
//     // 测试前的设置
// }

// void tearDown(void) {
//     // 测试后的清理
// }

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_hann_window);
//     RUN_TEST(test_frame0_windowed);
//     RUN_TEST(test_frame_windowed);
//     return UNITY_END();
// }

// void setup() {
//     // Wait ~2 seconds before the Unity test runner
//     // if you don't want to wait for the debugger attach
//     Serial.begin(115200);
//     delay(2000);

//     runUnityTests();
// }

// void loop() {
//     // 测试运行完成后不需要循环执行
//     delay(1000);
// }



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
// max abs diff: 0.000000
// test/test_debug_hann.cpp:143:test_hann_window:PASS
// max abs diff: 0.000000
// max abs diff: 0.000000
// test/test_debug_hann.cpp:144:test_frame0_windowed:PASS
// num frames: 18
// total points: 7200C端前10个窗后值：
// 0.00000000
// 0.00000361
// 0.00002833
// 0.00009251
// 0.00020930
// 0.00038447
// 0.00061503
// 0.00088845
// 0.00118279
// 0.00146770
// Python端前10个窗后值：
// 0.00000000
// 0.00000361
// 0.00002832
// 0.00009251
// 0.00020930
// 0.00038447
// 0.00061503
// 0.00088845
// 0.00118279
// 0.00146770
// max abs diff: 0.000000
// 全量窗口后 diff = 0.000000
// test/test_debug_hann.cpp:145:test_frame_windowed:PASS
// -----------------------
// 3 Tests 0 Failures 0 Ignored 
// OK
// -------------------------------------------------------------------------------------------------------- esp32s3-n16r8:* [PASSED] Took 13.23 seconds --------------------------------------------------------------------------------------------------------
// ========================================================================================================================== SUMMARY ==========================================================================================================================
// Environment    Test    Status    Duration
// -------------  ------  --------  ------------
// esp32s3-n16r8  *       PASSED    00:00:13.226
// ========================================================================================================= 3 test cases: 3 succeeded in 00:00:13.226 =========================================================================================================
// (base) kend@kend-Guanxin:~/文档/PlatformIO/Projects/DogStopper$ 
// */