// /*
//  * 逐步调试测试
//  * 用于生成与Python端相同的中间结果，以便进行对比
//  * 
//  * 测试结果：
//  * 调用封装的hann_frame方法， 加权后的数据和py对比：PASS
//  */

// #include "unity.h"
// #include <string.h>
// #include <Arduino.h>
// #include <math.h>
// #include "hann_frame.h"

// // 容差
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


// // // ==================== 窗后信号测试[pass] ====================


// // ==================== 测试第一帧 [pass]====================
// void test_frame0_windowed(void)
// {
//     Serial.println("———— 开始测试第一帧窗后信号...");

//     float frame[FRAME_SIZE];   // 存放窗后的第一帧
//     const float *frame_src = test_input_signal; // 拿到3200点输入数据的第一帧起点

//     frames_win(frame_src, frame, FRAME_SIZE);
    
//     float diff_frame0 = max_abs_diff(frame, frame0_windowed, FRAME_SIZE); // 直接对比第一帧

//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff_frame0);

//     // 打印前十个值看看
//     Serial.println("C端前10个窗后值：");
//     for (int i = 0; i < 10; i++) {
//         Serial.println(frame[i], 8);
//     }
//     Serial.println("Python端前10个窗后值：");
//     for (int i = 0; i < 10; i++) {
//         Serial.println(frame0_windowed[i], 8);  
//     }
//     Serial.println("———— 结束测试第一帧窗后信号...");

// }


// void test_frame_windowed(void)
// {
    
//     Serial.println(".......开始测试全量窗后信号......");
//     int num = (TEST_SIGNAL_LENGTH - FRAME_SIZE) / FRAME_SHIFT + 1;
//     Serial.print("生成全量窗后信号帧数：");
//     Serial.println(num);
//     // 生成全量窗后信号
//     float frames_windowed_all_c[num * FRAME_SIZE];  // 存放全量窗后信号
//     const float *frame_src = test_input_signal; // 拿到3200点输入数据的第一帧起点

    
//     // 调用封装的hann_frame方法
//     int num_frames = frames_win(frame_src, frames_windowed_all_c, TEST_SIGNAL_LENGTH);
    


//     Serial.print("C端生成窗后信号帧数：");
//     Serial.println(num_frames);

//     // 打印前10个值
//     Serial.println("C端前10个窗后值：");
//     for (int i = 0; i < 10; i++) {
//         Serial.println(frames_windowed_all_c[i], 8);
//     }

//     // 打印后10个值
//     Serial.println("C端后10个窗后值：");
//     int total_len = num_frames * FRAME_SIZE;
//     for (int i = total_len - 10; i < total_len; i++) {
//         Serial.println(frames_windowed_all_c[i], 8);
//     }

//     // Python端同样打印前10个和后10个
//     Serial.println("Python端前10个窗后值：");
//     for (int i = 0; i < 10; i++) {
//         Serial.println(frame_windowed_all[i], 8);
//     }

//     Serial.println("Python端后10个窗后值：");
//     for (int i = total_len - 10; i < total_len; i++) {
//         Serial.println(frame_windowed_all[i], 8);
//     }
//     // int total_len = num_frames * FRAME_SIZE;

//     // 一次性与 Python 导出的窗后数组（已转置保存）进行对比
//     float diff = max_abs_diff(frames_windowed_all_c, frame_windowed_all, total_len);
//     Serial.print("全量窗口后 diff = ");
//     Serial.println(diff, 8);
//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff);
//     Serial.println(".......结束测试全量窗后信号......");
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
//     // RUN_TEST(test_frame0_windowed);
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
//     // delay(1000);
// }
