// #include "unity.h"
// #include <string.h>
// #include <Arduino.h>
// #include <math.h>
// #include "esp_dsp.h"
// #include "esp_heap_caps.h"  // PSRAM malloc
// #include "mel_filterbank.h"  // 单帧Mel 特征计算接口
// #include "../python/out/frame_mel_db.h"   // Python 导出的 log-Mel 数据 frame_mel_db[1152]
// #include "../python/out/powspec.h"        // Python 导出的 功率谱数据 powspec[4626]



// #define TOL_BD_DIFF 0.1f  // 精度容忍 db 值


// // ==================== 工具函数 ====================
// // static float max_abs_diff(const float *a, const float *b, int n)
// // {
// //     float maxd = 0;
// //     for (int i = 0; i < n; i++) {
// //         float d = fabsf(a[i] - b[i]);
// //         if (d > maxd) maxd = d;
// //     }
// //     return maxd;
// // }


// static void max_abs_diff_info(const float *a, const float *b, int n,
//                               float *out_maxdiff, int *out_idx)
// {
//     float maxd = 0;
//     int idx = -1;
//     for (int i = 0; i < n; i++) {
//         float d = fabsf(a[i] - b[i]);
//         if (d > maxd) {
//             maxd = d;
//             idx = i;
//         }
//     }
//     if (out_maxdiff) *out_maxdiff = maxd;
//     if (out_idx) *out_idx = idx;
// }


// // ==================== 测试函数 入口：c和py端验证过的 功率谱 误差1e-4f====================
// void test_logmel(void)
// {
//     Serial.println("=== test_logmel_start ===");
//     // step1. 准备输入功率谱数据（从 python 导入）
//     const float* power_spectrum = powspec; // python 导出数据
//     const float* mel_ref = frame_mel_db;   // python 导出的 log-Mel 参考结果

//     // step2. 功率谱分帧， 按照每帧调用 Mel 特征计算接口
//     const int n_fft_bins = 257;      // 一帧功率谱点数 (N_FFT/2+1)
//     const int n_mel_bins = 40;       // Mel 滤波器数量
//     const int n_frames = 18;         // 假设共有18帧 (4626 / 257 = 18)
//     float mel_out[64];               // 每帧 MCU 输出的 log-Mel
//     const int total_mel_values = n_frames * n_mel_bins;

//     // ====== Step3: 分配输出缓存 ======
//     float* mel_out_all = (float*)heap_caps_malloc(total_mel_values * sizeof(float), MALLOC_CAP_SPIRAM);
//     if (!mel_out_all) {
//         TEST_FAIL_MESSAGE("mel_out_all malloc failed");
//         return;
//     }

//     // ====== Step4: 全部帧计算 ======
//     for (int i = 0; i < n_frames; i++) {
//         const float* p_frame = power_spectrum + i * n_fft_bins;
//         float* mel_frame_out = mel_out_all + i * n_mel_bins;
//         u32_t start_time = micros();
//         // 单帧 Mel 特征计算
//         apply_log_mel(p_frame, mel_frame_out);
//         Serial.printf("logmel compute time: %d us\n", micros() - start_time);   // 稀疏之前：1.6ms  稀疏化矩阵之后：0.15ms
//     }
//     Serial.printf("-----------log-Mel 特征计算完成----------\n");
//     Serial.printf("total_mel_values应该是=%d\n", total_mel_values); // 1152
//     // ====== Step5: 一次性比较 Python 结果与 MCU 输出 ======

//     // 打印前16个 log-Mel 特征值对比
//     for (int i = 0; i < 16; i++) {
//         Serial.printf("mel_out_all[%d]=%.8f, mel_ref[%d]=%.8f\n", i, mel_out_all[i], i, mel_ref[i]);
//     }
//     // 打印后16个 log-Mel 特征值对比
//     for (int i = total_mel_values - 16; i < total_mel_values; i++) {
//         Serial.printf("mel_out_all[%d]=%.8f, mel_ref[%d]=%.8f\n", i, mel_out_all[i], i, mel_ref[i]);
//     }


//     // ====== Step6: 计算最大差异并打印详细信息 ======
//     float max_diff = 0;
//     int max_idx = -1;
//     max_abs_diff_info(mel_out_all, mel_ref, total_mel_values, &max_diff, &max_idx);

//     Serial.printf("Max abs diff = %.8f (at index %d)\n", max_diff, max_idx);
//     if (max_idx >= 0) {
//         Serial.printf("mel_out_all[%d] = %.8f\n", max_idx, mel_out_all[max_idx]);
//         Serial.printf("mel_ref[%d]     = %.8f\n", max_idx, mel_ref[max_idx]);
//         Serial.printf("abs diff        = %.8f\n", fabsf(mel_out_all[max_idx] - mel_ref[max_idx]));
//     }

//     // ====== Step7: 单元测试断言 ======
//     TEST_ASSERT_MESSAGE(max_diff <= TOL_BD_DIFF, "Mel diff exceeds tolerance");



//     // ====== Step6: 清理并结束 ======
//     heap_caps_free(mel_out_all);
//     Serial.println("=== test_logmel END ===");
//     TEST_PASS();
// }


// // ==================== Unity 框架 ====================
// void setUp(void) {}
// void tearDown(void) {}

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_logmel);
//     return UNITY_END();
// }

// // ==================== Arduino 主入口 ====================
// void setup() {
//     Serial.begin(115200);
//     delay(2000); // 等串口稳定
//     runUnityTests();
// }

// void loop() {
//     delay(1000);
// }


// /*
// === test_logmel_start ===
// -----------log-Mel 特征计算完成----------
// total_mel_values应该是=1152
// mel_out_all[0]=-60.25472260, mel_ref[0]=-60.25464630
// mel_out_all[1]=-57.28952408, mel_ref[1]=-57.28963852
// mel_out_all[2]=-51.94395065, mel_ref[2]=-51.94388580
// mel_out_all[3]=-52.66822433, mel_ref[3]=-52.66822815
// mel_out_all[4]=-49.32366180, mel_ref[4]=-49.32366180
// mel_out_all[5]=-39.14432907, mel_ref[5]=-39.14432907
// mel_out_all[6]=-38.46017456, mel_ref[6]=-38.46017456
// mel_out_all[7]=-24.23805428, mel_ref[7]=-24.23806000
// mel_out_all[8]=-6.50161457, mel_ref[8]=-6.50161457
// mel_out_all[9]=10.86481285, mel_ref[9]=10.86481285
// mel_out_all[10]=13.13775826, mel_ref[10]=13.13775921
// mel_out_all[11]=6.18362904, mel_ref[11]=6.18362808
// mel_out_all[12]=-12.49108887, mel_ref[12]=-12.49108696
// mel_out_all[13]=-36.01152802, mel_ref[13]=-36.01153183
// mel_out_all[14]=-37.94219589, mel_ref[14]=-37.94219589
// mel_out_all[15]=-42.84332657, mel_ref[15]=-42.84332275
// mel_out_all[1136]=-80.00000000, mel_ref[1136]=-80.00000000
// mel_out_all[1137]=-80.00000000, mel_ref[1137]=-80.00000000
// mel_out_all[1138]=-80.00000000, mel_ref[1138]=-80.00000000
// mel_out_all[1139]=-80.00000000, mel_ref[1139]=-80.00000000
// mel_out_all[1140]=-80.00000000, mel_ref[1140]=-80.00000000
// mel_out_all[1141]=-80.00000000, mel_ref[1141]=-80.00000000
// mel_out_all[1142]=-80.00000000, mel_ref[1142]=-80.00000000
// mel_out_all[1143]=-80.00000000, mel_ref[1143]=-80.00000000
// mel_out_all[1144]=-80.00000000, mel_ref[1144]=-80.00000000
// mel_out_all[1145]=-80.00000000, mel_ref[1145]=-80.00000000
// mel_out_all[1146]=-80.00000000, mel_ref[1146]=-80.00000000
// mel_out_all[1147]=-80.00000000, mel_ref[1147]=-80.00000000
// mel_out_all[1148]=-80.00000000, mel_ref[1148]=-80.00000000
// mel_out_all[1149]=-80.00000000, mel_ref[1149]=-80.00000000
// mel_out_all[1150]=-80.00000000, mel_ref[1150]=-80.00000000
// mel_out_all[1151]=-80.00000000, mel_ref[1151]=-80.00000000
// Max abs diff = 0.01402283 (at index 28)
// mel_out_all[28] = -79.76953125
// mel_ref[28]     = -79.75550842
// abs diff        = 0.01402283
// === test_logmel END ===
// test/test_debug_log_mel.cpp:119:test_logmel:PASS

// -----------------------
// 1 Tests 0 Failures 0 Ignored 
// OK

// 注意：
// C 端和 Python 端对 线性 Mel 能量（mel_linear） 已经非常一致（误差 ~3e-6）
// 误差放大出现在 log 步骤（10*log10(x)）且仅在 x 极小（比如 <1e-8）时。
// log 在极小数值处非常敏感（大小变化/浮点舍入会被放大为 dB 级差异）。Python/Librosa 多用 float64，而 MCU 用 float32，因此在超小数值下差异不可避免。
// */
