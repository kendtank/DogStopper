// #include "unity.h"
// #include <string.h>
// #include <Arduino.h>
// #include <math.h>
// #include "esp_dsp.h"
// #include "esp_heap_caps.h"  // PSRAM malloc
// #include "../python/out/frame_mfcc.h"
// #include "../python/out/powspec.h"
// #include "mel_filterbank.h"
// #include "mfcc.h"



// #define TOL_BD_DIFF 0.1f  // 精度容忍 db 值


// // ==================== 工具函数 ====================
// static float max_abs_diff(const float *a, const float *b, int n)
// {
//     float maxd = 0;
//     for (int i = 0; i < n; i++) {
//         float d = fabsf(a[i] - b[i]);
//         if (d > maxd) maxd = d;
//     }
//     return maxd;
// }



// // ==================== 测试函数 入口：c和py端验证过的 功率谱 误差1e-4f====================
// void test_mfcc(void)
// {
//     Serial.println("=== test_mfcc_start ===");
//     // step1. 准备输入功率谱数据（从 python 导入）
//     const float* power_spectrum = powspec; // python 导出数据

//     // step2. 功率谱分帧， 按照每帧调用 Mel 特征计算接口
//     const int n_fft_bins = 257;      // 一帧功率谱点数 (N_FFT/2+1)
//     const int n_mel_bins = 40;       // Mel 滤波器数量
//     const int n_frames = 18;         // 假设共有18帧 (4626 / 257 = 18)
//     float mel_out[40];               // 每帧 MCU 输出的 log-Mel
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

//         // 单帧 Mel 特征计算
//         apply_log_mel(p_frame, mel_frame_out);
//     }

//     // 拿到log-mel 特征值
//     Serial.printf("-----------log-Mel 特征计算完成----------\n");

//     // ====== Step5: 应用 DCT 变换 ======
//     const int n_mfcc_coeffs = 13;  // MFCC 系数数量
//     float* mfcc_out_all = (float*)heap_caps_malloc(n_frames * n_mfcc_coeffs * sizeof(float), MALLOC_CAP_SPIRAM);
//     if (!mfcc_out_all) {
//         heap_caps_free(mel_out_all);
//         TEST_FAIL_MESSAGE("mfcc_out_all malloc failed");
//         return;
//     }

//     // compute_mfcc(mel_out_all, mfcc_out_all);

//     // 修复：对每一帧单独计算MFCC
//     for (int i = 0; i < n_frames; i++) {
//         const float* logmel_frame = mel_out_all + i * MEL_BANDS;  // 指向第i帧的log-Mel
//         float* mfcc_frame = mfcc_out_all + i * n_mfcc_coeffs;    // 指向第i帧的MFCC输出
        
//         compute_mfcc(logmel_frame, mfcc_frame);  // 正确调用
//     }
//     Serial.println("-----------MFCC 计算完成----------");

//     Serial.println("-----------MFCC 计算完成----------");

//     // ====== Step6: 计算最大差异并打印详细信息 ======
//     float max_diff = 0.0f;
//     for (int i = 0; i < n_frames; i++) {
//         const float* mfcc_frame = mfcc_out_all + i * n_mfcc_coeffs;
//         const float* mfcc_py_frame = frame_mfcc + i * n_mfcc_coeffs; // Python 导出的 MFCC

//         float frame_max = max_abs_diff(mfcc_frame, mfcc_py_frame, n_mfcc_coeffs);
//         if (frame_max > max_diff) max_diff = frame_max;

//         // 打印每帧前8个 MFCC 对比
//         Serial.printf("Frame %2d: MCU=", i);
//         for (int j = 0; j < 8; j++) Serial.printf("% .5f ", mfcc_frame[j]);
//         Serial.print(" | PY=");
//         for (int j = 0; j < 8; j++) Serial.printf("% .5f ", mfcc_py_frame[j]);
//         Serial.println();
//     }

//     Serial.printf("最大绝对误差: %f\n", max_diff);

//     // ====== Step7: 单元测试断言 ======
//     TEST_ASSERT_TRUE_MESSAGE(max_diff < TOL_BD_DIFF, "MFCC 最大误差超出容差");

//     // ====== Step8: 清理并结束 ======
//     heap_caps_free(mel_out_all);
//     heap_caps_free(mfcc_out_all);
//     Serial.println("=== test_mfcc END ===");
//     TEST_PASS();

// }


// // ==================== Unity 框架 ====================
// void setUp(void) {}
// void tearDown(void) {}

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_mfcc);
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
// -----------MFCC 计算完成----------
// Frame  0: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  1: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  2: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  3: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  4: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  5: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  6: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  7: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  8: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame  9: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 10: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 11: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 12: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 13: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 14: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 15: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 16: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// Frame 17: MCU=-519.71252  141.21040  71.73590 -0.62178 -46.06634 -59.27390 -52.85446 -40.00343  | PY=-519.71191  141.21036  71.73501 -0.62170 -46.06517 -59.27383 -52.85602 -40.00394 
// 最大绝对误: 0.002865
// === test_mfcc END ===

// */