// #include "unity.h"
// #include <string.h>
// #include <Arduino.h>
// #include <math.h>
// #include "esp_dsp.h"
// #include "esp_heap_caps.h"  // PSRAM malloc
// #include "mel_filterbank.h"  // 单帧Mel 特征计算接口
// #include "../python/out/frame_mel_power.h"   // Python 导出的 log-Mel 数据 frame_mel_db[1152]
// #include "../python/out/powspec.h"        // Python 导出的 功率谱数据 powspec[4626]



// #define TOL_LOGMEL_DIFF 1e-4f  // 精度容忍


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
// void test_logmel(void)
// {
//     Serial.println("=== test_logmel_start ===");
//     // step1. 准备输入功率谱数据（从 python 导入）
//     const float* power_spectrum = powspec; // python 导出数据
//     const float* mel_ref = frame_mel_power;   // python 导出的 log-Mel 参考结果

//     // step2. 功率谱分帧， 按照每帧调用 Mel 特征计算接口
//     const int n_fft_bins = 257;      // 一帧功率谱点数 (N_FFT/2+1)
//     const int n_mel_bins = 64;       // Mel 滤波器数量
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

//         // 单帧 Mel 特征计算
//         apply_mel(p_frame, mel_frame_out);
//     }
//     Serial.printf("-----------Mel 特征计算完成----------\n");
//     Serial.printf("total_mel_values应该是=%d\n", total_mel_values); // 1152
//     // ====== Step5: 一次性比较 Python 结果与 MCU 输出 ======

//     // 打印前16个 Mel 特征值对比
//     for (int i = 0; i < 16; i++) {
//         Serial.printf("mel_out_all[%d]=%.8f, mel_ref[%d]=%.8f\n", i, mel_out_all[i], i, mel_ref[i]);
//     }


//     float max_diff = max_abs_diff(mel_out_all, mel_ref, total_mel_values);
//     Serial.printf("Max abs diff = %.8f\n", max_diff);

//     TEST_ASSERT_MESSAGE(max_diff <= TOL_LOGMEL_DIFF, "Mel diff exceeds tolerance");

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


/*
=== test_logmel_start ===
-----------Mel 特征计算完成----------
total_mel_values应该是=1152
mel_out_all[0]=0.00000094, mel_ref[0]=0.00000094
mel_out_all[1]=0.00000187, mel_ref[1]=0.00000187
mel_out_all[2]=0.00000639, mel_ref[2]=0.00000639
mel_out_all[3]=0.00000541, mel_ref[3]=0.00000541
mel_out_all[4]=0.00001169, mel_ref[4]=0.00001169
mel_out_all[5]=0.00012178, mel_ref[5]=0.00012178
mel_out_all[6]=0.00014256, mel_ref[6]=0.00014256
mel_out_all[7]=0.00376873, mel_ref[7]=0.00376872
mel_out_all[8]=0.22378890, mel_ref[8]=0.22378892
mel_out_all[9]=12.20341301, mel_ref[9]=12.20341492
mel_out_all[10]=20.59566307, mel_ref[10]=20.59566689
mel_out_all[11]=4.15300941, mel_ref[11]=4.15300798
mel_out_all[12]=0.05634963, mel_ref[12]=0.05634966
mel_out_all[13]=0.00025052, mel_ref[13]=0.00025052
mel_out_all[14]=0.00016061, mel_ref[14]=0.00016061
mel_out_all[15]=0.00005196, mel_ref[15]=0.00005196
Max abs diff = 0.00000381
=== test_logmel END ===
test/test_debug_mel.cpp:86:test_logmel:PASS

*/