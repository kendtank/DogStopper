// #include "unity.h"
// #include <string.h>
// #include <Arduino.h>
// #include <math.h>
// #include "mel_filterbank.h"
// #include "esp_heap_caps.h"


// #define TOL_POWSPEC 1e-4f


// // ==================== 工具函数 ====================
// static float max_abs_diff(const float *a, const float *b, int n)
// {
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


// static double hz_to_mel(double f) {
//     const double f_min = 0.0;
//     const double f_sp = 200.0 / 3.0;
//     const double f_break = 1000.0;
//     const double logstep = pow(6.4, 1.0 / 27.0);

//     if (f < f_break) {
//         return (f - f_min) / f_sp;
//     } else {
//         return (f_break - f_min) / f_sp + log(f / f_break) / log(logstep);
//     }
// }
// static double mel_to_hz(double m) {
//     const double f_min = 0.0;
//     const double f_sp = 200.0 / 3.0;
//     const double f_break = 1000.0;
//     const double logstep = pow(6.4, 1.0 / 27.0);
//     const double min_log_mel = (f_break - f_min) / f_sp;

//     if (m < min_log_mel) {
//         return f_min + f_sp * m;
//     } else {
//         return f_break * exp(log(logstep) * (m - min_log_mel));
//     }
// }

// static void build_mel_filters(float W[MEL_BANDS][NUM_BINS]) {
//     const int n_mel = MEL_BANDS;
//     const int n_fft = 512;
//     const int nfft_bins = NUM_BINS;
//     const int sample_rate = 16000;

//     memset(W, 0, sizeof(float) * n_mel * nfft_bins);

//     // 1️⃣ FFT bin 对应频率（线性）
//     double fft_freqs[nfft_bins];
//     for (int k = 0; k < nfft_bins; k++) {
//         fft_freqs[k] = (double)sample_rate * k / n_fft;
//     }

//     // 2️⃣ Mel 频率点
//     double m_min = hz_to_mel(0.0);
//     double m_max = hz_to_mel(sample_rate * 0.5);
//     double m_pts[n_mel + 2];
//     double f_pts[n_mel + 2];

//     for (int i = 0; i < n_mel + 2; i++) {
//         m_pts[i] = m_min + (m_max - m_min) * (i / (double)(n_mel + 1));
//         f_pts[i] = mel_to_hz(m_pts[i]);
//     }

//     // 3️⃣ 构建每个 Mel 滤波器
//     for (int m = 0; m < n_mel; m++) {
//         double f0 = f_pts[m];
//         double f1 = f_pts[m + 1];
//         double f2 = f_pts[m + 2];

//         if (f1 <= f0) f1 = f0 + 1e-9;
//         if (f2 <= f1) f2 = f1 + 1e-9;

//         for (int k = 0; k < nfft_bins; k++) {
//             double fk = fft_freqs[k];
//             float w = 0.0f;

//             if (fk <= f0 || fk >= f2)
//                 w = 0.0f;
//             else if (fk <= f1)
//                 w = (float)((fk - f0) / (f1 - f0));
//             else
//                 w = (float)((f2 - fk) / (f2 - f1));

//             // 归一化（Slaney 风格）
//             if (w > 0.0f)
//                 W[m][k] = w * (float)(2.0 / (f2 - f0));
//         }
//     }
// }

// // ==================== 测试函数 ====================
// void test_mel_filterbank(void) {

//     Serial.println("=== DIAG: test_mel_filterbank start ===");
//     // TODO: 测试py和c生成的mel filterbank是否一致

//     float (*W)[NUM_BINS] = (float (*)[NUM_BINS])heap_caps_malloc(sizeof(float) * MEL_BANDS * NUM_BINS, MALLOC_CAP_SPIRAM);
//         if (!W) {
//         Serial.println("堆分配失败！");
//         return;
//     }

//     build_mel_filters(W);

//     // 逐元素比较 Python 生成的数组
//     int total = MEL_BANDS * NUM_BINS;
//     const float* py_data = mel_filterbank; // Python生成的线性数组
//     float* c_data = &W[0][0];

//     // 分别打印前后10个元素对比
//     Serial.println("First 10 elements comparison:");
//     for (int i = 0; i < 10; i++) {
//         Serial.print("py: "); Serial.print(py_data[i], 6);
//         Serial.print(" | c: "); Serial.println(c_data[i], 6);
//     }
//     Serial.println("Last 10 elements comparison:");
//     for (int i = total - 10; i < total; i++) {
//         Serial.print("py: "); Serial.print(py_data[i], 6);
//         Serial.print(" | c: "); Serial.println(c_data[i], 6);
//     }

//     float diff = max_abs_diff(py_data, c_data, total);
//     TEST_ASSERT_TRUE(diff < TOL_POWSPEC);

//     Serial.println("=== test_mel_filterbank PASS ===");
//     TEST_PASS();
// }

// // ==================== Unity 测试框架 ====================
// void setUp(void) {}
// void tearDown(void) {}

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_mel_filterbank);
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
// First 10 elements comparison:
// py: 0.000000 | c: 0.000000
// py: 0.014511 | c: 0.014511
// py: 0.014076 | c: 0.014076
// py: 0.000000 | c: 0.000000
// py: 0.000000 | c: 0.000000
// py: 0.000000 | c: 0.000000
// py: 0.000000 | c: 0.000000
// py: 0.000000 | c: 0.000000
// py: 0.000000 | c: 0.000000
// py: 0.000000 | c: 0.000000
// Last 10 elements comparison:
// py: 0.002061 | c: 0.002061
// py: 0.001832 | c: 0.001832
// py: 0.001603 | c: 0.001603
// py: 0.001374 | c: 0.001374
// py: 0.001145 | c: 0.001145
// py: 0.000916 | c: 0.000916
// py: 0.000687 | c: 0.000687
// py: 0.000458 | c: 0.000458
// py: 0.000229 | c: 0.000229
// py: 0.000000 | c: 0.000000
// max abs diff: 0.000000
// === test_mel_filterbank PASS ===

// */