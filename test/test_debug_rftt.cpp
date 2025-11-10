// /*
//  * 逐步调试测试
//  * 用于生成与Python端相同的中间结果，以便进行对比
//  * 
//  * 测试结果：
//  * 1. Hann窗函数生成测试：PASS
//  */

// #include "unity.h"
// #include "mfcc.h"
// #include <string.h>
// #include <Arduino.h>
// #include <math.h>
// #include "esp_dsp.h"
// #include "../python/test_data.h"
// #include "../python/out/hann_window.h"
// #include "../python/out/frame_windowed_all.h"
// #include "../python/out/powspec_all.h"


// // ====== 与 Python 一致的配置 ======
// #define NFFT        400
// #define HOP         160
// #define NBIN        (NFFT / 2 + 1)
// #define NUM_FRAMES  (sizeof(frame_windowed_all) / sizeof(float) / NFFT)

// static float frame_in[NFFT];
// static float fft_buf[2 * NFFT];    // interleaved real/imag
// static float powspec_c[NBIN];



// #define TOL_WIN   1e-6f  // 允许的误差范围                    // FFT临时buffer

// void print_vector(const char *name, const float *x, int n) {
//     Serial.printf("%s: ", name);
//     for (int i = 0; i < n; i++) {
//         Serial.printf("%.6f ", x[i]);
//     }
//     Serial.println();
// }



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


// void test_full_power_spectrum(void)
// {
//     Serial.println("=== DIAG: test_full_power_spectrum start ===");

//     // === 0. 初始化 FFT ===
//     esp_err_t r = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
//     TEST_ASSERT_EQUAL_MESSAGE(ESP_OK, r, "FFT init failed");

//     // === 1. 验证 Hann 窗 ===
//     float win_c[FRAME_SIZE];
//     make_hann(win_c);
//     float hann_diff = max_abs_diff(win_c, hann_window, NFFT);
//     Serial.printf("Hann diff = %.9f\n", hann_diff);
//     TEST_ASSERT_LESS_OR_EQUAL_FLOAT(TOL_WIN, hann_diff);

//     // === 2. 加窗帧 ===
//     float frame_ref[NFFT];
//     memcpy(frame_ref, &frame_windowed_all[0], sizeof(float) * NFFT);

//     float frame_py_power = 0;
//     for (int i = 0; i < NFFT; i++)
//         frame_py_power += frame_ref[i] * frame_ref[i];
//     TEST_ASSERT_GREATER_THAN_MESSAGE(0.0f, frame_py_power, "Frame power zero!");

//     Serial.printf("Frame0 time-domain energy = %.9f\n", frame_py_power);

//     // === 3. 执行 FFT ===
//     float *fft_buf_heap = (float *)malloc(sizeof(float) * 2 * NFFT);
//     TEST_ASSERT_NOT_NULL_MESSAGE(fft_buf_heap, "malloc FFT buf failed");

//     for (int i = 0; i < NFFT; i++) {
//         fft_buf_heap[2 * i] = frame_ref[i];
//         fft_buf_heap[2 * i + 1] = 0.0f;
//     }

//     dsps_fft2r_fc32(fft_buf_heap, NFFT);
//     dsps_bit_rev_fc32(fft_buf_heap, NFFT);

//     // === 4. 功率谱计算 ===
//     float *powspec_c = (float *)malloc(sizeof(float) * NBIN);
//     TEST_ASSERT_NOT_NULL_MESSAGE(powspec_c, "malloc powspec buf failed");

//     float total_power_c = 0.0f;

//     // Hann 窗能量补偿系数（经验值约 1 / 0.638 ≈ 1.567）
//     const float HANN_ENERGY_COMP = 1.0f / 0.638f;

//     float scale = (1.0f / NFFT) * HANN_ENERGY_COMP;
//     for (int i = 0; i < NBIN; i++) {
//         float re = fft_buf_heap[2 * i];
//         float im = fft_buf_heap[2 * i + 1];
//         float val = (re * re + im * im) * scale;

//         if (i != 0 && i != NFFT / 2)
//             val *= 2.0f;

//         powspec_c[i] = val;
//         total_power_c += val;
//     }


//     Serial.printf("Total C-frame energy = %.9f\n", total_power_c);

//     // === 5. Python 功率谱 ===
//     float total_power_py = 0.0f;
//     for (int i = 0; i < NBIN; i++)
//         total_power_py += powspec_all[i];

//     Serial.printf("Total Py energy = %.9f\n", total_power_py);

//     float energy_ratio = total_power_c / total_power_py;
//     Serial.printf("Energy ratio C/Py = %.6f\n", energy_ratio);

//     // 容差设宽松一点，防止浮点微差
//     TEST_ASSERT_FLOAT_WITHIN_MESSAGE(0.02f, 1.0f, energy_ratio, "Total energy mismatch!");

//     // === 6. 打印前 12 个频点对比 ===
//     Serial.println("--- compare first 12 bins ---");
//     for (int k = 0; k < 12; k++) {
//         float diff = fabsf(powspec_c[k] - powspec_all[k]);
//         Serial.printf("bin %2d: C=%.9e Py=%.9e diff=%.9e\n",
//                       k, powspec_c[k], powspec_all[k], diff);
//     }

//     // === 7. 基本非零检查 ===
//     float max_val = 0.0f;
//     for (int i = 0; i < NBIN; i++)
//         if (powspec_c[i] > max_val) max_val = powspec_c[i];
//     TEST_ASSERT_GREATER_THAN_MESSAGE(1e-12f, max_val, "All-zero power spectrum!");

//     free(fft_buf_heap);
//     free(powspec_c);

//     Serial.println("=== test_full_power_spectrum PASS ===");
//     TEST_PASS();
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
//     RUN_TEST(test_full_power_spectrum);
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