// #include "dsp_platform.h"
// #include "dsps_fft2r.h"
// #include <stdio.h>
// // #include "../python/out/frame0_windowed.h"  // 包含 signal_data 数组
// #include "../python/signal_data.h"  // 包含 signal_data 数组
// #include <Arduino.h>
// #include "unity.h"

// #define N_FFT 64

// float x[N_FFT];
// float y_cf[2 * N_FFT];     // 复数数组：实部/虚部交替
// float twiddles[N_FFT];

// void test_fft_compare()
// {
//     // 1初始化输入信号
//     for (int i = 0; i < N_FFT; i++) {
//         // x[i] = frame0_windowed[i];
//         x[i] = signal_data[i];
//         y_cf[2 * i] = x[i];     // 实部
//         y_cf[2 * i + 1] = 0.0f; // 虚部
//     }

//     // 初始化旋转因子
//     dsps_fft2r_init_fc32(twiddles, N_FFT);

//     // 执行 FFT
//     dsps_fft2r_fc32(y_cf, N_FFT);
//     dsps_bit_rev_fc32(y_cf, N_FFT);
//     // ⚠️ 不要调用 dsps_cplx2reC_fc32，否则顺序会被打乱！

//     // 打印全部 64 点输出结果2
//     printf("=== MCU FFT 输出（ 64 个 bin） ===\n");
//     for (int i = 0; i < 64; i++) {
//         float re = y_cf[2 * i]; //频率分量的实部
//         float im = y_cf[2 * i + 1];
//         float mag = sqrtf(re * re + im * im);
//         printf("Bin %02d: Re=%+.8f, Im=%+.8f, |X|=%+.8f\n", i, re, im, mag);
//     }

// }

// // ==================== Unity Runner ====================
// void setUp(void) {}
// void tearDown(void) {}

// int runUnityTests(void) {
//     UNITY_BEGIN();
//     RUN_TEST(test_fft_compare);
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





