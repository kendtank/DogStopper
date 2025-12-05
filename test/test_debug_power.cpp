// #include "unity.h"
// #include "esp_dsp.h"
// #include <Arduino.h>
// #include <math.h>
// #include "hann_frame.h"
// #include "fft_power.h"
// #include "../python/test_data.h"
// #include "../python/out/frame_windowed_all.h" // Python 输出的窗后帧信号 frame_windowed_all
// #include "../python/out/powspec.h"   // Python 输出的功率谱  第一帧 400 点 FFT 后的功率谱powspec =np.abs(X)**2  # np.abs(X) 就是幅度 |X|
// // #include "../python/signal_data.h"

// #define TOL_WIN 1e-6f  // 窗口化误差容限
// #define TOL_POWSPEC 1e-3f // 功率谱误差容限


// static float max_abs_diff(const float *a, const float *b, int n)
// {
//     float maxd = 0;
//     for (int i = 0; i < n; i++) {
//         float d = fabsf(a[i] - b[i]);
//         if (d > maxd) maxd = d;
//     }
//     return maxd;
// }



// void test_fft()
// {
//     Serial.println("=== test_fft_begin ===");


//     int num = (TEST_SIGNAL_LENGTH - FRAME_SIZE) / FRAME_SHIFT + 1;  // 18帧
//     int total_len = num * FRAME_SIZE;
//     // 生成全量窗后信号
//     // ---- 分配 frames_windowed_c 到 PSRAM
//     float *frames_windowed_c = (float*) heap_caps_malloc(sizeof(float) * total_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); // 存放全量窗后信号  // 18*400 =7200
//     if (!frames_windowed_c) {
//         Serial.println("alloc frames_windowed_c failed");
//         return;
//     }
//     const float *frame_src = test_input_signal; // 拿到3200点输入数据的第一帧起点

//     // 调用封装的hann_frame方法
//     int num_frames = frames_win(frame_src, frames_windowed_c, TEST_SIGNAL_LENGTH);
//     Serial.print("C端生成窗后信号帧数：");
//     Serial.println(num_frames);

    
//     // 一次性与 Python 导出的窗后数组（已转置保存）进行对比
//     float diff = max_abs_diff(frames_windowed_c, frame_windowed_all, total_len);
//     Serial.print("全量窗口后 diff = ");
//     Serial.println(diff, 8);
//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff);

//     // -----------------------------
//     // Step 2. === 3. 执行 FFT === 并得到功率谱
//     // -----------------------------
//     int nfft = 512; // 通常为 FRAME_SIZE 的下一个2的幂

//     fft_power_init(nfft);

//     int num_bins = nfft / 2 + 1;


//     // 输出功率谱数组 (num_frames x num_bins)
//     float *power_out = (float*) heap_caps_malloc(sizeof(float) * num_frames * num_bins, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
//     if (!power_out) {
//         Serial.println("alloc power_out failed");
//         fft_power_free();
//         heap_caps_free(frames_windowed_c);
//         return;
//     }
//     // 使用相同的测试数据
    
//     u32_t start_time = micros();
//     // 调用 FFT + 功率谱计算函数
//     int ret = fft_power_compute(frames_windowed_c,   // 输入窗后帧信号（PSRAM）
//                             num_frames,          // 一共多少帧
//                             FRAME_SIZE,          // 每帧点数
//                             nfft,                // FFT长度（初始化时相同）
//                             power_out);          // 输出功率谱（PSRAM）
//     Serial.print("FFT cost time: ");
//     Serial.print(micros() - start_time);
//     Serial.println("us");

//     if (ret != 0) {
//     Serial.printf("fft_power_compute failed, ret=%d\n", ret);
//     fft_power_free();
//     heap_caps_free(frames_windowed_c);
//     heap_caps_free(power_out);
//     return;
//     }

//     Serial.print("功率谱计算完成，输出大小：");
//     Serial.print(num_frames);
//     Serial.print(" x ");
//     Serial.println(num_bins);

//     // 打印前16个数据的功率谱
//     for (int i = 0; i < 16; i++) {
//         Serial.printf("MCU Bin %02d: P=%+.9f\n", i, power_out[i]);
//     }
    
//     // -----------------------------
//     // Step 4. === 5. 与 Python 导出的功率谱进行对比 ===
//     // -----------------------------
//     float diff_all = max_abs_diff(power_out, powspec, num_frames * num_bins);
//     Serial.print("功率谱 diff = ");
//     Serial.println(diff_all, 8);

//     // 设置断言
//     TEST_ASSERT_LESS_THAN_FLOAT(TOL_POWSPEC, diff_all);


//     // -----------------------------
//     // 清理内存
//     // -----------------------------
//     fft_power_free();
//     heap_caps_free(frames_windowed_c);
//     heap_caps_free(power_out);
// }


// // ==================== Unity Runner ====================
// void setUp(void) {}
// void tearDown(void) {}

// int runUnityTests(void)
// {
//     UNITY_BEGIN();
//     RUN_TEST(test_fft);
//     return UNITY_END();
// }

// void setup()
// {
//     Serial.begin(115200);
//     delay(2000);
//     runUnityTests();
// }

// void loop()
// {
//     // delay(1000);
// }
