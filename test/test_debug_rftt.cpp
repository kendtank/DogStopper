/*
 * 逐步调试测试
 * 用于生成与Python端相同的中间结果，以便进行对比
 * 
 * 测试结果：
 * 1. Hann窗函数生成测试：PASS
 */

#include "unity.h"
#include "mfcc.h"
#include <string.h>
#include <Arduino.h>
#include <math.h>
#include "dsps_fft2r.h"
#include "../python/test_data.h"
#include "../python/out/powspec_all.h"




#define TOL_WIN   1e-6f  // 允许的误差范围
#define MAX_FRAMES 18
#define NFFT 400
#define NBIN (NFFT/2+1)

static float frames_windowed_all_c[MAX_FRAMES*NFFT];  // 分帧+加窗后的信号
static float fft_power_all[MAX_FRAMES*NBIN];          // 全量功率谱
static float fft_buf[2*NFFT];                         // FFT临时buffer


static float max_abs_diff(const float *a, const float *b, int n)
{
    // 计算两个数组的绝对差， 返回最大差值
    float maxd = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > maxd)
            maxd = d;
    }
    Serial.print("max abs diff: ");
    Serial.println(maxd, 6);

    return maxd;
}



void test_full_power_spectrum(void)
{
    // 初始化 DSP FFT
    esp_err_t r = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    
    float win[NFFT];
    make_hann(win);

    int num_frames = (TEST_SIGNAL_LENGTH - NFFT) / FRAME_SHIFT + 1;
    int total_points = num_frames * NFFT;

    // -------------------------------
    // 1. 分帧 + Hann
    // -------------------------------
    for (int f = 0; f < num_frames; f++) {
        const float *frame_src = test_input_signal + f * FRAME_SHIFT;
        float *frame_dst = frames_windowed_all_c + f*NFFT;
        for (int i=0; i<NFFT; i++)
            frame_dst[i] = frame_src[i] * win[i];
    }
    


    // -------------------------------
    // 2. 每帧 FFT -> 功率谱
    // -------------------------------
    for (int f=0; f<num_frames; f++) {
        const float *x = frames_windowed_all_c + f*NFFT;
        float *P = fft_power_all + f*NBIN;

        // 复数buffer填充
        for (int i=0;i<NFFT;i++){
            fft_buf[2*i] = x[i];       // re
            fft_buf[2*i+1] = 0.0f;     // im
        }

        dsps_fft2r_fc32(fft_buf, NFFT);
        dsps_bit_rev_fc32(fft_buf, NFFT);

        // 计算功率谱
        for(int k=0;k<NBIN;k++){
            float re = fft_buf[2*k];
            float im = fft_buf[2*k+1];
            P[k] = re*re + im*im;  // /NFFT可根据Python做法调整
        }
    }

    // -------------------------------
    // 3. 对比 Python 全量功率谱
    // -------------------------------
    float diff = max_abs_diff(fft_power_all, powspec_all, num_frames*NBIN);
    Serial.print("全量功率谱 diff = ");
    Serial.println(diff, 6);
    TEST_ASSERT_LESS_THAN_FLOAT(TOL_WIN, diff);
}





// ==================== 主入口 ====================
void setUp(void) {
    // 测试前的设置
}

void tearDown(void) {
    // 测试后的清理
}

int runUnityTests(void) {
    UNITY_BEGIN();
    RUN_TEST(test_full_power_spectrum);
    return UNITY_END();
}

void setup() {
    // Wait ~2 seconds before the Unity test runner
    // if you don't want to wait for the debugger attach
    Serial.begin(115200);
    delay(2000);

    runUnityTests();
}

void loop() {
    // 测试运行完成后不需要循环执行
    delay(1000);
}