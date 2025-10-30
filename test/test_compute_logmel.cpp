// test/test_logmel.cpp
#include <Arduino.h>
#include <unity.h>
#include "compute_logmel.h"



// ------------------- 辅助函数 -------------------
void generate_sine(int16_t *buf, int len, float freq, float amplitude) {
    for (int i = 0; i < len; i++) {
        buf[i] = (int16_t)(amplitude * sinf(2.0f * M_PI * freq * i / SAMPLE_RATE));
    }
}



// ------------------- Unity 测试用例 -------------------

void setUp(void) {
    // 每个测试前调用（可选）
    Serial.println("[TEST] Setting up...");
}

void tearDown(void) {
    // 每个测试后调用（可选）
    Serial.println("[TEST] Clean up.");
}

/// @brief 测试 Log-Mel 特征提取是否正常运行，不崩溃且输出合理
void test_logmel_computation() {
    int16_t test_input[INPUT_SAMPLES];
    float logmel_out[NUM_FRAMES * N_MEL_BINS];

    Serial.println("=== Running: test_logmel_computation ===");

    // 生成 440Hz 正弦波（A4 音符）
    generate_sine(test_input, INPUT_SAMPLES, 440.0f, 1000.0f);

    // 执行计算
    compute_logmel_200ms(test_input, logmel_out);

    // 断言：检查前几个输出值是否正常（非 NaN、非 Inf）
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_FALSE(isnan(logmel_out[i]));
        TEST_ASSERT_FALSE(isinf(logmel_out[i]));
    }

    // 断言：能量应在合理范围（比如 -50dB ~ 0dB）
    for (int i = 0; i < NUM_FRAMES * N_MEL_BINS; i++) {
        TEST_ASSERT_GREATER_THAN(-120.0f, logmel_out[i]);  // 下限
        TEST_ASSERT_LESS_THAN(100.0f, logmel_out[i]);       // 上限（静音接近 -∞）
    }

    // 打印前两帧用于观察（调试用）
    Serial.println("First 2 frames of Log-Mel spectrum:");
    for (int f = 0; f < 2; f++) {
        Serial.printf("Frame %d: ", f);
        for (int m = 0; m < N_MEL_BINS; m++) {
            Serial.printf("%.2f ", logmel_out[f * N_MEL_BINS + m]);
        }
        Serial.println();
    }

    Serial.println("=== test_logmel_computation PASSED ===");
}


/// @brief 简单测试静音输入是否产生低能量
void test_logmel_silence_input() {
    int16_t silence[INPUT_SAMPLES] = {0};
    float output[NUM_FRAMES * N_MEL_BINS];

    Serial.println("=== Running: test_logmel_silence_input ===");

    compute_logmel_200ms(silence, output);

    // 静音时所有 Mel 能量应非常小（对数后为大负数）
    for (int i = 0; i < NUM_FRAMES * N_MEL_BINS; i++) {
        TEST_ASSERT_LESS_THAN(40.0f, output[i]);  // 应小于 -30dB
        TEST_ASSERT_GREATER_THAN(-120.0f, output[i]);
    }

    Serial.println("Silence test passed.");
}


// ------------------- Unity 主入口 -------------------
void setup() {
    delay(2000); // 给串口点时间启动
    Serial.begin(115200);
    while (!Serial && millis() < 3000); // 等待串口监视器（仅开发时）

    UNITY_BEGIN();

    RUN_TEST(test_logmel_computation);
    RUN_TEST(test_logmel_silence_input);

    UNITY_END();
}


void loop() {
    // 必须存在，但空着即可
}

