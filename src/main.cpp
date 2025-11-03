// #include <Arduino.h>
// #include "audio_features.h"

// // 包含调试数据用于比较
// #include "../python/dbg_data.h"

// // 定义输入输出缓冲区
// #define AUDIO_BUFFER_SIZE 16000  // 1秒 @ 16kHz
// #define LOGMEL_FEATURE_SIZE 64

// int16_t audio_buffer[AUDIO_BUFFER_SIZE];
// float logmel_features[LOGMEL_FEATURE_SIZE];

// // 生成测试信号（正弦波）
// void generate_test_signal(int16_t* buffer, int length, int frequency, int sample_rate) {
//     const float amplitude = 0.3f * 32768.0f; // 30% 幅度
//     for (int i = 0; i < length; i++) {
//         float t = (float)i / sample_rate;
//         buffer[i] = (int16_t)(amplitude * sinf(2.0f * M_PI * frequency * t));
//     }
// }

// // 比较两个浮点数组，允许一定误差
// bool compare_float_arrays(const float* a, const float* b, int length, float tolerance = 1e-3f) {
//     for (int i = 0; i < length; i++) {
//         if (fabsf(a[i] - b[i]) > tolerance) {
//             Serial.printf("Mismatch at index %d: %.6f vs %.6f (diff: %.6f)\n", 
//                           i, a[i], b[i], fabsf(a[i] - b[i]));
//             return false;
//         }
//     }
//     return true;
// }

// void setup() {
//     // 初始化串口
//     Serial.begin(115200);
//     while (!Serial && millis() < 4000); // 等待串口监视器（仅用于调试）

//     Serial.println("[INFO] DogStopper System Starting...");
    
// #if defined(DEBUG_FEATURES) && DEBUG_FEATURES
//     // 打印初始内存信息
//     Serial.println("[INFO] 初始内存信息:");
//     print_memory_info();
// #endif

//     // 示例1：用静音数据测试 compute_logmel_200ms
//     memset(audio_buffer, 0, sizeof(audio_buffer));
//     memset(logmel_features, 0, sizeof(logmel_features));

//     int num = compute_logmel_200ms(audio_buffer, logmel_features);
//     Serial.printf("[INFO] Computed %d log-mel frames.\n", num);
//     Serial.println("[INFO] Log-Mel computation with silence:");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("Feature[%d] = %.6f\n", i, logmel_features[i]);
//     }

//     // 示例2：用测试信号测试
//     generate_test_signal(audio_buffer, AUDIO_BUFFER_SIZE, 500, 16000); // 500Hz 正弦波
    
//     memset(logmel_features, 0, sizeof(logmel_features));
//     num = compute_logmel_200ms(audio_buffer, logmel_features);
//     Serial.printf("\n[INFO] Computed %d log-mel frames with test signal.\n", num);
//     Serial.println("[INFO] Log-Mel computation with 500Hz sine wave:");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("Feature[%d] = %.6f\n", i, logmel_features[i]);
//     }

//     // 测试mfcc
//     memset(logmel_features, 0, sizeof(logmel_features));
//     num = compute_mfcc_200ms(audio_buffer, logmel_features);
//     Serial.printf("\n[INFO] Computed %d mfcc frames.\n", num);
//     Serial.println("[INFO] MFCC computation with 500Hz sine wave:");
//     for (int i = 0; i < 10; i++) {
//         Serial.printf("MFCC[%d] = %.6f\n", i, logmel_features[i]);
//     }
    
//     // 使用调试数据进行比较
//     Serial.println("\n[INFO] Comparing with debug data...");
    
//     // 复制调试输入信号到音频缓冲区
//     memcpy(audio_buffer, input_signal, INPUT_SIGNAL_LENGTH * sizeof(int16_t));
    
//     // 填充剩余部分为0
//     if (INPUT_SIGNAL_LENGTH < AUDIO_BUFFER_SIZE) {
//         memset(audio_buffer + INPUT_SIGNAL_LENGTH, 0, (AUDIO_BUFFER_SIZE - INPUT_SIGNAL_LENGTH) * sizeof(int16_t));
//     }
    
//     // 计算Log-Mel特征并与调试数据比较
//     memset(logmel_features, 0, sizeof(logmel_features));
//     num = compute_logmel_200ms(audio_buffer, logmel_features);
    
//     if (num > 0) {
//         Serial.println("[INFO] Comparing Log-Mel features:");
//         bool logmel_match = compare_float_arrays(logmel_features, frame0_logmel, FRAME0_LOGMEL_LENGTH);
//         if (logmel_match) {
//             Serial.println("  ✓ Log-Mel features match!");
//         } else {
//             Serial.println("  ✗ Log-Mel features do not match!");
//         }
//     }
    
//     // 计算MFCC特征并与调试数据比较
//     float mfcc_features[LOGMEL_FEATURE_SIZE];
//     memset(mfcc_features, 0, sizeof(mfcc_features));
//     num = compute_mfcc_200ms(audio_buffer, mfcc_features);
    
//     if (num > 0) {
//         Serial.println("[INFO] Comparing MFCC features:");
//         // 注意: 我们只有部分调试数据用于比较
//         bool mfcc_match = compare_float_arrays(mfcc_features, frame0_logmel, FRAME0_LOGMEL_LENGTH); // 比较MFCC调试数据
//         if (mfcc_match) {
//             Serial.println("  ✓ MFCC features match!");
//         } else {
//             Serial.println("  ✗ MFCC features do not match!");
//         }
//     }

// #if defined(DEBUG_FEATURES) && DEBUG_FEATURES
//     // 打印最终内存信息
//     Serial.println("\n[INFO] 最终内存信息:");
//     print_memory_info();
// #endif
// }

// void loop() {
//     // 主循环
//     delay(10000); // 每10秒运行一次
// }