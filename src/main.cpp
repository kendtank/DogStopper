#include <Arduino.h>
#include "compute_logmel.h"

// 定义输入输出缓冲区
#define AUDIO_BUFFER_SIZE 16000  // 1秒 @ 16kHz
#define LOGMEL_FEATURE_SIZE 64

int16_t audio_buffer[AUDIO_BUFFER_SIZE];
float logmel_features[LOGMEL_FEATURE_SIZE];

void setup() {
    // 初始化串口
    Serial.begin(115200);
    while (!Serial && millis() < 4000); // 等待串口监视器（仅用于调试）

    Serial.println("[INFO] DogStopper System Starting...");

    // 示例：用静音数据测试 compute_logmel_200ms
    memset(audio_buffer, 0, sizeof(audio_buffer));
    memset(logmel_features, 0, sizeof(logmel_features));

    compute_logmel_200ms(audio_buffer, logmel_features);

    Serial.println("[INFO] Log-Mel computation completed.");
    for (int i = 0; i < 10; i++) {
        Serial.printf("Feature[%d] = %.6f\n", i, logmel_features[i]);
    }
}

void loop() {
    // 主循环
    delay(5000); // 每5秒运行一次
}