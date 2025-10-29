#include <Arduino.h>
#include "logmel_extractor.h"

#define SEGMENT_SAMPLES 16000   // 1 秒音频
#define MAX_FRAMES_PER_SEG 160  // 最大帧数

float logmel_out[MAX_FRAMES_PER_SEG * N_MEL];  // 输出数组

float test_audio[SEGMENT_SAMPLES];  // 随机音频

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=== Log-Mel MCU Test (Random 1s Audio) ===");

    // 检查 PSRAM 可用性
    Serial.printf("Heap free: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("PSRAM free: %d bytes\n", ESP.getFreePsram());

    if (ESP.getFreePsram() < 1000000) {
        Serial.println(" PSRAM not available or too small! Check board settings.");
        while (1); 
    }

    // 生成测试音频
    for (int i = 0; i < SEGMENT_SAMPLES; i++) {
        test_audio[i] = ((float)random(-1000, 1000)) / 1000.0f;
    }

    // 调用 Log-Mel 提取
    int frames = compute_logmel(test_audio, SEGMENT_SAMPLES, logmel_out, MAX_FRAMES_PER_SEG);
    
    if (frames < 0) {
        Serial.printf("compute_logmel returned error code: %d\n", frames);
    } else {
        Serial.printf("Extracted %d frames from 1s random audio\n", frames);

        // 打印前 3 帧
        for (int f = 0; f < min(3, frames); f++) {
            Serial.printf("Frame %d: ", f);
            for (int m = 0; m < N_MEL; m++) {
                Serial.printf("%.2f ", logmel_out[f * N_MEL + m]);
            }
            Serial.println();
        }
    }
}

void loop() {

}
