// main.cpp - 在 ESP32 上测试 int8 狗吠模型（随机输入）
#include <Arduino.h>
// TensorFlow Lite Micro 头文件
#include <Arduino.h>
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "../python/tinyml/model/dog_bark_model.cc"  // 包含模型数组


// === 全局对象（和你原来一致）===
tflite::MicroErrorReporter micro_error_reporter;
constexpr int kTensorArenaSize = 8 * 1024; // 8KB 足够大多数小模型
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {

  Serial.begin(115200);
  while (!Serial);
  Serial.println("Dog Bark Model Test (Random Input)");
  #ifdef ESP_NN
  Serial.println("✅ Using esp_nn acceleration!");
    #endif

  model = tflite::GetModel(g_bark_model); // 传入模型数组！


  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // 打印类型和形状
  Serial.print("Input type: ");
  Serial.println(input->type == kTfLiteFloat32 ? "float32" : (input->type == kTfLiteInt8 ? "int8" : "unknown"));
  Serial.print("Output type: ");
  Serial.println(output->type == kTfLiteFloat32 ? "float32" : (output->type == kTfLiteInt8 ? "int8" : "unknown"));

  Serial.print("Input shape: ");
  for (int i = 0; i < input->dims->size; i++) Serial.print(input->dims->data[i]), Serial.print(" ");
  Serial.println();

}




void loop() {
  // 根据模型类型填充随机输入
  if (input->type == kTfLiteFloat32) {
    int num_floats = input->bytes / sizeof(float);
    for (int i = 0; i < num_floats; i++) {
      input->data.f[i] = (float)random(100) / 100.0f; // 0.0 ~ 1.0
    }
  } else if (input->type == kTfLiteInt8) {
    int num_int8 = input->bytes;
    for (int i = 0; i < num_int8; i++) {
      input->data.int8[i] = (int8_t)(random(256) - 128); // -128 ~ 127
    }
  } else {
    Serial.println("❓ Unsupported input type");
    delay(2000);
    return;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }

  // 读取输出
  if (input->type == kTfLiteFloat32) {
    int num_floats = input->bytes / sizeof(float);
    for (int i = 0; i < num_floats; i++) {
      input->data.f[i] = (float)random(100) / 100.0f;
    }
  } else if (input->type == kTfLiteInt8) {
    int num_int8 = input->bytes;
    for (int i = 0; i < num_int8; i++) {
      input->data.int8[i] = (int8_t)(random(256) - 128);
    }
  } else {
    Serial.println("❓ Unsupported input type");
    delay(2000);
    return;
  }

  // ====== 开始计时 ======
  uint32_t start_us = micros();

  TfLiteStatus invoke_status = interpreter->Invoke();

  uint32_t end_us = micros();
  uint32_t inference_time_us = end_us - start_us;
  // =====================

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }

  // 输出结果（保持不变）
  if (output->type == kTfLiteFloat32) {
    float prob = output->data.f[0];
    Serial.printf("Inference: %.4f (%.2f ms) → %s\n",
                  prob, inference_time_us / 1000.0, (prob > 0.7f) ? "BARK" : "NO");
  } else if (output->type == kTfLiteInt8) {
    int8_t raw = output->data.int8[0];
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    float prob = (raw - zero_point) * scale;
    Serial.printf("Inference: raw=%d, prob=%.4f (%.2f ms) → %s\n",
                  raw, prob, inference_time_us / 1000.0, (prob > 0.7f) ? "BARK" : "NO");
  }

  delay(1000);
}