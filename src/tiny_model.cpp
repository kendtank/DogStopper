#include "tiny_model.h"
#include "mfcc_model_data.h"  // 模型 FlatBuffer C 数组  python端量化生成
#include "mfcc_quant_params.h"  // 模型量化参数

// 引入tflite micro库
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"



// ===== MFCC 模型全局对象 =====
static tflite::MicroErrorReporter mfcc_error_reporter;    // TinyML Micro 版本的错误报告器，用于打印模型推理错误 
static uint8_t mfcc_arena[8 * 1024] __attribute__((aligned(16)));  // 模型运行时需要的工作内存（tensor buffer），Micro 端必须自己分配连续内存
static tflite::AllOpsResolver mfcc_resolver;                  // 包含模型中需要的算子（Conv, FullyConnected 等），让 interpreter 知道怎么执行
static const tflite::Model* mfcc_model = nullptr;           // 指向需要加载的 FlatBuffer 模型
static tflite::MicroInterpreter* mfcc_interpreter = nullptr;    // 推理解释器对象，用于执行模型
static TfLiteTensor* mfcc_input_tensor = nullptr;   // 输入张量指针（方便直接 memcpy 特征进去）
static TfLiteTensor* mfcc_output_tensor = nullptr;    // 输出张量指针（方便直接读取推理结果）



// ===== 初始化mfcc模型 =====
int mfcc_model_init() {
    mfcc_model = tflite::GetModel(quantized_model_mfcc_int8);
    if (!mfcc_model) return -1;
    if (mfcc_model->version() != TFLITE_SCHEMA_VERSION) return -2;

    // 用 static 对象避免栈释放
    static tflite::MicroInterpreter static_interpreter(
        mfcc_model, mfcc_resolver, mfcc_arena, sizeof(mfcc_arena), &mfcc_error_reporter
    );
    mfcc_interpreter = &static_interpreter;

    if (mfcc_interpreter->AllocateTensors() != kTfLiteOk) return -3;

    mfcc_input_tensor  = mfcc_interpreter->input(0);
    mfcc_output_tensor = mfcc_interpreter->output(0);

    return 0;
}


// -----------------------------------------
// MFCC 推理函数
// input: float MFCC 输入特征
// return: 狗吠概率 float
// -----------------------------------------
float mfcc_model_infer(const float* input_float) {
    // -----------------------------------------
    // 1. 输入归一化 + 量化 (float -> int8)
    // -----------------------------------------
    int8_t input_int8[MFCC_INPUT_SIZE];

    for (size_t i = 0; i < MFCC_INPUT_SIZE; i++) {

        // 归一化 先做缩放再做偏移
        // float norm = (input_float[i] - NORM_OFFSET) * NORM_SCALE;
        float norm = input_float[i] * NORM_SCALE + NORM_OFFSET;

        // 截断
        if (norm > NORM_TARGET_MAX) norm = NORM_TARGET_MAX;
        if (norm < NORM_TARGET_MIN) norm = NORM_TARGET_MIN;

        // 量化到 int8
        int32_t q = (int32_t)(norm / MFCC_INPUT_SCALE + MFCC_INPUT_ZERO_POINT);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        input_int8[i] = (int8_t)q;
    }

    // -----------------------------------------
    // 2. 拷贝到 TFLite Micro 输入张量
    // -----------------------------------------
    memcpy(mfcc_input_tensor->data.int8, input_int8, MFCC_INPUT_SIZE);

    // -----------------------------------------
    // 3. 推理 // 核心函数：将输入传入模型进行前向计算，结果放到输出张量
    // -----------------------------------------
    mfcc_interpreter->Invoke();    //  所以初始化加上static不然报空指针

    // -----------------------------------------
    // 4. 获取输出并反量化 (int8 -> float)
    // -----------------------------------------
    // 获取输出张量的 int8 值
    int8_t q_out = mfcc_output_tensor->data.int8[0];
    // 反量化 输出狗吠的概率
    float prob = (q_out - MFCC_OUTPUT_ZERO_POINT) * MFCC_OUTPUT_SCALE;

    return prob;

}


// === LogMel模型实现 ===
int logmel_model_init() {
  return 0;
}

int logmel_model_infer(const float* features, float* probability) {
  return 0;
}