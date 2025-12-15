
#include <math.h>
#include "tiny_model.h"
#include "mfcc_model_data.h"  // 模型 FlatBuffer C 数组  python端量化生成
#include "mfcc_norm_params.h"  // 模型归一化参数
#include "embed_model_data.h"

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
static float mfcc_in_scale = 0.0f;    // 输入量化 scale
static int mfcc_in_zero    = 0;      // 输入量化 zero_point
static float mfcc_out_scale = 0.0f;   // 输出反量化 scale
static int mfcc_out_zero    = 0;      // 输出反量化 zero_point


// ===== embed 模型全局对象 =====
static tflite::MicroErrorReporter embed_error_reporter;    // TinyML Micro 版本的错误报告器，用于打印模型推理错误 
static uint8_t embed_arena[16 * 1024] __attribute__((aligned(16)));  // 模型运行时需要的工作内存（tensor buffer），Micro 端必须自己分配连续内存
static tflite::AllOpsResolver embed_resolver;                  // 包含模型中需要的算子（Conv, FullyConnected 等），让 interpreter 知道怎么执行
static const tflite::Model* embed_model = nullptr;           // 指向需要加载的 FlatBuffer 模型
static tflite::MicroInterpreter* embed_interpreter = nullptr;    // 推理解释器对象，用于执行模型
static TfLiteTensor* embed_input_tensor = nullptr;   // 输入张量指针（方便直接 memcpy 特征进去）
static TfLiteTensor* embed_output_tensor = nullptr;    // 输出张量指针（方便直接读取推理结果）
static float embed_in_scale = 0.0f;    // 输入量化 scale
static int embed_in_zero    = 0;      // 输入量化 zero_point
static float embed_out_scale = 0.0f;   // 输出反量化 scale
static int embed_out_zero    = 0;      // 输出反量化 zero_point



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

    // 量化参数可以直接从模型输入张量获取
    mfcc_in_scale = mfcc_input_tensor->params.scale;
    mfcc_in_zero    = mfcc_input_tensor->params.zero_point;
    mfcc_out_scale = mfcc_output_tensor->params.scale;
    mfcc_out_zero    = mfcc_output_tensor->params.zero_point;

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
        int32_t q = (int32_t)(norm / mfcc_in_scale + mfcc_in_zero);
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
    float prob = (q_out - mfcc_out_zero) * mfcc_out_scale;

    return prob;

}


// ===== 初始化embed模型 =====
int logmel_model_init() {
    embed_model = tflite::GetModel(embed_model_int8_tflite);
    if (!embed_model) return -1;
    if (embed_model->version() != TFLITE_SCHEMA_VERSION) return -2;

    // 用 static 对象避免栈释放
    static tflite::MicroInterpreter static_interpreter_embed(
        embed_model, embed_resolver, embed_arena, sizeof(embed_arena), &embed_error_reporter
    );
    embed_interpreter = &static_interpreter_embed;

    if (embed_interpreter->AllocateTensors() != kTfLiteOk) return -3;

    embed_input_tensor  = embed_interpreter->input(0);
    embed_output_tensor = embed_interpreter->output(0);
    // 量化参数直接从模型输入张量获取
    embed_in_scale = embed_input_tensor->params.scale;
    embed_in_zero    = embed_input_tensor->params.zero_point;
    embed_out_scale = embed_output_tensor->params.scale;
    embed_out_zero    = embed_output_tensor->params.zero_point;

    return 0;
}


int embed_model_infer(const float* features, float* embedding) {
    // 默认完成了初始化  注意： 这个模型没有输入特征的归一化，因为训练时没有做归一化处理

    // -----------------------------------------
    // 1. 输入量化 (float -> int8)
    // ----------
    int8_t* in_buf = embed_input_tensor->data.int8;

    // // 获取 int8 元素数量
    // int input_len = EMBED_INPUT_SIZE;

    for (int i = 0; i < EMBED_INPUT_SIZE; i++) {

        // float → int32（中间值）
        float x = features[i] / embed_in_scale;

        int q = (x >= 0 ? x + 0.5f : x - 0.5f);  // 等价 roundf(x)

        q += embed_in_zero;

        // clamp 到 int8
        if (q > 127) q = 127;
        if (q < -128) q = -128;

        in_buf[i] = (int8_t)q;
    }

    // === 2. 推理 === //
    if (embed_interpreter->Invoke() != kTfLiteOk) {
        return -1;
    }

    // === 3. 输出反量化 === //
    int8_t* out_buf = embed_output_tensor->data.int8;

    // int out_len = embed_output_tensor->bytes / sizeof(int8_t);

    for (int i = 0; i < EMBED_OUTPUT_SIZE; i++) {
        embedding[i] = (out_buf[i] - embed_out_zero) * embed_out_scale;
    }

    return 0;
}


// // 计算两个向量的余弦相似度， 用于验证embedding的相似性
// float cosine_similarity(const float *a, const float *b, int size) {
//     float dot = 0.0f;
//     float norm_a = 0.0f;
//     float norm_b = 0.0f;

//     for (int i = 0; i < size; i++) {
//         float ai = a[i];
//         float bi = b[i];
//         dot     += ai * bi;
//         norm_a  += ai * ai;
//         norm_b  += bi * bi;
//     }

//     // 避免除零
//     float denom = sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f;
//     return dot / denom;
// }
