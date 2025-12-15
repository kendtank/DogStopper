#ifndef TINY_MODEL_H
#define TINY_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MFCC_INPUT_SIZE 234  // 输入特征数量  18 * 13 = 234
#define EMBED_INPUT_SIZE 720  // 输入特征数量  18 * 40 = 720
#define EMBED_OUTPUT_SIZE 32


// 初始化mfcc模型， 作为接口暴露出来
int mfcc_model_init();

/* 
    推理mfcc模型
    input: 输入mfcc特征  [18 * 13] 特征归一化和量化为int8_t逻辑也包含在里面
    output: 输出概率  输出节点是狗吠的int8_t概率，需要反量化为float输出
*/
float mfcc_model_infer(const float* features);


// 初始化logmel模型
int logmel_model_init();


/* 
    推理logmel-embed模型
    input: 输入logmel特征  [18 * 40]
    output: 32D embedding
*/
int embed_model_infer(const float* features, float* embedding);


// // 计算两个向量的余弦相似度
// float cosine_similarity(const float *a, const float *b, int size);


#ifdef __cplusplus
}
#endif

#endif // TINY_MODEL_H
