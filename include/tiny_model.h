#ifndef TINY_MODEL_H
#define TINY_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MFCC_INPUT_SIZE 234  // 输入特征数量  18 * 13 = 234


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
TODO:logmel模型还未完成， 先直接return
    推理logmel模型
    input: 输入logmel特征  [18 * 64] 输入归一化和量化为int8_t逻辑也包含在里面
    output: 输出
*/
int logmel_model_infer(const float* features, float* probability);


#ifdef __cplusplus
}
#endif

#endif // TINY_MODEL_H
