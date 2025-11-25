#ifndef MFCC_MODEL_DATA_H
#define MFCC_MODEL_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// 声明 MFCC TinyML 模型数据
// 注意：实际字节在 mfcc_model_data.cc 中定义
// -----------------------------------------------------------------------------
extern const unsigned char quantized_model_mfcc_int8[];
extern const unsigned int quantized_model_mfcc_int8_len;

#ifdef __cplusplus
}
#endif

#endif // MFCC_MODEL_DATA_H
