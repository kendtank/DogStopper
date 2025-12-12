#ifndef EMBED_MODEL_DATA_H
#define EMBED_MODEL_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// 声明 EMBED TinyML 模型数据
// 注意：实际字节在 embed_model_int8.cc 中定义
// -----------------------------------------------------------------------------
extern const unsigned char embed_model_int8_tflite[];
extern const unsigned int embed_model_int8_tflite_len;

#ifdef __cplusplus
}
#endif

#endif // EMBED_MODEL_DATA_H
