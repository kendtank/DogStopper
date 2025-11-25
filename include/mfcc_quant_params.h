/**
    文档说明：
    接口compute_mfcc_200ms返回的 float 类型的mfcc特征 [18* 13] 矩阵
    模型训练使用了归一化操作，需要先把mfcc特征进行归一化
    tinyml模型是量化的int8 类型，需要将float类型的mfcc特征进行量化
    tinyml的输出也是int8类型，需要先将int8的输出进行反量化
 */


#ifndef MFCC_QUANT_PARAMS_H
#define MFCC_QUANT_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

// 注意: 以下的参数全部由python模型训练端得出，模型重新训练，需要实时更新

// -------------------------
// MFCC 输入归一化参数
// -------------------------
#define NORM_SCALE  (0.00365884f)
#define NORM_OFFSET (0.66899937f)
#define NORM_TARGET_MIN (-1.0f)
#define NORM_TARGET_MAX (1.0f)

// -------------------------
// MFCC 输入量化参数（float -> int8）
// -------------------------
#define MFCC_INPUT_SCALE  (0.00784314f)
#define MFCC_INPUT_ZERO_POINT  (-1)

// -------------------------
// MFCC 输出量化参数（int8 -> float）
// -------------------------
#define MFCC_OUTPUT_SCALE (0.00390625f)
#define MFCC_OUTPUT_ZERO_POINT (-128)

#ifdef __cplusplus
}
#endif

#endif // MFCC_QUANT_PARAMS_H
