#ifndef MFCC_H
#define MFCC_H

#include <Arduino.h>

#define MEL_BANDS 40      // 输入 log-mel 特征维度
#define MFCC_COEFFS 13    // 输出 MFCC 特征维度

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief 计算 MFCC 特征
 * @param logmel_in 输入 log-mel 特征数组 [MEL_BANDS]
 * @param mfcc_out 输出 MFCC 特征数组 [MFCC_COEFFS]
 */
void compute_mfcc(const float* logmel_in, float* mfcc_out);


#ifdef __cplusplus
}
#endif


#endif // MFCC_H

