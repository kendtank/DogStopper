#include "mfcc.h"
#include "esp_dsp.h"  // dsps_dotprod_f32
#include <math.h>


// ==================== 全局 DCT-II 矩阵 存放到 psram 内存 13 * 63====================
// MEL_BANDS x MFCC_COEFFS
static const float dct_matrix[MFCC_COEFFS * MEL_BANDS]
    __attribute__((section(".psram"))) = {
    #include "dct_matrix.h"  // 这里是Python生成的DCT-II矩阵数据  已经验证过mcu和py差距在1e-5以内
};



// ==================== 计算 MFCC ====================
void compute_mfcc(const float* logmel_in, float* mfcc_out) {
    // dct_matrix 是一维数组，行主序存储 (row-major)
    // numpy 默认列优先，如果生成dct_matrix时已经按行展平，就不需要转置
    for (int k = 0; k < MFCC_COEFFS; k++) {
        // 每行偏移 k * MEL_BANDS
        const float* dct_row = &dct_matrix[k * MEL_BANDS];
        float acc = 0.0f;
        // ESP-DSP 最新 API，第三个参数是结果指针
        dsps_dotprod_f32_aes3(dct_row, logmel_in, &acc, MEL_BANDS);
        mfcc_out[k] = acc;
    }
}


