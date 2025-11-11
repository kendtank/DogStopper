#ifndef MEL_FILTERBANK_H
#define MEL_FILTERBANK_H

#include <cstdint>



#ifdef __cplusplus
extern "C" {
#endif


/// ================================
/// 配置参数
/// ================================
#define MEL_BANDS 64        // Mel 滤波器数量
#define N_FFT 512            // FFT 长度
#define NUM_BINS (N_FFT/2+1) // 功率谱长度


/// ================================
/// 全局 Mel 滤波器数组（PSRAM）， 烧写就一次性存入，不提供初始化方法， 需要验证c和python结果一致性[pass]
/// ================================
/// 1. 使用 const 避免修改，提高安全性
/// 2. 数据放在 PSRAM 节区（在 cpp 中定义时加 __attribute__((section(".psram")))）
/// 3. extern 声明表示数组在别的 cpp 文件中定义，避免重复占用空间
// const float mel_filterbank[MEL_BANDS * NUM_BINS];
// 注意： 只是测试使用，测试完，应该在cpp中加static 保护数组



/// ================================
/// log-Mel 特征计算接口， 单帧输入
/// ================================
/// 根据功率谱计算 log-Mel 特征
/// @param power_spectrum 输入功率谱数组，每帧长度 NUM_BINS
/// @param log_mel_out 输出 log_mel_out 特征数组，长度 MEL_BANDS
// void apply_mel(const float* power_spectrum, float* mel_out);   // 测试使用， 测试结果一致性[pass]， 融合logmel会提速10%

// 计算log-mel
void apply_log_mel(const float* power_spectrum, float* log_mel_out);


#ifdef __cplusplus
}
#endif

#endif // MEL_FILTERBANK_H

