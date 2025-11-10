#include "mel_filterbank.h"
#include <math.h>

/// ================================
/// Mel滤波器数组定义
/// 数据直接由Python生成的.h文件包含
/// 放在PSRAM节区，避免占用内部RAM
/// 验证c和python结果一致性[pass]  1e-5f 误差范围内一致  TODO: 是否需要加static 修饰
/// ================================
const float mel_filterbank[MEL_BANDS * NUM_BINS] 
    __attribute__((section(".psram"))) = {
    #include "mel_filterbank_data.h"   // 这里是Python生成的纯数据内容文件
};



/// ================================
/// Mel 特征计算函数
/// 功能：将线性功率谱转换为 Mel 频率能量谱。
/// mel[i]=j∑​W[i,j]⋅P[j]  功率转Mel
/// 每个 mel 滤波器（第 i 个三角窗）去「加权求和」功率谱上的能量。
/// 输入：
///   power_spectrum — 功率谱数组（长度 NUM_BINS = NFFT/2+1）
/// 输出：
///   mel_out — Mel 滤波后能量数组（长度 MEL_BANDS）
/// ================================
void apply_mel(const float* power_spectrum, float* mel_out) {
    // 遍历每一个 mel 滤波器（共 MEL_BANDS 个）
    for (int i = 0; i < MEL_BANDS; i++) {
        float sum = 0.0f;
        // 遍历功率谱的每一个频率 bin（共 NUM_BINS 个）
        for (int j = 0; j < NUM_BINS; j++) {
            // mel_filterbank[i * NUM_BINS + j] 表示第 i 个滤波器在第 j 个频点的权重
            // power_spectrum[j] 是该频点的功率值
            // 两者相乘并累加，即为第 i 个 mel 滤波器的加权能量
            sum += mel_filterbank[i * NUM_BINS + j] * power_spectrum[j];
        }
        // 得到第 i 个 mel 滤波器的总能量
        mel_out[i] = sum;
        // mel_out[i] = logf(sum + 1e-8f);  // log-mel
    }
}


// 注意：apply_mel() 是计算单帧的 Mel 能量谱。它的输入 power 是一帧的功率谱（比如从 25ms 的音频帧 FFT 得到的频谱能量），输出 mel_out 就是这帧对应的 Mel 滤波能量向量。



/// ================================
/// Log-Mel 特征计算（以 dB 形式输出）
/// log_mel[i] = log10(mel[i] + eps)
/// ================================
// void apply_log_mel(const float* power_spectrum, float* logmel_out) {
//     float mel_linear[MEL_BANDS];
//     apply_mel(power_spectrum, mel_linear);
//     for (int i = 0; i < MEL_BANDS; i++) {
//         float x = mel_linear[i];

//         if (x < 1e-10f) x = 1e-10f;   // librosa 对齐
//         // if (x < 1e-6f) x = 1e-6f;
//         // 稍微提高下限，避免浮点数进入极端区
//         logmel_out[i] = 10.0f * log10f(x);
//     }
// }
void apply_log_mel(const float* power_spectrum, float* logmel_out)
{
    float mel_linear[MEL_BANDS];
    apply_mel(power_spectrum, mel_linear);  // MCU 端计算 Mel 滤波

    // step1: 找到本帧最大值，用于 top_db
    float max_val = mel_linear[0];
    for (int i = 1; i < MEL_BANDS; i++) {
        if (mel_linear[i] > max_val)
            max_val = mel_linear[i];
    }

    for (int i = 0; i < MEL_BANDS; i++) {
        float x = mel_linear[i];

        // 避免 log(0)
        if (x < 1e-8f) x = 1e-8f;

        // 计算 dB
        float db = 10.0f * log10f(x);

        // top_db 限制
        float min_db = 10.0f * log10f(max_val) - 100.0f;  // top_db = 100
        if (db < min_db)
            db = min_db;

        logmel_out[i] = db;
    }
}
