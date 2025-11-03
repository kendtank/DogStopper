/*
 * audio_features.c
 *
 * 2025 © Kend.tank
 *
 * 功能：Log-Mel + MFCC 特征提取
 * 平台：ESP32 + Arduino
 * 优化：
 *  - FFT 使用 esp-dsp
 *  - 大数组默认放在 PSRAM
 *  - 可预计算 DCT 矩阵，加速 MFCC
 * #TODO: 待功能一致性测试通过后重新封装功能
 */

#include "audio_features.h"
#include <Arduino.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_heap_caps.h"
#include "dsps_fft2r.h"

// Optional: if you later want to use esp-dsp dotprod functions
// #include "dsps_dotprod.h"

// -------------------- 模块状态 --------------------
static int g_inited = 0;


// -------------------- 全局缓存/表 --------------------
static float *g_hann_win = NULL;       // 汉宁窗长度 N_FFT
static float *g_mel_filter = NULL;     // Mel滤波器 N_MEL_BINS*(N_FFT/2+1)
static float *g_fft_buf = NULL;        // FFT临时buffer N_FFT*2
static float *g_logmel_buf = NULL;     // 临时 log-mel buffer NUM_FRAMES*N_MEL_BINS
#if PRECOMPUTE_DCT
static float *g_dct_matrix = NULL;     // DCT矩阵 N_MFCC*N_MEL_BINS
#endif


// -------------------- 内存分配 --------------------
/**
 * @brief 在PSRAM中分配内存
 *
 * 根据配置决定是否优先使用PSRAM分配内存，如果PSRAM分配失败则回退到普通内存分配
 *
 * @param bytes: 需要分配的字节数
 * @return 分配的内存指针，如果分配失败返回NULL
 */
static inline void *alloc_psram(size_t bytes) {
// 是否启动 PSRAM 分配
#if USE_PSRAM_BUFFERS
    void *p = heap_caps_calloc(1, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) {
        // 申请失败，退回内存
        p = calloc(1, bytes);
    }
    return p;
#else
    return calloc(1, bytes);
#endif
}

/**
 * @brief 释放PSRAM内存
 *
 * 释放通过alloc_psram分配的内存
 *
 * @param p: 指向需要释放的内存的指针
 */
static inline void free_psram(void *p) {
    if (p) heap_caps_free(p);
}

/**
 * @brief 打印内存使用信息
 *
 * 显示堆内存和PSRAM的使用情况
 */
void print_memory_info(void) {
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
    Serial.printf("堆内存剩余: %d bytes\n", ESP.getFreeHeap());
    if (psramFound()) {
        Serial.printf("PSRAM总大小: %d bytes\n", ESP.getPsramSize());
        Serial.printf("PSRAM可用: %d bytes\n", ESP.getFreePsram());
    }
#endif
}

// -------------------- 构建汉宁窗和Mel滤波器 --------------------

/**
 * @brief 生成汉宁窗
 *
 * 生成长度为N的汉宁窗系数，用于音频分帧时减少频谱泄漏
 *
 * @param w: 指向存储汉宁窗系数的数组指针
 * @param N: 汉宁窗的长度
 */
static void make_hann(float *w, int N) {
    for (int n = 0; n < N; n++) {
        // w[n] = 0.5 - 0.5 * cos(2*pi*n/(N-1))
        w[n] = 0.5f - 0.5f * cosf(2.0f * M_PI * n / (N - 1));
    }
}

/**
 * @brief 将频率（Hz）转换为梅尔值
 *
 * 用于生成梅尔滤波器的边界频率，使用的是HTK公式
 *
 * @param f: 输入频率（Hz）
 * @return 对应的梅尔值
 */
static double hz_to_mel(double f) {
    const double f_min = 0.0;
    const double f_sp = 200.0/3.0;
    const double f_break = 1000.0;
    const double logstep = pow(6.4, 1.0/27.0);
    
    if (f < f_break) {
        return (f - f_min) / f_sp;
    } else {
        return (f_break - f_min) / f_sp + log(f / f_break) / log(logstep);
    }
}

/**
 * @brief 将梅尔值转换回频率（Hz）
 *
 * hz_to_mel 的逆函数，用于生成梅尔滤波器的边界频率。
 *
 * @param m: 输入梅尔值
 * @return 对应的频率（Hz）
 */
static double mel_to_hz(double m) {
    const double f_min = 0.0;
    const double f_sp = 200.0/3.0;
    const double f_break = 1000.0;
    const double logstep = pow(6.4, 1.0/27.0);
    const double min_log_mel = (f_break - f_min) / f_sp;
    
    if (m < min_log_mel) {
        return f_min + f_sp * m;
    } else {
        return f_break * exp(log(logstep) * (m - min_log_mel));
    }
}

/**
 * @brief 构建Mel滤波器组
 *
 * 根据给定参数生成Mel滤波器组矩阵，用于将线性频谱转换为Mel频谱
 *
 * @param W: 指向存储滤波器组矩阵的数组指针
 * @param n_mel: Mel滤波器的数量
 * @param n_fft: FFT点数
 * @param sample_rate: 音频采样率
 */
static void build_mel_filters(float *W, int n_mel, int n_fft, int sample_rate) {
    int nfft_bins = n_fft/2 + 1;
    memset(W, 0, sizeof(float) * n_mel * nfft_bins);

    double fft_freqs[nfft_bins];
    for (int k = 0; k < nfft_bins; k++) {
        fft_freqs[k] = (double)sample_rate * k / n_fft;
    }

    double m_min = hz_to_mel(0.0);
    double m_max = hz_to_mel(sample_rate * 0.5);
    double mpts[n_mel + 2];
    for (int i = 0; i < n_mel + 2; i++) {
        mpts[i] = m_min + (m_max - m_min) * (i / (double)(n_mel + 1));
    }
    
    double fpts[n_mel + 2];
    for (int i = 0; i < n_mel + 2; i++) {
        fpts[i] = mel_to_hz(mpts[i]);
    }

    for (int m = 0; m < n_mel; m++) {
        double f0 = fpts[m];
        double f1 = fpts[m+1];
        double f2 = fpts[m+2];
        
        if (f1 <= f0) f1 = f0 + 1e-12;
        if (f2 <= f1) f2 = f1 + 1e-12;
        
        for (int k = 0; k < nfft_bins; k++) {
            double fk = fft_freqs[k];
            float w = 0.0f;
            if (fk <= f0 || fk >= f2) {
                w = 0.0f;
            } else if (fk <= f1) {
                w = (float)((fk - f0) / (f1 - f0));
            } else {
                w = (float)((f2 - fk) / (f2 - f1));
            }
            
            if (w < 0.0f) w = 0.0f;
            W[m * nfft_bins + k] = w * (float)(2.0 / (f2 - f0));
        }
    }
}

// -------------------- FFT -> 功率谱 --------------------
/**
 * @brief 实数FFT并计算功率谱
 *
 * 对输入信号进行实数FFT变换并计算功率谱
 *
 * @param x: 输入信号
 * @param P: 输出功率谱
 * @param fft_buf: FFT计算缓冲区
 * @param n_fft: FFT点数
 */
static void rfft_power(const float *x, float *P, float *fft_buf, int n_fft) {
    // fill complex buffer: re=x, im=0
    for (int i = 0; i < n_fft; i++) {
        fft_buf[2*i] = x[i];
        fft_buf[2*i + 1] = 0.0f;
    }
    dsps_fft2r_fc32(fft_buf, n_fft);
    dsps_bit_rev_fc32(fft_buf, n_fft);
    dsps_cplx2reC_fc32(fft_buf, n_fft);
    int nbin = n_fft/2 + 1;
    for (int k = 0; k < nbin; k++) {
        float re = fft_buf[2*k];
        float im = fft_buf[2*k + 1];
        // ESP example divides by N (not N^2)
        P[k] = (re*re + im*im) / (float)n_fft;
    }
}

// -------------------- DCT (precompute & runtime) --------------------
#if PRECOMPUTE_DCT
static int dct_ready = 0;

/**
 * @brief 预计算DCT矩阵
 *
 * 预计算DCT变换矩阵以加速MFCC计算
 */
static void precompute_dct_matrix(void) {
    if (dct_ready) return;
    // allocate matrix: N_MFCC rows x N_MEL_BINS cols
    size_t bytes = sizeof(float) * (size_t)N_MFCC * (size_t)N_MEL_BINS;
    g_dct_matrix = (float*) alloc_psram(bytes);
    if (!g_dct_matrix) {
        return;
    }
    for (int k = 0; k < N_MFCC; k++) {
        float norm = (k == 0) ? sqrtf(1.0f / (float)N_MEL_BINS) : sqrtf(2.0f / (float)N_MEL_BINS);
        for (int n = 0; n < N_MEL_BINS; n++) {
            float v = cosf(M_PI / (float)N_MEL_BINS * (n + 0.5f) * (float)k);
            g_dct_matrix[k * N_MEL_BINS + n] = norm * v;
        }
    }
    dct_ready = 1;
}
#endif

/**
 * @brief 计算DCT-II变换
 *
 * 计算输入信号的DCT-II变换，用于MFCC计算
 *
 * @param input: 输入信号
 * @param output: 输出DCT系数
 * @param n_input: 输入信号长度
 * @param n_output: 输出系数数量
 */
static void compute_dct_ii_internal(const float *input, float *output, int n_input, int n_output) {
#if PRECOMPUTE_DCT
    if (!dct_ready) precompute_dct_matrix();
    if (!g_dct_matrix) {
        // fallback simple compute
        for (int k = 0; k < n_output; k++) {
            float sum = 0.0f;
            for (int n = 0; n < n_input; n++) {
                sum += input[n] * cosf(M_PI * k * (2*n + 1) / (2.0f * n_input));
            }
            float alpha = (k == 0) ? sqrtf(1.0f / (float)n_input) : sqrtf(2.0f / (float)n_input);
            output[k] = alpha * sum;
        }
        return;
    }
    // matrix multiply (rows = n_output)
    for (int k = 0; k < n_output; k++) {
        float sum = 0.0f;
        const float *row = &g_dct_matrix[k * N_MEL_BINS];
        for (int n = 0; n < n_input; n++) {
            sum += row[n] * input[n];
        }
        output[k] = sum;
    }
#else
    // On-the-fly DCT compute
    for (int k = 0; k < n_output; k++) {
        float sum = 0.0f;
        for (int n = 0; n < n_input; n++) {
            sum += input[n] * cosf(M_PI * k * (2*n + 1) / (2.0f * n_input));
        }
        float alpha = (k == 0) ? sqrtf(1.0f / (float)n_input) : sqrtf(2.0f / (float)n_input);
        output[k] = alpha * sum;
    }
#endif
}

/**
 * @brief 计算DCT-II变换（对外接口）
 *
 * 计算输入信号的DCT-II变换，用于MFCC计算
 *
 * @param input: 输入信号
 * @param output: 输出DCT系数
 * @param n_input: 输入信号长度
 * @param n_output: 输出系数数量
 */
void compute_dct_ii(const float *input, float *output, int n_input, int n_output) {
    // n_input should equal N_MEL_BINS for speed, but we allow arbitrary
    compute_dct_ii_internal(input, output, n_input, n_output);
}

// -------------------- Initialization / Free --------------------
/**
 * @brief 初始化特征提取模块
 *
 * 初始化特征提取所需的各种资源，包括FFT模块、窗口函数、滤波器组等
 *
 * @return 0表示成功，负数表示失败
 */
int feature_extractor_init(void) {
    if (g_inited) return 0;

    // init DSP FFT
    esp_err_t r = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    if (r != ESP_OK) {
        return -1;
    }

    // allocate windows & filters
    g_hann_win = (float*) alloc_psram(sizeof(float) * (size_t)N_FFT);
    if (!g_hann_win) { 
        return -2; 
    }
    make_hann(g_hann_win, N_FFT);

    int nfft_bins = N_FFT/2 + 1;
    g_mel_filter = (float*) alloc_psram(sizeof(float) * (size_t)N_MEL_BINS * (size_t)nfft_bins);
    if (!g_mel_filter) { 
        feature_extractor_free(); 
        return -3; 
    }
    build_mel_filters(g_mel_filter, N_MEL_BINS, N_FFT, SAMPLE_RATE);

    g_fft_buf = (float*) alloc_psram(sizeof(float) * (size_t)N_FFT * 2);
    if (!g_fft_buf) { 
        feature_extractor_free(); 
        return -4; 
    }

    g_logmel_buf = (float*) alloc_psram(sizeof(float) * (size_t)NUM_FRAMES * (size_t)N_MEL_BINS);
    if (!g_logmel_buf) { 
        feature_extractor_free(); 
        return -5; 
    }

#if PRECOMPUTE_DCT
    // DCT matrix will be lazy-allocated in precompute_dct_matrix()
    precompute_dct_matrix();
#endif

#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
    // 在初始化完成后进行调试比较
    // debug_compare_step_by_step();  // 启用调试比较
#endif

    g_inited = 1;
    return 0;
}

/**
 * @brief 释放特征提取模块资源
 *
 * 释放初始化时分配的所有资源
 */
void feature_extractor_free(void) {
    if (!g_inited) return;
    free_psram(g_hann_win); g_hann_win = NULL;
    free_psram(g_mel_filter); g_mel_filter = NULL;
    free_psram(g_fft_buf); g_fft_buf = NULL;
    free_psram(g_logmel_buf); g_logmel_buf = NULL;
#if PRECOMPUTE_DCT
    free_psram(g_dct_matrix); g_dct_matrix = NULL;
#endif
    g_inited = 0;
}

// -------------------- Log-Mel main --------------------
/**
 * @brief 计算200ms音频的Log-Mel特征
 *
 * 对输入的200ms音频计算Log-Mel特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点）
 * @param output: 输出Log-Mel特征
 * @return 成功返回帧数，失败返回负数
 */
int compute_logmel_200ms(const int16_t *input, float *output) {
    unsigned long start_time = 0;
    
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
    start_time = millis();
#endif
    
    if (!g_inited) {
        if (feature_extractor_init() != 0) return -1;
    }
    if (!input || !output) return -2;

    // local frame buffer on stack (small)
    float frame[N_FFT];
    int nfft_bins = N_FFT/2 + 1;
    float *P = (float*)alloca(sizeof(float) * nfft_bins); // small, OK on stack

    for (int f = 0; f < NUM_FRAMES; f++) {
        int offset = f * FRAME_SHIFT;
        // copy & window (assume input length >= required)
        for (int i = 0; i < N_FFT; i++) {
            if (i < FRAME_LENGTH) {
                frame[i] = (float)input[offset + i] * g_hann_win[i];
            } else {
                frame[i] = 0.0f;
            }
        }
        
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
        // 保存第0帧数据用于调试
        if (f == 0) {
            // 比较第0帧数据
            float frame0_ref[FRAME_LENGTH];
            for (int i = 0; i < FRAME_LENGTH; i++) {
                frame0_ref[i] = (float)input[offset + i] / 32768.0f;
            }
            // compare_with_reference(frame0_ref, ::frame0, FRAME_LENGTH, "第0帧数据(实际输入)");
            
            // 比较第0帧加窗后数据
            // compare_with_reference(frame, ::frame0_windowed, FRAME_LENGTH, "第0帧加窗后数据");
        }
#endif

        // compute power spectrum -> P
        rfft_power(frame, P, g_fft_buf, N_FFT);

#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
        // 保存第0帧功率谱用于调试
        if (f == 0) {
            // compare_with_reference(P, ::frame0_power, nfft_bins, "第0帧功率谱");
        }
#endif

        // mel filtering
        for (int m = 0; m < N_MEL_BINS; m++) {
            float mel_energy = 0.0f;
            const float *mf_row = &g_mel_filter[m * nfft_bins];
            for (int k = 0; k < nfft_bins; k++) {
                mel_energy += mf_row[k] * P[k];
            }
            // 提高数值稳定性
            const float min_energy = 1e-10f;
            mel_energy = fmaxf(mel_energy, min_energy);
            output[f * N_MEL_BINS + m] = 10.0f * log10f(mel_energy);
        }
        
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
        // 保存第0帧Log-Mel特征用于调试
        if (f == 0) {
            // compare_with_reference(&output[f * N_MEL_BINS], ::frame0_logmel, N_MEL_BINS, "第0帧Log-Mel特征");
        }
#endif
    }
    
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
    unsigned long end_time = millis();
    Serial.printf("[PERFORMANCE] Log-Mel计算耗时: %lu ms\n", end_time - start_time);
#endif
    
    return NUM_FRAMES;
}

// -------------------- MFCC main --------------------
/**
 * @brief 计算200ms音频的MFCC特征
 *
 * 对输入的200ms音频计算MFCC特征
 *
 * @param input: 输入音频数据（16kHz采样率，3200个采样点）
 * @param output: 输出MFCC特征
 * @return 成功返回帧数，失败返回负数
 */
int compute_mfcc_200ms(const int16_t *input, float *output) {
    unsigned long start_time = 0;
    
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
    start_time = millis();
#endif
    
    if (!g_inited) {
        if (feature_extractor_init() != 0) return -1;
    }
    if (!input || !output) return -2;

    // compute logmel into g_logmel_buf (PSRAM)
    int frames = compute_logmel_200ms(input, g_logmel_buf);
    if (frames <= 0) return -3;

    // for each frame compute DCT -> MFCC
    for (int f = 0; f < frames; f++) {
        const float *mel = &g_logmel_buf[f * N_MEL_BINS];
        float *mfcc = &output[f * N_MFCC];
        compute_dct_ii_internal(mel, mfcc, N_MEL_BINS, N_MFCC);
    }
    
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
    unsigned long end_time = millis();
    Serial.printf("[PERFORMANCE] MFCC计算耗时: %lu ms\n", end_time - start_time);
#endif
    
    return frames;
}

// 添加调试数据保存函数
#if defined(DEBUG_FEATURES) && DEBUG_FEATURES
// 包含Python生成的参考数据
#include "dbg_data.h"

static int compare_with_reference(const float* data, const float* reference, int length, const char* name, float tolerance = 1e-4f) {
    Serial.printf("比较 %s: ", name);
    
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    
    for (int i = 0; i < length; i++) {
        float diff = fabsf(data[i] - reference[i]);
        avg_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    
    avg_diff /= length;
    
    Serial.printf("最大差异 = %.6f, 平均差异 = %.6f\n", max_diff, avg_diff);
    
    if (max_diff < tolerance) {
        Serial.println("  ✓ 通过");
        return 1; // 通过
    } else {
        Serial.println("  ✗ 失败");
        // 找到第一个差异较大的位置
        for (int i = 0; i < length; i++) {
            if (fabsf(data[i] - reference[i]) > tolerance) {
                Serial.printf("    位置[%d]: 实际=%.6f, 参考=%.6f, 差异=%.6f\n", 
                             i, data[i], reference[i], fabsf(data[i] - reference[i]));
                break;
            }
        }
        return 0; // 失败
    }
}

static void debug_compare_step_by_step() {
    Serial.println("=== 开始逐步调试比较 ===");
    
    // 生成测试信号
    const int test_length = 3200;
    float test_input[test_length];
    const float frequency = 500.0f;
    const float amplitude = 0.3f;
    const float sample_rate = 16000.0f;
    
    for (int i = 0; i < test_length; i++) {
        float t = i / sample_rate;
        test_input[i] = amplitude * sinf(2.0f * M_PI * frequency * t);
    }
    
    // 比较输入信号
    bool passed = true;
    float max_diff = 0.0f;
    for (int i = 0; i < test_length; i++) {
        float diff = fabsf(test_input[i] - input_signal[i]);
        if (diff > max_diff) max_diff = diff;
    }
    Serial.printf("输入信号比较 - 最大差异: %.6f\n", max_diff);
    if (max_diff < 1e-5f) {
        Serial.println("  ✓ 输入信号匹配");
    } else {
        Serial.println("  ✗ 输入信号不匹配");
        passed = false;
    }
    
    // 比较汉宁窗
    max_diff = 0.0f;
    for (int i = 0; i < FRAME_LENGTH; i++) {
        float diff = fabsf(g_hann_win[i] - hann_window[i]);
        if (diff > max_diff) max_diff = diff;
    }
    Serial.printf("汉宁窗比较 - 最大差异: %.6f\n", max_diff);
    if (max_diff < 1e-5f) {
        Serial.println("  ✓ 汉宁窗匹配");
    } else {
        Serial.println("  ✗ 汉宁窗不匹配");
        passed = false;
    }
    
    // 比较Mel滤波器组
    int nfft_bins = N_FFT/2 + 1;
    max_diff = 0.0f;
    for (int i = 0; i < MEL_FILTER_ROWS; i++) {
        for (int j = 0; j < MEL_FILTER_COLS && j < nfft_bins; j++) {
            float diff = fabsf(g_mel_filter[i * nfft_bins + j] - mel_filter[i][j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    Serial.printf("Mel滤波器组比较 - 最大差异: %.6f\n", max_diff);
    if (max_diff < 1e-5f) {
        Serial.println("  ✓ Mel滤波器组匹配");
    } else {
        Serial.println("  ✗ Mel滤波器组不匹配");
        passed = false;
    }
    
    // 比较第0帧处理过程
    float frame0_data[FRAME_LENGTH];
    for (int i = 0; i < FRAME_LENGTH; i++) {
        frame0_data[i] = test_input[i] / 32768.0f;  // 归一化
    }
    
    max_diff = 0.0f;
    for (int i = 0; i < FRAME_LENGTH; i++) {
        float diff = fabsf(frame0_data[i] - frame0[i]);
        if (diff > max_diff) max_diff = diff;
    }
    Serial.printf("第0帧数据比较 - 最大差异: %.6f\n", max_diff);
    if (max_diff < 1e-5f) {
        Serial.println("  ✓ 第0帧数据匹配");
    } else {
        Serial.println("  ✗ 第0帧数据不匹配");
        passed = false;
        // 显示前几个值作参考
        Serial.println("  前10个值比较:");
        for (int i = 0; i < 10 && i < FRAME_LENGTH; i++) {
            Serial.printf("    [%d] 实际:%.6f 参考:%.6f 差异:%.6f\n", 
                         i, frame0_data[i], frame0[i], fabsf(frame0_data[i] - frame0[i]));
        }
    }
    
    // 比较第0帧加窗后数据
    float frame0_windowed_data[FRAME_LENGTH];
    for (int i = 0; i < FRAME_LENGTH; i++) {
        frame0_windowed_data[i] = frame0_data[i] * g_hann_win[i];
    }
    
    max_diff = 0.0f;
    for (int i = 0; i < FRAME_LENGTH; i++) {
        float diff = fabsf(frame0_windowed_data[i] - frame0_windowed[i]);
        if (diff > max_diff) max_diff = diff;
    }
    Serial.printf("第0帧加窗后数据比较 - 最大差异: %.6f\n", max_diff);
    if (max_diff < 1e-5f) {
        Serial.println("  ✓ 第0帧加窗后数据匹配");
    } else {
        Serial.println("  ✗ 第0帧加窗后数据不匹配");
        passed = false;
        // 显示前几个值作参考
        Serial.println("  前10个值比较:");
        for (int i = 0; i < 10 && i < FRAME_LENGTH; i++) {
            Serial.printf("    [%d] 实际:%.6f 参考:%.6f 差异:%.6f\n", 
                         i, frame0_windowed_data[i], frame0_windowed[i], 
                         fabsf(frame0_windowed_data[i] - frame0_windowed[i]));
        }
    }
    
    // 计算第0帧功率谱并与参考值比较
    float frame_for_fft[N_FFT];
    for (int i = 0; i < N_FFT; i++) {
        if (i < FRAME_LENGTH) {
            frame_for_fft[i] = frame0_windowed_data[i];
        } else {
            frame_for_fft[i] = 0.0f;
        }
    }
    
    int nfft_bins_power = N_FFT/2 + 1;
    float power_spectrum[nfft_bins_power];
    float fft_buffer[N_FFT * 2];
    
    // 复制复数缓冲区: re=frame_for_fft, im=0
    for (int i = 0; i < N_FFT; i++) {
        fft_buffer[2*i] = frame_for_fft[i];
        fft_buffer[2*i + 1] = 0.0f;
    }
    
    dsps_fft2r_fc32(fft_buffer, N_FFT);
    dsps_bit_rev_fc32(fft_buffer, N_FFT);
    dsps_cplx2reC_fc32(fft_buffer, N_FFT);
    
    for (int k = 0; k < nfft_bins_power; k++) {
        float re = fft_buffer[2*k];
        float im = fft_buffer[2*k + 1];
        power_spectrum[k] = (re*re + im*im) / (float)N_FFT;
    }
    
    max_diff = 0.0f;
    for (int i = 0; i < nfft_bins_power; i++) {
        float diff = fabsf(power_spectrum[i] - frame0_power[i]);
        if (diff > max_diff) max_diff = diff;
    }
    Serial.printf("第0帧功率谱比较 - 最大差异: %.6f\n", max_diff);
    if (max_diff < 1e-5f) {
        Serial.println("  ✓ 第0帧功率谱匹配");
    } else {
        Serial.println("  ✗ 第0帧功率谱不匹配");
        passed = false;
        // 显示前几个值作参考
        Serial.println("  前10个值比较:");
        for (int i = 0; i < 10 && i < nfft_bins_power; i++) {
            Serial.printf("    [%d] 实际:%.6f 参考:%.6f 差异:%.6f\n", 
                         i, power_spectrum[i], frame0_power[i], 
                         fabsf(power_spectrum[i] - frame0_power[i]));
        }
    }
    
    // 计算第0帧Mel特征并与参考值比较
    float mel_features[N_MEL_BINS];
    for (int m = 0; m < N_MEL_BINS; m++) {
        float mel_energy = 0.0f;
        const float *mf_row = &g_mel_filter[m * nfft_bins_power];
        for (int k = 0; k < nfft_bins_power; k++) {
            mel_energy += mf_row[k] * power_spectrum[k];
        }
        const float min_energy = 1e-10f;
        mel_energy = fmaxf(mel_energy, min_energy);
        mel_features[m] = 10.0f * log10f(mel_energy);
    }
    
    max_diff = 0.0f;
    for (int i = 0; i < N_MEL_BINS; i++) {
        float diff = fabsf(mel_features[i] - frame0_logmel[i]);
        if (diff > max_diff) max_diff = diff;
    }
    Serial.printf("第0帧Log-Mel特征比较 - 最大差异: %.6f\n", max_diff);
    if (max_diff < 1e-3f) {  // 略微放宽容差
        Serial.println("  ✓ 第0帧Log-Mel特征匹配");
    } else {
        Serial.println("  ✗ 第0帧Log-Mel特征不匹配");
        passed = false;
        // 显示前几个值作参考
        Serial.println("  前10个值比较:");
        for (int i = 0; i < 10 && i < N_MEL_BINS; i++) {
            Serial.printf("    [%d] 实际:%.6f 参考:%.6f 差异:%.6f\n", 
                         i, mel_features[i], frame0_logmel[i], 
                         fabsf(mel_features[i] - frame0_logmel[i]));
        }
    }
    
    if (passed) {
        Serial.println("=== 所有调试比较通过! ===");
    } else {
        Serial.println("=== 存在调试比较失败项 ===");
    }
}
#endif
