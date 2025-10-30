/*
 * 2025 © Kend.tank
 *
 * 功能：计算 200ms 音频片段的 Log-Mel Spectrogram
 * 平台：ESP32 (使用 esp-dsp 库)
 */

#include "compute_logmel.h"
#include <Arduino.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_heap_caps.h"
#include "dsps_fft2r.h"


// -------------------- 全局缓存 --------------------
static bool dsp_inited = false;                // DSP 库是否初始化状态, 必须初始化，否则会报空指针错误
static float hann_window[N_FFT];              // 汉宁窗系数
static float mel_filter[N_MEL_BINS * (N_FFT/2 + 1)];  // 梅尔滤波器组权重


// -------------------- 工具函数 --------------------

/**
 * @brief 生成汉宁窗（Hanning Window）
 *
 * 汉宁窗用于分帧时平滑加窗，减少频谱泄漏。
 * 公式：w[n] = 0.5 - 0.5 * cos(2πn / (N-1))
 *
 * @param win: 输出窗函数数组，长度为 N_FFT
 */
static void make_hann(float *win)
{
    int n;
    for (n = 0; n < N_FFT; n++) {
        win[n] = 0.5f - 0.5f * cosf(2.0f * M_PI * n / (N_FFT - 1));
    }
}


/**
 * @brief 将频率（Hz）转换为梅尔（mel）刻度
 *
 * 梅尔刻度模拟人耳对频率的非线性感知：
 *  - 低频区域分辨率高
 *  - 高频区域分辨率低
 *
 * 使用分段线性-对数转换（常见于语音处理）：
 *  - f < 1000Hz: 线性映射
 *  - f >= 1000Hz: 对数映射
 *
 * @param f: 输入频率（Hz）
 * @return 对应的梅尔值
 */
static double hz_to_mel(double f)
{
    const double f_min = 0.0;           // 最低频率偏移
    const double f_sp = 200.0 / 3.0;    // 线性段斜率（~66.67 Hz/mel）
    const double f_break = 1000.0;      // 分段转折点（1000 Hz）
    const double logstep = pow(6.4, 1.0 / 27.0);  // 对数步长因子

    if (f < f_break) {
        return (f - f_min) / f_sp;  // 线性段
    } else {
        return (f_break - f_min) / f_sp + log(f / f_break) / log(logstep);  // 对数段
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
static double mel_to_hz(double m)
{
    const double f_min = 0.0;
    const double f_sp = 200.0 / 3.0;
    const double f_break = 1000.0;
    const double logstep = pow(6.4, 1.0 / 27.0);
    const double min_log_mel = (f_break - f_min) / f_sp;  // 转折点对应的 mel 值

    if (m < min_log_mel) {
        return f_min + f_sp * m;  // 线性段
    } else {
        return f_break * exp(log(logstep) * (m - min_log_mel));  // 对数段
    }
}


/**
 * @brief 构建梅尔滤波器组（Mel Filter Bank）
 *
 * 生成一组三角形滤波器，用于将线性频率谱映射到梅尔刻度。
 * 每个滤波器覆盖一段频率范围，在其峰值处响应最大，两侧线性衰减。
 *
 * 步骤：
 *  1. 在梅尔刻度上均匀取 N_MEL_BINS + 2 个点（含边界）
 *  2. 转换回线性频率，得到每个滤波器的左、中、右边界
 *  3. 对每个频点计算三角形权重，并归一化（面积为1）
 *
 * 输出：W[m*(N_FFT/2+1) + k] 表示第 m 个滤波器在第 k 个FFT频点的权重
 *
 * @param W: 输出滤波器权重矩阵，大小为 N_MEL_BINS × (N_FFT/2+1)
 */
static void build_mel_filters(float *W)
{
    int m, k, i;
    double fft_freqs[N_FFT / 2 + 1];   // 每个FFT bin 对应的实际频率（Hz）
    double m_min, m_max;                // 梅尔刻度的最小/最大值
    double mpts[N_MEL_BINS + 2];        // 梅尔刻度上的等距点
    double fpts[N_MEL_BINS + 2];        // 转换回线性频率的边界点
    double f0, f1, f2;                  // 当前滤波器的左、中、右边界
    double fk;                          // 当前FFT频点频率
    float w;                            // 权重

    // 1. 初始化所有权重为 0
    memset(W, 0, sizeof(float) * N_MEL_BINS * (N_FFT / 2 + 1));

    // 2. 计算每个FFT频点对应的实际频率（Hz）
    //    实数FFT只有前 N_FFT/2+1 个点有效（0 ~ Nyquist）
    for (k = 0; k < N_FFT / 2 + 1; k++) {
        fft_freqs[k] = (double)SAMPLE_RATE * k / N_FFT;
    }

    // 3. 定义梅尔刻度范围
    m_min = hz_to_mel(0.0);
    m_max = hz_to_mel(SAMPLE_RATE * 0.5);  // Nyquist 频率

    // 4. 在梅尔刻度上均匀取 N_MEL_BINS + 2 个点
    for (i = 0; i < N_MEL_BINS + 2; i++) {
        mpts[i] = m_min + (m_max - m_min) * (i / (double)(N_MEL_BINS + 1));
    }

    // 5. 转换回线性频率，作为滤波器边界
    for (i = 0; i < N_MEL_BINS + 2; i++) {
        fpts[i] = mel_to_hz(mpts[i]);
    }

    // 6. 构建每个三角形滤波器
    for (m = 0; m < N_MEL_BINS; m++) {
        f0 = fpts[m];     // 左边界
        f1 = fpts[m + 1]; // 峰值（中心）
        f2 = fpts[m + 2]; // 右边界

        // 防止频率点重合（避免除零）
        if (f1 <= f0) f1 = f0 + 1e-12;
        if (f2 <= f1) f2 = f1 + 1e-12;

        for (k = 0; k < N_FFT / 2 + 1; k++) {
            fk = fft_freqs[k];
            w = 0.0f;

            // 三角形滤波器权重
            if (fk <= f0 || fk >= f2) {
                w = 0.0f;  // 范围外
            } else if (fk <= f1) {
                w = (float)((fk - f0) / (f1 - f0));  // 上升段
            } else {
                w = (float)((f2 - fk) / (f2 - f1));  // 下降段
            }

            // 归一化：乘以 2.0/(f2-f0)，使三角形面积为1
            W[m * (N_FFT / 2 + 1) + k] = w * (float)(2.0 / (f2 - f0));
        }
    }
}


/**
 * @brief 计算实数FFT功率谱(快速傅里叶变换)
 *
 * 使用 esp-dsp 库进行实数FFT计算。
 * 流程：
 *  1. 实部填充，虚部为0
 *  2. 执行 RFFT
 *  3. 位翻转
 *  4. 提取实数部分（复数转实数表示）
 *  5. 计算功率：P[k] = (re² + im²) / N_FFT
 *
 * @param x: 输入时域信号，长度 N_FFT
 * @param P: 输出功率谱，长度 N_FFT/2+1
 * @param fft_buf: 临时缓冲区，长度 N_FFT*2（复数格式）
 */
static void rfft_power(const float *x, float *P, float *fft_buf)
{
    int i, k;

    // 1. 填充实数输入（实部 = x[i], 虚部 = 0）
    for (i = 0; i < N_FFT; i++) {
        fft_buf[2 * i]     = x[i];        // 实部
        fft_buf[2 * i + 1] = 0.0f;        // 虚部
    }

    // 2. 执行实数FFT
    dsps_fft2r_fc32(fft_buf, N_FFT);

    // 3. 位翻转（bit-reverse）以得到正确顺序
    dsps_bit_rev_fc32(fft_buf, N_FFT);

    // 4. 将复数结果转换为交错实数格式（仅保留正频率）
    dsps_cplx2reC_fc32(fft_buf, N_FFT);

    // 5. 计算功率谱（只取前 N_FFT/2+1 个点）
    for (k = 0; k < N_FFT / 2 + 1; k++) {
        float re = fft_buf[2 * k];
        float im = fft_buf[2 * k + 1];
        P[k] = (re * re + im * im) / N_FFT;  // 归一化功率， ESP官方实现方式
    }
}



// -------------------- 主函数 --------------------

/**
 * @brief 计算 200ms 音频的 Log-Mel 特征
 *
 * 输入：16-bit PCM 音频数据（200ms）
 * 输出：NUM_FRAMES × N_MEL_BINS 的 Log-Mel 特征图
 *
 * 流程：
 *  1. 初始化 DSP（首次调用）
 *  2. 分帧 + 加窗（每帧 N_FFT，步长 FRAME_SHIFT）
 *  3. 对每帧计算 RFFT 功率谱
 *  4. 应用梅尔滤波器组 → 得到 Mel 能量
 *  5. 取对数 → Log-Mel Spectrogram
 *
 * @param input:  输入音频数据，int16_t，长度 = NUM_FRAMES * FRAME_SHIFT + (N_FFT - FRAME_SHIFT)
 * @param output: 输出特征，float，长度 = NUM_FRAMES * N_MEL_BINS
 */
void compute_logmel_200ms(const int16_t *input, float *output)
{
    float frame[N_FFT];           // 当前帧数据
    float P[N_FFT / 2 + 1];       // 功率谱
    float *fft_buf;               // FFT 临时缓冲区（分配在 SPIRAM）
    int f, i, m, k;
    int offset;

    // // 每次都清零，避免旧数据残留  这样会报空指针
    // memset(fft_buf, 0, sizeof(float) * N_FFT * 2);
    // memset(P, 0, sizeof(float) * (N_FFT / 2 + 1));

    // --- 初始化（仅第一次调用执行） ---
    if (!dsp_inited) {
        dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);  // 初始化 DSP 库 必须要初始化
        make_hann(hann_window);
        build_mel_filters(mel_filter);
        dsp_inited = true;
    }

    // --- 分配 FFT 缓冲区（SPIRAM）---
    fft_buf = (float*) heap_caps_aligned_calloc(16, N_FFT * 2, sizeof(float),
                                                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!fft_buf) {
        // 分配失败：此处仅返回，不处理错误
        return;
    }

    // --- 处理每一帧 ---
    for (f = 0; f < NUM_FRAMES; f++) {
        offset = f * FRAME_SHIFT;

        // 1. 提取一帧并加汉宁窗
        for (i = 0; i < N_FFT; i++) {
            frame[i] = (float)input[offset + i] * hann_window[i];
        }

        // 2. 计算该帧的功率谱
        rfft_power(frame, P, fft_buf);

        // 3. 应用梅尔滤波器组
        for (m = 0; m < N_MEL_BINS; m++) {
            float mel_energy = 0.0f;

            // 对每个FFT频点加权求和
            for (k = 0; k < N_FFT / 2 + 1; k++) {
                mel_energy += mel_filter[m * (N_FFT / 2 + 1) + k] * P[k];
            }

            // 4. 取对数：Log-Mel 能量（防止 log(0)）
            // output[f * N_MEL_BINS + m] = 10.0f * log10f(mel_energy + 1e-10f);
            output[f * N_MEL_BINS + m] = 10.0f * log10f(fmaxf(mel_energy, 1e-8f));  // 保证不会出现 log(0), 避免完全静音导致 -inf
        }
    }

    // --- 释放临时缓冲区 ---
    heap_caps_free(fft_buf);
}