#include "fft_power.h"
#include "esp_dsp.h"
#include "Arduino.h"


// 注意：dsps_fft2r_init_fc32(NULL, max_size) 已经自动分配并准备好了 twiddle（旋转因子）表和 bit-reversal 表

// ================================
// 模块内部静态资源， 只对本文件可见
// ================================
static bool fft_initialized = false;  // 是都完成旋转因子表的初始化
// static int fft_size = 0;            // FFT长度


// 全局静态指针，函数调用结束，并不会释放内存，数据还在，  内部RAM缓存，用于FFT输入、临时数据、复数输出
static float *fft_input_mfcc = NULL;   // 输入缓冲区  mfcc计算使用
static float *fft_input_logmel = NULL;   // 输入缓冲区 logmel计算使用

// static float *twiddles = NULL;     // 旋转因子表 
// static uint16_t *bitrev_table = NULL;  // 位逆序查找表 1kb


// 注意：输入和输出在外部提供，这里只负责内部计算
// Twiddle 因子表不需要显式释放，由 ESP-DSP 内部管理


// ================================
// 初始化fft旋转因子和位置逆序表
// ================================
void fft_power_init(int nfft)
{
    // 如果已经初始化且相同大小，直接返回
    if (fft_initialized) {
        // Serial.println("fft_power_init: 已初始化，无需重复");
        return;
    }

    // 分配内部 RAM 缓冲（速度快）
    fft_input_mfcc = (float *)heap_caps_malloc(sizeof(float) * nfft * 2, MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);  // 复数数组：实部/虚部交替，使用32位对齐， 4字节。
    fft_input_logmel = (float *)heap_caps_malloc(sizeof(float) * nfft * 2, MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);
    // twiddles   = (float *)heap_caps_malloc(sizeof(float) * fft_size, MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);  // 旋转因子表
    // bitrev_table = (uint16_t *)heap_caps_malloc(fft_size * sizeof(uint16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_32BIT);  // 位逆序查找表
    if (!fft_input_mfcc || !fft_input_logmel) {
        return;
    }

    // 初始化 ESP-DSP FFT 表
    // 注意：因为logmel计算和mfcc都是nftt相同， 所以只需要初始化一次即可， 因为函数内部本质就是只读的表。
    dsps_fft2r_init_fc32(NULL, nfft);  // 只需要fft_size大小的size
    fft_initialized = true;
}



// ==========================
// 功率谱计算（多帧）  优化前：7418us
// 完全对齐 scipy.fftpack.fft
// ==========================
int fft_power_compute(const float *frames_in, int num_frames, int frame_size, int nfft, int logmel_flag,  float *power_out)
{

    if (!fft_initialized) {
        // Serial.println("fft_power_compute: FFT未初始化！");
        return -1;
    }

    int num_bins = nfft / 2 + 1;  // 功率谱点数， 只需要前半部分


    // 循环多帧
    for (int f = 0; f < num_frames; f++) {

        // 拿到当前帧指针
        const float *frame = frames_in + f * frame_size;
        float *out_frame = power_out + f * num_bins;

        // === Step 1: 对当前帧拷贝帧数据并补零  循环nfft次 ===
        // for (int i = 0; i < nfft; i++) {
        //     fft_input[2 * i]     = (i < frame_size) ? frame[i] : 0.0f; // 实部
        //     fft_input[2 * i + 1] = 0.0f;                                // 虚部
        // }


        // 更新12-12：根据logmel_flag选择不同的输入缓冲区
        if (logmel_flag == 0) {
            // 优化：1204
            // === Step 1: memset直接把向量置为0，因为虚部是0，后面直接处理实部 ===
            memset(fft_input_mfcc, 0, sizeof(float) * 2 * nfft);
            // 把 frame 拷贝到偶数位（real），多余的补0
            for (int i = 0; i < frame_size; ++i) {
                fft_input_mfcc[2 * i] = frame[i]; // real
                // fft_input[2 * i + 1] already zeroed
            }
            // 注意： nfft， 短时傅里叶变换的长度， 一定是我们设置的长度，不能取输出的功率长度， 即使输入只有前 400 个有效样本、后 112 个补零。
            // === Step 2: 执行 FFT ===
            dsps_fft2r_fc32(fft_input_mfcc, nfft);    // 实数FFT 自动使用加速版	
            dsps_bit_rev_fc32(fft_input_mfcc, nfft);  // 位逆序调整   自动选择了加速版
            // === Step 3: 对齐 scipy.fftpack.fft 输出格式 （前 N/2+1 个 bin） ===
            // scipy.fftpack.fft(x, NFFT)[:NFFT//2 + 1]  
            for (int k = 0; k < num_bins; k++) {
                float re = fft_input_mfcc[2 * k];    // 实部
                float im = fft_input_mfcc[2 * k + 1];  // 虚部
                out_frame[k] = re * re + im * im; // 功率谱值
                // out_frame[k] = mag;  // 功率谱值
            }
        }
        // 计算logmel使用 默认只传0或者1
        else {
            memset(fft_input_logmel, 0, sizeof(float) * 2 * nfft);
            for (int i = 0; i < frame_size; ++i) {
                fft_input_logmel[2 * i] = frame[i]; 
            }
            dsps_fft2r_fc32(fft_input_logmel, nfft);
            dsps_bit_rev_fc32(fft_input_logmel, nfft);
            for (int k = 0; k < num_bins; k++) {
                float re = fft_input_logmel[2 * k];
                float im = fft_input_logmel[2 * k + 1];
                out_frame[k] = re * re + im * im;
            }
        }
        // 这一帧处理完成... 下一帧开始
    }

    // Serial.println("fft_power_compute: 功率谱计算完成");
    return 0;
}



// ====== 释放资源 ======
void fft_power_free(void)
{
    if (!fft_initialized) return;

    if (fft_input_logmel || fft_input_mfcc ) {
        heap_caps_free(fft_input_logmel);
        heap_caps_free(fft_input_mfcc);
        fft_input_logmel = NULL;
        fft_input_mfcc = NULL;
    }
    // if (twiddles) {
    //     heap_caps_free(twiddles);
    //     twiddles = NULL;
    // }

    fft_initialized = false;
    // Serial.println("fft_power_free: 已释放资源");
}

