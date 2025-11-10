#include "fft_power.h"
#include "esp_dsp.h"
#include "Arduino.h"
// #include "dsps_fft2r_platform.h"  // 位置逆序调整使用dsp硬件加速

// TODO：位置逆序调整使用dsp硬件加速， 使用预生成 bitrev 表， 加速

// ================================
// 模块内部静态资源， 只对本文件可见
// ================================
static bool fft_initialized = false;  // 是都完成旋转因子表的初始化
static int fft_size = 0;            // FFT长度

// 全局静态指针，函数调用结束，并不会释放内存，数据还在，  内部RAM缓存，用于FFT输入、临时数据、复数输出
static float *fft_input = NULL;   // 输入缓冲区  
static float *twiddles = NULL;     // 旋转因子表 
// static uint16_t *bitrev_table = NULL;  // 位逆序查找表 1kb


// 注意：输入和输出在外部提供（PSRAM），这里只负责内部计算
// Twiddle 因子表不需要显式释放，由 ESP-DSP 内部管理


// ================================
// 初始化fft旋转因子和位置逆序表
// ================================
void fft_power_init(int nfft)
{
    // 如果已经初始化且相同大小，直接返回
    if (fft_initialized && fft_size == nfft) {
        Serial.println("fft_power_init: 已初始化，无需重复");
        return;
    }

    fft_size = nfft;

    // 分配内部 RAM 缓冲（速度快）
    fft_input = (float *)heap_caps_malloc(sizeof(float) * fft_size * 2, MALLOC_CAP_INTERNAL);  // 复数数组：实部/虚部交替
    twiddles   = (float *)heap_caps_malloc(sizeof(float) * fft_size, MALLOC_CAP_INTERNAL);  // 旋转因子表

    if (!fft_input || !twiddles) {
        Serial.println("fft_power_init: 内部RAM分配失败！");
        return;
    }

    // 初始化 ESP-DSP FFT 表
    dsps_fft2r_init_fc32(twiddles, fft_size);  // 只需要fft_size大小的size

    // 生成位逆序查找表 (只调用1次)
    // dsps_gen_bitrev2r_table(nfft, 1, bitrev_table);  // nfft = FFT点数 (512点也是占用1kb字节)


    fft_initialized = true;

    Serial.printf("fft_power_init: 初始化完成, FFT大小 = %d\n", nfft);
}



// ==========================
// 功率谱计算（多帧）
// 完全对齐 scipy.fftpack.fft
// ==========================
int fft_power_compute(const float *frames_in, int num_frames, int frame_size, int nfft, float *power_out)
{
    if (!fft_initialized) {
        Serial.println("fft_power_compute: FFT未初始化！");
        return -1;
    }

    if (nfft != fft_size) {
        Serial.println("fft_power_compute: nfft 与初始化不一致！");
        return -2;
    }

    int num_bins = nfft / 2 + 1;  // 功率谱点数， 只需要前半部分


    // 循环多帧
    for (int f = 0; f < num_frames; f++) {

        // 拿到当前帧指针
        const float *frame = frames_in + f * frame_size;
        float *out_frame = power_out + f * num_bins;

        // === Step 1: 对当前帧拷贝帧数据并补零  循环nfft次 ===
        for (int i = 0; i < nfft; i++) {
            fft_input[2 * i]     = (i < frame_size) ? frame[i] : 0.0f; // 实部
            fft_input[2 * i + 1] = 0.0f;                                // 虚部
        }

        // 注意： nfft， 短时傅里叶变换的长度， 一定是我们设置的长度，不能取输出的功率长度， 即使输入只有前 400 个有效样本、后 112 个补零。
        // === Step 2: 执行 FFT ===
        dsps_fft2r_fc32(fft_input, nfft);    // 实数FFT 自动使用加速版	
        dsps_bit_rev_fc32(fft_input, nfft);  // 位逆序调整
        // dsps_bit_rev_lookup_fc32(fft_input, nfft, (uint16_t*)dsps_bitrev_table);  // 位逆序 (加速)


        // === Step 3: 对齐 scipy.fftpack.fft 输出格式 （前 N/2+1 个 bin） ===
        // scipy.fftpack.fft(x, NFFT)[:NFFT//2 + 1]  
        for (int k = 0; k < num_bins; k++) {
            float re = fft_input[2 * k];    // 实部
            float im = fft_input[2 * k + 1];  // 虚部
            float mag = re * re + im * im;
            out_frame[k] = mag;  // 功率谱值
        }
        // 这一帧处理完成... 下一帧开始
    }

    Serial.println("fft_power_compute: 功率谱计算完成");
    return 0;
}





// ====== 释放资源 ======
void fft_power_free(void)
{
    if (!fft_initialized) return;

    if (fft_input) {
        heap_caps_free(fft_input);
        fft_input = NULL;
    }
    if (twiddles) {
        heap_caps_free(twiddles);
        twiddles = NULL;
    }

    fft_initialized = false;
    Serial.println("fft_power_free: 已释放资源");
}

