#include <Arduino.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "dsps_fft2r.h"
#include "logmel_extractor.h"

// FFT初始化标志
static bool fft_initialized = false;

// -------------------- 工具函数 --------------------
static void make_hann(float *w){
    for(int n=0;n<NFFT;n++) w[n] = 0.5f - 0.5f * cosf(2.0f*M_PI*n/NFFT);
}

static int reflect_pad(const float *y, int L, int pad, float *out) {
    if (L <= 1 || pad <= 0) {
        memcpy(out, y, L * sizeof(float));
        return L;
    }
    int T = L + 2 * pad;
    // 左填充：y[pad-1], y[pad-2], ..., y[0]
    for (int i = 0; i < pad; i++) {
        out[i] = y[pad - 1 - i];  // 从 pad-1 开始
    }
    memcpy(out + pad, y, L * sizeof(float));
    // 右填充：y[L-1], y[L-2], ..., y[0]
    for (int i = 0; i < pad; i++) {
        out[pad + L + i] = y[L - 1 - i];  // 从 L-1 开始
    }
    return T;
}

static double hz_to_mel(double f){
    const double f_min=0.0, f_sp=200.0/3.0, f_break=1000.0, logstep=pow(6.4,1.0/27.0);
    if(f<f_break) return (f-f_min)/f_sp;
    return (f_break-f_min)/f_sp + log(f/f_break)/log(logstep);
}

static double mel_to_hz(double m){
    const double f_min=0.0, f_sp=200.0/3.0, f_break=1000.0, logstep=pow(6.4,1.0/27.0);
    const double min_log_mel=(f_break-f_min)/f_sp;
    if(m<min_log_mel) return f_min + f_sp*m;
    return f_break*exp(log(logstep)*(m-min_log_mel));
}

// 构建 Mel 滤波器组
static void build_mel_filters(float *W){
    memset(W,0,sizeof(float)*N_MEL*NFFT_BINS);
    double fft_freqs[NFFT_BINS];
    for(int k=0;k<NFFT_BINS;k++) fft_freqs[k]=(double)SR*k/NFFT;

    double m_min = hz_to_mel(0.0);
    double m_max = hz_to_mel(SR*0.5);
    double mpts[N_MEL+2];
    for(int i=0;i<N_MEL+2;i++) mpts[i]=m_min + (m_max-m_min)*(i/(double)(N_MEL+1));
    double fpts[N_MEL+2];
    for(int i=0;i<N_MEL+2;i++) fpts[i]=mel_to_hz(mpts[i]);

    for(int m=0;m<N_MEL;m++){
        double f0=fpts[m], f1=fpts[m+1], f2=fpts[m+2];
        if(f1<=f0) f1=f0+1e-12; if(f2<=f1) f2=f1+1e-12;
        for(int k=0;k<NFFT_BINS;k++){
            double fk=fft_freqs[k]; float w=0.0f;
            if(fk<=f0 || fk>=f2) w=0.0f;
            else if(fk<=f1) w=(float)((fk-f0)/(f1-f0));
            else w=(float)((f2-fk)/(f2-f1));
            if(w<0.0f) w=0.0f;
            W[m*NFFT_BINS + k]=w*(float)(2.0/(f2-f0));
        }
    }
}

// -------------------- FFT 使用dsps库（专门用于实数 FFT运算加速）--------------------
static void rfft_power(float *x, float *P) {
    // 初始化FFT库（只初始化一次）
    if (!fft_initialized) {
        esp_err_t ret = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
        if (ret != ESP_OK) {
            Serial.printf("Failed to initialize FFT library. Error = %i\n", ret);
            return;
        }
        fft_initialized = true;
    }

    // 使用PSRAM分配内存
    float* fft_buf = (float*) heap_caps_aligned_calloc(16, NFFT * 2, sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    if (!fft_buf) {
        Serial.println("rfft_power: heap_caps_aligned_calloc failed!");
        return;
    }

    // 额外检查指针是否有效（防止 malloc 返回 0x0）
    if ((uint32_t)fft_buf < 0x1000) {
        Serial.printf("rfft_power: fft_buf is suspiciously low address: 0x%08x\n", (uint32_t)fft_buf);
        heap_caps_free(fft_buf);
        return;
    }

    // 填充前先检查 x 是否有效
    if (!x) {
        Serial.println("rfft_power: input x is NULL!");
        heap_caps_free(fft_buf);
        return;
    }

    for(int i = 0; i < NFFT; i++) {
        fft_buf[2*i]     = x[i];
        fft_buf[2*i + 1] = 0.0f;
    }

    dsps_fft2r_fc32(fft_buf, NFFT);
    dsps_bit_rev_fc32(fft_buf, NFFT);
    dsps_cplx2reC_fc32(fft_buf, NFFT);

    for(int k = 0; k < NFFT_BINS; k++) {
        P[k] = (fft_buf[2*k] * fft_buf[2*k] + fft_buf[2*k + 1] * fft_buf[2*k + 1]) / NFFT;
    }

    // free
    heap_caps_free(fft_buf);
}

static void free_all_and_return(float* buf, float* melW, float* mel_buf, float* x, float* win, float* P) {
    heap_caps_free(buf);
    heap_caps_free(melW);
    heap_caps_free(mel_buf);
    heap_caps_free(x);
    heap_caps_free(win);
    heap_caps_free(P);
}

// -------------------- Log-Mel --------------------
int compute_logmel(const float *audio, int num_samples, float *logmel_out, int max_frames) {
    if (!audio || !logmel_out || num_samples < NFFT) {
        Serial.println("compute_logmel: Invalid input args (null or too short)");
        return -1;
    }

    int pad = NFFT / 2;
    int T = num_samples + 2 * pad;
    int frames = (T - NFFT) / HOP + 1;
    if (frames > max_frames) frames = max_frames;

    // 所有大数组分配到 PSRAM，避免栈溢出
    float *buf      = (float*) heap_caps_malloc(sizeof(float) * T, MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);
    if (!buf) { Serial.println("buf == NULL"); return -2; }

    float *melW     = (float*) heap_caps_malloc(sizeof(float) * N_MEL * NFFT_BINS, MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);
    if (!melW) { Serial.println("melW == NULL"); heap_caps_free(buf); return -2; }

    float *mel_buf  = (float*) heap_caps_malloc(sizeof(float) * frames * N_MEL, MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);
    if (!mel_buf) { Serial.println("mel_buf == NULL"); heap_caps_free(buf); heap_caps_free(melW); return -2; }

    float *x        = (float*) heap_caps_malloc(sizeof(float) * NFFT, MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);
    if (!x) { Serial.println("x == NULL"); free_all_and_return(buf, melW, mel_buf, nullptr, nullptr, nullptr); return -2; }

    float *win      = (float*) heap_caps_malloc(sizeof(float) * NFFT, MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);
    if (!win) { Serial.println("win == NULL"); free_all_and_return(buf, melW, mel_buf, x, nullptr, nullptr); return -2; }

    float *P        = (float*) heap_caps_malloc(sizeof(float) * NFFT_BINS, MALLOC_CAP_SPIRAM | MALLOC_CAP_32BIT);
    if (!P) { Serial.println("P == NULL"); free_all_and_return(buf, melW, mel_buf, x, win, nullptr); return -2; }

    // 检查所有指针是否分配成功（任何为 NULL 都失败）
    if (!buf || !melW || !mel_buf || !x || !win || !P) {
        Serial.println("compute_logmel: Failed to allocate memory! Possible causes:");
        if (!buf)     Serial.println("   → buf (padded audio) is NULL");
        if (!melW)    Serial.println("   → melW (Mel filter bank) is NULL");
        if (!mel_buf) Serial.println("   → mel_buf (temp Mel features) is NULL");
        if (!x)       Serial.println("   → x (windowed frame) is NULL");
        if (!win)     Serial.println("   → win (Hann window) is NULL");
        if (!P)       Serial.println("   → P (FFT power) is NULL");

        // 安全释放（只释放非 NULL 的）
        heap_caps_free(buf);
        heap_caps_free(melW);
        heap_caps_free(mel_buf);
        heap_caps_free(x);
        heap_caps_free(win);
        heap_caps_free(P);
        return -2;
    }

    // 执行计算
    reflect_pad(audio, num_samples, pad, buf);
    make_hann(win);
    build_mel_filters(melW);

    for (int f = 0; f < frames; f++) {
        const float *x0 = buf + f * HOP;
        for (int i = 0; i < NFFT; i++) {
            x[i] = x0[i] * win[i];  // 如果 x 或 win 为 NULL，这里会崩溃
        }
        rfft_power(x, P);  // 内部已检查 fft_buf
        float *mel = mel_buf + f * N_MEL;
        for (int m = 0; m < N_MEL; m++) {
            float s = 0.0f;
            for (int k = 0; k < NFFT_BINS; k++) {
                s += melW[m * NFFT_BINS + k] * P[k];  // 如果 melW 或 P 为 NULL，这里崩溃
            }
            mel[m] = s;
        }
    }

    // Power → dB
    const double amin = 1e-10;
    for (int f = 0; f < frames; f++) {
        for (int m = 0; m < N_MEL; m++) {
            double v = mel_buf[f * N_MEL + m];
            if (v < amin) v = amin;
            logmel_out[f * N_MEL + m] = (float)(10.0 * log10(v));
        }
    }

    // 释放所有内存
    heap_caps_free(buf);
    heap_caps_free(melW);
    heap_caps_free(mel_buf);
    heap_caps_free(x);
    heap_caps_free(win);
    heap_caps_free(P);

    return frames;
}