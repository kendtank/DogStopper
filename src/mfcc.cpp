#include "mfcc.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ==================================================
// 工具函数（mel 相关）
// ==================================================
float hz_to_mel(float f){ return 1127.0f * logf(1.0f + f/700.0f); }
float mel_to_hz(float m){ return 700.0f * (expf(m/1127.0f) - 1.0f); }

// Slaney mel scale（htk=False）
static double hz_to_mel_slaney_d(double f){
    const double f_sp = 200.0/3.0;
    const double f_break = 1000.0;
    const double logstep = pow(6.4, 1.0/27.0);
    if (f < f_break) return f / f_sp;
    return (f_break / f_sp) + log(f / f_break) / log(logstep);
}

static double mel_to_hz_slaney_d(double m){
    const double f_sp = 200.0/3.0;
    const double f_break = 1000.0;
    const double logstep = pow(6.4, 1.0/27.0);
    const double min_log_mel = f_break / f_sp;
    if (m < min_log_mel) return f_sp * m;
    return f_break * exp(log(logstep) * (m - min_log_mel));
}

// ==================================================
// 核心计算组件
// ==================================================
void make_hann(float *w){
    for(int n=0;n<FRAME_SIZE;n++)
        w[n] = 0.5f - 0.5f * cosf(2.0f * M_PI * n / FRAME_SIZE);
}

void rfft_power_400(const float *x, float *P){
    for(int k=0;k<NFFT_BINS;k++){
        float re=0.0f, im=0.0f;
        float ang = -2.0f * (float)M_PI * k / FRAME_SIZE;
        for(int n=0;n<FRAME_SIZE;n++){
            float a = ang * n;
            re += x[n] * cosf(a);
            im += x[n] * sinf(a);
        }
        P[k] = re*re + im*im;
    }
}

void build_mel_filters(float W[N_MEL][NFFT_BINS]){
    memset(W, 0, sizeof(float)*N_MEL*NFFT_BINS);
    double fft_freqs[NFFT_BINS];
    for(int k=0;k<NFFT_BINS;k++)
        fft_freqs[k] = (double)SAMPLE_RATE * k / (double)FRAME_SIZE;

    double m_min = hz_to_mel_slaney_d(0.0);
    double m_max = hz_to_mel_slaney_d((double)SAMPLE_RATE * 0.5);
    double mpts[N_MEL+2];
    for(int i=0;i<N_MEL+2;i++)
        mpts[i] = m_min + (m_max - m_min) * (double)i / (N_MEL+1);

    double fpts[N_MEL+2];
    for(int i=0;i<N_MEL+2;i++)
        fpts[i] = mel_to_hz_slaney_d(mpts[i]);

    for(int m=0;m<N_MEL;m++){
        double f0=fpts[m], f1=fpts[m+1], f2=fpts[m+2];
        if(f1<=f0) f1=f0+1e-12;
        if(f2<=f1) f2=f1+1e-12;
        for(int k=0;k<NFFT_BINS;k++){
            double fk = fft_freqs[k];
            float w = 0.0f;
            if(fk > f0 && fk < f1) w = (float)((fk - f0)/(f1 - f0));
            else if(fk >= f1 && fk < f2) w = (float)((f2 - fk)/(f2 - f1));
            if(w < 0.0f) w = 0.0f;
            W[m][k] = w;
        }
        float fe = (float)(2.0 / (f2 - f0));
        for(int k=0;k<NFFT_BINS;k++) W[m][k] *= fe;
    }
}

void dct_ortho_1x_double(const double *x, float *c){
    const double s0 = sqrt(1.0 / (double)N_MEL);
    const double sk = sqrt(2.0 / (double)N_MEL);
    for(int k=0;k<N_MFCC;k++){
        double sum=0.0;
        for(int n=0;n<N_MEL;n++)
            sum += x[n]*cos(M_PI*((n+0.5)*k)/(double)N_MEL);
        double v = (k==0? s0: sk)*sum;
        c[k] = (float)v;
    }
}

// ==================================================
// 主函数 compute_mfcc()
// ==================================================
int compute_mfcc(const float *audio, int num_samples, float *mfcc_out, int max_frames){
    if(!audio || !mfcc_out || num_samples < FRAME_SIZE) return -1;

    int frames = (num_samples - FRAME_SIZE) / FRAME_SHIFT + 1;
    if(frames > max_frames) frames = max_frames;

    float win[FRAME_SIZE];
    make_hann(win);

    float melW[N_MEL][NFFT_BINS];
    build_mel_filters(melW);

    float *mel_buf = (float*)malloc(sizeof(float)*frames*N_MEL);
    if(!mel_buf) return -2;

    // 逐帧：FFT + Mel
    for(int f=0; f<frames; f++){
        const float *x0 = audio + f*FRAME_SHIFT;
        float x[FRAME_SIZE];
        for(int i=0;i<FRAME_SIZE;i++) x[i] = x0[i]*win[i];
        float P[NFFT_BINS];
        rfft_power_400(x, P);
        float *mel = mel_buf + f*N_MEL;
        for(int m=0;m<N_MEL;m++){
            float s=0.0f;
            for(int k=0;k<NFFT_BINS;k++) s += melW[m][k]*P[k];
            mel[m]=s;
        }
    }

    // power_to_db + DCT
    const double amin=1e-10;
    for(int f=0; f<frames; f++){
        double mel_db[N_MEL];
        for(int m=0;m<N_MEL;m++){
            double v = (double)mel_buf[f*N_MEL+m];
            if(v<amin) v = amin;
            mel_db[m] = 10.0 * log10(v);
        }
        dct_ortho_1x_double(mel_db, mfcc_out + f*N_MFCC);
    }

    free(mel_buf);
    return frames;
}
