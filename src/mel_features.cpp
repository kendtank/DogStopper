// #include "mel_features.h"
// #include "dsps_fft2r.h"
// #include "dsps_bit_rev.h"
// #include <math.h>

// // =======================
// // 全局变量定义
// // =======================
// float hann_window[MEL_N_FFT];
// float mel_basis[MEL_N_MELS * MEL_NUM_BINS];
// float fft_input[2 * MEL_N_FFT];
// float mel_out[MEL_N_MELS * MEL_MAX_FRAMES];



// void build_mel_filters(float W[N_MEL][NFFT_BINS]){
//     memset(W, 0, sizeof(float)*N_MEL*NFFT_BINS);
//     double fft_freqs[NFFT_BINS];
//     for(int k=0;k<NFFT_BINS;k++)
//         fft_freqs[k] = (double)SAMPLE_RATE * k / (double)FRAME_SIZE;

//     double m_min = hz_to_mel_slaney_d(0.0);
//     double m_max = hz_to_mel_slaney_d((double)SAMPLE_RATE * 0.5);
//     double mpts[N_MEL+2];
//     for(int i=0;i<N_MEL+2;i++)
//         mpts[i] = m_min + (m_max - m_min) * (double)i / (N_MEL+1);

//     double fpts[N_MEL+2];
//     for(int i=0;i<N_MEL+2;i++)
//         fpts[i] = mel_to_hz_slaney_d(mpts[i]);

//     for(int m=0;m<N_MEL;m++){
//         double f0=fpts[m], f1=fpts[m+1], f2=fpts[m+2];
//         if(f1<=f0) f1=f0+1e-12;
//         if(f2<=f1) f2=f1+1e-12;
//         for(int k=0;k<NFFT_BINS;k++){
//             double fk = fft_freqs[k];
//             float w = 0.0f;
//             if(fk > f0 && fk < f1) w = (float)((fk - f0)/(f1 - f0));
//             else if(fk >= f1 && fk < f2) w = (float)((f2 - fk)/(f2 - f1));
//             if(w < 0.0f) w = 0.0f;
//             W[m][k] = w;
//         }
//         float fe = (float)(2.0 / (f2 - f0));
//         for(int k=0;k<NFFT_BINS;k++) W[m][k] *= fe;
//     }
// }


// // =======================
// // 示例 Hann窗初始化
// // =======================
// void mel_init() {
//     // 简单Hann窗示例
//     for (int i = 0; i < MEL_N_FFT; i++) {
//         hann_window[i] = 0.5f - 0.5f * cosf(2.0f * PI * i / (MEL_N_FFT - 1));
//     }

//     // Mel矩阵需要用Python生成并替换，示例全1
//     for (int i = 0; i < MEL_N_MELS * MEL_NUM_BINS; i++) {
//         mel_basis[i] = 1.0f / MEL_NUM_BINS; // 占位，可用Python生成
//     }

//     Serial.println("mel_init: 完成 Hann 窗和 Mel 矩阵初始化");
// }

// // =======================
// // 功率谱 → Mel → Log
// // frames: (num_frames x frame_size) 按帧连续存储
// // mel_out: (num_frames x n_mels) 按帧连续存储
// // =======================
// int mel_compute(const float *frames, int num_frames, int frame_size, float *mel_out_buf) {
//     if (frame_size != MEL_N_FFT) {
//         Serial.println("mel_compute: frame_size 与 n_fft 不一致！");
//         return -1;
//     }

//     for (int f = 0; f < num_frames; f++) {
//         const float *frame = frames + f * frame_size;
//         float *out_frame = mel_out_buf + f * MEL_N_MELS;

//         // === Step 1: Hann窗 + 补零到 n_fft ===
//         for (int i = 0; i < MEL_N_FFT; i++) {
//             fft_input[2*i]   = (i < frame_size) ? frame[i] * hann_window[i] : 0.0f; // 实部
//             fft_input[2*i+1] = 0.0f; // 虚部
//         }

//         // === Step 2: FFT ===
//         dsps_fft2r_fc32(fft_input, MEL_N_FFT);
//         dsps_bit_rev_fc32(fft_input, MEL_N_FFT);

//         // === Step 3: 功率谱 ===
//         float power[MEL_NUM_BINS];
//         for (int k = 0; k < MEL_NUM_BINS; k++) {
//             float re = fft_input[2*k];
//             float im = fft_input[2*k + 1];
//             power[k] = re*re + im*im;
//         }

//         // === Step 4: Mel滤波器矩阵乘法 ===
//         for (int m = 0; m < MEL_N_MELS; m++) {
//             float sum = 0.0f;
//             for (int k = 0; k < MEL_NUM_BINS; k++) {
//                 sum += mel_basis[m * MEL_NUM_BINS + k] * power[k];
//             }
//             // === Step 5: Log，防止 log(0) ===
//             out_frame[m] = logf(sum + 1e-10f);
//         }
//     }

//     return 0;
// }
