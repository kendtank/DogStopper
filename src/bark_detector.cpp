/**
 * 
 * 
 */

#include "bark_detector.h"
#include "tiny_model.h"   // 包含了 mfcc_model_init 和 mfcc_model_infer
#include "audio_features.h"  // 提取mfcc和logmel的200ms窗的函数
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include <Arduino.h>
#include <esp_log.h>
#include "esp_heap_caps.h"



static const char* MFCCTAG = "MFCC_MODULE";


// -------------------- 全局静态变量 --------------------
static QueueHandle_t g_bark_queue = nullptr;   // 外部定义，内部只接受句柄，绑定

// 滑窗缓存，长度 2*BARK_WIN_LEN
static int16_t g_mid_buffer[2 * BARK_WIN_LEN];   // 滑窗缓存,  用于在两个mid概率窗找出中心窗push
static size_t g_mid_len = 0;                    // 当前缓冲有效长度 
// 存放mfcc推理的结果，用于推理
static float test_output_mfcc[MFCC_SIZE];
// 声明推送事件的结构体
static BarkEvent bark_evt;   // 单线程处理，不会存在抢夺push
// 声明个用来处理队满的结构体，放在PSRAM
static BarkEvent* discard_evt = NULL;    // 6.3kb

// 全局 200ms 窗口缓冲，防止栈溢出（6.4KB）
static int16_t g_window[BARK_WIN_LEN];




// 内部函数声明
// 调用封装好的模型推理，对样本数据，做转float，推理mfcc特征，调整增益，输出狗吠概率
static float tinyml_bark_inference(int16_t* samples, int16_t len);

// 组装结构体，并推入队列（int16的原始样本数据）
static void push_bark_event(int16_t* samples, int16_t len, uint32_t timestamp_ms);



// -------------------- 初始化函数 --------------------
bool bark_detector_init(QueueHandle_t bark_queue) {
    if (!bark_queue) return false;
    // 申请psram，返回地址
    discard_evt = (BarkEvent*) heap_caps_malloc(sizeof(BarkEvent), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!discard_evt) {
        ESP_LOGE("APP", "Failed to allocate discard_evt in PSRAM!");
        return false;
    }
    // 清空bark_evt
    memset(&bark_evt, 0, sizeof(bark_evt));
    // 清空discard_evt
    memset(discard_evt, 0, sizeof(BarkEvent));
    // 绑定队列句柄
    g_bark_queue = bark_queue;
    //  缓冲区长度初始化
    g_mid_len = 0;
    //  缓冲区清零
    memset(g_mid_buffer, 0, sizeof(g_mid_buffer));
    memset(g_window, 0, sizeof(g_window));
    // 初始化特征提取函数
    if (!init_feature_buffers()){
        free_feature_buffers();
        return false;
    }
    // 初始化tinyml_mfcc模型
    if (mfcc_model_init() != 0){
        return false;
    }
    return true;
}


// -------------------- TinyML推理函数 输出窗的概率--------------------
static float tinyml_bark_inference(int16_t* samples, int16_t len) {

    if (!samples || !len  || (len != INPUT_SAMPLES)) return 0.0f;

    // 分别打印一下mfcc特征提取的时间开销和模型推理时间开销
    uint32_t  start_ms = millis();

    // 调用audio_feature的提取200msmfcc的函数(包含了数据转换和输出mfcc特征)
    compute_mfcc_200ms(samples, test_output_mfcc);

    uint32_t  mfcc_end_ms = millis();

    // Serial.printf("MFCC特征提取时间: %d ms\n", mfcc_end_ms - start_ms);   // 57 ms

    // 推理
    float prob = mfcc_model_infer(test_output_mfcc);

    uint32_t  tinyml_end_ms = millis();

    // Serial.printf("tinyml模型推理时间: %d ms\n", tinyml_end_ms - mfcc_end_ms);   // 15 ms

    // 增益补偿
    prob += GAIN_THRESHOLD;
    // 限幅， 避免超出
    if (prob < 0) prob = 0.0f;
    if (prob > 1) prob = 1.0f;
    return prob;
}


// -------------------- 输出 BarkEvent --------------------
static void push_bark_event(int16_t* samples, int16_t len, uint32_t timestamp_ms) {
    // 检查长度
    if (len != BARK_WIN_LEN || !g_bark_queue) return;

    // 拼接BarkEvent
    memcpy(bark_evt.samples, samples, BARK_WIN_LEN * sizeof(int16_t));
    bark_evt.length = BARK_WIN_LEN;
    bark_evt.timestamp_ms = timestamp_ms;

    // push BarkEvent到队列bark_queue
    BaseType_t ret = xQueueSend(g_bark_queue, &bark_evt, 0);   // 非阻塞发送， 直接拷贝发送，不使用指针
    if (ret != pdTRUE) {
        // 队列满了， 直接覆盖旧的数据
        // 栈上声明一个，用来接收丢弃的旧数据
        xQueueReceive(g_bark_queue, discard_evt, 0); // 丢弃最旧事件
        // 再次发送
        if (xQueueSend(g_bark_queue, &bark_evt, 0) != pdTRUE) {
            ESP_LOGE(MFCCTAG, "tinyml_queue still full after discard!"); // 再次发送失败，打印错误
        }
    }
}



// -------------------- 主处理函数 --------------------
void bark_detector_process_event(TinyMLEvent* event) {

    if (!event || !g_bark_queue) return;

    int16_t* pcm = event->samples;
    int total_len = event->samples_length;
    uint32_t ts = event->timestamp_ms;

    // ------- 1. 小于150ms：丢弃 -------
    if (total_len < BARK_LOW) { // <150ms @16kHz, 实际呢， VAD最短都是13 *  block  = 3328
        // 丢弃
        return;
    }


    // ------- 2. 150~200ms：平均填充到200ms -------
    else if (total_len < BARK_WIN_LEN)
    {
        int pad_len = BARK_WIN_LEN - total_len;
        int pad_left = pad_len / 2;
        int pad_right = pad_len - pad_left;

        memset(g_window, 0, pad_left * sizeof(int16_t));
        memcpy(g_window + pad_left, pcm, total_len * sizeof(int16_t));
        memset(g_window + pad_left + total_len, 0, pad_right * sizeof(int16_t));

        float prob = tinyml_bark_inference(g_window, BARK_WIN_LEN);
        // 打印概率
        Serial.printf("当前窗狗吠概率为： %.3f\n", prob);

        if (prob >= BARK_HIGH_THRESHOLD) {
            push_bark_event(g_window, BARK_WIN_LEN, ts);
        }

        return;
    }

    // ------- 3. 大于等于200ms：滑窗推理 -------
    else {
        int start = 0;   // 滑窗起始位置

        while (start < total_len) {

            int remain = total_len - start;  // 剩余长度
            int win_len = fmin(remain, (int)BARK_WIN_LEN);

            // 清空全局缓冲
            memset(g_window, 0, BARK_WIN_LEN * sizeof(int16_t));

            // 尾部窗， 尾窗长度小于200ms， 前填充(和训练保持一致)
            if (win_len < BARK_WIN_LEN) {
                // 尾窗前填充真实音频
                int pad_len = BARK_WIN_LEN - win_len;  // 需要填充的长度
                int copy_from = start - pad_len;     // 计算填充的起始位置
                if (copy_from < 0) copy_from = 0; // 边界保护， 正常不可能发生
                int pad_actual = start - copy_from; // 从填充位置开始，实际需要填充的长度
                memcpy(g_window, pcm + copy_from, pad_actual * sizeof(int16_t));   // 前填充的真实音频
                memcpy(g_window + pad_actual, pcm + start, win_len * sizeof(int16_t));  // 追加当前窗
            } 
            // 正常窗
            else {
                memcpy(g_window, pcm + start, BARK_WIN_LEN * sizeof(int16_t));
            }
            // 推理
            float prob = tinyml_bark_inference(g_window, BARK_WIN_LEN);
            Serial.printf("当前窗狗吠概率为： %.3f\n", prob);

            // 判断这个窗的概率事件
            if (prob >= BARK_HIGH_THRESHOLD) {
                // 高概率 → 直接输出
                push_bark_event(g_window, BARK_WIN_LEN, ts);
                g_mid_len = 0; // 清空中概率缓冲
            } 
            // 中概率窗
            else if (prob >= BARK_LOW_THRESHOLD) {
                // 中概率窗 → 累积到缓冲
                memcpy(g_mid_buffer + g_mid_len, g_window, BARK_WIN_LEN * sizeof(int16_t));
                g_mid_len += BARK_WIN_LEN;

                // 如果中概率缓冲 ≥ 2 窗 → 输出中心窗
                if (g_mid_len >= 2 * BARK_WIN_LEN) {
                    // 暂时使用中心窗，// TODO:后续需要使用能量峰值检测
                    size_t center_offset = (g_mid_len - BARK_WIN_LEN) / 2;

                    push_bark_event(g_mid_buffer + center_offset, BARK_WIN_LEN, ts);

                    // 只保留后一半继续累积
                    memmove(g_mid_buffer, g_mid_buffer + BARK_WIN_LEN, BARK_WIN_LEN * sizeof(int16_t));
                    g_mid_len = BARK_WIN_LEN;
                }
            } 
            else {
                // 低概率 → 丢弃事件 + 并清空缓冲
                g_mid_len = 0;
            }

            start += BARK_STRIDE;
        }
    }
}