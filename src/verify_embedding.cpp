#include "verify_embedding.h"
#include "tiny_model.h"   // embedding推理
#include "audio_features.h"  // 提取logmel的200ms窗的函数
#include "learning_core.h"   // 自学习/模版管理模块
#include "flash_storage.h"  // flash存储模块
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include <Arduino.h>
#include <esp_log.h>
#include "esp_heap_caps.h"

/*
本系统将 Flash 中的 embedding 视为不可变的原始观测数据。
自学习模块仅在 batch 级别对数据进行一致性筛选，
学习失败不会回滚或删除历史 embedding，
而是通过时间滑窗自然淘汰异常样本，
以保证模板更新的稳定性与可解释性。
*/



static const char* EMBEDTAG = "EMBED_MODULE";


// 存放embed推理的结果，用于验证和自学习模块
static float logmel_mebedding[EMBED_OUTPUT_SIZE];

static float logmel_feature[LOGMEL_SIZE];  // 提取出来的logmel特征， 720点浮点数 约2.8KB



// 初始化模块
bool verify_embedding_init() {
    // 初始化 logmel/embedding 模型  必须还要初始化init_feature_buffers， 因为logmel提取需要buffer
    if (logmel_model_init() != 0) {
        Serial.println("[verify_embedding] logmel model init failed!");
        return false;
    }
    // 初始化自学习和flash持久化模块

    // 先上电状态机
    if (!flash_storage_init()) {
        Serial.println("[verify_embedding] flash storage init failed!");
        return false;
    }
    // 再上电模版
    learning_core_init();

    // 擦除flash的namespace
    reset_storage();

    return true;
}



// 推理 pcm数据 -> embedding
bool tinyml_embedding_inference(int16_t* pcm_samples, int16_t len, float* embed_out) {
    if (!pcm_samples || len != INPUT_SAMPLES || !embed_out) {
        Serial.println("[verify_embedding] invalid input to embedding inference");
        return false;
    }
    // 1. 提取 logmel 特征
    int frames = compute_logmel_200ms(pcm_samples, logmel_feature, 1);  // 1表示logmel提取使用
    if (frames != NUM_FRAMES) {
        Serial.println("[verify_embedding] logmel compute failed");
        return false;
    }

    // 2. embedding 推理
    int re = embed_model_infer(logmel_feature, embed_out);
    if (re != 0) {
        Serial.println("[verify_embedding] embedding inference failed");
        return false;
    }
    return true;
}


static const char* learn_result_str(LearnResult r)
{
    switch (r) {
    case LEARN_SKIP:    return "SKIP";
    case LEARN_FAIL:    return "FAIL";
    case LEARN_SUCCESS: return "SUCCESS";
    case LEARN_FROZEN:  return "FROZEN";
    default:            return "UNKNOWN";
    }
}



// -------------------- 主处理函数 --------------------
void verify_embedding_process(BarkEvent *event)
{
    if (!event) {
        return;
    }

    /* ==================================================
     * 1. embedding 推理
     * ================================================== */
    if (!tinyml_embedding_inference(
            event->samples,
            event->length,
            logmel_mebedding)) {
        return;
    }


    // 打印延迟
    // Serial.printf("[verify_embedding] delay = %d ms\n",
    //              millis() - event->timestamp_ms);

    /* ==================================================
     * 2. 学习期：embedding 落盘（原始事实）
     * ================================================== */
    if (!flash_state.close_learning || flash_state.total_embed_counter <= 100 ) {
        // Serial.println("embedding save to flash");
        if (flash_save_embedding(logmel_mebedding)) {
        flash_state.total_embed_counter++;
        flash_state.batch_embed_counter++;
        } else {
            Serial.println("flash save embedding failed");
        }

    }

    /* ==================================================
     * 3. batch 满 → 尝试自学习（只在学习期）
     * ================================================== */
    if (!flash_state.close_learning && flash_state.batch_embed_counter >= EMBED_BATCH_SIZE) {

        uint32_t start_index = flash_state.total_embed_counter - EMBED_BATCH_SIZE;
        // Serial.printf("开始一次自学习，start_index = %d", start_index);
        // batch内部自学习，自动聚类
        LearnResult r = learning_core_try_learn(start_index, EMBED_BATCH_SIZE);
        // 打印自学习结果，是结构体
        Serial.printf("learning result = %s (%d)",learn_result_str(r), r);
  

        // batch 生命周期结束，无论成功失败都清零
        flash_state.batch_embed_counter = 0;

        // switch (r) {

        // case LEARN_SUCCESS:
        //     ESP_LOGI(EMBEDTAG, "learning success");
        //     break;

        // case LEARN_FROZEN:
        //     ESP_LOGI(EMBEDTAG, "template frozen, close learning");
        //     flash_state.close_learning = true;
        //     break;

        // case LEARN_FAIL:
        //     ESP_LOGW(EMBEDTAG, "learning failed (batch ignored)");
        //     break;

        // case LEARN_SKIP:
        // default:
        //     break;
        // }
    }

    /* ==================================================
     * 4. 状态持久化（轻量，幂等）
     * ================================================== */
    flash_save_state();

    /* ==================================================
     * 5. 声纹验证（与学习完全解耦）
     * ================================================== */

    
    float sim = learning_core_calc_similarity(logmel_mebedding);
    Serial.printf("[verify_embedding] sim = %f\n", sim);

    // 如果是等于-2，则说明没有模板，则进行弱惩罚狗狗
    if (sim == -2) {
        Serial.println("没有模版，进行弱惩罚干预");
        // 弱惩罚
        return;
    }
    if (sim > 0.75) {
        Serial.println("匹配成功模版，进行狗吠惩罚");
        // 狗吠
        return;
    }
    Serial.printf("不是模版狗，不干预处理");

}
