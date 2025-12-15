#include "learning_core.h"
#include "flash_storage.h"
#include <string.h>
#include <math.h>
#include "Arduino.h"


/* ============ 模版结构 需要持久化在Flash中 =======*/
typedef struct {
    float centroid[EMBED_DIM];
    int   batch_count;       // 已融合 batch 数
    bool  frozen;           // 是否冻结，达到上限不再更新
} TemplateModel;

// 当前模板（唯一实例）
static TemplateModel g_template;



/* ================== 工具函数 ================== */

static float dot(const float* a, const float* b) {
    float s = 0.f;
    for (int i = 0; i < EMBED_DIM; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

static float norm(const float* v) {
    return sqrtf(dot(v, v) + 1e-8f);  // 防止除零
}

// 计算余弦相似度
static float cosine_sim(const float* a, const float* b) {
    return dot(a, b) / (norm(a) * norm(b));
}


// 计算 EMA 衰减率
static float calc_ema_alpha(int batch_count)
{
    float a = EMA_ALPHA_START - batch_count * EMA_DECAY_STEP;
    if (a < EMA_ALPHA_MIN) {
        a = EMA_ALPHA_MIN;
    }
    return a;
}

// 冷启动阈值
static float calc_dynamic_sim_th(int batch_count)
{
    const float SIM_START = 0.4f;
    const float SIM_STEP  = 0.05f;
    const float SIM_MAX   = 0.6f;

    float th = SIM_START + batch_count * SIM_STEP;
    if (th > SIM_MAX) {
        th = SIM_MAX;
    }
    return th;
}



// batch内部聚类
static bool batch_cluster_core(float embeds[EMBED_BATCH_SIZE][EMBED_DIM],
                               uint32_t batch_size,
                               float out_centroid[EMBED_DIM])
    /*
    通过构造 batch 特征空间的中心向量，
    利用其作为一致性代理，
    以降低两两相似度计算的复杂度。”
    */
{
    float mean[EMBED_DIM] = {0};

    /* ---------- 1. 计算 batch 均值 ---------- */
    for (uint32_t i = 0; i < batch_size; i++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            mean[d] += embeds[i][d];
        }
    }
    for (int d = 0; d < EMBED_DIM; d++) {
        mean[d] /= batch_size;
    }

    /* ---------- 2. 每个 embedding 与 mean 的相似度 ---------- */
    float sim[EMBED_BATCH_SIZE];
    for (uint32_t i = 0; i < batch_size; i++) {
        sim[i] = cosine_sim(embeds[i], mean);
    }

    /* ---------- 3. 选 TopK ---------- */
    int topk_idx[EMBED_CORE_TOPK];
    for (int k = 0; k < EMBED_CORE_TOPK; k++) {
        float max_val = -1e9f;
        int max_idx = -1;

        for (uint32_t i = 0; i < batch_size; i++) {
            bool used = false;
            for (int t = 0; t < k; t++) {
                if (topk_idx[t] == i) {
                    used = true;
                    break;
                }
            }
            if (!used && sim[i] > max_val) {
                max_val = sim[i];
                max_idx = i;
            }
        }
        
        if (max_idx < 0) {
            return false;
        }
        topk_idx[k] = max_idx;
    }



    /* ---------- 4. TopK 阈值校验 ---------- */

    float th = calc_dynamic_sim_th(g_template.batch_count);
    for (int k = 0; k < EMBED_CORE_TOPK; k++) {
        if (sim[topk_idx[k]] < th) {
            return false;   // 聚类失败
        }
    }

    /* ---------- 5. 用 TopK 重新算 centroid ---------- */
    memset(out_centroid, 0, EMBED_DIM * sizeof(float));
    for (int k = 0; k < EMBED_CORE_TOPK; k++) {
        int idx = topk_idx[k];
        for (int d = 0; d < EMBED_DIM; d++) {
            out_centroid[d] += embeds[idx][d];
        }
    }
    for (int d = 0; d < EMBED_DIM; d++) {
        out_centroid[d] /= EMBED_CORE_TOPK;
    }

    return true;
}



/* ================== 初始化模版 ================== */
void learning_core_init(void)
{
    // 初始化内存的模版
    memset(&g_template, 0, sizeof(g_template));

    // 从 Flash 加载模版到内存结构体
    if (flash_storage_load_template(&g_template, sizeof(g_template))) {
        // 加载模版成功。 这里其实不需要设置，因为状态机已经设置了，但是还是可以设置一下以防万一
        flash_state.template_ready = true;
    }
}



/* ================== 对外接口 1：声纹验证 ================== */
float learning_core_calc_similarity(const float* embed)
{
    // 判断模版是否建立完毕
    if (!flash_state.template_ready || !embed) {
        return -2.0f;
    }

    // 计算与模版的相似度
    return cosine_sim(embed, g_template.centroid); 
}




/* ================== 难点： 对外接口 2：自学习接口 ================== */
LearnResult learning_core_try_learn(uint32_t start_idx,
                                    uint32_t batch_size)
{
    if (batch_size != EMBED_BATCH_SIZE) {
        return LEARN_FAIL;
    }

    // 已冻结
    if (g_template.frozen) {
        return LEARN_SKIP;
    }

    /* ---------- 1. 读取 batch embedding ---------- */
    float embeds[EMBED_BATCH_SIZE][EMBED_DIM];

    for (uint32_t i = 0; i < batch_size; i++) {
        if (!flash_read_embedding(start_idx + i, embeds[i])) {
            return LEARN_FAIL;
        }
    }

    /* ---------- 2. batch 内部聚类 ---------- */
    float batch_centroid[EMBED_DIM];

    if (!batch_cluster_core(embeds,
                            batch_size,
                            batch_centroid)) {
        return LEARN_FAIL;   // 聚类失败, 已经return了
    }

    /* ---------- 3. 与全局模版一致性 ---------- */
    if (flash_state.template_ready) {
        float sim =
            cosine_sim(batch_centroid,
                       g_template.centroid);

        // if (sim < TEMPLATE_UPDATE_SIM_TH) {
        //     return LEARN_FAIL;
        // }

        float th = calc_dynamic_sim_th(g_template.batch_count);

        Serial.printf(
            "[LEARN] batch=%d sim=%.3f th=%.3f\n",
            g_template.batch_count,
            sim,
            th
        );

        if (sim < th) {
            Serial.println("[LEARN] global template consistency check FAILED");
            return LEARN_FAIL;
        }

        Serial.printf(
            "[LEARN] batch centroid built, batch_count=%d\n",
            g_template.batch_count
        );
    }

    /* ---------- 4. 初始化 or EMA 更新 ---------- */
    if (!flash_state.template_ready) {
        // 第一次聚类，直接更新模版，不需要EMA
        memcpy(g_template.centroid,
               batch_centroid,
               sizeof(batch_centroid));

        g_template.batch_count = 1;
        flash_state.template_ready = true;

    } else {
        // 模版已经建立，需要EMA更新，防止漂移
        float alpha =
            calc_ema_alpha(g_template.batch_count);

        for (int d = 0; d < EMBED_DIM; d++) {
            g_template.centroid[d] =
                (1.0f - alpha) * g_template.centroid[d]
                + alpha * batch_centroid[d];
        }

        g_template.batch_count++;
    }

    /* ---------- 5. 冻结判断 ---------- */
    if (g_template.batch_count >= MAX_BATCH_NUM) {
        g_template.frozen = true;
        flash_storage_save_template(&g_template,
                                    sizeof(g_template));
        return LEARN_FROZEN;
    }

    /* ---------- 6. 持久化 ---------- */
    flash_storage_save_template(&g_template,
                                sizeof(g_template));

    return LEARN_SUCCESS;
}