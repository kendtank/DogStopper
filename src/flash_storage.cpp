#include "flash_storage.h"
#include <Preferences.h>
#include "Arduino.h"

/*
持久化 embedding（append）
持久化 template
持久化 state
不做聚类
不做学习
不知道“batch 是什么算法含义”
*/

// 声明 NVS 命名空间全局变量
Preferences nvs;

// 声明状态机全局变量
FlashState flash_state;


// 保存状态机
bool flash_save_state(void)
{
    return nvs.putBytes("state", &flash_state, sizeof(flash_state));
}


// 加载状态机或者初始化状态机
// ----------------- 初始化（程序上电必须执行） -----------------
bool flash_storage_init(void)
{
    // 打开一个 NVS 命名空间， flash_store 为命名空间名，false 表示可读写， 如果有这个空间就打开它，没有就创建一个新空间。
    // 空间里面里面可以存很多 key/value 对
    if (!nvs.begin("flash_store", false)) {
        return false;
    }

    size_t len = nvs.getBytes("state", &flash_state, sizeof(flash_state));
    if (len != sizeof(flash_state)) {
        flash_state.total_embed_counter = 0;
        flash_state.batch_embed_counter = 0;
        flash_state.template_ready = false;
        flash_state.close_learning = false;
        flash_save_state();
    }
    return true;
}



// 关闭 NVS空间， 一般不关闭。
void flash_storage_close() {
    // 关闭空间, 关闭不等于清理数据，数据还在
    nvs.end();
}


/* ============ embedding 日志 ============ */
bool flash_save_embedding(const float* embed)
{
    char key[16];
    snprintf(key, sizeof(key), "e%u", flash_state.total_embed_counter);

    if (!nvs.putBytes(key, embed,
                      EMBED_OUTPUT_SIZE * sizeof(float))) {
        return false;
    }
    return true;
}

bool flash_read_embedding(uint32_t index, float* out_embed)
{
    char key[16];
    snprintf(key, sizeof(key), "e%u", index);

    return nvs.getBytes(key, out_embed,
                        EMBED_OUTPUT_SIZE * sizeof(float))
           == EMBED_OUTPUT_SIZE * sizeof(float);
}



/* ============ template ============ */
bool flash_storage_save_template(const void* data, size_t size)
{
    return nvs.putBytes("template", data, size);
}

bool flash_storage_load_template(void* data, size_t size)
{
    return nvs.getBytes("template", data, size) == size;
}


// 重置状态机， 模版， 还需要删除所有 embedding， 作为测试使用
bool reset_storage(void)
{
    Serial.printf("FLASH", "RESET STORAGE: erase all embeddings, template and state");

    // 1. 关闭当前 NVS（如果已打开）
    flash_storage_close();

    // 2. 重新打开 namespace
    if (!nvs.begin("flash_store", false)) {
        Serial.printf("FLASH", "reset_storage: nvs begin failed");
        return false;
    }

    // 3. 擦除整个 namespace
    esp_err_t err = nvs.clear();  // 直接擦除 namespace所有的key
    if (err != ESP_OK) {
        Serial.printf("FLASH", "reset_storage: nvs clear failed (%d)", err);
        return false;
    }

    // 4. 重置 RAM 中的状态机
    memset(&flash_state, 0, sizeof(flash_state));
    flash_state.total_embed_counter = 0;
    flash_state.batch_embed_counter = 0;
    flash_state.template_ready = false;
    flash_state.close_learning = false;

    // 5. 持久化一个“干净状态”
    if (!flash_save_state()) {
        Serial.printf("FLASH", "reset_storage: save state failed");
        return false;
    }

    Serial.printf("FLASH", "RESET STORAGE DONE");
    return true;
}



    