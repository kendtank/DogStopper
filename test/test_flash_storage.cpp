// #include <Arduino.h>
// #include <unity.h>
// #include "flash_storage.h"

// /*
// ESP32 上电
// ↓
// 测试固件再次运行
// ↓
// test_flash_init_and_persist()  ← 第 2 次执行
// ↓
// print_flash_state("BOOT")      ← 看到的是“上一次写入后的值”
// ↓
// flash_save_state()             ← 又写了一次
// */

// void print_flash_state(const char* tag) {
//     Serial.printf(
//         "[%s] embed_counter=%d, total=%d, template_ready=%s, close_learning=%s\n",
//         tag,
//         flash_state.embed_counter,
//         flash_state.total_embed_counter,
//         flash_state.template_ready ? "YES" : "NO",
//         flash_state.close_learning ? "YES" : "NO"
//     );
// }

// void test_flash_init_and_persist(void) {
//     // 1. 初始化
//     TEST_ASSERT_TRUE(flash_storage_init());

//     // 2. 打印当前状态（可能是恢复的）
//     print_flash_state("BOOT");  // 应该是上次断电保存的11， 断电读取的状态

//     // // 3. 模拟一次自学习完成
//     flash_state.embed_counter += 1;
//     flash_state.total_embed_counter += 1;

//     // 4. 保存状态机
//     TEST_ASSERT_TRUE(flash_save_state());   // 这里保存后

//     print_flash_state("RUNING ADD + 1");    // 12

//     TEST_ASSERT_TRUE(flash_load_state());

//     // 再次加载一次，和上次一样
//     TEST_ASSERT_TRUE(flash_load_state());
//     print_flash_state("RUNING LOAD");

//     // 清零状态机
//     flash_storage_reset();
//     print_flash_state("RESET");
// }

// void setup() {
//     delay(2000);
//     Serial.begin(115200);
//     UNITY_BEGIN();

//     RUN_TEST(test_flash_init_and_persist);

//     UNITY_END();
// }

// void loop() {
//     delay(1000);
// }
