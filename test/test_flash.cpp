// #include <Arduino.h>
// #include <unity.h>
// #include <Preferences.h>

// // ---------------- 配置 ----------------
// #define TEST_ARRAY_LEN 32

// // ---------------- 全局变量 ----------------
// Preferences prefs;
// float test_array[TEST_ARRAY_LEN];
// float read_back[TEST_ARRAY_LEN];

// // ---------------- 测试函数 ----------------
// void test_flash_save_and_read(void) {
//     Serial.println("=== Flash Save/Read Test Start ===");

//     // 初始化数据
//     for (int i = 0; i < TEST_ARRAY_LEN; i++) {
//         test_array[i] = i * 1.1f;
//         read_back[i] = 0;
//     }

//     // 打开 Flash（NVS 命名空间 "mytest"）
//     if (!prefs.begin("mytest", false)) {
//         TEST_FAIL_MESSAGE("Failed to open NVS namespace");
//         return;
//     }

//     // 保存数组到 Flash
//     size_t written = prefs.putBytes("array32", test_array, sizeof(test_array));
//     TEST_ASSERT_EQUAL(sizeof(test_array), written);
//     Serial.println("Saved array to Flash");

//     // 读取回数组
//     size_t read_size = prefs.getBytes("array32", read_back, sizeof(read_back));
//     TEST_ASSERT_EQUAL(sizeof(read_back), read_size);
//     Serial.println("Read array back from Flash");

//     // 验证数据
//     for (int i = 0; i < TEST_ARRAY_LEN; i++) {
//         TEST_ASSERT_FLOAT_WITHIN(0.0001, test_array[i], read_back[i]);
//     }

//     Serial.println("Data verification passed");

//     // 关闭 Flash
//     prefs.end();
// }

// void setup() {
//     Serial.begin(115200);
//     delay(2000);

//     UNITY_BEGIN();
//     RUN_TEST(test_flash_save_and_read);
//     UNITY_END();
// }

// void loop() {
//     // Nothing to do
// }
