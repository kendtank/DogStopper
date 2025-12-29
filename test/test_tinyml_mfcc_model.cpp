/*
*/

#include "unity.h"
#include "audio_features.h"
#include <string.h>
#include <math.h>
#include <Arduino.h>
#include "../python/tinyml/aug_3.h"
#include "../python/tinyml/aug_10.h"
#include "../python/tinyml/aug_20.h"
#include "tiny_model.h"



static float test_input[INPUT_SAMPLES];
static float test_output[LOGMEL_SIZE];
static float test_output_mfcc[MFCC_SIZE];



// 测试MFCC模型推理， 在py端同事推理查看结果，使用三段狗吠音频以及三段非狗吠音频

void test_mfcc_model_aug_3(void) { 

    bool is_init = init_feature_buffers();
    TEST_ASSERT_TRUE(is_init);

    // 直接使用python生成的数据头文件
    int result = compute_mfcc_200ms(aug_3, test_output_mfcc);
    TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);


    // 调用推理接口
    // 初始化模型
    if (mfcc_model_init() != 0) {
    Serial.println("MFCC model init failed!");
    while (1);
    }
    Serial.println("MFCC model init success!");

    // 推理
    float prob = mfcc_model_infer(test_output_mfcc);

    // 打印结果
    Serial.printf("Dog_AUG_3 Bark Probability: %.3f -> %s\n", prob, (prob > 0.7f) ? "BARK" : "NO BARK");

}

void test_mfcc_model_aug_10(void) { 

    bool is_init = init_feature_buffers();
    TEST_ASSERT_TRUE(is_init);

    // 直接使用python生成的数据头文件
    int result = compute_mfcc_200ms(aug_10, test_output_mfcc);
    TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);


    // 调用推理接口
    // 初始化模型
    if (mfcc_model_init() != 0) {
    Serial.println("MFCC model init failed!");
    while (1);
    }
    Serial.println("MFCC model init success!");

    // 推理
    float prob = mfcc_model_infer(test_output_mfcc);

    // 打印结果
    Serial.printf("Dog_AUG_10 Bark Probability: %.3f -> %s\n", prob, (prob > 0.7f) ? "BARK" : "NO BARK");

}

void test_mfcc_model_aug_20(void) { 

    bool is_init = init_feature_buffers();
    TEST_ASSERT_TRUE(is_init);

    // 直接使用python生成的数据头文件
    int result = compute_mfcc_200ms(aug_20, test_output_mfcc);
    TEST_ASSERT_EQUAL_INT(NUM_FRAMES, result);


    // 调用推理接口
    // 初始化模型
    if (mfcc_model_init() != 0) {
    Serial.println("MFCC model init failed!");
    while (1);
    }
    Serial.println("MFCC model init success!");

    // 推理
    float prob = mfcc_model_infer(test_output_mfcc);

    // 打印结果
    Serial.printf("Dog_AUG_20 Bark Probability: %.3f -> %s\n", prob, (prob > 0.7f) ? "BARK" : "NO BARK");

}


void setUp(void) {
}

void tearDown(void) {
}

int runUnityTests(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mfcc_model_aug_3);
    RUN_TEST(test_mfcc_model_aug_10);
    RUN_TEST(test_mfcc_model_aug_20);
    return UNITY_END();
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    runUnityTests();
}

void loop() {
    delay(1000);
}



/* 测试结果
MFCC model init success!
Dog_AUG_3 Bark Probability: 0.246 -> NO BARK
Dog_AUG_10 Bark Probability: 0.309 -> NO BARK
Dog_AUG_20 Bark Probability: 0.691 -> NO BARK
*/