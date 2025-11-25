# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/17 18:28
@Author  : Kend
@FileName: mfcc_model_int8_infer
@Software: PyCharm
@modifier:
"""



"""
和mcu端测试六组音频片段，并直接打印对应的狗吠概率也就是模型的输出。 对比测试结果
"""

import numpy as np
import tensorflow as tf
import importlib.util
import os
from tinyml.audio_features import compute_mfcc
from tinyml.dataset.norm_mfcc import norm_mfcc



# ==============================
# 配置
# ==============================
# 头文件音频数据，每个 h 文件定义一个 int16 数组名为 aug_x
H_FILES = [
    "tinyml/aug_3.h",
    "tinyml/aug_10.h",
    "tinyml/aug_20.h"
]

MODEL_INT8_PATH = "tinyml/train/quantized_model_mfcc_int8.tflite"
MODEL_FLOAT_PATH = "tinyml/train/float_model_mfcc.tflite"

# INT8 量化参数
INPUT_SCALE, INPUT_ZP = 0.00784314, -1
OUTPUT_SCALE, OUTPUT_ZP = 0.00390625, -128



# ==============================
# 工具函数
# ==============================
def load_h_array(h_file):
    """
    读取 MCU 风格 .h 文件数组，返回 np.int16 数组
    """
    with open(h_file, 'r') as f:
        content = f.read()
    # 找到大括号里的数字
    import re
    nums = re.findall(r'-?\d+', content)
    arr = np.array(nums, dtype=np.int16)
    return arr


def extract_mfcc_from_int16(int16_array):
    """
    MCU 风格：输入 int16，转换到 float32
    """
    float_audio = int16_array.astype(np.float32) / 32768.0
    # 假设 compute_mfcc 已经存在
    mfcc = compute_mfcc(float_audio)       # shape (18,13)
    mfcc = norm_mfcc(mfcc)                 # 归一化
    mfcc = np.expand_dims(mfcc, axis=-1)   # (18,13,1)
    return mfcc

def quantize_input(mfcc):
    """
    float -> int8
    """
    input_int8 = (mfcc / INPUT_SCALE + INPUT_ZP).round().astype(np.int8)
    return np.expand_dims(input_int8, axis=0)  # (1,18,13,1)

def float_input(mfcc):
    return np.expand_dims(mfcc.astype(np.float32), axis=0)

# ==============================
# 主函数
# ==============================
def detect_barks():
    # 加载模型
    interpreter_int8 = tf.lite.Interpreter(model_path=MODEL_INT8_PATH)
    interpreter_int8.allocate_tensors()
    input_detail_int8 = interpreter_int8.get_input_details()[0]
    output_detail_int8 = interpreter_int8.get_output_details()[0]

    interpreter_fp = tf.lite.Interpreter(model_path=MODEL_FLOAT_PATH)
    interpreter_fp.allocate_tensors()
    input_detail_fp = interpreter_fp.get_input_details()[0]
    output_detail_fp = interpreter_fp.get_output_details()[0]

    # 逐个 h 文件测试
    for hfile in H_FILES:
        print(f"\n=== Testing {hfile} ===")
        audio = load_h_array(hfile)
        mfcc = extract_mfcc_from_int16(audio)

        # INT8 推理
        input_i8 = quantize_input(mfcc)
        interpreter_int8.set_tensor(input_detail_int8['index'], input_i8)
        interpreter_int8.invoke()
        out_i8 = interpreter_int8.get_tensor(output_detail_int8['index'])[0][0]
        prob_int8 = (out_i8 - OUTPUT_ZP) * OUTPUT_SCALE

        # Float32 推理
        input_fp = float_input(mfcc)
        interpreter_fp.set_tensor(input_detail_fp['index'], input_fp)
        interpreter_fp.invoke()
        prob_fp = interpreter_fp.get_tensor(output_detail_fp['index'])[0][0]

        print(f"INT8 probability : {prob_int8:.4f}")
        print(f"Float32 probability : {prob_fp:.4f}")

if __name__ == "__main__":
    detect_barks()



"""
=== Testing tinyml/aug_3.h ===
INT8 probability : 0.2031
Float32 probability : 0.1988

=== Testing tinyml/aug_10.h ===
INT8 probability : 0.2344
Float32 probability : 0.2409

=== Testing tinyml/aug_20.h ===
INT8 probability : 0.6367
Float32 probability : 0.6199


需要比较mcu测试结果：
ROOT/test/test_tinyml_mfcc_model.cpp
Dog_AUG_3 Bark Probability: 0.246 -> NO BARK
Dog_AUG_10 Bark Probability: 0.309 -> NO BARK
Dog_AUG_20 Bark Probability: 0.691 -> NO BARK



误差在10%都是可以接受的
"""