# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/14 15:31
@Author  : Kend
@FileName: convert_quant_int8
@Software: PyCharm
@modifier:
"""

"""
使用验证集的数据进行int8模型的量化
"""

import numpy as np
import tensorflow as tf
import glob
import soundfile as sf
from sklearn.model_selection import train_test_split
from train_float_model_mfcc import load_dataset



VAL_LIST = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/tinyml/dataset/val.txt"  # 验证集文件列表
MODEL_PATH = "best_float_model_mfcc.keras"   # 训练好的 float 模型， mfcc。 保持在同一个目录
OUTPUT_TFLITE = "quantized_model_mfcc_int8.tflite"   # 导出的量化的int8 模型文件为转cc数组做mcu部署准备


def from_val_dataset_get_mfcc_data():
    mfcc_segment, mfcc_label = load_dataset(VAL_LIST)   # 获取验证集数据 验证集: (3141, 18, 13, 1), 标签分布: [2434  707]
    # 对数据进行挑选200个狗吠 200个非狗吠进行mfcc数据提取和归一化，作为int8量化的数据入口
    # 转为 numpy array 方便索引
    # mfcc_segment = np.array(mfcc_segment)
    # mfcc_label = np.array(mfcc_label)
    # 分离狗叫（label=1）和非狗叫（label=0）的索引
    bark_indices = np.where(mfcc_label == 1)[0]  # shape: (707,)
    non_bark_indices = np.where(mfcc_label == 0)[0]  # shape: (2434,)
    print(f"狗叫样本数: {len(bark_indices)}, 非狗叫样本数: {len(non_bark_indices)}")
    # 随机采样 200 个（如果不够就全取）
    sampled_bark = np.random.choice(bark_indices, size=min(200, len(bark_indices)), replace=False)
    sampled_non_bark = np.random.choice(non_bark_indices, size=min(200, len(non_bark_indices)), replace=False)
    # 合并索引
    selected_indices = np.concatenate([sampled_bark, sampled_non_bark])
    # 打乱顺序（重要！避免量化时 bias）
    np.random.shuffle(selected_indices)
    # 提取对应的 MFCC 数据
    representative_data = mfcc_segment[selected_indices]  # shape: (~400, 18, 13, 1)
    print(f"最终用于量化的样本数: {len(representative_data)}")
    """ 注意： 量化不需要标签， 只做 forward pass（不计算 loss，不更新参数）完全不关心预测对不对 → 所以不需要知道真实标签"""
    return representative_data




if __name__ == "__main__":
    representative_data = from_val_dataset_get_mfcc_data()
    """
    狗叫样本数: 707, 非狗叫样本数: 2434
    最终用于量化的样本数: 400
    """
    print("正在定义 representative dataset 生成器...")


    def representative_dataset():
        for i in range(len(representative_data)):
            # 注意：必须加 batch 维度 → (1, 18, 13, 1)
            yield [representative_data[i:i + 1]]   # 因为后续需要部署到mcu， 所以batch固定位1


    print("正在加载 float 模型...")
    model = tf.keras.models.load_model(MODEL_PATH)   # 加载的keras 模型

    print("正在配置 TFLite 转换器...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 默认
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("正在执行 INT8 量化...")
    quantized_tflite_model = converter.convert()

    print("正在保存量化模型...")
    with open(OUTPUT_TFLITE, "wb") as f:
        f.write(quantized_tflite_model)

    print(f"量化完成！模型已保存为: {OUTPUT_TFLITE}")

    # ==============================
    # 打印量化参数（用于 MCU 端还原概率）
    # ==============================
    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print("\n 量化参数（MCU 部署需要）:")
    print(f"  输入 scale:    {input_details['quantization'][0]:.8f}")
    print(f"  输入 zero_point: {input_details['quantization'][1]}")
    print(f"  输出 scale:    {output_details['quantization'][0]:.8f}")
    print(f"  输出 zero_point: {output_details['quantization'][1]}")

"""
正在保存量化模型...
量化完成！模型已保存为: quantized_model_mfcc_int8.tflite

 量化参数（MCU 部署需要）:
  输入 scale:    0.00784314
  输入 zero_point: -1
  输出 scale:    0.00390625
  输出 zero_point: -128
"""

