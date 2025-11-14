# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/14 16:08
@Author  : Kend
@FileName: evaluate_quantized_model
@Software: PyCharm
@modifier:
"""


"""
模型量化为int8之后，写一个脚本，测试量化int8模型和训练得到的float32位的模型的量化误差
"""

"""
评估量化模型 vs 原始 float 模型在完整验证集上的性能
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================
# 配置路径
# ==============================
VAL_LIST = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/tinyml/dataset/val.txt"
FLOAT_MODEL_PATH = "best_float_model_mfcc.keras"
QUANTIZED_MODEL_PATH = "quantized_model_mfcc_int8.tflite"

from train_float_model_mfcc import load_dataset


def evaluate_models():

    print("正在加载完整验证集...")
    X_val, y_val = load_dataset(VAL_LIST)
    print(f"验证集大小: {X_val.shape}, 标签分布: {np.bincount(y_val)}")

    # ==============================
    # 1. Float32 TFLite 模型预测
    # ==============================
    print("\n正在运行 Float32 TFLite 模型预测...")
    float_interpreter = tf.lite.Interpreter(model_path="float_model_mfcc.tflite")
    float_interpreter.allocate_tensors()

    input_details_float = float_interpreter.get_input_details()[0]
    output_details_float = float_interpreter.get_output_details()[0]

    float_probs = []
    for i in range(len(X_val)):
        x_batch = X_val[i:i + 1]  # (1, 18, 13, 1), float32
        float_interpreter.set_tensor(input_details_float['index'], x_batch)
        float_interpreter.invoke()
        prob = float_interpreter.get_tensor(output_details_float['index'])[0][0]
        float_probs.append(prob)

    float_probs = np.array(float_probs)
    float_preds = (float_probs > 0.5).astype(int)


    # ==============================
    # 2. INT8 模型预测
    # ==============================
    print("\n正在运行 INT8 模型预测...")
    interpreter = tf.lite.Interpreter(model_path=QUANTIZED_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    # 获取输入量化参数
    input_scale, input_zero_point = input_details['quantization']
    print(f"输入量化参数: scale={input_scale:.8f}, zero_point={input_zero_point}")

    int8_probs = []
    for i in range(len(X_val)):
        # NOTE : 这里拿到的float类型的mfcc数据，已经归一化
        # 输入必须是 batch=1 的 float32
        x_float = X_val[i:i + 1]  # shape: (1, 18, 13, 1)
        # 2. 手动量化到 int8， 喂给int8模型需要量化到INT8的输入
        x_int8 = np.round(x_float / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details['index'], x_int8)
        interpreter.invoke()

        # 获取 int8 输出并反量化为概率
        output_int8 = interpreter.get_tensor(output_details['index'])[0][0]
        scale, zero_point = output_details['quantization']
        # 获取的INT8输出，需要反量化为概率
        prob = (output_int8 - zero_point) * scale
        int8_probs.append(prob)

    int8_probs = np.array(int8_probs)
    int8_preds = (int8_probs > 0.5).astype(int)

    # ==============================
    # 3. 计算并打印指标
    # ==============================
    def metrics_str(y_true, y_pred, name):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return f"{name} → Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"

    print("\n" + "=" * 70)
    print("量化效果对比（完整验证集）")
    print("=" * 70)
    print(metrics_str(y_val, float_preds, "Float32"))
    print(metrics_str(y_val, int8_preds, "INT8   "))

    # 计算性能损失
    f1_drop = f1_score(y_val, float_preds) - f1_score(y_val, int8_preds)
    print(f"\nF1 下降: {f1_drop:.4f} ({'可接受' if f1_drop < 0.03 else '需警惕'})")

    # 概率误差统计
    prob_diff = np.abs(float_probs - int8_probs)
    print(f"最大概率误差: {prob_diff.max():.5f}")
    print(f"平均概率误差: {prob_diff.mean():.5f}")
    print(f"\n评估使用的样本总数: {len(y_val)}")
    counts = np.bincount(y_val)
    print(f"   - 非狗叫: {counts[0]}")
    print(f"   - 狗叫:   {counts[1]}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_models()

"""
验证集大小: (3141, 18, 13, 1), 标签分布: [2434  707]

正在运行 Float32 TFLite 模型预测...

正在运行 INT8 模型预测...
输入量化参数: scale=0.00784314, zero_point=-1

======================================================================
量化效果对比（完整验证集）
======================================================================
Float32 → Acc: 0.9242 | Prec: 0.8355 | Rec: 0.8260 | F1: 0.8307
INT8    → Acc: 0.9242 | Prec: 0.8394 | Rec: 0.8204 | F1: 0.8298

F1 下降: 0.0010 (可接受)
最大概率误差: 0.04535
平均概率误差: 0.00290

评估使用的样本总数: 3141
   - 非狗叫: 2434
   - 狗叫:   707
======================================================================

结论： INT8 量化模型在完整验证集上，输出概率与 float 模型平均仅偏差 0.29%，最大偏差 4.5%，分类性能几乎无损。
"""
