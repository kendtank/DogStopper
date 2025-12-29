# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/9 14:14
@Author  : Kend
@FileName: convert_quantize_embedding
@Software: PyCharm
@modifier:
"""

"""
全量量化脚本：复用训练时完全一致的滑窗、logmel 特征提取流程
"""

import os
import numpy as np
import tensorflow as tf

# 导入你训练时的所有流程
from train.dataset_utils import (
    read_wave_mcu_style_float,
    sliding_windows_200ms_50ms,
    compute_logmel,
)

# -----------------------------
# 1. 生成校准数据集（全量）
# -----------------------------
def build_calibration_set(dataset_root: str, limit=None):
    """
    dataset_root: 训练集根目录，每个子目录是一个 speaker
    limit: 若设置，将最多取 N 个特征避免爆内存（可选）
    """
    calib_feats = []

    class_dirs = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])

    print("Found speakers:", class_dirs)

    count = 0

    for spk in class_dirs:
        spk_dir = os.path.join(dataset_root, spk)
        wavs = [
            f for f in os.listdir(spk_dir)
            if f.lower().endswith((".wav", ".wave"))
        ]

        for wf in wavs:
            wav_path = os.path.join(spk_dir, wf)

            # 读取音频
            wave = read_wave_mcu_style_float(wav_path)

            # 滑窗 200ms/50ms
            segs = sliding_windows_200ms_50ms(wave)  # list of 3200 samples

            for seg in segs:
                # logmel 特征（与你训练时完全一致）
                S_db, t = compute_logmel(seg)

                # t = (time_frames, n_mels) → 加一个通道维度
                t = t.astype(np.float32)[..., np.newaxis]  # (T, n_mels, 1)

                calib_feats.append(t)
                count += 1

                if limit and count >= limit:
                    print("Reached limit =", limit)
                    return calib_feats

    print(f"Calibration features collected: {len(calib_feats)}")
    return calib_feats


# -----------------------------
# 2. 构建 Representative Dataset
# -----------------------------
def representative_dataset_gen(calib_feats):
    for feat in calib_feats:
        # TFLite 要求 batch size = 1
        feat = feat[np.newaxis, ...]  # (1, T, n_mels, 1)
        yield [feat]


# -----------------------------
# 3. 执行 INT8 量化
# -----------------------------
# def quantize_int8(keras_model_path: str, calib_feats, output_tflite="model_int8.tflite"):
#     model = tf.keras.models.load_model(keras_model_path)

#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]

#     # 使用校准集
#     converter.representative_dataset = lambda: representative_dataset_gen(calib_feats)

#     # 全 int8 推理
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#     converter.inference_input_type = tf.int8
#     converter.inference_output_type = tf.int8

#     tflite_model = converter.convert()

#     with open(output_tflite, "wb") as f:
#         f.write(tflite_model)

#     print("Saved:", output_tflite)


def quantize_int8(
    keras_model_path: str,
    calib_feats,
    output_tflite: str = "model_int8.tflite",
):
    # 1. 加载模型（不编译，避免引入无关 state）
    model = tf.keras.models.load_model(keras_model_path, compile=False)

    # 2. 创建 TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # ====== 核心：MCU 必须的约束 ======

    # 启用新 converter（图更干净）
    converter.experimental_new_converter = True

    # 禁止 resource variables（Stateful 图的来源）
    converter.experimental_enable_resource_variables = False

    # 禁止 custom ops（否则 MCU 会 fallback）
    converter.allow_custom_ops = False

    # 禁止 TensorList / 动态 list（SHAPE / PACK 常见来源）
    converter._experimental_lower_tensor_list_ops = False

    # ====== 标准 INT8 量化 ======

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 使用真实校准数据
    converter.representative_dataset = lambda: representative_dataset_gen(calib_feats)

    # 强制全 INT8（MCU 必须）
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    converter.experimental_new_quantizer = True  # ← 这一行

    # ====== 转换 ======
    tflite_model = converter.convert()

    # ====== 保存 ======
    with open(output_tflite, "wb") as f:
        f.write(tflite_model)

    print(f"[OK] INT8 TFLite saved to: {output_tflite}")




def dump_tflite_quant_params(tflite_path: str):
    """读取 tflite 的量化 scale / zero_point"""
    print("\n== Dump INT8 Quantization Params ==")

    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n-- Input tensors --")
    for idx, det in enumerate(input_details):
        qp = det['quantization']
        print(f"Input[{idx}] name={det['name']}")
        print(f"  scale={qp[0]}, zero_point={qp[1]}")
        print(f"  shape={det['shape']}  dtype={det['dtype']}")

    print("\n-- Output tensors --")
    for idx, det in enumerate(output_details):
        qp = det['quantization']
        print(f"Output[{idx}] name={det['name']}")
        print(f"  scale={qp[0]}, zero_point={qp[1]}")
        print(f"  shape={det['shape']}  dtype={det['dtype']}")

    print("\n=========================================\n")


# -----------------------------
# 4. 主入口
# -----------------------------
if __name__ == "__main__":
    DATASET_ROOT = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog/"   # 修改成你的训练集路径
    KERAS_MODEL = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/dog_bark_embed32_export/embedding_model.keras" # 修改成你导出的模型
    OUT_FILE = "embed_model_int8.tflite"

    print("== Building calibration dataset ==")
    calibs = build_calibration_set(DATASET_ROOT, limit=None)  # limit=None = 全量

    print("== Quantizing to INT8 ==")
    quantize_int8(KERAS_MODEL, calibs, OUT_FILE)

    dump_tflite_quant_params(OUT_FILE)

    print("Done.")


    '''
    量化与反量化参数
    == Dump INT8 Quantization Params ==

    -- Input tensors --
    Input[0] name=serving_default_logmel_input:0
      scale=0.3959104120731354, zero_point=74
      shape=[ 1 18 40  1]  dtype=<class 'numpy.int8'>
    
    -- Output tensors --
    Output[0] name=StatefulPartitionedCall_1:0
      scale=0.000658028875477612, zero_point=-33
      shape=[ 1 32]  dtype=<class 'numpy.int8'>


xxd -i embed_model_int8.tflite > embed_model_int8.cc

=========================================
    '''
