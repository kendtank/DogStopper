# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/11 16:25
@Author  : Kend
@FileName: ver_diff_dog
@Software: PyCharm
@modifier:
"""

# -*- coding: utf-8 -*-
"""
支持 float32 模型 + INT8 全整形量化模型推理测试对比
"""

import numpy as np
import tensorflow as tf

from train_embedding_v4 import compute_logmel
from train_embedding_v4 import read_wave_mcu_style_float
from train_embedding_v4 import sliding_windows_200ms_50ms

SR = 16000


# =========================================================
# 加载 TFLite 模型（自动识别 float or int8）
# =========================================================
def load_tflite(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 检查输入类型
    inp_dtype = input_details[0]["dtype"]
    model_type = "int8" if inp_dtype == np.int8 else "float"

    print(f"Loaded model: {tflite_path}")
    print(f"Model type detected: {model_type}")

    return interpreter, input_details, output_details, model_type


# =========================================================
# INT8 量化 / 反量化
# =========================================================
def quantize_input(x_float, input_detail):
    """float32 → int8"""
    scale = input_detail["quantization"][0]
    zero = input_detail["quantization"][1]

    x_int8 = np.clip(
        np.round(x_float / scale + zero),
        -128, 127
    ).astype(np.int8)
    return x_int8


def dequantize_output(x_int8, output_detail):
    """int8 → float32"""
    scale = output_detail["quantization"][0]
    zero = output_detail["quantization"][1]

    return (x_int8.astype(np.float32) - zero) * scale


# =========================================================
# TFLite 推理（自动适配 float / int8）
# =========================================================
def infer_tflite_embedding(interpreter, input_details, output_details, model_type, logmel_T):
    """
    logmel_T shape: (T, n_mels)
    """

    # reshape → (1, T, n_mels, 1)
    x = logmel_T[np.newaxis, ..., np.newaxis].astype(np.float32)

    if model_type == "int8":
        # === 输入量化 ===
        x = quantize_input(x, input_details[0])

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])

    if model_type == "int8":
        # === 输出反量化 ===
        out = dequantize_output(out, output_details[0])

    return out[0].astype(np.float32)


# =========================================================
# 提取所有滑窗的 embedding
# =========================================================
def extract_embeddings(wave, sr, interpreter, input_details, output_details, model_type, do_l2=True):
    frames = sliding_windows_200ms_50ms(wave, sr)
    embeddings = []

    for seg in frames:
        S_db, mel_T = compute_logmel(seg, sr)
        emb = infer_tflite_embedding(interpreter, input_details, output_details, model_type, mel_T)

        if do_l2:
            emb = emb / (np.linalg.norm(emb) + 1e-9)

        embeddings.append(emb)

    return np.array(embeddings)


# =====================================================
# 余弦相似度
# =====================================================
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =========================================================
# 主逻辑
# =========================================================
def test_similarity_tflite(audio1, audio2, tflite_model, do_l2=True):
    print("\n=== Loading audio ===")
    wav1 = read_wave_mcu_style_float(audio1)
    wav2 = read_wave_mcu_style_float(audio2)

    print("\n=== Loading TFLite model ===")
    interpreter, inp, out, model_type = load_tflite(tflite_model)

    print("\n=== Extracting embeddings ===")
    emb1 = extract_embeddings(wav1, SR, interpreter, inp, out, model_type, do_l2)
    emb2 = extract_embeddings(wav2, SR, interpreter, inp, out, model_type, do_l2)

    print(f"Audio1 segments = {emb1.shape[0]}")
    print(f"Audio2 segments = {emb2.shape[0]}")

    print("\n=== 相似度矩阵（cosine）===\n")

    for i in range(emb1.shape[0]):
        row = []
        for j in range(emb2.shape[0]):
            sim = cosine(emb1[i], emb2[j])
            row.append(f"{sim:.3f}")
        print(f"[A1_seg{i:02d}] -> {row}")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=str, required=True, help="audio 1")
    parser.add_argument("--a2", type=str, required=True, help="audio 2")
    parser.add_argument("--tflite", type=str, required=True, help="tflite model path")
    parser.add_argument("--nol2", action="store_true")
    args = parser.parse_args()

    test_similarity_tflite(args.a1, args.a2, args.tflite, do_l2=not args.nol2)



# python ver_diff_dog.py --a1 /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog/dog07/dog_bark_044.WAV --a2 /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog/dog07/dog_bark_048.WAV --tflite /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/dog_bark_embed32_export/embedding_model_fp32.tflite

# python ver_diff_dog.py --a1 /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog/dog07/dog_bark_044.WAV --a2 /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog/dog07/dog_bark_048.WAV --tflite /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/embed_model_int8.tflite

# python ver_diff_dog.py --a1 /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog/dog02/dog_bark_009.WAV --a2 /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/train/compare_dog/dog02/dog_bark_017.WAV --tflite /home/kend/文档/PlatformIO/Projects/DogStopper/python/verify/embed_model_int8.tflite


"""
声纹测试结果：(滑窗) （fp32模型测试结果）

    同一狗不同叫声：
    Audio1 segments = 3
    Audio2 segments = 3
    === 相似度矩阵（cosine）===
    [A1_seg00] -> ['0.836', '0.823', '0.644']
    [A1_seg01] -> ['0.855', '0.912', '0.771']
    [A1_seg02] -> ['0.574', '0.713', '0.773']
    
    
    同一狗不同叫声：
    Audio1 segments = 4
    Audio2 segments = 6
    
    === 相似度矩阵（cosine）===
    
    [A1_seg00] -> ['0.799', '0.800', '0.691', '0.756', '0.680', '0.699']
    [A1_seg01] -> ['0.843', '0.904', '0.800', '0.865', '0.731', '0.820']
    [A1_seg02] -> ['0.770', '0.834', '0.870', '0.908', '0.735', '0.732']
    [A1_seg03] -> ['0.752', '0.758', '0.664', '0.851', '0.791', '0.889']
    
    === INT8量化模型 ====
    [A1_seg00] -> ['0.773', '0.742', '0.617', '0.702', '0.644', '0.640']
    [A1_seg01] -> ['0.826', '0.874', '0.720', '0.884', '0.699', '0.843']
    [A1_seg02] -> ['0.771', '0.798', '0.831', '0.924', '0.736', '0.764']
    [A1_seg03] -> ['0.736', '0.720', '0.549', '0.878', '0.756', '0.882']


    同一狗同一叫声：
    Audio1 segments = 4
    Audio2 segments = 4
    === 相似度矩阵（cosine）===
    [A1_seg00] -> ['1.000', '0.855', '0.800', '0.814']
    [A1_seg01] -> ['0.855', '1.000', '0.839', '0.748']
    [A1_seg02] -> ['0.800', '0.839', '1.000', '0.801']
    [A1_seg03] -> ['0.814', '0.748', '0.801', '1.000']
    
    不同狗叫声：
    Audio1 segments = 9
    Audio2 segments = 4
    
    === 相似度矩阵（cosine）===
    
    [A1_seg00] -> ['0.013', '-0.083', '0.258', '0.201']
    [A1_seg01] -> ['0.025', '-0.052', '0.206', '0.229']
    [A1_seg02] -> ['0.049', '-0.040', '0.245', '0.212']
    [A1_seg03] -> ['0.155', '-0.006', '0.188', '0.342']
    [A1_seg04] -> ['0.078', '-0.162', '0.066', '0.239']
    [A1_seg05] -> ['0.255', '0.020', '0.343', '0.419']
    [A1_seg06] -> ['0.214', '0.038', '0.371', '0.330']
    [A1_seg07] -> ['0.184', '0.085', '0.438', '0.273']
    [A1_seg08] -> ['0.190', '0.041', '0.382', '0.258']
    
    不同狗叫声：
    Audio1 segments = 4
    Audio2 segments = 4
    
    === 相似度矩阵（cosine）===
    
    [A1_seg00] -> ['0.158', '0.130', '-0.066', '0.183']
    [A1_seg01] -> ['0.132', '0.089', '-0.096', '0.249']
    [A1_seg02] -> ['-0.099', '-0.111', '-0.344', '-0.035']
    [A1_seg03] -> ['0.339', '0.247', '0.058', '0.278']

    
"""