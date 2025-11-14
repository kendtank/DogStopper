# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/14 13:31
@Author  : Kend
@FileName: train_float_model_mfcc
@Software: PyCharm
@modifier:
"""


"""
使用和mcu对齐的mfcc特征进行数据集的训练
    归一化特征到[-1, 1]
    训练float32模型为后续的int8量化模型做准备
"""



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
# 将上级目录（包含 audio_features.py 的目录）加入 Python 模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import argparse
from tqdm import tqdm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# 归一化mfcc特征
from dataset.norm_mfcc import norm_mfcc
# 对齐mcu的加载音频文件和mfcc特征的计算
from dataset.compute_robust_norm_params import load_audio_file, extract_mfcc, sliding_windows




# ==============================
# 数据加载函数
# ==============================
def load_dataset(file_list_path, max_segments_per_file=50):
    """
    加载数据集：滑窗切分 + MFCC + 归一化
    返回: X (N, 18, 13, 1), y (N,)
    """
    X, y = [], []
    
    with open(file_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in tqdm(lines, desc=f"Loading {os.path.basename(file_list_path)}"):
        parts = line.split()
        if len(parts) < 2:
            continue
        wav_path, label = parts[0], int(parts[1])
        if not os.path.exists(wav_path):
            continue
        
        try:
            audio = load_audio_file(wav_path)
            if audio is None:
                continue
            
            segments = sliding_windows(audio)
            count = 0
            for seg in segments:
                if count >= max_segments_per_file:   # 滑窗不能超过50
                    break
                # 确保长度为 3200
                target_len = 3200
                if len(seg) != target_len:
                    print(f"滑窗处理后还是没有对齐3200点")  # 这里只是防御性编程
                mfcc = extract_mfcc(seg)  # (18, 13)
                mfcc_norm = norm_mfcc(mfcc)  # (18, 13), [-1, 1]
                
                X.append(mfcc_norm[..., np.newaxis])  # (18, 13, 1)
                y.append(label)
                count += 1
                
        except Exception as e:
            print(f"跳过 {wav_path}: {e}")
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)



# ==============================
# 加载模型  需要和model/bark_mfcc_tinyml.py中模型一致
# ==============================
from model.bark_mfcc_tinyml import build_dog_bark_tiny_model

# def build_dog_bark_tiny_model(input_shape=(18, 13, 1)):
#     model = tf.keras.Sequential([
#         tf.keras.layers.SeparableConv2D(4, (3, 3), activation='relu', padding='same', input_shape=input_shape),
#         tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
#         tf.keras.layers.SeparableConv2D(8, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model



# ==============================
# 主训练流程
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default="train.txt")
    parser.add_argument("--val_list", default="val.txt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_save", default="best_float_model_mfcc.keras")
    args = parser.parse_args()

    print("正在加载训练数据...")
    X_train, y_train = load_dataset(args.train_list)
    print(f"训练集: {X_train.shape}, 标签分布: {np.bincount(y_train)}")

    print("正在加载验证数据...")
    X_val, y_val = load_dataset(args.val_list)
    print(f"验证集: {X_val.shape}, 标签分布: {np.bincount(y_val)}")

    # 计算类别权重（处理不平衡）
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"类别权重: {class_weight_dict}")

    # 构建模型
    model = build_dog_bark_tiny_model()
    model.summary()


    # 回调函数
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),   # 早停的阈值参数
        tf.keras.callbacks.ModelCheckpoint(args.model_save, save_best_only=True, monitor='val_accuracy')  # 自动保存最好的模型
    ]

    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 训练完成，评估模型
    print("模型训练完成... 评估中")
    # ==============================
    # 评估不同阈值下的模型性能
    # ==============================
    print("\n正在评估不同阈值下的性能...")
    best_model = tf.keras.models.load_model(args.model_save)
    y_val_pred_prob = best_model.predict(X_val).flatten()  # shape: (N,)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    thresholds = [0.5, 0.6, 0.7]
    print(f"{'阈值':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 55)

    for th in thresholds:
        y_pred = (y_val_pred_prob > th).astype(int)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        print(f"{th:<6} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")


    # 保存 TFLite float 模型（用于后续量化）
    print("正在加载最佳模型用于导出...")
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()
    with open("float_model_mfcc.tflite", "wb") as f:
        f.write(tflite_model)
    print("TFLite float 模型已保存: float_model.tflite")

    print("训练完成！")


if __name__ == "__main__":
    main()


"""
run: python train_float_model_mfcc.py --train_list ../dataset/train.txt --val_list ../dataset/val.txt 

正在评估不同阈值下的性能...
99/99 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step   
阈值     Accuracy   Precision  Recall     F1-score  
-------------------------------------------------------
0.5    0.9242     0.8355     0.8260     0.8307    
0.6    0.9147     0.8604     0.7412     0.7964    
0.7    0.9013     0.8767     0.6535     0.7488    
正在加载最佳模型用于导出...
Saved artifact at '/tmp/tmphygys2y1'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 18, 13, 1), dtype=tf.float32, name='input_layer')
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  135302548286096: TensorSpec(shape=(), dtype=tf.resource, name=None)
  135303637879088: TensorSpec(shape=(), dtype=tf.resource, name=None)
  135303637878912: TensorSpec(shape=(), dtype=tf.resource, name=None)
  135302544214848: TensorSpec(shape=(), dtype=tf.resource, name=None)
  135302213113104: TensorSpec(shape=(), dtype=tf.resource, name=None)
  135302154294368: TensorSpec(shape=(), dtype=tf.resource, name=None)
  135302153330000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  135302153329120: TensorSpec(shape=(), dtype=tf.resource, name=None)
W0000 00:00:1763104984.629068  362254 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1763104984.629080  362254 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
I0000 00:00:1763104984.632335  362254 mlir_graph_optimization_pass.cc:425] MLIR V1 optimization pass is not enabled
TFLite float 模型已保存: float_model.tflite
训练完成！



"""