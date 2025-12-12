# -*- coding: utf-8 -*-
"""
@Time    : 2025/12/9 11:06
@Author  : Kend
@FileName: embedding_model
@Software: PyCharm
@modifier:
"""



"""
模型训练以及部署的主要事项：
    做“狗吠声纹验证”，而且 MCU 上空间极小 → 可以做到“不训练主模型”
    用一个 固定的、小模型 提取 embedding（你现在的 CNN+Depthwise 结构）
    模型 不用学每只狗是谁，只是负责把声音压成 16 维声纹向量
    真正的 “狗是谁” 是你 MCU 或后端用模板库做：
        ✔ 欧氏距离
        ✔ 余弦相似度
        ✔ 阈值判断
    也就是说：
    模型不训练区分不同狗
    狗吠声纹的分类逻辑，不在模型里，而在你 MCU 的模板库里
    所以这个方案：
    无监督 + 模板库声纹验证
    模型只抽特征，不做分类，模型不需要训练成“识别不同狗”的任务。
"""


"""
狗吠声纹验证对比 tinyml模型的设计
注意：实际部署是验证是不是模版库的声纹特征， 只是1对多
    1. 输入是 18×40
        18 = 200ms / 10ms hop
        40 mel 是声纹任务的最稳选择（MFCC 反而损失 timbre 信息）
        logmel 是声纹识别里的行业标准
    2. Conv2D + DepthwiseConv
        Conv2D → TFLM 最稳定的主力算子
        DepthwiseConv → 大幅减少参数（比普通Conv2D少9倍）
        无 BatchNorm（TFLM 不支持）
        无 LayerNorm（TFLM 不支持）
        无 SE/ResNet block（Flash 超限）
    3. pooling 两次
        18×40 的输入太大
        特征图必须在 MCU 内降低维度，否则 Dense 层会爆 RAM
        两次 pool 之后形状变成大概 4×10
        再卷积一次 → 足够表达声纹特征
    4. 输出 16D embedding
        每只狗只有 4–6 条吠声样本
        目前总共 11 只狗
        MCU RAM 只有 150KB
    5. 最后用 L2 Norm
        sim = dot(emb1 , emb2)     # 余弦相似度
        保证余弦相似度有效，embedding 必须 单位化
        练端就直接加 L2 Norm，MCU 端可以省略
"""



import tensorflow as tf
from tensorflow.keras import layers, models


# 自定义 L2Norm 层  ？ tf.math.l2_normalize 不能直接作用在 KerasTensor 上 ，必须封装成 Keras 层
class L2NormLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)



def build_dog_bark_embedding_model():
    """
    TinyML-compatible LogMel → 16D embedding 模型
    输入： (18, 40, 1)
    输出： (16,) L2-normalized embedding
    """

    inputs = layers.Input(shape=(18, 40, 1), name="logmel_input")

    # -------------------------
    # 1) 3×3 标准卷积：提取局部时间-频率特征
    # -------------------------
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(inputs)

    # -------------------------
    # 2) Depthwise 让模型更轻 → 类似 MobileNet
    # -------------------------
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # (9, 20, 16)

    # -------------------------
    # 3) 第二组卷积 + 深度可分离卷积
    # -------------------------
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # (4, 10, 32)

    # -------------------------
    # 4) 进一步压缩信息
    # -------------------------
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    # -------------------------
    # 5) 全局平均池化 → 将所有时间+频率信息 squeeze 到 1×1×32
    # -------------------------
    x = layers.AveragePooling2D(
        pool_size=(4, 10)
    )(x)  # → (1,1,32)

    x = layers.Reshape((32,))(x)  # → shape: (32,)

    # -------------------------
    # 6) 全连接 → 最终 embedding
    # -------------------------
    x = layers.Dense(16)(x)  # 16D 特征

    # -------------------------
    # 7) L2 Normalize → 单位向量，适合做相似度比较
    # -------------------------
    x = L2NormLayer()(x)

    model = models.Model(inputs, x, name="tiny_dog_embedding")
    return model



def build_tflm_dog_bark_embedding_model():
    """
    TinyML-compatible LogMel -> 16D Feature Extractor Model
    - Uses SeparableConv2D for better feature representation.
    - Output is the 16D embedding (before L2 Norm).
    - Designed for int8 quantization using TFLM-friendly operators.
    """

    inputs = layers.Input(shape=(18, 40, 1), name="logmel_input")

    # --- 1) 初始特征提取与维度压缩 (18x40 -> 9x20) ---
    x = layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(inputs)

    # 替换 DepthwiseConv 以增强特征融合 (Depthwise + 1x1 Pointwise Conv)
    x = layers.SeparableConv2D(
        filters=16,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Shape: (None, 9, 20, 16)

    # --- 2) 第二组特征提取与二次压缩 (9x20 -> 4x10) ---
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    x = layers.SeparableConv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Shape: (None, 4, 10, 32)

    # --- 3) 最终特征提炼与全局压缩 ---
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)

    # 全局平均池化 (GAP) 将所有空间信息压缩到 1x1x32
    x = layers.AveragePooling2D(
        pool_size=(4, 10)  # 对应当前的特征图尺寸
    )(x)  # Shape: (None, 1, 1, 32)

    x = layers.Reshape((32,))(x)  # Shape: (None, 32)

    # --- 4) 最终 Embedding ---
    # 16D 特征输出
    # embedding_output = layers.Dense(16, name="embedding_output")(x)
    # 将输出维度改为 32
    embedding_output = layers.Dense(32, name="embedding_output")(x)  # <--- 唯一修改点

    # 注意：这里不包含 L2 Norm 层。
    # L2 Norm 应该在训练时使用，并在 TFLite 转换后，
    # 在 MCU 的 C/C++ 代码中，对这个 'embedding_output' 进行归一化。

    model = models.Model(inputs, embedding_output, name="tiny_dog_embedding_v2")
    return model






if __name__ == "__main__":
    model = build_tflm_dog_bark_embedding_model()
    model.summary()





