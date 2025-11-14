
# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/12 下午03:59
@Author  : Kend
@FileName: bark_mfcc_tinyml.py
@Software: PyCharm
@modifier:
"""


import tensorflow as tf
from tensorflow.keras import layers

"""
注意：为什么设计模型的输入是18 * 13
- input_shape=(13, 18, 1)  # 频率在前，时间在后
- 卷积核 (3,3) 会在 13（频率）×18（时间） 平面上滑动
    但 频率维只有 13，3×3 卷积会：过度平滑频谱细节（丢失高频特征）在边缘产生 padding 假象
为什么时间维应在前？
    狗吠是 时间上的突发事件，模型需要检测 “某几帧是否出现能量爆发”。如果时间在后，卷积无法有效捕捉这种时序模式。
MaxPooling 可能丢掉短吠叫
    MaxPool((2,2)) 把 (18,13) → (9,6)
    如果狗吠只持续 2–3 帧（<100ms），可能被池化“稀释”
"""


def build_dog_bark_tiny_model(input_shape=(18, 13, 1)):
    """
    超轻量狗吠检测模型（二分类）
    - 输入：MFCC 特征 (时间帧, 频率系数, 1) 时间在前 → 卷积捕捉声音突发模式， 频率在后 → 保留狗叫音色特征 mfcc特征需要保持行列式 18 * 13
    - 输出：sigmoid 概率 [0.0 ~ 1.0]，>0.7 判为狗吠 (直接舍弃了非狗吠的背景输出节点)
    
    设计目标：
        模型 < 1KB (int8)
        推理 < 2ms on ESP32-S3
        完全兼容 TFLite Micro + int8 量化
        对噪声鲁棒（配合多样数据 + 后处理规则） 多样数据，后续需要更多数据 # TODO

    算子兼容性说明（全部支持 TFLite Micro int8）：
      - SeparableConv2D → TFLM 支持（Depthwise + 1x1 Conv）
      - MaxPooling2D   → 支持
      - GlobalAveragePooling2D → 支持（关键！替代 Flatten）
      - Dense          → 支持
      - ReLU / Sigmoid → 支持（Sigmoid 在 int8 下近似实现）
    """
    model = tf.keras.Sequential([
        # ────────────────────────────────────────
        # 第1层：轻量时空特征提取（SeparableConv）
        # 原理：SeparableConv = DepthwiseConv（每通道独立卷积） + PointwiseConv（1x1融合）
        # 优势：参数量仅为普通 Conv 的 ~1/N，保留局部时频模式（如“能量突增”）
        # 输入: (18, 13, 1) → 输出: (18, 13, 4)
        # ────────────────────────────────────────
        layers.SeparableConv2D(
            filters=4,                # 极少通道，防过拟合
            kernel_size=(3, 3),       # 3x3 捕捉局部时频上下文
            activation='relu',
            padding='same',           # 保持时间/频率维度不变
            input_shape=input_shape
        ),

        # ────────────────────────────────────────
        # 第2层：时间维降采样（保护频率细节！）
        # 原理：狗吠是时间上的突发信号，频率结构（13维）必须保留完整
        # 所以只在时间维压缩（18 → 9），频率维（13）不动
        # 输出: (9, 13, 4)
        # ────────────────────────────────────────
        layers.MaxPooling2D(pool_size=(2, 1)),  # (time↓, freq→)

        # ────────────────────────────────────────
        # 第3层：进一步抽象特征
        # 输出: (9, 13, 8)
        # ────────────────────────────────────────
        layers.SeparableConv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ),

        # ────────────────────────────────────────
        # 第4层：全局平均池化（GAP）— TinyML 核心技巧！
        # 原理：对每个通道求均值 → (8,)
        # 优势：
        #   - 替代 Flatten + 大 Dense，参数从数百降至个位数
        #   - 对输入长度微小变化鲁棒（适合滑窗）
        #   - TFLM 高效支持
        # ────────────────────────────────────────
        layers.GlobalAveragePooling2D(),

        # ────────────────────────────────────────
        # 第5层：二分类输出
        # 使用 sigmoid（非 softmax），因为：
        #   - 二分类只需 1 个输出节点
        #   - 可设阈值（如 0.7）灵活控制灵敏度
        # 注意：TFLite Micro int8 会将 sigmoid 近似为查表或分段线性
        # ────────────────────────────────────────
        layers.Dense(1, activation='sigmoid')   # # 二分类标签必须是：必须是 0/1 整数  输出label是1的概率
    ])

    # 编译（仅训练用，TFLite 导出时不依赖）
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',     # 二分类用 binary，非 sparse_categorical
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model





# 测试模型能够兼容tflite, 算子是否都兼容
if __name__ == '__main__':
    import tensorflow as tf
    # 创建模型
    model = build_dog_bark_tiny_model(input_shape=(18, 13, 1))
    model.summary()

    # 随机输入测试
    import numpy as np
    dummy_input = np.random.rand(1, 18, 13, 1).astype(np.float32)
    dummy_output = model.predict(dummy_input)
    print("输出 shape:", dummy_output.shape, "值示例:", dummy_output)

    # 导出为 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 只测试算子兼容性，先不量化, 因为量化需要真实数据
    tflite_model = converter.convert()

    # 保存
    with open("dog_bark_tiny_fp32.tflite", "wb") as f:
        f.write(tflite_model)

    print("TFLite 模型导出成功 (FP32)")



"""
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ separable_conv2d (SeparableConv2D)   │ (None, 18, 13, 4)           │              17 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 9, 13, 4)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ separable_conv2d_1 (SeparableConv2D) │ (None, 9, 13, 8)            │              76 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_average_pooling2d             │ (None, 8)                   │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │               9 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 102 (408.00 B)
 Trainable params: 102 (408.00 B)
 Non-trainable params: 0 (0.00 B)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1762934221.906686  142805 service.cc:152] XLA service 0x7fe7580040b0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1762934221.906707  142805 service.cc:160]   StreamExecutor device (0): NVIDIA RTX A4000, Compute Capability 8.6
2025-11-12 15:57:01.925150: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:269] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
I0000 00:00:1762934221.955701  142805 cuda_dnn.cc:529] Loaded cuDNN version 90501
I0000 00:00:1762934222.489297  142805 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 652ms/step
输出 shape: (1, 1) 值示例: [[0.50030047]]
Saved artifact at '/tmp/tmpmx1m6wq_'. The following endpoints are available:

* Endpoint 'serve'
  args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 18, 13, 1), dtype=tf.float32, name='keras_tensor')
Output Type:
  TensorSpec(shape=(None, 1), dtype=tf.float32, name=None)
Captures:
  140637825060528: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140637825059296: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140637825135360: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140637825138000: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140637825138352: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140637825137648: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140637824423536: TensorSpec(shape=(), dtype=tf.resource, name=None)
  140637824423360: TensorSpec(shape=(), dtype=tf.resource, name=None)
W0000 00:00:1762934222.648880  142722 tf_tfl_flatbuffer_helpers.cc:365] Ignored output_format.
W0000 00:00:1762934222.648892  142722 tf_tfl_flatbuffer_helpers.cc:368] Ignored drop_control_dependency.
2025-11-12 15:57:02.649074: I tensorflow/cc/saved_model/reader.cc:83] Reading SavedModel from: /tmp/tmpmx1m6wq_
2025-11-12 15:57:02.649384: I tensorflow/cc/saved_model/reader.cc:52] Reading meta graph with tags { serve }
2025-11-12 15:57:02.649388: I tensorflow/cc/saved_model/reader.cc:147] Reading SavedModel debug info (if present) from: /tmp/tmpmx1m6wq_
I0000 00:00:1762934222.652252  142722 mlir_graph_optimization_pass.cc:425] MLIR V1 optimization pass is not enabled
2025-11-12 15:57:02.652675: I tensorflow/cc/saved_model/loader.cc:236] Restoring SavedModel bundle.
2025-11-12 15:57:02.668640: I tensorflow/cc/saved_model/loader.cc:220] Running initialization op on SavedModel bundle at path: /tmp/tmpmx1m6wq_
2025-11-12 15:57:02.672985: I tensorflow/cc/saved_model/loader.cc:471] SavedModel load for tags { serve }; Status: success: OK. Took 23912 microseconds.
TFLite 模型导出成功 (FP32)
(yolo) kend@kend-Guanxin:~/文档/PlatformIO/Projects/DogStopper/python/tinyml/model$ 

"""

