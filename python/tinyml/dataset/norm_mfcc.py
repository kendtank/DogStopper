# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/13 16:31
@Author  : Kend
@FileName: norm_mfcc
@Software: PyCharm
@modifier:
"""

"""
对提取的mfcc特征进行[-1, 1]的归一化
"""

import numpy as np
import os

# 默认参数路径
DEFAULT_NORM_PARAMS_PATH = "/home/kend/文档/PlatformIO/Projects/DogStopper/python/tinyml/dataset/norm_params.npy"    # 归一化mfcc特征的参数

_normalizer_instance = None
_normalizer_params_path = None


class MFCCNormalizer:
    """
    MFCC 归一化器，加载预计算的 Robust Min-Max 参数，
    将原始 MFCC 映射到 [-1, 1] 区间。
    
    用法：
        normalizer = MFCCNormalizer("norm_params.npy")
        norm_mfcc = normalizer(raw_mfcc)
    """

    def __init__(self, params_path=DEFAULT_NORM_PARAMS_PATH):
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"归一化参数文件未找到: {params_path}")
        
        params = np.load(params_path, allow_pickle=True).item()
        self.scale = float(params["scale"])
        self.offset = float(params["offset"])
        self.method = params.get("method", "robust_minmax")
        self.original_range = (params["original_low"], params["original_high"])
        self.target_range = (params["target_min"], params["target_max"])


    def __call__(self, mfcc_features):
        """
        归一化 MFCC 特征。

        Args:
            mfcc_features (np.ndarray): 原始 MFCC，shape 应为 (18, 13)

        Returns:
            np.ndarray: 归一化后的 MFCC，shape (18, 13)，值域 ≈ [-1, 1]
        """
        if not isinstance(mfcc_features, np.ndarray):
            mfcc_features = np.array(mfcc_features, dtype=np.float32)
        
        if mfcc_features.shape != (18, 13):
            raise ValueError(f"期望输入形状 (18, 13)，但得到 {mfcc_features.shape}")
        
        # 线性映射: x_norm = x_raw * scale + offset
        normalized = mfcc_features * self.scale + self.offset
        
        # 裁剪到目标范围（这里使用防御性编程）
        # 注意：由于使用 percentile(1%,99%)，约有 2% 的值会略超出 [-1,1]
        # 裁剪与否取决于模型鲁棒性，通常不裁剪也可，但裁剪更安全
        target_min, target_max = self.target_range
        normalized = np.clip(normalized, target_min, target_max)
        
        return normalized.astype(np.float32)


# 全局函数接口（方便直接调用）
def norm_mfcc(mfcc_features, params_path=DEFAULT_NORM_PARAMS_PATH):
    """
    输入：
        mfcc_features: 从 3200 点音频片段提取的 MFCC，形状 (18, 13)
    输出：
        归一化后的 MFCC，形状 (18, 13)，值在 [-1, 1] 附近
    """
    # 使用全局缓存的 normalizer，避免重复加载和实例化。
    global _normalizer_instance, _normalizer_params_path
    # 如果还没初始化，或者参数路径变了，就重新创建
    if _normalizer_instance is None or _normalizer_params_path != params_path:
        _normalizer_instance = MFCCNormalizer(params_path)
        _normalizer_params_path = params_path

    return _normalizer_instance(mfcc_features)


if __name__ == "__main__":
    # 模拟一个原始 MFCC
    raw_mfcc = np.random.uniform(-500, 100, (18, 13)).astype(np.float32)
    
    try:
        normed = norm_mfcc(raw_mfcc)
        normed2 = norm_mfcc(raw_mfcc)  # 第二次调用不会新建实例
        print("归一化成功！")
        print(f"原始范围: [{raw_mfcc.min():.2f}, {raw_mfcc.max():.2f}]")
        print(f"归一化后: [{normed.min():.3f}, {normed.max():.3f}]")
        print(f"输出形状: {normed.shape}")
    except Exception as e:
        print(f"错误: {e}")

"""
归一化成功！
原始范围: [-496.86, 94.73]
归一化后: [-1.000, 1.000]
输出形状: (18, 13)
"""