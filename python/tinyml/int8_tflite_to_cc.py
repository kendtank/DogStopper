# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/14 16:40
@Author  : Kend
@FileName: int8_tflite_to_cc.py
@Software: PyCharm
@modifier:
"""

"""
完成量化脚本测试的int模型和完成test_int8_model_by_wav测试的int模型，运行这个int8模型转cc数组的脚本，得到cc数组，嵌入在cpp工程中
"""

import re
import os

def tflite_to_cc(tflite_path, cc_path):
    with open(tflite_path, 'rb') as f:
        data = f.read()

    # 从文件名生成合法C变量名（只保留字母、数字、下划线）
    base_name = os.path.basename(tflite_path)  # 如 "quantized_model_mfcc_int8.tflite"
    var_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)  # 替换所有非法字符为下划线
    var_name = re.sub(r'_+', '_', var_name)               # 合并多个下划线
    var_name = var_name.strip('_')                        # 去掉首尾下划线
    if var_name[0].isdigit():
        var_name = 'model_' + var_name                    # 确保不以数字开头

    with open(cc_path, 'w') as f:
        f.write(f'// Generated from: {tflite_path}\n')
        f.write(f'const unsigned char {var_name}[] __attribute__((aligned(8))) = {{\n')
        # 每行16个字节，提升可读性
        hex_values = [f'0x{b:02x}' for b in data]
        for i in range(0, len(hex_values), 16):
            f.write('  ' + ', '.join(hex_values[i:i+16]) + ',\n')
        f.write('};\n\n')
        f.write(f'const unsigned int {var_name}_len = {len(data)};\n')

    print(f"Successfully generated: {cc_path}")
    print(f"C symbol name: {var_name}")
    print(f"Model size: {len(data)} bytes")

if __name__ == "__main__":
    tflite_to_cc(
        "/home/kend/文档/PlatformIO/Projects/DogStopper/python/tinyml/train/quantized_model_mfcc_int8.tflite",
        "mfcc_int8_model.cc"
    )