# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/13 14:12
@Author  : Kend
@FileName: split_dataset
@Software: PyCharm
@modifier:
"""



"""
训练模型的第一步

对狗吠数据集进行划分

用法：
python split_dataset.py --data_root "/home/kend/Guanxin/Datasets/dataset/classes/train_brak_1113" --train_ratio 0.7
输出：
train.txt  包含了音频路径和文件标签类别
val.txt
"""


import os
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="数据根目录，如 ./train_brak_1113")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例 (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    # 获取所有文件
    dog_files = [str(p) for p in Path(args.data_root).glob("dog_bark/*.wav")]   # 狗叫文件
    no_dog_files = [str(p) for p in Path(args.data_root).glob("no_bark/*.wav")]   # 非狗叫文件

    print(f"狗叫样本: {len(dog_files)}")
    print(f"非狗叫样本: {len(no_dog_files)}")

    # 分别打乱并划分
    random.shuffle(dog_files)
    random.shuffle(no_dog_files)

    # 按照类别划分训练集的样本数
    n_train_dog = int(len(dog_files) * args.train_ratio)
    n_train_no = int(len(no_dog_files) * args.train_ratio)

    # 训练集
    train_files = [(f, 1) for f in dog_files[:n_train_dog]] + \
                  [(f, 0) for f in no_dog_files[:n_train_no]]

    # 验证集
    val_files = [(f, 1) for f in dog_files[n_train_dog:]] + \
                [(f, 0) for f in no_dog_files[n_train_no:]]

    # 打乱训练/验证列表
    random.shuffle(train_files)
    random.shuffle(val_files)


    # 保存
    def save_list(file_list, path):
        with open(path, 'w') as f:
            for wav_path, label in file_list:
                f.write(f"{wav_path} {label}\n")
        print(f"已保存 {len(file_list)} 行到 {path}")

    save_list(train_files, "train.txt")
    save_list(val_files, "val.txt")


if __name__ == "__main__":
    main()


"""
狗叫样本: 445
非狗叫样本: 1008
已保存 1016 行到 train.txt
已保存 437 行到 val.txt

"""