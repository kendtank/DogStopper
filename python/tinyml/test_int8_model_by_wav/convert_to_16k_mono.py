# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/14 17:33
@Author  : Kend
@FileName: convert_to_16k_mono
@Software: PyCharm
@modifier:
"""

"""
将任意音频文件（mp3/wav/m4a等）转换为 16kHz、单声道、16-bit PCM WAV
用法:
    python convert_to_16k_mono.py dog_in_home_001.mp3 dog_in_home_001.wav
"""

import sys
import os
from pydub import AudioSegment


def convert_to_16k_mono(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"输入文件不存在: {input_path}")
        sys.exit(1)

    try:
        # 自动根据扩展名加载音频（支持 mp3, wav, m4a, ogg, flac...）
        audio = AudioSegment.from_file(input_path)

        # 转换为单声道 + 16kHz
        audio = audio.set_channels(1).set_frame_rate(16000)

        # 导出为 16-bit PCM WAV
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])

        duration_sec = len(audio) / 1000.0
        print(f"转换成功!")
        print(f"   输入: {input_path}")
        print(f"   输出: {output_path}")
        print(f"   时长: {duration_sec:.2f} 秒 | 采样率: 16000 Hz | 声道: 1 (mono)")

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_to_16k_mono(input_file, output_file)