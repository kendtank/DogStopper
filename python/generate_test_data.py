"""
生成测试数据并保存到C语言头文件中
"""

import numpy as np

def generate_sine_wave():
    """生成500Hz正弦波测试信号"""
    # 参数设置
    sr = 16000        # 采样率 16kHz
    duration = 0.2    # 持续时间 200ms
    frequency = 500   # 频率 500Hz
    amplitude = 0.3   # 幅度 0.3
    
    # 生成测试信号 (3200个采样点)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    test_signal = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return test_signal

def save_to_header_file(signal, filename):
    """将信号保存到C语言头文件"""
    with open(filename, "w") as f:
        f.write("/*\n")
        f.write(" * 测试数据头文件\n")
        f.write(" * 自动生成的正弦波测试信号\n")
        f.write(" */\n\n")
        f.write("#ifndef TEST_DATA_H\n")
        f.write("#define TEST_DATA_H\n\n")
        f.write(f"#define TEST_SIGNAL_LENGTH {len(signal)}\n\n")
        f.write("static const float test_input_signal[TEST_SIGNAL_LENGTH] = {\n")
        
        # 每行写入8个值
        for i in range(0, len(signal), 8):
            values = signal[i:i+8]
            line = ", ".join([f"{val:.6f}f" for val in values])
            f.write(f"    {line}")
            if i + 8 < len(signal):
                f.write(",\n")
            else:
                f.write("\n")
        
        f.write("};\n\n")
        f.write("#endif // TEST_DATA_H\n")
    
    print(f"测试数据已保存到 {filename}")

def main():
    # 生成正弦波测试信号
    test_signal = generate_sine_wave()
    
    # 显示一些统计信息
    print("生成测试数据:")
    print(f"采样率: 16000 Hz")
    print(f"持续时间: 0.2 秒")
    print(f"频率: 500 Hz")
    print(f"幅度: 0.3")
    print(f"采样点数: {len(test_signal)}")
    print()
    
    # 显示前10个采样点
    print("前10个采样点:")
    for i in range(10):
        print(f"  [{i}]: {test_signal[i]:.6f}")
    print()
    
    # 保存到头文件
    save_to_header_file(test_signal, "test_data.h")
    
    print("完成!")

if __name__ == "__main__":
    main()