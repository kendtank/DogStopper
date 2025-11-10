import numpy as np

N = 64
fs = 64  # 采样率 = 点数，方便正弦整周期
t = np.arange(N) / fs

# 合成两个频率分量：1Hz + 3Hz
x = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*3*t)

print("float test_input_signal[64] = {")
for i in range(N):
    end = "," if i < N-1 else ""
    print(f"    {x[i]:.9f}f{end}")
print("};")
