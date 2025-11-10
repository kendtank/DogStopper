import re
import numpy as np
from scipy.fftpack import fft

# =========================================================
# 1️⃣ 从 C 头文件中读取浮点数组
# =========================================================
header_file = "signal_data.h"  # 修改为你的路径
with open(header_file, "r") as f:
    content = f.read()

# 提取花括号里的数值
match = re.search(r"\{([^}]*)\}", content, re.S)
if not match:
    raise ValueError("❌ 未找到数组数据段")

# 清理字符串，去掉 f、空格、换行
data_str = match.group(1)
data_str = data_str.replace("f", "").replace("F", "").replace("\n", " ").replace("\r", " ")

# 转成 float 数组
nums = [float(x) for x in data_str.split(",") if x.strip()]
x = np.array(nums, dtype=np.float32)

print(f"✅ 成功加载 {len(x)} 个样本")

# =========================================================
# 2️⃣ 执行 FFT（scipy.fftpack，最接近 ESP-DSP）
# =========================================================
N = len(x)
X = fft(x)

# =========================================================
# 3️⃣ 打印完整 64 点复数结果
# =========================================================
print("\n=== FFT 输出（全部 33 点） ===")
for i in range(int((N / 2) + 1)):
    re_part = X[i].real  # 频率分量的实部
    im_part = X[i].imag  # 频率分量的虚部
    # mag = np.sqrt(re_part**2 + im_part**2)  # 幅度（magnitude）
    mag = np.abs(X[i]) ** 2  # 功率谱（magnitude squared）
    """
    re_part：表示该频率点的 余弦分量（cos部分，对应信号的相位为0°）
    im_part：表示该频率点的 正弦分量（sin部分，对应信号的相位为90°）
    mag：表示该频率点的 幅度大小（能量强度）
    Mel 特征提取（Log-Mel、MFCC）只关心 频率能量分布，不关心相位。
    因此：
    最终需要的值是 mag 或功率谱 mag**2
    """
    print(f"py Bin {i:02d}: Re={re_part:+.8f}, Im={im_part:+.8f}, |X|={mag:.8f}")
    # 结果：
    # mcu端能对齐，现在需要对3200个点进行
