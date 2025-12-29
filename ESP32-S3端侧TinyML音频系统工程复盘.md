# ESP32-S3端侧TinyML音频系统工程复盘

> 本项目是一个基于 ESP32-S3 的端侧 MCU 音频智能系统实现，
> 聚焦实时音频流处理、VAD 事件建模、TinyML 推理链路、
> 声纹验证与自学习机制，以及嵌入式系统级架构解耦设计。


## 目录导航

- [ESP32-S3端侧TinyML音频系统工程复盘](#esp32-s3端侧tinyml音频系统工程复盘)
  - [目录导航](#目录导航)
  - [一、基本信息](#一基本信息)
  - [二、遇到的问题](#二遇到的问题)
    - [1. 串口打印无输出](#1-串口打印无输出)
    - [2. 板子无法启用PSRAM内存](#2-板子无法启用psram内存)
    - [3. platformio无法使用TensorflowLite for MicroController插件](#3-platformio无法使用tensorflowlite-for-microcontroller插件)
    - [4. 快速傅里叶变换(FFT)计算运行速度很慢](#4-快速傅里叶变换fft计算运行速度很慢)
    - [5. 使用ESP的dsp库加速FFT计算, 遇到空指针错误](#5-使用esp的dsp库加速fft计算-遇到空指针错误)
    - [6. 使用unity进行单元测试遇到的问题](#6-使用unity进行单元测试遇到的问题)
    - [7. 使用platformio进行debug遇到问题](#7-使用platformio进行debug遇到问题)
    - [8. c语言实现的log-mel算法的提取和python的实现保持算法功能上的一致性问题(难点)](#8-c语言实现的log-mel算法的提取和python的实现保持算法功能上的一致性问题难点)
    - [9. mfcc特征值和python的实现保持一致性的问题](#9-mfcc特征值和python的实现保持一致性的问题)
    - [10. 自定义实现的mfcc特征值的归一化问题(重点)](#10-自定义实现的mfcc特征值的归一化问题重点)
    - [11. tensorflow-lite模型的设计与训练策略以及导出cc数组的步骤和注意事项](#11-tensorflow-lite模型的设计与训练策略以及导出cc数组的步骤和注意事项)
    - [12. tensorflow-lite模型的量化以及部署mcu中保证推理结果误差不超过5%的注意事项和步骤(难点)](#12-tensorflow-lite模型的量化以及部署mcu中保证推理结果误差不超过5的注意事项和步骤难点)
    - [13. 产品的架构设计和模块间的解耦设计（重点）](#13-产品的架构设计和模块间的解耦设计重点)
      - [13.1 整体数据流](#131-整体数据流)
      - [13.2 音频采集模块（Audio Producer）](#132-音频采集模块audio-producer)
      - [13.3 VAD 模块（滑窗消费者）](#133-vad-模块滑窗消费者)
      - [13.4 TinyML 推理模块（事件级推理）](#134-tinyml-推理模块事件级推理)
      - [13.5 Bark 事件消费者（声纹验证模块）](#135-bark-事件消费者声纹验证模块)
      - [13.6 模块解耦设计总结](#136-模块解耦设计总结)
    - [14. VAD 初筛算法的实现和音频流滑窗架构设计（难点）](#14-vad-初筛算法的实现和音频流滑窗架构设计难点)
      - [14.1 音频 Block 与滑窗参数设计](#141-音频-block-与滑窗参数设计)
      - [14.2 滑窗内部数据流与缓冲区关系](#142-滑窗内部数据流与缓冲区关系)
      - [14.3 VAD 状态机设计](#143-vad-状态机设计)
      - [14.4 事件拼接与去重策略](#144-事件拼接与去重策略)
      - [14.5 事件终止与保护机制](#145-事件终止与保护机制)
      - [14.6 设计总结](#146-设计总结)
    - [15. 声纹验证及自学习：自动建立声纹模版与 Flash 持久化（难点）](#15-声纹验证及自学习自动建立声纹模版与-flash-持久化难点)
      - [15.1 模块职责划分](#151-模块职责划分)
      - [15.2 数据语义约定（核心设计原则）](#152-数据语义约定核心设计原则)
      - [15.3 声纹自学习流程（Batch 驱动）](#153-声纹自学习流程batch-驱动)
      - [15.4 Batch 内部一致性聚类策略](#154-batch-内部一致性聚类策略)
      - [15.5 模版更新与冻结机制](#155-模版更新与冻结机制)
      - [15.6 声纹验证逻辑（与学习完全解耦）](#156-声纹验证逻辑与学习完全解耦)
      - [15.7 状态持久化与系统鲁棒性](#157-状态持久化与系统鲁棒性)
      - [15.8 为什么这个模块一开始最容易写乱，以及我是如何把它拆清楚的](#158-为什么这个模块一开始最容易写乱以及我是如何把它拆清楚的)
    - [16. 算法误判率减少的方法和整体的性能优化(重点)\*\*](#16-算法误判率减少的方法和整体的性能优化重点)
    - [17. 程序部署在mcu后，仍然存在的优化点](#17-程序部署在mcu后仍然存在的优化点)


## 一、基本信息
- 系统：ubuntu22.04
- 芯片型号：ESP32-S3
- 开发板：ESP32-S3-DevKitC-1-N16R8
- IDE/工具：VSCode
- 框架：Arduino

## 二、遇到的问题
### 1. 串口打印无输出
   - 问题描述：platformio连接开发板, 调用serial.print()打印数据，串口监测不到数据
   - 解决方法：在platformio.ini中添加如下代码：
   ```
   upload_speed = 921600
   ```
   并且在main.cpp中添加如下代码：
   Serial.begin(115200) // 串口打印
   如果编译烧录成功之后，还是没有数据打印，就进行下面的步骤
   上传镜像之后打开串口监控的控制台，再按下reset按键，就会显示打印的数据


### 2. 板子无法启用PSRAM内存
   - 问题描述：开发板选择[Espressif ESP32-S3-DevKitC-1-N8 (8 MB QD, No PSRAM)]之后无法启用PSRAM内存
   - 解决方法：由于板子是N16R8型号，platformio板子中没有这个型号，只能使用[Espressif ESP32-S3-DevKitC-1-N8 (8 MB QD, No PSRAM)]， 然后再platformio.ini中添加如下配置：
   ```
   platform = espressif32
   board    = esp32-s3-devkitc-1    ; 选通用 S3 DevKitC
   framework= arduino
   ; --- Flash & PSRAM 接口配置 ---
   board_build.flash_mode    = qio       ;  指定FLASH和PSRAM的运行模式 Flash 用 QIO
   board_build.arduino.memory_type = qio_opi  ; Flash QIO + PSRAM Octal
   board_build.psram_type    = opi       ; PSRAM 用 Octal mode
   ; --- 大小定义 ---
   board_upload.flash_size   = 16MB
   board_upload.maximum_size = 16777216  ; 一定要和 16MB 对齐
   board_build.partitions    = default_16MB.csv  ; 用 16MB 分区表
   ; 指定为16MB的FLASH分区表
   board_build.arduino.partitions = default_16MB.csv
   ; --- 启用 PSRAM 宏 ---
   board_build.extra_flags   = -DBOARD_HAS_PSRAM
   ```
  配置完成之后，重新拔插板子，再编译测试是否可以申请PSRAM的代码进行测试。
  

### 3. platformio无法使用TensorflowLite for MicroController插件
   - 问题描述：platformio无法使用TFLM插件
   - 解决方法：添加官方的tflm git仓库运行测试，会提示缺少很多的第三方库，一般是flatbuffers以及别的库，手动去下载很麻烦，但是Arduino IDE是有官方的ESP32_TFLM插件的全称[TensorFlowLite_ESP32], 所以可以尝试使用Arduino IDE下载插件，再把源码拷贝到lib目录中，引入如下一行代码：
   ```
   #include <TensorFlowLite_ESP32.h>
   ```
   添加完之后，就可以使用TFLM插件了
   注意：正常的tf-lite的代码还是需要自己引入的, 如果需要使用硬件级别的加速tflm, 需要使用esp官方的TFLM插件且搭配esp-idf工具使用，我目前使用的库中没有使用硬件加速，只能使用软件加速， 因为速度已经满足了。

### 4. 快速傅里叶变换(FFT)计算运行速度很慢
   - 问题描述：手写FFT算法，因为需要大量的复数运算，导致运行速度很慢(测试10s 16k的音频需要70s左右)
   - 解决方法：采用ESP的dsp库进行加速，采用dsps_fft2r进行快速傅里叶变换，速度大大提升， 可以详细查看[ESP32-DSP-Library](https://github.com/espressif/esp-dsp)。

### 5. 使用ESP的dsp库加速FFT计算, 遇到空指针错误
   - 问题描述：使用dsps_fft2r_fc32(fft_buf, N_FFT);进行快速傅里叶变换，但是运行过程中会报空指针错误
   - 解决方法：在使用dsps_fft2r_fc32()之前，必须调用dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);进行初始化查看，否则在申请内存空间时会报空指针错误

### 6. 使用unity进行单元测试遇到的问题
   - 问题描述：使用unity进行单元测试，遇到了很多的问题
   - 解决方法：platformio创建项目会生成一个test 目录，里面就是为了存放了unity的测试代码，在test目录下创建一个测试的cpp文件，按照main.cpp的格式进行编码，运行项目会报错，找不到自己写的主功能代码，因为头文件默认是指定的include目录，但是unity测试默认不编译src目录下的代码。所以需要在platformio.ini中指定：
   ```
   test_framework = unity
   test_build_project_src = yes  
   ```
   unity测试默认不编译src目录下的代码，启用此选项以编译src目录下的代码， 但是src中有main.cpp, 需要屏蔽其中的setup和loop函数, 这样才能进行烧录测试。否则会报错找到两个setup和loop函数。

### 7. 使用platformio进行debug遇到问题
   - 问题描述：使用platformio进行debug报异常
   - 解决方法：配置platformio.ini文件，添加如下代码：
   ```
   debug_tool = esp-builtin
   debug_init_break = tbreak setup
   build_type = debug
   ```
   添加了esp-builtin调试工具，添加了debug_init_break，指定了在setup函数开始时断点，添加了build_type = debug，指定了debug模式，最后打开终端执行：
   ```
   curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules | sudo tee /etc/udev/rules.d/99-platformio-udev.rules
   sudo service udev restart
   ```
   重新拔插板子，这样platformio就会在setup函数开始时断点，然后进行debug。使用vscode进行正常的板子层级别的debug功能, 想要使用更高层级的debug功能需要使用esp-idf工具进行debug。


### 8. c语言实现的log-mel算法的提取和python的实现保持算法功能上的一致性问题(难点)

  - 问题描述：c语言实现的log-mel算法的提取和python的实现保持算法功能一致性的问题，导致提取的特征值不一致，影响后续的模型推理结果。
  - 解决方法：
    1. 确定算法的实现逻辑，按照算法步骤，分布debug操作。python实现和mcu中的c语言实现，找到不一致的地方，进行修改。确保python算法和mcu算法功能一致。
    2. 确定输入一致，随机生成500Hz正弦波测试信号， 截断为200ms, 3200点采样点，进行测试，数组输入一致(pass)
    3. 生成汉宁窗函数，确定汉宁窗权值一致， 误差不超过：1e-6f(pass)
    4. 对音频进行分帧，对每一帧进行汉宁窗加权计算(对每个窗口内的采样点进行汉宁系数的加权)，确定python端和c端计算结果一致，误差不超过：1e-6f(pass)
    5. 采用ESP32的dsp库进行快速傅里叶变换，确定python端和c端功率谱总能量误差不超过0.1%(pass)
    6. 如下步骤：
    ``` python
     x = frames_win.astype(np.float32)  # 汉宁窗后的音频数据
     n_fft = 512                          # 功率谱的点数
     num_bins = n_fft // 2 + 1            # 功率谱的频谱点数

     # scipy.fftpack.fft → 对齐 ESP-DSP 的 dsps_fft2r_fc32
     X = fft(x, n=n_fft, axis=0)  # 默认 scipy.fftpack.fft(x) 会把整个二维数组当作 扁平数组 做 FFT（按行展开）。
     # 功率谱（对齐 MCU：只取前 N/2+1 个 bin）
     powspec = np.abs(X[:num_bins, :])**2  # Power Spectrum    p.abs(X)**2 是功率 # np.abs(X) 就是幅度 |X|
     print(f"powspec shape: {powspec.shape}")
     powspec = powspec.astype(np.float32)

     # mcu是一维数组，要比较数据，需要转置
     save_to_header(powspec.T.flatten(), "powspec", "out/powspec.h")
    ```

    ``` CPP
     int fft_power_compute(const float *frames_in, int num_frames, int frame_size, int nfft, int logmel_flag,  float *power_out)
     {
        int num_bins = nfft / 2 + 1;  // 功率谱点数， 只需要前半部分
        // 循环多帧
        for (int f = 0; f < num_frames; f++) {

           // 拿到当前帧指针
           const float *frame = frames_in + f * frame_size;
           float *out_frame = power_out + f * num_bins;
           // 更新12-12：根据logmel_flag选择不同的输入缓冲区
           if (logmel_flag == 0) {
                 // 优化：1204
                 // === Step 1: memset直接把向量置为0，因为虚部是0，后面直接处理实部 ===
                 memset(fft_input_mfcc, 0, sizeof(float) * 2 * nfft);
                 // 把 frame 拷贝到偶数位（real），多余的补0
                 for (int i = 0; i < frame_size; ++i) {
                    fft_input_mfcc[2 * i] = frame[i]; // real
                    // fft_input[2 * i + 1] already zeroed
                 }
                 // 注意： nfft， 短时傅里叶变换的长度， 一定是我们设置的长度，不能取输出的功率长度， 即使输入只有前 400 个有效样本、后 112 个补零。
                 // === Step 2: 执行 FFT ===
                 dsps_fft2r_fc32(fft_input_mfcc, nfft);    // 实数FFT 自动使用加速版	
                 dsps_bit_rev_fc32(fft_input_mfcc, nfft);  // 位逆序调整   自动选择了加速版
                 // === Step 3: 对齐 scipy.fftpack.fft 输出格式 （前 N/2+1 个 bin） ===
                 // scipy.fftpack.fft(x, NFFT)[:NFFT//2 + 1]  
                 for (int k = 0; k < num_bins; k++) {
                    float re = fft_input_mfcc[2 * k];    // 实部
                    float im = fft_input_mfcc[2 * k + 1];  // 虚部
                    out_frame[k] = re * re + im * im; // 功率谱值
                 }
           }
           // 这一帧处理完成... 下一帧开始
        }
        return 0;
     }

    ```

    注意：计算出来的功率谱每个 bin 是完全一致的(误差在1e-4), 为了对齐这个功率谱算法， 作者在mcu中debug了一周的时间 ，对齐好之后后续处理（Mel 滤波器、log-Mel、MFCC）就不会出问题。
    总能量对齐(误差在1e-3以内): C 端总能量 ≈ Python 总能量, 这样经过 Mel 滤波器和 log-Mel 后，特征值差异很小，可以用于训练或推理, 换句话说，mcu中实现的 MFCC / log-Mel 特征提取必须和 Python 保持对齐。这是模型在mcu中部署的关键前提。

### 9. mfcc特征值和python的实现保持一致性的问题
   - 问题描述：mfcc特征值和python的实现保持一致性的问题，导致提取的特征值不一致，影响后续的模型推理结果。
   - 解决方法：
     1. 确定输入一致(功率谱)，(pass)
     2. 确定mel滤波器权值一致， py端和c端实现的mel滤波器矩阵权值单点误差不超过1e-6(pass)， 后续可以固定使用，避免多次开销。
     3. 确定logmel计算逻辑一致，(pass)(重点)
      在确实第一步和第二步骤正确的前提下，进行mel特征值计算，并和python的实现保持一致。（pass）
      在py端使用librosa默认计算 log10(mel_energy)，然后常常还乘上 10（单位是 dB）转为db单位，但是在mcu端无法达到每个bin保持相同的 db 单位。librosa的内部实现不清楚。（fail）
      之后转为使用numpy实现
      ```python
      amin = 1e-8      # 最小功率值，避免 log(0)
      top_db = 100.0   # 最大动态范围
      # 计算 dB
      S_db = 10.0 * np.log10(np.maximum(mel_power.astype(np.float32), amin))
      # top_db 限制
      S_db = np.maximum(S_db, S_db.max() - top_db)
      Sdb_py = S_db.astype(np.float32)
      save_to_header(Sdb_py.T.flatten(), "frame_mel_db", "out/frame_mel_db.h")
      ```

      ``` CPP
      void apply_log_mel(const float* power_spectrum, float* logmel_out) {
         const float amin = 1e-8f;
         float max_db = -INFINITY;

         for (int i = 0; i < MEL_BANDS; i++) {
            float sum = 0.0f;
            for (int k = 0; k < TOP_K; k++) {
                  int idx = sparse_indices[i * TOP_K + k];     // 稀疏矩阵的索引
                  sum += sparse_filters[i * TOP_K + k] * power_spectrum[idx];    // 稀疏矩阵的权值*功率谱值
            }
            if (sum < amin) sum = amin;
            float db = 10.0f * logf(sum) * 0.43429448f;
            logmel_out[i] = db;
            if (db > max_db) max_db = db;
         }
         float min_db = max_db - 100.0f;
         for (int i = 0; i < MEL_BANDS; i++) {
            if (logmel_out[i] < min_db) logmel_out[i] = min_db;
            if (logmel_out[i] < -79.9f) logmel_out[i] = -80.0f;
         }
      }
      ```
      测试结果：mcu和python的实现一致,误差在0.01, (pass)  误差放大出现在 log 步骤（且仅在 x 极小（比如 <1e-6）时。所以做了处理，在接近-80.0f 时，会变成 -80.0f)


### 10. 自定义实现的mfcc特征值的归一化问题(重点)
   详情查看[为什么mfcc特征需要归一化](python/tinyml/readme.md), 文档中的困惑点有详细说明。

### 11. tensorflow-lite模型的设计与训练策略以及导出cc数组的步骤和注意事项
   详情查看[tinyml mfcc 模型的设计到量化转换为字节cc数组移植到mcu的闭环流程](python/tinyml/readme.md)

### 12. tensorflow-lite模型的量化以及部署mcu中保证推理结果误差不超过5%的注意事项和步骤(难点)
   1. 问题描述：模型训练量化得到cc数组之后，在mcu中实现需要掌握一些细腻的步骤
   2. 解决方法：
   ```cpp
   // -----------------------------------------
   // MFCC 推理函数
   // input: float MFCC 输入特征
   // return: 狗吠概率 float
   // -----------------------------------------
   float mfcc_model_infer(const float* input_float) {
      // -----------------------------------------
      // 1. 输入归一化 + 量化 (float -> int8)
      // -----------------------------------------
      int8_t input_int8[MFCC_INPUT_SIZE];

      for (size_t i = 0; i < MFCC_INPUT_SIZE; i++) {

         // 归一化 先做缩放再做偏移
         float norm = input_float[i] * NORM_SCALE + NORM_OFFSET; // 权重由数据集得到

         // 截断
         if (norm > NORM_TARGET_MAX) norm = NORM_TARGET_MAX;
         if (norm < NORM_TARGET_MIN) norm = NORM_TARGET_MIN;

         // 输入量化到 int8
         int32_t q = (int32_t)(norm / mfcc_in_scale + mfcc_in_zero);  
         if (q > 127) q = 127;
         if (q < -128) q = -128;
         input_int8[i] = (int8_t)q;
      }

      // -----------------------------------------
      // 2. 拷贝到 TFLite Micro 输入张量
      // -----------------------------------------
      memcpy(mfcc_input_tensor->data.int8, input_int8, MFCC_INPUT_SIZE);

      // -----------------------------------------
      // 3. 推理 // 核心函数：将输入传入模型进行前向计算，结果放到输出张量
      // -----------------------------------------
      mfcc_interpreter->Invoke();    //  初始化加上static不然报空指针

      // -----------------------------------------
      // 4. 获取输出并反量化 (int8 -> float)
      // -----------------------------------------
      // 获取输出张量的 int8 值
      int8_t q_out = mfcc_output_tensor->data.int8[0];
      // 反量化 输出狗吠的概率
      float prob = (q_out - mfcc_out_zero) * mfcc_out_scale;
      return prob;
   }
   ```
   整体的量化步骤如上代码。



### 13. 产品的架构设计和模块间的解耦设计（重点）

本产品采用多级 Producer–Consumer 流水线架构，将实时音频处理拆分为四个相互解耦的模块。模块之间仅通过 buffer / queue 传递数据，不直接调用彼此逻辑，从而保证系统的实时性、稳定性与可扩展性。
最初的架构设计详情查看[MCU实时音频处理架构设计文档](MCU实时音频处理架构设计文档.md)

#### 13.1 整体数据流

Audio Producer → PCM Ring Buffer → VAD Consumer → VAD / Bark Tinyml Queue → TinyML Inference → Bark Event Queue → Bark Verification（声纹验证）-> other modules


#### 13.2 音频采集模块（Audio Producer）
音频采集模块作为数据生产者，负责从麦克风持续采集 PCM 音频数据，并写入缓冲区。
- 采样率：16kHz，int16，单通道
- 以固定帧长（64 * 4 / 256 samples）写入 buffer  
- 不做任何语义判断，仅做简单的去DC偏移，保证稳定、连续的数据输入  
该模块不依赖任何下游模块，是整个系统的入口基准。

#### 13.3 VAD 模块（滑窗消费者）

VAD 模块从 PCM  buffer 中按滑窗方式消费音频数据，用于检测是否存在有效发声片段。

- 滑窗长度：256 samples * 10ms = 2560 samples
- 步长：256 samples * 3 = 768 samples
- 基于能量，过零率轻量级声学特征进行判断 
由于单次窗的VAD 计算耗时远小于生产时间，能够保证消费速度始终快于音频采集速度，仅在命中时生成 VAD / Bark Tinyml 事件并推入Bark Tinyml队列。


#### 13.4 TinyML 推理模块（事件级推理）

TinyML 推理模块以事件驱动方式工作，仅在接收到 VAD 事件时执行模型推理。

- 输入：VAD 事件携带的音频片段（约 150–500ms）  
- 对音频进行滑窗步长或填充以适配模型输入长度  
- 输出：狗吠概率 P(bark)  
- 后处理: 对输出的狗吠概率窗进行top-K的后处理，只输出高质量的候选片段到bark 事件队列  

该设计避免了持续流式推理，有效控制 MCU 端算力与功耗， 并有效的减少声纹模型的推理次数，减少事件延迟和功耗开销 。



#### 13.5 Bark 事件消费者（声纹验证模块）

声纹验证模块作为最终消费者，对 TinyML 判定为 bark 的候选事件进行二次确认。

- 提取音频声纹特征（log-mel embedding）  
- 基于相似度策略进行验证  
- 用于区分“非目标狗吠”与“目标狗吠”，降低误触发率  
- 采用自学习和聚类的策略进行模版的建立，使用ema机制和batch分批学习的策略，实现无监督学习， 使得产品越用越准确的特性。



#### 13.6 模块解耦设计总结

- 模块之间仅通过 buffer / queue 传递数据  
- 不共享状态，不跨模块调用  
- 每个模块只负责单一职责，可独立替换与优化  

该架构适用于 MCU / TinyML 场景，支持产品级稳定运行及后续算法持续演进。


### 14. VAD 初筛算法的实现和音频流滑窗架构设计（难点）

本模块用于在 MCU 端对连续 PCM 音频流进行实时粗筛，目标是在极低算力和有限内存条件下，以“高召回”为第一原则，捕获尽可能完整的狗吠候选事件，并为后续 TinyML 推理提供连续、无碎片的音频片段。
系统整体采用“Block 输入 + 重叠滑窗 + 事件缓存 + 状态机控制”的流式架构，避免单点判断和瞬时能量波动导致的漏检。



#### 14.1 音频 Block 与滑窗参数设计

音频采集模块以固定大小 Block 向 VAD 推送 PCM 数据：

- Block 大小：256 samples（约 16ms @16kHz）
- 滑窗大小：10 Blocks（2560 samples，约 160ms）
- 滑动步长：3 Blocks（768 samples，约 48ms）
- 窗口重叠：7 Blocks

滑窗以 Block 为单位构建，当窗口填满后执行一次判定，
随后窗口整体左移一个步长，并继续接收新的 Block 填充尾部。

这种重叠滑窗设计可以确保：
- 狗吠起始点不会被恰好切在窗口边缘
- 短时爆破音（120~500ms）至少被多个窗口覆盖
- 能量/ZCR 的统计更加稳定



#### 14.2 滑窗内部数据流与缓冲区关系

系统中涉及的关键缓冲区如下：

- window_buf：
  当前滑窗完整数据，用于计算短时能量与 ZCR

- prev_stride_buf：
  上一个窗口滑出的 stride（3 Blocks），
  用于在事件刚开始时作为前置 padding

- cur_stride_buf：
  当前窗口即将滑出的 stride，
  在事件结束或衔接时直接拼接到事件中

- event_buf：
  当前事件的累计缓冲区，只追加“新出现的数据”，避免重复

整体数据流逻辑为：

1. 生产者每次推送一个 Block
2. Block 被顺序拷贝进 window_buf 尾部
3. 当 window_buf 填满：
   - 计算能量与 ZCR
   - 结合 EMA 做平滑判定
   - 根据状态机决定是否拼接数据
4. 滑窗左移一个 stride：
   - 移出的数据保存为 cur_stride_buf
   - 其余数据前移
5. 等待新的 Block 进入，继续循环



#### 14.3 VAD 状态机设计
为适配连续音频流并减少碎片化，本模块采用显式状态机控制事件生命周期：

状态定义：

- IDLE：
  当前不在事件中，仅做滑窗检测

- ACTIVE（in_event = true）：
  已检测到有效发声，正在累计事件数据

状态转移规则：

- IDLE -> ACTIVE：
  当前滑窗判定为命中（高能量 + 高 ZCR）
  此时：
    - 将 prev_stride_buf 作为前置 padding
    - 拼接整个窗口数据
    - 进入事件累计状态

- ACTIVE -> ACTIVE：
  当前窗口继续命中
  仅拼接窗口中新出现的 stride 数据

- ACTIVE -> END：
  连续 EVENT_END_NOHIT 次窗口未命中
  认为事件结束，推送事件到 TinyML 队列

该状态机设计的核心目标是：
- 抵抗能量瞬时下降
- 避免事件被错误切断
- 保证事件整体连续性


#### 14.4 事件拼接与去重策略

事件拼接遵循“只追加新数据”的原则：

- 事件开始时：
  prev_stride_buf + 第一次命中的完整窗口

- 事件进行中：
  每个新窗口仅追加 cur_stride_buf

- 第一次非命中窗口：
  直接将 cur_stride_buf 拼接进事件中，
  作为事件尾部的自然衰减

- 若下一窗口重新命中：
  该 stride 视为事件连续部分，继续累计

- 若连续非命中达到阈值：
  事件结束并推送

这种设计取消了独立 post-padding 缓冲，
降低了内存占用，同时保持了事件边缘完整性。

#### 14.5 事件终止与保护机制

为避免异常情况，系统引入以下保护规则：
- 最大事件长度限制：
  当 event_buf 超过 EVENT_MAX_SAMPLES 时强制切断并推送
- 队列保护：
  TinyML 队列满时覆盖最旧事件，保证新事件优先
- 全流程无 malloc：
  所有缓冲区静态分配，避免内存碎片与泄漏


#### 14.6 设计总结
该 VAD 初筛模块通过滑窗重叠、状态机控制和事件实时拼接，
在 MCU 资源受限条件下实现了：
- 高召回率的狗吠捕获
- 连续、完整的事件输出
- 稳定、可控的实时数据流

该设计为后续 TinyML 推理提供了可靠的数据入口，同时保持了系统整体的实时性与鲁棒性。


### 15. 声纹验证及自学习：自动建立声纹模版与 Flash 持久化（难点）

本模块负责对 TinyML 初筛后的狗吠事件进行二次确认，
并在无需人工标注的前提下，逐步建立并优化目标狗的声纹模版。
整体设计遵循 **“先记录事实，再尝试学习，学习失败不影响系统运行”** 的原则，
保证在 MCU 资源受限和真实环境噪声复杂的条件下，系统行为稳定、可解释。
最初的设计思路详情查看：[Mcu端狗吠声纹自学习系统设计说明](Mcu端狗吠声纹自学习系统设计说明.MD)


#### 15.1 模块职责划分

声纹模块由三个完全解耦的子模块组成：

- **verify_embedding**
  - 负责编排整体流程
  - 调用 embedding 推理
  - 控制学习时机
  - 执行声纹验证并触发业务行为
  - 不持有任何长期状态

- **learning_core**
  - 管理声纹模版的生命周期
  - 负责 batch 级自学习与一致性校验
  - 执行 EMA 更新与冻结策略
  - 不涉及 Flash 读写细节

- **flash_storage**
  - 负责 embedding / template / state 的持久化
  - 只提供 append / load 接口
  - 不理解 batch、相似度或学习语义

模块之间仅通过明确的接口交互，不共享隐式状态。


#### 15.2 数据语义约定（核心设计原则）

为避免学习逻辑失控，本模块对不同数据的语义做了强约束：

- **embedding 是原始事实**
  - 每一次 Bark 事件生成的 embedding 都视为不可变观测
  - Flash 中的 embedding 只追加、不修改、不回滚
  - 学习失败不会删除历史 embedding

- **batch 是学习窗口，而非存储概念**
  - batch 仅用于聚合一段时间内的 embedding
  - batch 生命周期结束即被丢弃
  - batch 内 embedding 不参与跨 batch 的比较

- **模版是唯一长期语义状态**
  - 系统中始终只有一个 TemplateModel
  - 模版仅在 batch 成功时更新
  - 达到上限后进入冻结状态，不再漂移


#### 15.3 声纹自学习流程（Batch 驱动）

自学习仅在 learning 开启状态下，并且 batch 满足条件时触发：

1. embedding 持续 append 到 Flash
2. batch 满（EMBED_BATCH_SIZE）后触发学习
3. batch 内部聚类，构造 batch centroid
4. 与全局模版做一致性校验（动态阈值）
5. 成功则通过 EMA 更新模版
6. 失败则忽略该 batch，不影响系统运行
7. 达到最大 batch 数后冻结模版

学习过程只影响模版，不会反向修改任何 embedding 数据。


#### 15.4 Batch 内部一致性聚类策略

batch 内部聚类采用“中心一致性”策略：

- 先计算 batch embedding 的均值向量
- 计算每个 embedding 与均值的余弦相似度
- 选取 Top-K 相似 embedding 作为核心样本
- 对 Top-K 执行动态阈值校验
- 使用 Top-K 重新计算 batch centroid

该方法避免了 O(N²) 两两相似度计算，
同时对异常样本具备天然鲁棒性。



#### 15.5 模版更新与冻结机制

- 初次学习：
  - 直接使用 batch centroid 初始化模版

- 后续学习：
  - 采用 EMA 进行平滑更新
  - EMA 衰减系数随 batch_count 递减，防止后期漂移

- 冻结策略：
  - 当 batch_count 超过 MAX_BATCH_NUM
  - 模版进入 frozen 状态
  - 后续 embedding 仅用于验证，不再更新模版


#### 15.6 声纹验证逻辑（与学习完全解耦）

声纹验证与学习流程完全独立：

- 若模版不存在：
  - 返回特殊值，不做强惩罚

- 若模版存在：
  - 计算 embedding 与模版的余弦相似度
  - 超过阈值判定为目标狗吠
  - 否则视为非目标狗吠

验证阶段不关心 embedding 是否来自学习期，
也不依赖 batch 状态。



#### 15.7 状态持久化与系统鲁棒性

Flash 中持久化以下三类数据：

- embedding（事实日志）
- template（唯一长期语义）
- state（计数器与学习开关）

所有状态写入均为幂等操作，
支持断电恢复，不依赖 RAM 状态连续性。



#### 15.8 为什么这个模块一开始最容易写乱，以及我是如何把它拆清楚的

声纹验证与自学习模块是整个系统中最容易失控的部分，
根本原因不在算法复杂度，而在于**状态、时间与职责的高度耦合**。

最初实现时，常见混乱包括：
- embedding 生成、学习判断、模板更新、Flash 写入混在同一流程
- 学习失败是否回滚、是否删除历史数据语义不清
- batch 既像算法概念，又像存储结构
- 模版是否存在、是否可用、是否更新由多处逻辑隐式决定

最终的重构突破点在于**强制拆清三类角色**：

1. **事实层**
   - embedding 是只增不减的原始观测

2. **学习层**
   - 只在 batch 边界做判断
   - 学习失败不产生副作用

3. **语义层**
   - 模版是唯一长期状态
   - 其演进路径完全可追踪

在明确“谁拥有状态、谁可以修改状态、谁只能读取状态”之后，
整个模块的控制流变得线性、可解释，也更易于维护。

这部分设计带来的最大收益是：**系统稳定性来自对状态语义的克制，而不是更复杂的算法。**


### 16. 算法误判率减少的方法和整体的性能优化(重点)**

  - 1. 音频生产者模块的优化点：
    - 1.1. 取消音频DC滤波器锁， 保证只有音频生产者会使用dc滤波器， 减少锁的开销。
    - 1.2. 使用Q15定点计算滤波器加速， 减少浮点运算。
  
  - 2. 音频消费者模块的优化点：
    - 2.1. VAD滑窗检测，取消post_padding, 因为第一次非吠叫，直接把cur_stride_buf拼接到事件中，后续只有两种可能：1.下次继续非吠叫，直接不处理， push。2.下次是吠叫，直接继续使用cur_stride_buf拼接到event_buff.(节省了2kb ram开销)。
    - 2.2. 优化音频特征提取：
      - hann加窗改为dsp库实现 dsps_mul_f32(frame_in, hann_win, frame_out, FRAME_SIZE, 1, 1, 1);窗口长度400samples 步长160samples 加窗。 从952us优化到627us
      - 功率谱的计算预优化：输入向量的复部实部的填充由循环填充改为先使用memset填充0，再把帧数据使用循环填充到实部，减少了一半的循环次数。为了防止多线程环境多次调用fft操作，上层加入了一个变量，对不同的功能进行判断，使用不同的全局static buffer， 保护全局变量的访问。
      - 计算logmel变换的优化：使用稀疏矩阵的方式，预处理mel三角滤波器，使用topk-k矩阵乘法， 减少矩阵乘法运算。实际测试k=32就能满足和py端的结果一致（误差在1e-4以内）。
      - mfcc的dct变换，取消矩阵乘法，改为dsp的dsps_dotprod_f32_aes3点积， 减少矩阵乘法运算。
      - 结论：**最终的mfcc特征提取从最初的75ms优化到如今的12ms**    
   
  - 3. VAD事件消费者模块的优化点：
    - 3.1. 后处理的机制优化：原来对于vad时间超过200ms窗的事件，会滑窗推理，使用100ms窗，覆盖率50%，设置了一个缓冲buffer，设置高低阈值，对高阈值的窗直接push到bark event队列。低阈值的需要连续的两个窗满足，才会拼接，并找到buffer中rms最集中的200ms片段，push。如今的处理改为：依旧是滑窗推理，不过是对所有的窗进行推理后，使用topk进行排序，以及设置阈值，保证push的数量，以及push窗的质量。减少验证端的推理次数以及窗质量。
    - 3.2. 模型量化参数优化：使用mfcc_in_scale = mfcc_input_tensor->params.scale; mfcc_in_zero    = mfcc_input_tensor->params.zero_point;不在使用py端生成的量化参数，保存为宏、直接使用模型内部的量化参数。

  - 4. 声纹验证的优化点：
    - 4.1. 声纹验证模型量化参数优化：使用量化参数保存为宏，直接使用模型内部的量化参数。
    - 4.2. 聚类学习的flash结构体， 使用固定的十个key，防止key值过大，flash抛异常，实际测试100个key左右就会抛异常。
    - 4.3. 对于聚类算法的优化开销，不再计算相似度矩阵，采样先计算mean(feature)， 再计算向量与mean的相似度， 减少向量之间的距离计算， 后续后处理依旧使用topk进行排序，使用冷启动阈值和阈值变换，保证每次聚类的质量。
    - 4.4. 对于后续的声纹验证，需要设计一个状态机来减少推理以及乘法的分级处理。这是一个未完成项。


### 17. 程序部署在mcu后，仍然存在的优化点

1. paltformio集成esp-nn的难点，由于该项目是基于框架Arduino开发，截至 2025年12月11号，Espressif 仅在 ESP-IDF 框架中提供对 ESP-NN 加速的完整、可靠支持。Arduino 框架（包括 PlatformIO）没有官方支持的 TFLite Micro + ESP-NN 集成方案。已经验证过，无法配置成功，第三方库要集成需要花费大量时间，且无法保证正确性。
2. 后续需要转为esp-idf框架，使用esp-nn， 并替换掉esp库，因为esp-idf能提供更高的性能和功耗。
3. 减少功耗，目前对于前端的vad算法检测，在安静的环境中，功耗电流在72mA左右， 检测到狗吠和声纹验证时，功耗电流在95mA左右。目标优化功耗降低10%。
4. 蓝牙功能的编写，和app以及其他外部的硬件模块的通讯还需开发。
5. 考虑在前端设计一个检测入口在闲置状态时，使用低功耗模式，减少功耗， 只有触发条件时候才启用音频流检测。目标能够保持在1个月的使用。
......    

2025-12-22 kend 完成