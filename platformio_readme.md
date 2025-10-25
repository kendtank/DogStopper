# 基于 ESP32-S3 止吠器项目结构说明

本项目基于 **PlatformIO** 开发环境，目标是实现一个运行在 **ESP32-S3** 上的止吠器系统（包含声音识别与控制逻辑）。

---

## 项目结构说明

项目创建完成后，主要的目录和文件结构如下：

```
.
├── .pio/                  # PlatformIO 的工作目录（自动生成）
├── .vscode/               # VSCode 的配置目录（自动生成）
├── include/               # 存放头文件（.h）
├── lib/                   # 存放项目依赖库
├── src/                   # 存放主程序源码（.cpp/.c）
│   └── main.cpp           # 主程序入口（等价于 Arduino 的 .ino 文件）
├── test/                  # 存放测试代码或单元测试
├── platformio.ini         # PlatformIO 项目配置文件
└── .gitignore             # Git 忽略规则文件
```

---

## 各目录作用说明

### `.pio/`
PlatformIO 自动生成的中间文件目录，包含编译后的二进制文件、缓存、日志等。  
> 注意：无需手动修改或提交到 Git。

---

### `.vscode/`
如果使用 VSCode + PlatformIO 插件，会在此目录生成项目调试和构建相关的配置文件（如任务、断点、调试设置等）。

---

### `include/`
用于存放 **头文件（.h）**。  
头文件中一般写：
- 函数、类、结构体、宏的声明；
- 供其他源码文件引用的公共接口。

> 编译时，编译器会自动在此目录下查找头文件。

---

### `lib/`
用于存放 **项目依赖库**。  
你可以通过 PlatformIO 的库管理器安装第三方库，也可以将自己编写的库（带源码）放在这里。

例如：
```
lib/
└── MyAudioLib/
    ├── MyAudioLib.cpp
    └── MyAudioLib.h
```

---

### `src/`
主代码目录。  
主要代码文件应放在这里，例如：
```
src/
└── main.cpp
```

PlatformIO 中的 `main.cpp` 与 Arduino IDE 中的 `.ino` 文件功能相同。  
唯一的区别是：  
在文件顶部需要手动引入头文件：

```cpp
#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    Serial.println("Hello ESP32-S3!");
}

void loop() {
    // 主逻辑循环
}
```

---

### `test/`
用于存放单元测试或集成测试代码，验证模块或整体功能是否正确。  
> 可选目录，非必须。

---

### `platformio.ini`
PlatformIO 的项目配置文件，用于指定：
- 目标硬件平台；
- 框架（如 Arduino、ESP-IDF）；
- 上传方式；
- 编译选项等。

示例：

```ini
[env:esp32-s3-devkitm-1]
platform = espressif32
board = esp32-s3-devkitm-1
framework = arduino
upload_speed = 921600
monitor_speed = 115200
```

---

## 目录使用建议总结

| 目录名 | 用途 | 示例内容 |
|--------|------|----------|
| `.pio/` | 编译缓存与临时文件 | 自动生成 |
| `.vscode/` | VSCode 调试配置 | launch.json, tasks.json |
| `include/` | 头文件声明 | `config.h`, `audio_preproc.h` |
| `lib/` | 自定义库或第三方库 | `TinyBarkLib/` |
| `src/` | 主程序源代码 | `main.cpp` |
| `test/` | 单元测试 | `test_main.cpp` |
| `platformio.ini` | 项目配置 | 编译/上传参数 |

---

## 小贴士

- **主逻辑写在 `src/main.cpp` 中**
- **公共声明放在 `include/`**
- **模块化逻辑封装进 `lib/`**
- **不要手动修改 `.pio/` 下的内容**

---

## 示例：快速验证串口输出

在 `src/main.cpp` 中输入以下代码：

```cpp
#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    Serial.println("ESP32-S3 止吠器启动成功！");
}

void loop() {
    Serial.println("正在运行...");
    delay(1000);
}
```

然后点击 **“Build → Upload → Monitor”**，即可在串口终端中看到输出。

---

**作者提示**
> 如果你是从 Arduino IDE 迁移过来的开发者，`PlatformIO` 提供了更清晰的项目结构、更强的依赖管理能力和更专业的调试支持。

## 鸣谢
本项目基于极客侠GeeksMan的学习资料创建。网址：https://docs.geeksman.com/esp32/Arduino/13.esp32-arduino-platformio.html#_3-%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8-platformio
