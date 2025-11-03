/*
 * 生成测试数据并保存到C语言头文件中
 * 与Python脚本生成相同的数据用于对比
 */

#include <stdio.h>
#include <math.h>

#define SAMPLE_RATE 16000
#define DURATION 0.2
#define FREQUENCY 500
#define AMPLITUDE 0.3
#define SAMPLES ((int)(SAMPLE_RATE * DURATION))

static float test_input_signal[SAMPLES];

void generate_sine_wave() {
    // 生成500Hz正弦波测试信号
    for (int i = 0; i < SAMPLES; i++) {
        float t = i / (float)SAMPLE_RATE;
        test_input_signal[i] = AMPLITUDE * sinf(2.0f * M_PI * FREQUENCY * t);
    }
}

void save_to_header_file() {
    FILE* f = fopen("test_data_c.h", "w");
    if (!f) {
        printf("无法创建文件 test_data_c.h\n");
        return;
    }
    
    fprintf(f, "/*\n");
    fprintf(f, " * 测试数据头文件 (C版本)\n");
    fprintf(f, " * 自动生成的正弦波测试信号\n");
    fprintf(f, " */\n\n");
    fprintf(f, "#ifndef TEST_DATA_C_H\n");
    fprintf(f, "#define TEST_DATA_C_H\n\n");
    fprintf(f, "#define TEST_SIGNAL_LENGTH %d\n\n", SAMPLES);
    fprintf(f, "static const float test_input_signal_c[TEST_SIGNAL_LENGTH] = {\n");
    
    // 每行写入8个值
    for (int i = 0; i < SAMPLES; i += 8) {
        int remaining = SAMPLES - i;
        int count = remaining < 8 ? remaining : 8;
        
        fprintf(f, "    ");
        for (int j = 0; j < count; j++) {
            fprintf(f, "%.6ff", test_input_signal[i + j]);
            if (j < count - 1) {
                fprintf(f, ", ");
            }
        }
        
        if (i + 8 < SAMPLES) {
            fprintf(f, ",\n");
        } else {
            fprintf(f, "\n");
        }
    }
    
    fprintf(f, "};\n\n");
    fprintf(f, "#endif // TEST_DATA_C_H\n");
    
    fclose(f);
    printf("测试数据已保存到 test_data_c.h\n");
}

void print_signal_info() {
    printf("生成测试数据:\n");
    printf("采样率: %d Hz\n", SAMPLE_RATE);
    printf("持续时间: %.1f 秒\n", DURATION);
    printf("频率: %d Hz\n", FREQUENCY);
    printf("幅度: %.1f\n", AMPLITUDE);
    printf("采样点数: %d\n", SAMPLES);
    printf("\n");
    
    // 显示前10个采样点
    printf("前10个采样点:\n");
    for (int i = 0; i < 10 && i < SAMPLES; i++) {
        printf("  [%d]: %.6f\n", i, test_input_signal[i]);
    }
    printf("\n");
}

int main() {
    generate_sine_wave();
    print_signal_info();
    save_to_header_file();
    printf("完成!\n");
    return 0;
}