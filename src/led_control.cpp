#include "led_control.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* ====== 硬件引脚，只在这里定义 ====== */
#define LED_R 12
#define LED_G 9
#define LED_B 10

/* ====== LED 控制状态（唯一） ====== */
static LedState current_state = LED_IDLE;
static uint32_t state_expire_ms = 0;

/* ====== 初始化 ====== */
void led_init(void)
{
    pinMode(LED_R, OUTPUT);
    pinMode(LED_G, OUTPUT);
    pinMode(LED_B, OUTPUT);

    digitalWrite(LED_R, LOW);
    digitalWrite(LED_G, LOW);
    digitalWrite(LED_B, LOW);
}

/* ====== 只允许“设置状态” ====== */
void led_set_state(LedState state, uint32_t duration_ms)
{
    current_state = state;
    state_expire_ms = millis() + duration_ms;
}

/* ====== 唯一控制 LED 的任务 ====== */
void LedTask(void* param)
{
    LedState last = LED_IDLE;

    while (true) {

        /* 到期自动熄灭 */
        if (current_state != LED_IDLE &&
            millis() > state_expire_ms) {
            current_state = LED_IDLE;
        }

        /* 状态变化才操作 GPIO */
        if (current_state != last) {
            last = current_state;

            /* 全灭 */
            digitalWrite(LED_R, LOW);
            digitalWrite(LED_G, LOW);
            digitalWrite(LED_B, LOW);

            switch (last) {
                case LED_NO_TEMPLATE:
                    digitalWrite(LED_B, HIGH);
                    break;
                case LED_MATCH:
                    digitalWrite(LED_R, HIGH);
                    break;
                case LED_NO_MATCH:
                    digitalWrite(LED_G, HIGH);
                    break;
                default:
                    break;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
