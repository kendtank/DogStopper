#pragma once
#include <Arduino.h>

typedef enum {
    LED_IDLE = 0,
    LED_NO_TEMPLATE,
    LED_MATCH,
    LED_NO_MATCH,
} LedState;

void led_init(void);
void led_set_state(LedState state, uint32_t duration_ms);
void LedTask(void* param);

