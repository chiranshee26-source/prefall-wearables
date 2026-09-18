// Stand-in declarations so firmware/esp32/src can be syntax-checked on a PC. NOT the real Arduino API.
#pragma once
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#define OUTPUT 1
#define HIGH 1
#define LOW 0
#define portMAX_DELAY 0xFFFFFFFFu
#define configMAX_PRIORITIES 25
#define pdMS_TO_TICKS(ms) (ms)
typedef uint32_t TickType_t;
struct HardwareSerial {
  void begin(int) {}
  void println(const char*) {}
  void println() {}
  int printf(const char*, ...) __attribute__((format(printf, 2, 3))) { return 0; }
};
static HardwareSerial Serial;
inline uint32_t millis() { return 0; }
inline void delay(uint32_t) {}
inline void pinMode(int, int) {}
inline void digitalWrite(int, int) {}
inline void ledcAttach(int, int, int) {}
inline void ledcWriteTone(int, int) {}
inline TickType_t xTaskGetTickCount() { return 0; }
inline void vTaskDelayUntil(TickType_t*, TickType_t) {}
inline void vTaskDelay(TickType_t) {}
inline int xTaskCreatePinnedToCore(void (*)(void*), const char*, int, void*, int, void*, int) { return 1; }
void setup();
void loop();
