#pragma once
#include <cstddef>
#include <cstdint>
struct TwoWire {
  void begin(int, int, uint32_t) {}
  void beginTransmission(uint8_t) {}
  size_t write(uint8_t) { return 1; }
  uint8_t endTransmission(bool = true) { return 0; }
  size_t requestFrom(int, int, int) { return 0; }
  int read() { return 0; }
};
static TwoWire Wire;
