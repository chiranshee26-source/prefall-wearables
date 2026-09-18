// Minimal MPU-6050 driver over I2C (no external library). Configured as on slide 6:
// accel +/-8 g (4096 LSB/g), gyro +/-500 deg/s (65.5 LSB per deg/s), I2C fast mode 400 kHz.
#pragma once
#include <Arduino.h>
#include <Wire.h>
#include "board_config.h"

class Mpu6050 {
 public:
  bool begin(TwoWire& w = Wire) {
    w_ = &w;
    w_->begin(kPinSda, kPinScl, 400000);
    uint8_t who = 0;
    if (!readReg(0x75, &who, 1)) return false;
    who_ = who;                                   // 0x68 on a genuine MPU-6050; clones differ
    return writeReg(0x6B, 0x01)                   // wake up, clock = gyro X PLL
        && writeReg(0x19, 4)                      // sample-rate divider: 1 kHz / (1 + 4) = 200 Hz
        && writeReg(0x1A, 0x02)                   // DLPF ~94 Hz bandwidth (our own 25 Hz filter follows)
        && writeReg(0x1B, 0x08)                   // gyro +/-500 deg/s
        && writeReg(0x1C, 0x10);                  // accel +/-8 g
  }

  uint8_t whoAmI() const { return who_; }

  // Reads one sample, already mapped into the algorithm's frame: {ax, ay, az [g], gx, gy, gz [deg/s]}.
  bool read(float out[6]) {
    uint8_t b[14];
    if (!readReg(0x3B, b, 14)) return false;
    auto s16 = [&](int i) { return static_cast<int16_t>((b[i] << 8) | b[i + 1]); };
    const float acc[3] = {s16(0) / 4096.f, s16(2) / 4096.f, s16(4) / 4096.f};
    const float gyr[3] = {s16(8) / 65.5f, s16(10) / 65.5f, s16(12) / 65.5f};
    for (int k = 0; k < 3; ++k) {
      out[k] = kAccMap[k].sign * acc[kAccMap[k].src];
      out[3 + k] = kGyroMap[k].sign * gyr[kGyroMap[k].src];
    }
    return true;
  }

 private:
  bool writeReg(uint8_t reg, uint8_t v) {
    w_->beginTransmission(kMpuAddr);
    w_->write(reg);
    w_->write(v);
    return w_->endTransmission() == 0;
  }
  bool readReg(uint8_t reg, uint8_t* d, size_t n) {
    w_->beginTransmission(kMpuAddr);
    w_->write(reg);
    if (w_->endTransmission(false) != 0) return false;
    if (static_cast<size_t>(w_->requestFrom(static_cast<int>(kMpuAddr), static_cast<int>(n), 1)) != n) return false;
    for (size_t i = 0; i < n; ++i) d[i] = w_->read();
    return true;
  }
  TwoWire* w_ = nullptr;
  uint8_t who_ = 0;
};
