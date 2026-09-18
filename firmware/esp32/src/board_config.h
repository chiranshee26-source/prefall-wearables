// Board wiring and mounting. EDIT THESE for your hardware before the first flash.
#pragma once
#include <stdint.h>

// I2C to the MPU-6050 (ESP32-S3 DevKitC-1 default I2C pins; change if you wired elsewhere)
constexpr int kPinSda = 8;
constexpr int kPinScl = 9;
constexpr uint8_t kMpuAddr = 0x68;             // AD0 low; 0x69 if AD0 is tied high

// Alert outputs
constexpr int kPinBuzzer = 5;                  // piezo through a transistor, or an active buzzer
constexpr int kPinHaptic = 6;                  // haptic driver enable (see actuators.h note about LRA)
constexpr int kAlertMs = 400;                  // how long the buzzer/haptic stay on

constexpr uint16_t kDeviceId = 0x0A01;         // goes into the BLE payload

// Axis mapping: sensor axes -> the frame the algorithm expects (see ../../prefall/synth.py):
//   az = +1 g when the wearer stands upright and still
//   ax = +sin(theta) when the trunk tilts forward, and gy = +d(theta)/dt (forward tilt is positive)
// Each entry is {source axis 0..2, sign}. Verify with the bring-up checklist step 3 and fix the signs.
struct AxisMap { int src; float sign; };
constexpr AxisMap kAccMap[3]  = {{0, +1.f}, {1, +1.f}, {2, +1.f}};   // ax, ay, az
constexpr AxisMap kGyroMap[3] = {{0, +1.f}, {1, +1.f}, {2, +1.f}};   // gx, gy, gz
