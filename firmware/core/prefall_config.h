// Constants from the seminar deck (slides 6, 9, 13, 14). Portable C++17: no Arduino/ESP-IDF includes.
#pragma once

namespace prefall {

constexpr int kFsRaw = 200;      // IMU read rate before decimation
constexpr int kFs = 50;          // working rate (slide 9)
constexpr int kDecim = kFsRaw / kFs;
constexpr int kWin = 50;         // 1 s window
constexpr int kCh = 6;           // ax ay az (g), gx gy gz (deg/s)
constexpr int kFeat = 12;
constexpr float kG = 9.81f;
constexpr int kRefractory = 75;  // 1.5 s at 50 Hz: one physical event triggers once

}  // namespace prefall
