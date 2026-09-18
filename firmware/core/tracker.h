// Trunk angle (complementary filter) and descent speed (leaky integral of vertical acceleration).
// Mirrors prefall/threshold.py: Tracker. Slide 3: Vy(t) = integral of (a_vertical - g) dt.
#pragma once
#include <cmath>
#include "prefall_config.h"

namespace prefall {

class Tracker {
 public:
  // s = {ax, ay, az [g], gx, gy, gz [deg/s]}; outputs |trunk angle| in degrees and descent speed in m/s.
  void update(const float s[kCh], float* theta_abs_deg, float* vdesc_mps) {
    constexpr float kDt = 1.f / kFs, kAlpha = 0.98f, kLeakTau = 0.6f, kRad = 3.14159265358979f / 180.f;
    const float ax = s[0], ay = s[1], az = s[2], gy = s[4];
    const float a = std::sqrt(ax * ax + ay * ay + az * az);
    const float th_acc = std::atan2(ax, az) / kRad;
    if (!init_) {
      theta_ = th_acc;
      init_ = true;
    } else {
      const float pred = theta_ + gy * kDt;
      // near 1 g the accelerometer is a trustworthy tilt reference; in free fall only the gyro is
      theta_ = (a > 0.7f && a < 1.3f) ? kAlpha * pred + (1.f - kAlpha) * th_acc : pred;
    }
    const float r = theta_ * kRad;
    const float a_v = ax * std::sin(r) + az * std::cos(r);
    vy_ += (a_v - 1.f) * kG * kDt;
    vy_ -= vy_ * kDt / kLeakTau;
    *theta_abs_deg = std::fabs(theta_);
    *vdesc_mps = vy_ < 0.f ? -vy_ : 0.f;
  }

 private:
  bool init_ = false;
  float theta_ = 0.f, vy_ = 0.f;
};

}  // namespace prefall
