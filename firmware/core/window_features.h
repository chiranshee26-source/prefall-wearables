// 1 s window -> 12-D feature vector (slides 9 and 11). Mirrors prefall/features.py: window_features.
// w is row-major [kWin][kCh], oldest sample first.
#pragma once
#include <cmath>
#include "prefall_config.h"

namespace prefall {

inline void window_features(const float* w, float out[kFeat]) {
  constexpr float kRad = 3.14159265358979f / 180.f;
  float a[kWin], g[kWin], ac[kWin];
  float asum = 0.f, gsum = 0.f, sma = 0.f, a2 = 0.f;
  float amin = 1e30f, amax = -1e30f, gmax = 0.f, tilt_max = 0.f;
  for (int n = 0; n < kWin; ++n) {
    const float* s = w + n * kCh;
    a[n] = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    g[n] = std::sqrt(s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    asum += a[n];
    a2 += a[n] * a[n];
    gsum += g[n];
    sma += std::fabs(s[0]) + std::fabs(s[1]) + std::fabs(s[2]);
    amin = a[n] < amin ? a[n] : amin;
    amax = a[n] > amax ? a[n] : amax;
    gmax = g[n] > gmax ? g[n] : gmax;
    const float tilt = std::atan2(std::sqrt(s[0] * s[0] + s[1] * s[1]), s[2]) / kRad;
    tilt_max = tilt > tilt_max ? tilt : tilt_max;
  }
  const float amean = asum / kWin;
  float var = 0.f;
  for (int n = 0; n < kWin; ++n) {
    ac[n] = a[n] - amean;
    var += ac[n] * ac[n];
  }
  int zc = 0;
  for (int n = 1; n < kWin; ++n) zc += (std::signbit(ac[n]) != std::signbit(ac[n - 1])) ? 1 : 0;

  // dominant frequency: direct DFT bins 1..25 (1 Hz apart for a 50-sample window at 50 Hz)
  static float ct[kWin / 2 + 1][kWin], st[kWin / 2 + 1][kWin];
  static bool tables = false;
  if (!tables) {
    for (int k = 1; k <= kWin / 2; ++k)
      for (int n = 0; n < kWin; ++n) {
        const float ph = 2.f * 3.14159265358979f * k * n / kWin;
        ct[k][n] = std::cos(ph);
        st[k][n] = std::sin(ph);
      }
    tables = true;
  }
  int best_k = 1;
  float best = -1.f;
  for (int k = 1; k <= kWin / 2; ++k) {
    float re = 0.f, im = 0.f;
    for (int n = 0; n < kWin; ++n) {
      re += ac[n] * ct[k][n];
      im -= ac[n] * st[k][n];
    }
    const float mag = std::sqrt(re * re + im * im);
    if (mag > best) {
      best = mag;
      best_k = k;
    }
  }

  float c = 0.f, vd_max = -1e30f;                   // descent-speed proxy: max of -cumsum((|a|-1) g / fs)
  for (int n = 0; n < kWin; ++n) {
    c += (a[n] - 1.f) * kG / kFs;
    vd_max = (-c) > vd_max ? (-c) : vd_max;
  }

  out[0] = amean;
  out[1] = var / kWin;
  out[2] = std::sqrt(a2 / kWin);
  out[3] = amin;
  out[4] = amax;
  out[5] = sma / kWin;
  out[6] = static_cast<float>(zc) / (kWin - 1);
  out[7] = static_cast<float>(best_k) * kFs / kWin;
  out[8] = gsum / kWin;
  out[9] = gmax;
  out[10] = tilt_max;
  out[11] = vd_max;
}

}  // namespace prefall
