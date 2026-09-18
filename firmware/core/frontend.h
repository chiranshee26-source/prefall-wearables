// Slide 9 front end: 2nd-order Butterworth low-pass at the raw rate, then keep every 4th sample.
// Mirrors prefall/dsp.py: causal, and each filter is primed to the steady state of its first sample
// so there is no start-up transient (same as scipy lfilter with zi = lfilter_zi * x[0]).
#pragma once
#include "prefall_config.h"
#include "generated/lpf_coeffs.h"

namespace prefall {

struct Biquad {
  float z1 = 0.f, z2 = 0.f;
  bool primed = false;

  float process(float x) {
    const float b0 = kLpfB[0], b1 = kLpfB[1], b2 = kLpfB[2], a1 = kLpfA[1], a2 = kLpfA[2];
    if (!primed) {
      const float yss = (b0 + b1 + b2) / (1.f + a1 + a2);   // DC gain (1 for a Butterworth low-pass)
      z2 = (b2 - a2 * yss) * x;
      z1 = (b1 - a1 * yss + (b2 - a2 * yss)) * x;
      primed = true;
    }
    const float y = b0 * x + z1;                            // direct form II transposed
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
};

class FrontEnd {
 public:
  // Feed one raw 200 Hz sample. Returns true when a 50 Hz sample is ready in out[6].
  bool push(const float raw[kCh], float out[kCh]) {
    float y[kCh];
    for (int c = 0; c < kCh; ++c) y[c] = f_[c].process(raw[c]);
    const bool emit = (count_++ % kDecim) == 0;             // indices 0, 4, 8, ... like y[::4]
    if (emit)
      for (int c = 0; c < kCh; ++c) out[c] = y[c];
    return emit;
  }

 private:
  Biquad f_[kCh];
  unsigned count_ = 0;
};

}  // namespace prefall
