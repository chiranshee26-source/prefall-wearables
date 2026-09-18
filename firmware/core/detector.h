// Streaming detector: mirrors prefall/hybrid.py (episodes + confirmed_alarms), one 50 Hz sample per call.
//
//   Trigger::Threshold   detector A: the threshold gate starts the clock, the classifier confirms.
//   Trigger::Classifier  detector B: the classifier's own rising edge starts the clock.
//
// After a trigger at sample e the alert can only be sent at a = e + delay, and only if the classifier
// said "pre-fall" on every window ending in [a - persist, a]. Re-triggers within 1.5 s are ignored.
#pragma once
#include <cstring>
#include "window_features.h"
#include "generated/threshold_params.h"
#include "prefall_config.h"
#include "tracker.h"

namespace prefall {

// window: row-major [kWin][kCh], oldest first. Returns true for "pre-fall".
using ClassifyFn = bool (*)(void* ctx, const float* window);

enum class Trigger { Threshold, Classifier };

class Detector {
 public:
  Detector(Trigger trig, int delay_ms, int persist_ms, ClassifyFn fn, void* ctx)
      : trig_(trig), d_((delay_ms * kFs + 500) / 1000), m_((persist_ms * kFs + 500) / 1000), fn_(fn), ctx_(ctx) {}

  // Feed one 50 Hz sample. Returns true when an alert must be sent now; *alarm_sample is its index.
  bool step(const float s[kCh], int* alarm_sample) {
    const int i = n_++;
    std::memcpy(ring_ + (head_ * kCh), s, sizeof(float) * kCh);
    head_ = (head_ + 1) % kWin;
    tracker_.update(s, &theta_, &vd_);
    gate_ = theta_ > kThetaCritDeg && vd_ > kVThrMps && theta_ * vd_ > kTau;

    cls_valid_ = false;
    bool sig;
    if (trig_ == Trigger::Threshold) {
      sig = gate_;
    } else {
      sig = classify_now(i);
    }
    const bool edge = sig && !prev_sig_ && i >= 1;
    prev_sig_ = sig;
    if (edge && i - last_trigger_ >= kRefractory) {
      last_trigger_ = i;
      pending_ = true;
      target_ = i + d_;
    }
    if (pending_ && i >= target_ - m_ && i <= target_) {
      if (!classify_now(i)) {
        pending_ = false;                       // the model withdrew its support: no alert
      } else if (i == target_) {
        pending_ = false;
        *alarm_sample = i;
        return true;
      }
    }
    return false;
  }

  // debug / test access
  float theta() const { return theta_; }
  float vdesc() const { return vd_; }
  bool gate() const { return gate_; }
  bool have_window() const { return n_ >= kWin; }
  void window(float* out) const { copy_window(out); }

 private:
  void copy_window(float* out) const {           // ring -> contiguous, oldest first
    for (int k = 0; k < kWin; ++k)
      std::memcpy(out + k * kCh, ring_ + ((head_ + k) % kWin) * kCh, sizeof(float) * kCh);
  }

  bool classify_now(int i) {
    if (cls_valid_) return cls_val_;
    cls_valid_ = true;
    cls_val_ = false;
    if (i >= kWin - 1) {
      float w[kWin * kCh];
      copy_window(w);
      cls_val_ = fn_(ctx_, w);
    }
    return cls_val_;
  }

  Trigger trig_;
  int d_, m_;
  ClassifyFn fn_;
  void* ctx_;
  Tracker tracker_;
  float ring_[kWin * kCh] = {};
  int head_ = 0, n_ = 0, last_trigger_ = -1000000, target_ = 0;
  bool prev_sig_ = false, pending_ = false, gate_ = false, cls_valid_ = false, cls_val_ = false;
  float theta_ = 0.f, vd_ = 0.f;
};

}  // namespace prefall
