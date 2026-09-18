// ClassifyFn adapter: features -> SVM -> "pre-fall" if decision > 0. Also remembers the last score,
// which the firmware turns into the payload's confidence byte.
#pragma once
#include "detector.h"
#include "window_features.h"
#include "svm.h"

namespace prefall {

struct SvmClassifier {
  float last_decision = 0.f;

  static bool classify(void* ctx, const float* window) {
    auto* self = static_cast<SvmClassifier*>(ctx);
    float f[kFeat];
    window_features(window, f);
    self->last_decision = svm_decision(f);
    return self->last_decision > 0.f;
  }
};

}  // namespace prefall
