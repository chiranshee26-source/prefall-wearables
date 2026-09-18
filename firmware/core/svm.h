// RBF SVM confirmer exported from scikit-learn (400 support vectors, roughly 20 KB of flash).
// decision > 0 means "pre-fall". Mirrors sklearn: Pipeline(StandardScaler, SVC(kernel="rbf")).
#pragma once
#include <cmath>
#include "prefall_config.h"
#include "generated/svm_model.h"

namespace prefall {

inline float svm_decision(const float feat[kFeat]) {
  float z[kFeat];
  for (int j = 0; j < kFeat; ++j) z[j] = (feat[j] - kSvmMean[j]) / kSvmScale[j];
  float s = kSvmB;
  for (int i = 0; i < kSvmNsv; ++i) {
    float d2 = 0.f;
    for (int j = 0; j < kFeat; ++j) {
      const float t = z[j] - kSvmSv[i * kFeat + j];
      d2 += t * t;
    }
    s += kSvmCoef[i] * std::exp(-kSvmGamma * d2);
  }
  return s;
}

}  // namespace prefall
