// Host-side verification of the firmware core against the Python reference.
//   python tools/export_firmware_assets.py
//   g++ -std=c++17 -O2 -Wall -Wextra -I../core host_test.cpp -o host_test && ./host_test vectors.bin
// Each component is checked in isolation against Python (so a failure points at one file), then the
// whole chain runs end to end from raw 200 Hz samples to the alert bytes.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "alert.h"
#include "detector.h"
#include "window_features.h"
#include "frontend.h"
#include "svm_classifier.h"
#include "tracker.h"

using namespace prefall;

static FILE* g_f;
template <class T>
static std::vector<T> rd(size_t n) {
  std::vector<T> v(n);
  if (n && fread(v.data(), sizeof(T), n, g_f) != n) {
    std::fprintf(stderr, "short read\n");
    std::exit(2);
  }
  return v;
}
static int32_t rdi() { return rd<int32_t>(1)[0]; }

struct Policy {
  int delay, persist, src;
  std::vector<int> alarm;
  std::vector<std::vector<uint8_t>> payload;
};

static int g_fail = 0;
static void check(bool ok, const char* what, int trial, double detail = 0) {
  if (!ok) {
    ++g_fail;
    std::printf("  FAIL trial %d: %s (%.6g)\n", trial, what, detail);
  }
}

int main(int argc, char** argv) {
  g_f = std::fopen(argc > 1 ? argv[1] : "vectors.bin", "rb");
  if (!g_f) {
    std::fprintf(stderr, "cannot open vectors file (run tools/export_firmware_assets.py)\n");
    return 2;
  }
  const int ntr = rdi();
  double w_front = 0, w_track_th = 0, w_track_vd = 0, w_feat = 0, w_svm = 0, w_zcr = 0;
  int zcr_flips = 0;
  int nalarm_total = 0, npayload = 0;

  for (int t = 0; t < ntr; ++t) {
    const int nraw = rdi(), n50 = rdi();
    auto raw = rd<float>(size_t(nraw) * kCh), x50 = rd<float>(size_t(n50) * kCh);
    auto th = rd<float>(n50), vd = rd<float>(n50);
    auto mask = rd<uint8_t>(n50);
    auto dec = rd<float>(n50);
    auto feats = rd<float>(size_t(n50 - kWin + 1) * kFeat);
    Policy pol[3];
    for (auto& p : pol) {
      p.delay = rdi(); p.persist = rdi(); p.src = rdi();
      const int na = rdi();
      for (int k = 0; k < na; ++k) {
        p.alarm.push_back(rdi());
        p.payload.push_back(rd<uint8_t>(kAlertSize));
      }
    }

    // 1. front end: raw 200 Hz -> 50 Hz
    {
      FrontEnd fe;
      std::vector<float> out;
      float y[kCh];
      for (int i = 0; i < nraw; ++i)
        if (fe.push(&raw[size_t(i) * kCh], y)) out.insert(out.end(), y, y + kCh);
      check(out.size() == x50.size(), "front end output length", t, double(out.size()));
      for (size_t i = 0; i < out.size() && i < x50.size(); ++i)
        w_front = std::fmax(w_front, std::fabs(out[i] - x50[i]));
    }
    // 2. tracker on Python's 50 Hz stream
    {
      Tracker tr;
      for (int i = 0; i < n50; ++i) {
        float a, b;
        tr.update(&x50[size_t(i) * kCh], &a, &b);
        w_track_th = std::fmax(w_track_th, std::fabs(a - th[i]));
        w_track_vd = std::fmax(w_track_vd, std::fabs(b - vd[i]));
      }
    }
    // 3 + 4. features and SVM on Python's windows
    for (int j = kWin - 1; j < n50; ++j) {
      float f[kFeat];
      window_features(&x50[size_t(j - kWin + 1) * kCh], f);
      const float* ef = &feats[size_t(j - kWin + 1) * kFeat];
      for (int k = 0; k < kFeat; ++k) {
        const double e = std::fabs(f[k] - ef[k]) / std::fmax(1.f, std::fabs(ef[k]));
        if (k == 6) {                       // zero-crossing rate: a sample sitting exactly on the mean can
          w_zcr = std::fmax(w_zcr, e);      // flip sign between float32 and float64, moving it by 1/49
          zcr_flips += e > 1e-6;
        } else {
          w_feat = std::fmax(w_feat, e);
        }
      }
      w_svm = std::fmax(w_svm, std::fabs(svm_decision(ef) - dec[j]));
      check((svm_decision(f) > 0) == (dec[j] > 0), "SVM class agrees with sklearn", t, j);
    }
    // 5. detector: isolated (Python's 50 Hz stream in) and end to end (raw 200 Hz in)
    for (int mode = 0; mode < 2; ++mode) {
      for (int pi = 0; pi < 3; ++pi) {
        const Policy& p = pol[pi];
        SvmClassifier cls;
        Detector det(p.src == 0 ? Trigger::Threshold : Trigger::Classifier, p.delay, p.persist,
                     &SvmClassifier::classify, &cls);
        std::vector<int> got;
        std::vector<std::vector<uint8_t>> gotp;
        FrontEnd fe;
        float y[kCh];
        auto feed = [&](const float* s, int idx) {
          int a;
          if (det.step(s, &a)) {
            got.push_back(a);
            uint8_t b[kAlertSize];
            pack_alert(uint32_t(a) * 1000u / kFs, 0x0A01, 0.5f, uint8_t(p.src), b);
            gotp.emplace_back(b, b + kAlertSize);
          }
          (void)idx;
        };
        if (mode == 0)
          for (int i = 0; i < n50; ++i) feed(&x50[size_t(i) * kCh], i);
        else {
          int i50 = 0;
          for (int i = 0; i < nraw; ++i)
            if (fe.push(&raw[size_t(i) * kCh], y)) feed(y, i50++);
        }
        const bool same = got == p.alarm;
        check(same, mode == 0 ? "detector alarms (isolated) match Python" : "detector alarms (end to end) match Python", t, pi);
        if (!same) {
          std::printf("    policy %d expected:", pi);
          for (int a : p.alarm) std::printf(" %d", a);
          std::printf(" | got:");
          for (int a : got) std::printf(" %d", a);
          std::printf("\n");
        } else {
          nalarm_total += int(got.size());
          for (size_t k = 0; k < got.size(); ++k) {
            // Python packed confidence 0.5 as well, so the bytes must be identical
            check(gotp[k] == p.payload[k], "payload bytes match Python", t, double(k));
            ++npayload;
          }
        }
      }
    }
  }

  check(w_front < 2e-3, "front end max abs error", -1, w_front);
  check(w_track_th < 5e-2, "tracker angle max error (deg)", -1, w_track_th);
  check(w_track_vd < 1e-2, "tracker descent-speed max error (m/s)", -1, w_track_vd);
  check(w_feat < 2e-3, "features (all except ZCR) max relative error", -1, w_feat);
  check(w_zcr <= 2.5 / 49, "ZCR differs by more than 2 crossings", -1, w_zcr);
  check(w_svm < 5e-3, "SVM decision max abs error", -1, w_svm);

  std::printf("trials: %d\n", ntr);
  std::printf("front end  max |C++ - Python|      : %.2e g or deg/s\n", w_front);
  std::printf("tracker    max angle / speed error : %.2e deg / %.2e m/s\n", w_track_th, w_track_vd);
  std::printf("features   max relative error      : %.2e (all but ZCR)\n", w_feat);
  std::printf("ZCR        windows off by a crossing: %d, worst %.4f (1/49 = %.4f)\n", zcr_flips, w_zcr, 1.0 / 49);
  std::printf("SVM        max decision error      : %.2e\n", w_svm);
  std::printf("alerts matched (all policies, both modes): %d, payloads matched: %d\n", nalarm_total, npayload);
  std::printf(g_fail ? "RESULT: FAIL (%d checks)\n" : "RESULT: PASS\n", g_fail);
  return g_fail ? 1 : 0;
}
