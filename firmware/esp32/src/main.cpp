// Pre-fall prediction firmware for ESP32-S3 + MPU-6050. UNTESTED ON HARDWARE.
// The signal processing and detector logic live in ../core and are verified on a PC against the
// Python reference (firmware/test). This file only wires them to the sensor, buzzer and BLE.
#include <Arduino.h>
#include <cmath>
#include <esp_timer.h>

#include "actuators.h"
#include "alert.h"
#include "ble_alert.h"
#include "board_config.h"
#include "detector.h"
#include "frontend.h"
#include "mpu6050.h"
#include "svm_classifier.h"
#include "tflm_classifier.h"

using namespace prefall;

static Mpu6050 imu;
static Actuators actuators;
static BleAlert ble;
static FrontEnd frontend;
static float gyroBias[3] = {0, 0, 0};

#if defined(PREFALL_DETECTOR_B) && defined(PREFALL_USE_TFLM)
// Detector B: the INT8 CNN-BiLSTM decides every sample and must stay positive for 300 ms.
static TflmClassifier model;
static Detector detector(Trigger::Classifier, 300, 300, &TflmClassifier::classify, &model);
static const uint8_t kDetectorCode = 1;
static float confidence() { return model.last_prob; }
#else
// Detector A (the deck's design): threshold gate starts the clock, the SVM confirms immediately.
static SvmClassifier model;
static Detector detector(Trigger::Threshold, 0, 0, &SvmClassifier::classify, &model);
static const uint8_t kDetectorCode = 0;
static float confidence() { return 1.f / (1.f + std::exp(-model.last_decision)); }   // uncalibrated
#endif

static void onAlert(int sample) {
  const int64_t t0 = esp_timer_get_time();
  actuators.fire();                                  // local alert first (target < 10 ms)
  const int64_t t1 = esp_timer_get_time();
  uint8_t pkt[kAlertSize];
  pack_alert(millis(), kDeviceId, confidence(), kDetectorCode, pkt);
  ble.notify(pkt, kAlertSize);                       // then the caregiver (target < 50 ms)
  const int64_t t2 = esp_timer_get_time();
  Serial.printf("ALERT sample=%d local=%lld us ble_call=%lld us payload=", sample, (long long)(t1 - t0), (long long)(t2 - t0));
  for (int i = 0; i < kAlertSize; ++i) Serial.printf("%02x", pkt[i]);
  Serial.println();
}

static void calibrateGyroBias() {
  // Hold the device still for one second at boot: MPU-6050 gyro offsets are several deg/s.
  float sum[3] = {0, 0, 0};
  int n = 0;
  const uint32_t t0 = millis();
  while (millis() - t0 < 1000) {
    float s[6];
    if (imu.read(s)) {
      for (int k = 0; k < 3; ++k) sum[k] += s[3 + k];
      ++n;
    }
    delay(5);
  }
  for (int k = 0; k < 3; ++k) gyroBias[k] = n ? sum[k] / n : 0.f;
  Serial.printf("gyro bias (deg/s): %.2f %.2f %.2f from %d samples\n", gyroBias[0], gyroBias[1], gyroBias[2], n);
}

static void sampleTask(void*) {
  TickType_t last = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(1000 / kFsRaw);   // 5 ms; needs CONFIG_FREERTOS_HZ = 1000
  uint32_t rawCount = 0, i2cErr = 0, statT = millis();
  for (;;) {
    vTaskDelayUntil(&last, period);
    float raw[kCh];
    if (!imu.read(raw)) {
      ++i2cErr;
      continue;
    }
    for (int k = 0; k < 3; ++k) raw[3 + k] -= gyroBias[k];
    ++rawCount;
    float s[kCh];
    if (frontend.push(raw, s)) {
      int a;
      if (detector.step(s, &a)) onAlert(a);
    }
#ifdef PREFALL_LOG_TIMING
    if (millis() - statT >= 1000) {
      Serial.printf("rate %lu Hz, i2c errors %lu, theta %.1f deg, vdesc %.2f m/s\n", (unsigned long)rawCount,
                    (unsigned long)i2cErr, detector.theta(), detector.vdesc());
      rawCount = 0;
      statT = millis();
    }
#else
    (void)statT; (void)i2cErr;
#endif
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("prefall firmware (untested)");
  if (!imu.begin()) {
    Serial.println("MPU-6050 not found: check wiring, address (0x68/0x69) and pull-ups");
    for (;;) delay(1000);
  }
  Serial.printf("MPU WHO_AM_I = 0x%02x (0x68 expected)\n", imu.whoAmI());
  actuators.begin();
  ble.begin();
#if defined(PREFALL_DETECTOR_B) && defined(PREFALL_USE_TFLM)
  if (!model.begin()) {
    Serial.println("TFLite Micro init failed (arena too small or op missing)");
    for (;;) delay(1000);
  }
  Serial.printf("TFLM arena used: %u bytes\n", (unsigned)model.arenaUsed());
#endif
  calibrateGyroBias();
  xTaskCreatePinnedToCore(sampleTask, "sample", 8192, nullptr, configMAX_PRIORITIES - 2, nullptr, 1);
}

void loop() { vTaskDelay(portMAX_DELAY); }
