// Local alert (slide 15): buzzer + haptic, started immediately and switched off by a one-shot timer so
// the sampling task never blocks.
//
// NOTE on the haptic motor: the deck specifies an LRA. An LRA needs an AC drive signal, so a plain
// GPIO high will NOT drive it properly. For real use, put a driver such as the DRV2605L (I2C) in
// front of it and replace hapticOn()/hapticOff(). The GPIO below is enough for a bring-up test with
// an ERM motor or an LED standing in for it.
#pragma once
#include <Arduino.h>
#include <esp_arduino_version.h>
#include <esp_timer.h>
#include "board_config.h"

class Actuators {
 public:
  void begin() {
    pinMode(kPinHaptic, OUTPUT);
    digitalWrite(kPinHaptic, LOW);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcAttach(kPinBuzzer, 3000, 8);
#else
    ledcSetup(0, 3000, 8);
    ledcAttachPin(kPinBuzzer, 0);
#endif
    esp_timer_create_args_t a = {};
    a.callback = &Actuators::offCb;
    a.arg = this;
    a.name = "alert_off";
    esp_timer_create(&a, &timer_);
  }

  // Start both outputs now (a few microseconds), stop after kAlertMs.
  void fire() {
    hapticOn();
    buzzerOn();
    esp_timer_stop(timer_);
    esp_timer_start_once(timer_, static_cast<uint64_t>(kAlertMs) * 1000);
  }

 private:
  void hapticOn() { digitalWrite(kPinHaptic, HIGH); }
  void hapticOff() { digitalWrite(kPinHaptic, LOW); }
  void buzzerOn() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWriteTone(kPinBuzzer, 3000);
#else
    ledcWriteTone(0, 3000);
#endif
  }
  void buzzerOff() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWriteTone(kPinBuzzer, 0);
#else
    ledcWriteTone(0, 0);
#endif
  }
  static void offCb(void* self) {
    auto* s = static_cast<Actuators*>(self);
    s->hapticOff();
    s->buzzerOff();
  }
  esp_timer_handle_t timer_ = nullptr;
};
