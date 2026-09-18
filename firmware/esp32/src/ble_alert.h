// BLE alert to the caregiver phone: one custom GATT service, one characteristic that notifies the
// 8-byte payload from core/alert.h. Uses NimBLE-Arduino 2.x.
// The connection interval is chosen by the phone (central); the deck's 7.5-15 ms figure is the best
// case the spec allows, not something this device can force. Measure it (checklist step 8).
#pragma once
#include <NimBLEDevice.h>

class BleAlert {
 public:
  void begin() {
    NimBLEDevice::init("PreFall");
    NimBLEServer* server = NimBLEDevice::createServer();
    server->advertiseOnDisconnect(true);
    NimBLEService* svc = server->createService(kService);
    chr_ = svc->createCharacteristic(kChar, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
    uint8_t zero[8] = {0};
    chr_->setValue(zero, sizeof(zero));
    svc->start();
    NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
    adv->addServiceUUID(kService);
    adv->start();
  }

  void notify(const uint8_t* data, size_t n) {
    chr_->setValue(data, n);
    chr_->notify();
  }

 private:
  // custom 128-bit UUIDs, generated for this project
  static constexpr const char* kService = "6f1d0001-5a4b-4c1e-9d2a-7b8c9e0f1a2b";
  static constexpr const char* kChar = "6f1d0002-5a4b-4c1e-9d2a-7b8c9e0f1a2b";
  NimBLECharacteristic* chr_ = nullptr;
};
