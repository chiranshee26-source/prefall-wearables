#pragma once
#include <cstddef>
#include <cstdint>
namespace NIMBLE_PROPERTY { enum { READ = 1, NOTIFY = 2 }; }
struct NimBLECharacteristic { void setValue(const uint8_t*, size_t) {} bool notify() { return true; } };
struct NimBLEService { NimBLECharacteristic* createCharacteristic(const char*, uint32_t) { static NimBLECharacteristic c; return &c; } bool start() { return true; } };
struct NimBLEServer { void advertiseOnDisconnect(bool) {} NimBLEService* createService(const char*) { static NimBLEService s; return &s; } };
struct NimBLEAdvertising { bool addServiceUUID(const char*) { return true; } bool start() { return true; } };
struct NimBLEDevice {
  static bool init(const char*) { return true; }
  static NimBLEServer* createServer() { static NimBLEServer s; return &s; }
  static NimBLEAdvertising* getAdvertising() { static NimBLEAdvertising a; return &a; }
};
