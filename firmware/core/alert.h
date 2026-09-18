// 8-byte BLE alert payload, identical to prefall/alert.py:
//   u32 t_ms (ms since boot) | u16 device_id | u8 confidence (0..255) | u8 detector (0 = A, 1 = B)
// little-endian on the wire.
#pragma once
#include <cmath>
#include <cstdint>

namespace prefall {

constexpr int kAlertSize = 8;

inline void pack_alert(uint32_t t_ms, uint16_t device_id, float confidence, uint8_t detector,
                       uint8_t out[kAlertSize]) {
  const float c = confidence < 0.f ? 0.f : (confidence > 1.f ? 1.f : confidence);
  out[0] = t_ms & 0xFF;
  out[1] = (t_ms >> 8) & 0xFF;
  out[2] = (t_ms >> 16) & 0xFF;
  out[3] = (t_ms >> 24) & 0xFF;
  out[4] = device_id & 0xFF;
  out[5] = (device_id >> 8) & 0xFF;
  out[6] = static_cast<uint8_t>(std::nearbyint(c * 255.f));
  out[7] = detector;
}

}  // namespace prefall
