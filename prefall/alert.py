"""BLE alert payload (slide 15: timestamp, device ID, confidence score, kept minimal).

8 bytes, little-endian, so it fits a single BLE GATT notification with room to spare:
    u32 t_ms        milliseconds since device boot
    u16 device_id
    u8  confidence  model probability scaled 0..255
    u8  policy      0 = threshold + SVM confirm, 1 = ML-first persistence
The same layout is decoded in the demo page and will be encoded by the firmware.
"""
import struct
import numpy as np

FMT = "<IHBB"
SIZE = struct.calcsize(FMT)      # 8


def pack_alert(t_ms, device_id, confidence, policy=0):
    conf = int(round(float(np.clip(confidence, 0.0, 1.0)) * 255))
    return struct.pack(FMT, int(t_ms) & 0xFFFFFFFF, int(device_id) & 0xFFFF, conf, int(policy) & 0xFF)


def unpack_alert(b):
    t_ms, dev, conf, policy = struct.unpack(FMT, bytes(b))
    return dict(t_ms=t_ms, device_id=dev, confidence=conf / 255.0, policy=policy)
