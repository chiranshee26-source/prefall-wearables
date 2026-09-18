# Firmware

C++ implementation of the pre-fall detector for an **ESP32-S3 + MPU-6050**.

```
firmware/
├── core/     portable C++17, no Arduino or ESP-IDF includes. VERIFIED on a PC against the Python reference.
├── test/     host test (vectors exported from Python) and stand-in headers for a syntax check
└── esp32/    board layer: MPU-6050 driver, buzzer/haptic, BLE, main loop. NOT TESTED ON HARDWARE.
```

## What is verified, and what is not

| Part | Status |
|---|---|
| Low-pass filter and 200 Hz to 50 Hz decimation | Matches Python to 5e-5 |
| Trunk-angle / descent-speed tracker | Matches Python to 4e-5 deg and 4e-7 m/s |
| 12-D features | Match to 5e-7 (one window's zero-crossing count is off by a crossing from float32 rounding) |
| SVM confirmer (400 support vectors, about 20 KB) | Matches scikit-learn to 1e-4; same class on every window |
| Detector state machine and 8-byte alert payload | All alerts and payload bytes identical to Python, for detectors A and B |
| `esp32/src` | Compiles against stand-in headers only. Never built with the real toolchain, never run on a board |
| TFLite Micro adapter (`tflm_classifier.h`) | Written, never compiled. Optional: only detector B needs it |

Detector **A** (threshold gate, then SVM) needs nothing beyond `core/`: the SVM is plain C++. Detector **B**
(INT8 CNN-BiLSTM, ML-first) needs TensorFlow Lite Micro and is a second milestone.

## Run the PC verification

```bash
python scripts/run_step1.py                       # makes the training data
python tools/export_firmware_assets.py            # writes core/generated/*.h and test/vectors.bin
g++ -std=c++17 -O2 -Wall -Wextra -Ifirmware/core firmware/test/host_test.cpp -o host_test
./host_test firmware/test/vectors.bin             # expect: RESULT: PASS
```

On Windows use MSYS2/MinGW g++ or WSL, and `host_test.exe`.

## Bring-up checklist (do these in order, one at a time)

Wiring: MPU-6050 SDA to GPIO 8, SCL to GPIO 9, 3.3 V, GND (edit `esp32/src/board_config.h` if you wire differently).
Buzzer on GPIO 5 and haptic driver on GPIO 6 (through a transistor, never straight from the pin).

1. **Build and flash.** Install PlatformIO, open `firmware/esp32`, build. If `-I../core` is not found, copy `core/` into `esp32/lib/prefall_core/`.
2. **Sensor alive.** Serial monitor should print `WHO_AM_I = 0x68`. If not: wiring, address (0x69?), pull-up resistors, or a clone chip (WHO_AM_I differs; that is usually fine).
3. **Axis check (important).** With the board upright and still, `az` must be about +1 g. Tilt it forward: `ax` must go positive and `gy` positive while moving. Fix `kAccMap` / `kGyroMap` signs in `board_config.h` until both are true. A wrong sign here makes every detection wrong.
4. **Sample rate.** Build with `-DPREFALL_LOG_TIMING`. Expect 200 samples per second and 0 I2C errors. If the rate is off, check `CONFIG_FREERTOS_HZ` is 1000 (the 5 ms period needs it).
5. **Threshold detector on the bench.** Hold the board like a trunk-worn sensor and tilt it quickly forward while dropping it a few centimetres onto a soft surface: an ALERT line should print. Sitting down slowly and walking must not trigger it. Expect false alarms on fast sitting (that is what the SVM is for).
6. **SVM timing.** Log how long one `SvmClassifier::classify` call takes. It runs only after a trigger, so anything under a few ms is fine.
7. **Local alert latency.** The ALERT line prints `local=... us`. The deck claims under 10 ms; measure with a logic analyser on the buzzer pin if you can.
8. **BLE.** Install the free **nRF Connect** app, connect to `PreFall`, enable notifications on the second characteristic, trigger an alert. You should receive 8 bytes; decode them with `prefall/alert.py` (`unpack_alert`). Latency to the phone depends on the phone's connection interval; the deck's 7.5-15 ms and under-50 ms figures are unverified until you measure them.
9. **Haptic.** The deck specifies an LRA, which needs a driver chip (for example DRV2605L), not a GPIO. Replace `hapticOn/hapticOff` in `actuators.h` when you add one.
10. **Only then** consider detector B: add a TFLite Micro library, build with `-DPREFALL_DETECTOR_B -DPREFALL_USE_TFLM`, read the printed arena size and shrink `kArena`. Measure inference time; it must stay well under the 20 ms sample period at 50 Hz.

## Known gaps

- The Python pipeline was trained and tested on **simulated** signals. Real accelerometer data will differ (sensor noise, mounting, walking styles). Expect to collect real recordings and re-run the pipeline before trusting any number.
- The FSR insole variant is not implemented: the algorithm uses the IMU only.
- Battery life, power management and enclosure are out of scope here.
- The alert `confidence` byte for detector A is an uncalibrated logistic of the SVM score.
