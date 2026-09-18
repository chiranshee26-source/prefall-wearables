# Results

> [!IMPORTANT]
> All numbers come from **simulated** sensor data (`prefall/synth.py`), 400 trials (160 falls, 240 normal
> activity), evaluated on 200 held-out trials (83 falls, 117 normal). They are not clinical evidence and are not
> comparable to published benchmarks. Training is not bit-exact between runs (TensorFlow multithreading,
> scikit-learn versions), so expect a few points of drift in the model numbers.
> Regenerate everything with the commands in the [README](../README.md#quick-start).

## Threshold detector (step 1)

| Setting | Falls caught | No false alarm | Mean lead | Falls with 200 ms or more |
|---|---|---|---|---|
| Tuned for specificity (at least 90%) | 100% | 97.4% | 118 ms | 1.2% |
| Tuned for lead time (first pass) | 100% | 53.0% | 238 ms | 74.7% |

A lone threshold cannot have both a useful lead time and few false alarms. The first-pass setting sends fast sits
and stumbles (100% of them) to the confirmer.

![Threshold traces for a fall and a fast sit](images/threshold_traces.png)

## Classical models (step 2a), fired on every window

| Model | Falls caught | No false alarm | Mean lead |
|---|---|---|---|
| Random Forest (100 trees, depth 8) | 100% | 65-66% | 457-463 ms |
| SVM (RBF) | 100% | 70.9% | 467 ms |
| Decision Tree (depth 8) | 100% | 60.7% | 461 ms |

Top Random Forest features: gyro peak (0.36), peak tilt (0.27), gyro mean (0.17). Forest size is 5.5-6.9 thousand
nodes, roughly 65-80 KB at 12 bytes per node.

![Random Forest feature importance](images/rf_feature_importance.png)

## Deep models (step 2b)

| Model | Parameters | FP32 | Falls caught | No false alarm | Mean lead |
|---|---|---|---|---|---|
| 1D-CNN + BiLSTM | 20,850 | 81 KB | 100% | 56-63% | 448-463 ms |
| ConvLSTM | 9,506 | 37 KB | 100% | 54% | 458 ms |

## Two-stage and ML-first detectors (step 2c)

Threshold starts the clock, model confirms after waiting `d` ms:

| Confirmer | Wait 0 ms | Wait 100 ms |
|---|---|---|
| SVM | 98.8% caught, 87.2% no false alarm, 237 ms lead | 98.8%, 91.5%, 140 ms |
| CNN-BiLSTM | 100%, 71%, 238 ms | 100%, 97%, 138 ms |

Model starts the clock, must stay positive for `d` ms (no threshold):

| Model | d = 200 ms | d = 300 ms |
|---|---|---|
| SVM | 100% caught, 86.3% no false alarm, 351 ms lead | 98.8%, 91.5%, 250 ms |
| CNN-BiLSTM | 94%, 75%, 324 ms | 87-96% caught, 100% no false alarm, 211-225 ms |

In the ML-first setting with a 300 ms wait, both deep models reject every recovered stumble (0% false alarms) while
the SVM and Random Forest still flag 43% and 61%. On its own, no model separates a stumble from a fall.

![Specificity and lead time against confirmation wait](images/hybrid_tradeoff.png)

## Compression and export (step 3)

| CNN-BiLSTM version | File size | gzip | No false alarm | Agreement with FP32 |
|---|---|---|---|---|
| FP32 TFLite | 88.5 KB | 78 KB | n/a | n/a |
| INT8 | **32.8 KB** | 24 KB | 57-64% | 99.9-100% of windows |
| Pruned 35% + INT8 | 32.9 KB | 21 KB | 56-70% | 99.9% of windows |

- INT8 quantization costs no accuracy on this data. Ops used: CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE,
  REVERSE_V2, SOFTMAX, STRIDED_SLICE, UNIDIRECTIONAL_SEQUENCE_LSTM. All are registered in TFLite Micro's
  `MicroMutableOpResolver`; on-device behaviour is untested.
- Pruning does **not** shrink the `.tflite` file (zeros are still stored); only gzip sees the saving. In three
  runs the pruned model scored 6-10 points higher, but in a fresh retrain it did not (56% against 57%), so treat that
  as run-to-run noise, not a benefit of pruning.

## End to end with the deployable pieces (step 4)

Threshold detector + SVM, and the exported INT8 model file, run per sample on all 200 test trials:

| | Falls caught | No false alarm | Mean lead | Falls with 200 ms or more |
|---|---|---|---|---|
| A: threshold, then SVM | 98.8% | 87.2% | 237 ms | 73.5% |
| B: INT8 CNN-BiLSTM, 300 ms | 98.8% | 100% | 225 ms | 61.4% |

These are the numbers for the model committed in `export/`. **Detector A is deterministic** (threshold and SVM). **Detector B
depends on the training run:** retraining from scratch gave 89.2% of falls caught, 100% no false alarm and 210 ms lead
(and 86.7% to 96% in step 2c runs). Do not quote B's sensitivity as a single number.

With 117 normal-activity trials, a measured 100% specificity is compatible with a true value around 97%.

## Firmware core vs Python (`firmware/test`)

12 test trials, three detector configurations, both from Python's own 50 Hz stream and from raw 200 Hz samples:

| Component | Maximum difference from Python |
|---|---|
| Filter + decimation | 4.6e-05 |
| Trunk angle / descent speed | 3.8e-05 deg / 3.6e-07 m/s |
| Features (all but zero-crossing rate) | 4.8e-07 relative |
| SVM decision value | 1.0e-04, same class on every window |
| Alerts and 8-byte payloads | 18 of 18 identical |

One window's zero-crossing count differed by a crossing (float32 against float64 on a sample sitting on the mean);
no decision changed.
