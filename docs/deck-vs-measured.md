# Seminar deck vs. what we measured

For anyone presenting the project. Every "measured" figure is from simulated data (see
[results](results.md)), so present these as findings from our own simulation, not as facts about real falls.

| Slide | The deck says | What we found | Suggested wording |
|---|---|---|---|
| 9 | 25 Hz Butterworth low-pass at 50 Hz sampling | A 25 Hz digital filter at 50 Hz sampling sits exactly at Nyquist | "Applied at the raw sensor rate, then downsampled to 50 Hz" |
| 9 | 1 s window, 25-sample hop (500 ms) | The hop is longer than the whole 200-400 ms pre-impact window | "Model evaluated every sample; decisions are 20 ms apart" |
| 11 | Random Forest about 12 KB; SMA and COP sway are the top features | 5.5-6.9 thousand nodes, about 65-80 KB; top features were gyro peak and peak tilt (COP needs insole data we did not simulate) | Report measured size; drop the feature-ranking claim or label it dataset-dependent |
| 12 | The BiLSTM's backward context separates a recovered stumble from a fall | Alone it does not (100% of stumbles flagged). With a 300 ms persistence requirement both deep models reject them | "Separating a stumble needs about 300 ms of confirmation" |
| 12, 13, 22 | Final model is a quantized ConvLSTM | ConvLSTM1D does not convert to TFLite built-in ops. The CNN-BiLSTM does (32.8 KB INT8) | Name the CNN-BiLSTM as the deployable model |
| 13 | FP32 180 KB to 38 KB after pruning and INT8; pruning halves the size | Our model: 81 KB to 32.8 KB (2.7x). Pruning left the file size unchanged | Report the measured sizes; say pruning helps compressed storage, not flash |
| 13, 15 | Under 20 ms inference; under 10 ms and 50 ms alert latency | Not measured: needs hardware | Mark as targets until measured on the board |
| 14 | Threshold first pass, deep model confirms | Waiting to confirm costs lead time one for one (the threshold fires about 238 ms before impact). Best two-stage result: SVM confirming immediately, 87% no false alarm at 237 ms | "Confirmation trades false alarms against lead time" |
| 20, 22 | 96% sensitivity, 320-350 ms lead, 94% specificity, 38-42 KB | These read as literature targets. Measured (simulated): 98.8% caught, 87-100% no false alarm, 225-237 ms lead | Replace with measured values, labelled as simulated |
| 23 | 200-300 ms post-event confirmation window | Supported: it is exactly what rejects stumbles, at the cost of lead time | Keep, and cite the trade-off |
