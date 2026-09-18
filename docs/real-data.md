# Real datasets

Everything so far was trained and tested on **simulated** signals. This page explains which real datasets to use,
how to get them, and how to run the pipeline on them.

> [!IMPORTANT]
> The loaders in `prefall/realdata.py` were written from each dataset's published documentation and **tested only on
> generated files in the same formats** (`prefall/mockdata.py`). They have not been run on the real downloads, because
> SisFall is large and KFall is approval-gated. If anything looks wrong on the first real run, use `--inspect` (below)
> and the fix is usually a one-line change.

## The datasets

| Dataset | What it is | Sensor placement | Timing labels | Access |
|---|---|---|---|---|
| **SisFall** | 38 subjects (23 adults aged 19-30, 15 elderly aged 60-75), 19 daily activities + 15 fall types, 4,510 files, 200 Hz | Waist | None (impact must be estimated) | Free download |
| **KFall** | 32 young subjects, 21 daily activities + 15 fall types, 5,075 files (2,346 falls) | Low back | **Fall onset and impact frame for every fall**, from synchronized video | Request form, approval by email |
| **FallAllD** | 15 participants, 26,420 files, 238 Hz accelerometer and gyro, magnetometer, barometer | **Waist, wrist and neck** | Impact centred at 10 s in each 20 s file | IEEE DataPort, open access |

Things that matter when reading results:

- **SisFall's elderly subjects performed only daily activities** (one exception, SE06, who also simulated falls).
  So SisFall tells you about false alarms for elderly people, but its falls are almost all from young adults.
- **KFall is the only one here with real fall-timing labels**, so it is the one to trust for lead time.
- **SisFall and KFall are waist or back only.** They say nothing about a wristband. FallAllD has wrist data and is the
  natural next dataset for the insole-vs-wristband comparison. Its file format was not verified, so there is no loader
  yet: after you download it, send me a directory listing and the first lines of one file and I will write it.
- Other datasets exist (for example UNIVRFall on Zenodo, which follows the KFall format, and FARSEEING, real-world falls
  on request). They are not covered here.

## Recommended order

1. **Rehearse with generated data** (5 minutes, below). Proves your setup works before the real files arrive.
2. **SisFall first**: it needs no approval.
3. **Request KFall now**, since approval takes time.
4. Later: FallAllD for the wristband.

## Rehearsal with generated data

```
pip install openpyxl
python tools/make_mock_datasets.py
python scripts/run_real_data.py --dataset sisfall --path data/mock/sisfall
python scripts/run_real_data.py --dataset kfall   --path data/mock/kfall
```

These files are fake (made by the simulator, written in the real formats and in a scrambled axis frame). The numbers
mean nothing about real falls; the point is that the commands run.

## Getting SisFall

1. The dataset is described in Sucerquia et al., *Sensors* 2017 (doi 10.3390/s17010198). The authors' group is SISTEMIC at
   Universidad de Antioquia. A copy of the data with its readme is also on GitHub:
   `github.com/Fall-Prevention-Team/sisfallData` (a third-party mirror).
2. Put it under `data/real/SisFall`, for example:
   ```
   git clone --depth 1 https://github.com/Fall-Prevention-Team/sisfallData.git data/real/SisFall
   ```
   It is large (many files); expect a long download.
3. Check it: `dir data\real\SisFall\SA01` should list files like `D01_SA01_R01.txt`.

The expected layout is one folder per subject (`SA01` ... `SE15`), each holding files named
`<code>_<subject>_R<trial>.txt`. The loader searches recursively, so extra folder levels are fine.

## Getting KFall

1. Fill in the request form linked from `sites.google.com/view/kfalldataset`. The KFall team checks your identity and
   institution and emails you access. Use your university email.
2. Download `sensor_data.zip` and `label_data.zip` and unzip **both** into `data/real/KFall`, so you get:
   ```
   data/real/KFall/sensor_data/SA06/SA06T01R01.csv ...
   data/real/KFall/label_data/SA06_label.xlsx ...
   ```
3. **Do not redistribute it.** The terms forbid passing the data to third parties. That means: never commit it to GitHub
   (`data/` is already in `.gitignore`), and do not upload it anywhere, including to me. If something fails, `--inspect`
   prints just the first few lines of a file, which is enough.

## Running

```
python scripts/run_real_data.py --dataset sisfall --path data/real/SisFall
python scripts/run_real_data.py --dataset kfall   --path data/real/KFall
```

Add `--limit-subjects 6` for a quick first run, and `--inspect` to print the first lines of a few raw files
(nothing is trained). SisFall is big, so the full run takes a while; start with `--limit-subjects`.

What it does: loads every recording, converts units, works out the axis mapping, **splits by subject** (nobody is in both
train and test), calibrates the threshold on the training subjects, trains a Random Forest and an SVM, and scores the
held-out subjects with the same detectors as before. It saves `data/real_<dataset>_results.json`.

### Check the axis mapping first

Neither dataset documents which physical axis points up, so the loader estimates it and prints it:

```
axis map: accel: ax<-+z, ay<-+x, az<--y | gyro: ...
resting gravity vector (native axes, g): [-0.001, -0.993, -0.056], gyro match (sagittal): [-0.01, 0.01, 0.36]
```

- The **resting gravity vector** should have one component near 1 (or -1) and the others near 0. If none is near 1,
  the unit conversion is wrong.
- The **gyro match** numbers show how well each gyro channel follows the tilt seen by the accelerometer. One value should
  clearly stand out from the others (real data with more bending and sitting than the simulator should show a higher one).
  If they are all small, the gyro is probably in a different frame, and the threshold detector will be unreliable
  (the model-based detectors do not depend on this).
- The choice of the forward axis is a heuristic (the horizontal axis that swings more). If forward and sideways are
  swapped, sagittal falls will look like the wrong kind of tilt; the per-fall-type results below show it.

### Reading the output

The table has the same columns as the rest of the project (falls caught, no false alarm, lead time, share with 200 ms or
more). Also printed: which fall types are caught (detector A), the normal activities that trigger it most, and, for
SisFall, the false-alarm rate on elderly subjects separately.

## What to expect, and caveats

- **Lower numbers than the simulator, likely by a lot.** Real recordings have noise, different walking styles,
  sensor slippage and fall types the simulator does not model.
- **SisFall lead times are approximate.** Impact is estimated as the acceleration peak and the fall onset is assumed to be
  1 second earlier (`--onset-window`). Falls with no clear peak (under 2 g) are dropped and counted.
- **The threshold gate assumes forward or backward tilt.** Lateral falls tilt the other axis, so expect the threshold
  detector to miss more of them. The fall-type breakdown will show this (SisFall lateral falls are F03, F09, F12, F15).
  The model-based detectors use features that do not depend on the tilt direction.
- **Small numbers of subjects.** With 30% of subjects held out, a handful of people decide the result. Change `--seed` to
  see how much it moves.
- **Only classical models here.** The deep models can be trained on these windows too; that is a next step once the
  classical results look sensible.

## Citing

- Sucerquia A., Lopez J.D., Vargas-Bonilla J.F. SisFall: A Fall and Movement Dataset. *Sensors* 2017, 17(1), 198.
- Yu X., Jang J., Xiong S. A Large-scale Open Motion Dataset (KFall) and Benchmark Algorithms for Detecting Pre-impact
  Fall of the Elderly Using Wearable Inertial Sensors. *Frontiers in Aging Neuroscience* 2021.
- Saleh M., Abbas M., Le Jeannes R.B. FallAllD: An Open Dataset of Human Falls and Activities of Daily Living for Classical
  and Deep Learning Applications. *IEEE Sensors Journal* 2020.
