# Preserved experimental public-audio baseline

This document records the preserved public-data baseline for `noise`, `clap`, and `finger_snap`. Its held-out results justify shadow comparison only. The newer room-adapted local test model and recorded pipeline checks are described in [ROOM_MODEL_STATUS.md](ROOM_MODEL_STATUS.md); actual upload/runtime status is in [TEST_RESULTS.md](TEST_RESULTS.md).

## Reproducible model artifacts

The baseline artifacts are `data/hybrid/run/best.pt`, `data/hybrid/run/metrics.json`, `data/hybrid/export/model.tflite`, `data/hybrid/export/model_data.h`, `data/hybrid/export/export.json`, and `data/hybrid/export/sources.json`. These generated local artifacts are ignored by Git. The identical final run is preserved under `data/hybrid/public_full`; the earlier, smaller public-data run is under `data/hybrid/initial_public`.

- Model SHA256: `7cb2729f62cfc1ae2b102283baefb4a5a8b8b3816491befab045321b28a803ea`.
- Features SHA256: `3003a6bf7789acf86a3f853c7b9f7349289a85ca371989ec0f95d3e7b2b86c7d`.
- 6,139 parameters; convolution widths 8/16/32; 12,752-byte TFLite file.
- Requested 50 epochs; early stopping after 36; validation selected epoch 26. Selection used equal source weight within each class, with a validation noise false-accept target of 1%. Test results did not choose the checkpoint or threshold.
- Training samples classes and independent recording sources with equal probability. Training-only augmentation varies gain from -18 to +6 dB, timing by up to 20 ms, and occasionally masks up to four mel bands. Validation/test receive no augmentation.

## Data and provenance

388 FSD50K recordings were selectively acquired from the [official release](https://zenodo.org/records/4060432), using ZIP byte ranges instead of downloading the complete audio archive. Every selected WAV passed official ZIP CRC32, SHA256, and PCM-format checks. The subset has 249 uploaders and no uploader, source, or identical-file overlap across its splits. Only CC0 and CC-BY clips were selected; original attribution, license URLs, and source URLs are in `data/fsd50k/manifest.json`, `ATTRIBUTION.txt`, and `provenance.json`.

| FSD50K recording class | Train | Validation | Test |
|---|---:|---:|---:|
| Clap | 119 | 26 | 25 |
| Finger snap | 93 | 19 | 56 |
| Hard negatives | 35 | 8 | 7 |

The prepared dataset also uses ESC-50 noises from all environmental classes and clap recordings. Official ESC folds 1–3 are training, fold 4 validation, and fold 5 test. Source IDs `209698` and `234879` occur across official held-out splits and are excluded. Any original Freesound recording selected into FSD50K is excluded from ESC to prevent cross-dataset duplication. ESC split grouping is by original recording; its uploader identities are unavailable in the CSV. FSD uploader isolation therefore does not establish uploader isolation for the entire combined dataset.

ESC and FSD gesture labels describe whole clips. Automatic transient proposals are **weak labels**, especially for applause, speech mixtures, echoes, or clips containing several sound types. The test numbers measure performance on these proposals, not verified intentional gestures. New Lab data require closed, contiguous, raw PCM v2 capture at 16 kHz and `trainingReady: true`; old gain-adjusted sessions are skipped unless explicitly opted into for experiments. The original Lab WAV/session files were not changed.

| Prepared windows | Train | Validation | Test |
|---|---:|---:|---:|
| Noise | 7,246 | 2,383 | 2,364 |
| Clap | 843 | 219 | 241 |
| Finger snap | 118 | 115 | 576 |

FSD licenses do not cover ESC-50. ESC-50's [license file](https://github.com/karoldvl/ESC-50/blob/master/LICENSE) and original clip conditions must accompany redistribution decisions. This public baseline's generated artifacts and all training data remain excluded from the repository. Only the separately documented selected room-model header is distributed.

## Fixed held-out result

The validation-calibrated gate accepts when the highest-scoring class is a gesture and `1 - p_noise >= 0.9179978967`. The held-out test has 3,181 windows. INT8 classification accuracy is 78.72%; balanced three-class accuracy is 64.38%. These averages conceal substantial gesture rejection.

| Held-out INT8 result | Value |
|---|---:|
| Noise accepted by the gate | 32 / 2,364 = 1.35% |
| Clap windows accepted | 34 / 241 = 14.11% |
| Finger-snap windows accepted | 178 / 576 = 30.90% |
| Clap correctly classified (before gate) | 65.98% |
| Finger snap correctly classified (before gate) | 36.98% |

When independent sources have equal weight, the float gate accepts 22.69% of clap events and 44.20% of snap events. FSD-only test hard negatives produced four false accepts in 42 windows (9.52%), showing why the combined noise average is insufficient. These rates are not false light toggles per hour; the trigger and double-gesture state machine must be measured together on the actual microphone.

## Frontend and deployment contract

- Raw mono PCM, 16 kHz, signed 16-bit scaled by 32768; an event window is 6,400 samples of history and 1,600 samples after the trigger.
- 400-sample periodic Hann, 160-sample hop, 512-point FFT, 64 HTK mel bands from 125–7,500 Hz; natural log of **magnitude**, plus 0.001. No centering/padding inside the STFT. Feature layout is mel × time, 64 × 48.
- Mean/std are per-mel-band constants computed only from the training target region. Firmware normalizes features before INT8 quantization. Input tensor is NHWC `[1,64,48,1]`, scale `0.01899568736553192`, zero point `-18`. Softmax output is `[1,3]`, scale `1/256`, zero point `-128`.
- The model crops frames 32–47 **before** convolutions, so an earlier clap in the first 32 frames cannot validate a later noise event. This covers about 80 ms before the current trigger and 100 ms afterward. Reverberation that remains inside this region still requires hard-negative testing.
- Operator set: Conv2D, MaxPool2D, AveragePool2D, Reshape, Concatenation, FullyConnected, Softmax, StridedSlice. No float or custom operators. `model_version` is 3; label order is fixed.

PyTorch/Keras float softmax parity maximum error was `2.38e-7`. Full INT8 conversion calibrated on 300 **training** windows. Quantization acceptance used validation, including separate class-recall checks: maximum probability error was 0.0563, with no validation class-recall drop. Held-out INT8 behavior is reported afterward; it was not used to tune the quantizer or threshold.

Measured ESP32 heap, arena use, latency, watchdog stability, and audio/network continuity belong in the hardware test report. This model status does not claim those measurements before they are made.

For the boot inference plumbing check, the desktop INT8 reference gives these exact probabilities in noise/clap/snap order: all-zero PCM produces `[0.96484375, 0.015625, 0.01953125]`; zero PCM with samples 6400/6401 set to +24000/-24000 produces `[0.0078125, 0.01953125, 0.97265625]`. The latter is a synthetic impulse, not a labelled clap or snap. Small output-code differences can come from frontend/kernel arithmetic; large differences require investigating the deployment contract. Reference values are saved in `data/hybrid/export/selftest_expected.json`.

## Reproduce or retrain

Commands below use the tested Windows CPython 3.10 environment. CPU PyTorch avoids unnecessary GPU packages. Keep conversion in an isolated environment; TensorFlow is not needed to prepare data or train.

```powershell
python -m venv .venv-hybrid
.\.venv-hybrid\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-hybrid\Scripts\python.exe -m pip install -r requirements_hybrid_export.txt
.\.venv-hybrid\Scripts\python.exe -m hybrid.download_fsd50k --output data/fsd50k --positive-count 170 --negative-count 50
.\.venv-hybrid\Scripts\python.exe -m hybrid.prepare --fsd50k data/fsd50k --max-windows-per-file 6
.\.venv-hybrid\Scripts\python.exe train_custom.py --epochs 50 --patience 10 --threads 2
.\.venv-hybrid\Scripts\python.exe -m hybrid.export
.\.venv-hybrid\Scripts\python.exe -m pytest tests/test_hybrid_pipeline.py tests/test_hybrid_export.py -q
```

Preparation uses `data/esc50` by default, or accepts `--esc50 PATH`. `--local-only` omits ESC; it can still be combined with `--fsd50k`. For later room tuning, collect separate noise/clap/snap sessions in multiple conditions. `session.json` may contain verified `event_samples` (relative 16 kHz indices), `split_group` for shared capture runs, and explicit `split` values. FSM detections are not ground-truth annotations. Keep an independent room test set out of training and threshold selection.

The `--classes noise clap` preparation option creates an explicitly experimental binary baseline, excluding snap recordings. Default training requires every active class in train, validation, and test. Deployment export requires the complete three-class contract.

The supplied exporter freezes a concrete TensorFlow graph before conversion to avoid the TensorFlow 2.16/Keras 3 direct-conversion crash ([TensorFlow issue 63987](https://github.com/tensorflow/tensorflow/issues/63987)). Its successful run verifies the chosen package combination; future dependency changes require rerunning the export checks.
