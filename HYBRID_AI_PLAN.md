# ESP32 clap and finger-snap recognition: revised plan

Revised 7 October 2026. The goal is reliable double-gesture control on this ESP32 and INMP441, while keeping the local lamp controls and recording tools responsive. Both claps and finger snaps are target sounds. The current pairing policy accepts two verified target impulses, including a mixed clap/snap pair; their onsets must be 150–800 ms apart. A later same-class policy would require an explicit change.

The hybrid approach is useful, but the previous plan contained implementation errors and unsupported performance promises. Keep continuous audio capture and a cheap transient proposal stage, then test whether a classifier improves the complete detector. Accuracy, RAM, latency, and false toggles must be measured before AI takes control of the lamp.

## What was wrong with the previous plan

| Previous assumption | Correction and consequence |
| --- | --- |
| A DSP trigger guarantees zero missed events. | Any sound below the proposal gate is invisible to the classifier. Measure candidate recall on independently annotated recordings, particularly quiet and distant snaps. |
| AI guarantees zero false positives. | A classifier can confuse bags, keys, clicks, percussion, and claps. Test full double gestures and ordinary room activity; report false toggles per hour. |
| Sleep for 100 ms inside `clapTick()` before inference. | This interrupts microphone servicing and can miss the next impulse. Collect the trailing samples continuously and process a completed snapshot in a worker. |
| Use four classes, then train three outputs. | The only supported deployment map is `noise`, `clap`, `finger_snap`, in that order. Hands rubbing, thigh slaps, and other unintended hand sounds are negatives. |
| Either 250 ms or 500 ms is interchangeable. | The frontend, examples, model tensors, and firmware must share one versioned contract. This revision uses 500 ms. Changing it requires regenerated features, retraining, and fresh parity tests. |
| A small model file means little RAM use. | Weights, activations, scratch, queues, audio windows, stacks, networking, and the interpreter all consume different budgets. Flash size does not establish a viable tensor arena. |
| The CNN is perfect and always runs in 50–100 ms. | ResearchCNN is a reference candidate. Compare measured INT8 accuracy, arena use, and worst-case execution time before keeping it or choosing a smaller network. |
| Core 0 automatically owns every website operation. | Application tasks need explicit placement and synchronization. Core separation helps throughput but shared resources still cause contention. |
| Remove most negative sound types because they are irrelevant. | Broad negatives help prevent unexpected triggers. Add hard negatives from the actual room; curate only when a measured experiment justifies it. |
| Randomly split sliced windows. | Related windows can leak the same recording into training and testing. Split by source recording and capture group before feature normalization. |

The old suggested `delay(100)` integration and zero-error claims are superseded by this document.

## One audio and model contract

| Item | Value |
| --- | --- |
| Input | Mono 16,000 Hz; unamplified signed PCM16, normalized by 32768 |
| Window | 8,000 samples: 6,400 before the trigger and 1,600 starting at the trigger |
| Analysis | Periodic Hann, 400 samples; hop 160; FFT 512; no centering |
| Spectrum | Magnitude, not power |
| Features | 64 HTK mel bands, 125–7,500 Hz; natural log of magnitude sum plus 0.001 |
| Shape and layout | `[64, 48]`, mel then time |
| Model input | Normalized INT8 NHWC `[1,64,48,1]`; tensor quantization stored with model |
| Target region | Frames 32–47 cropped before convolutions; about 80 ms before this trigger and 100 ms afterward |
| Normalization | Per-mel mean/std learned from the training target region only |
| Classes | `0: noise`, `1: clap`, `2: finger_snap` |
| Contract version | `logmag_htk64_16k_500ms_v1` |

`hybrid/frontend.py` defines the Python contract. `clap_double/clap_frontend.h` and the generated sparse filter tables implement its portable C++ counterpart. Host tests compare actual C++ outputs with Python for silence, impulses, tones, random PCM, and clipped PCM. The present FFT is portable C++, so the plan does not claim an installed or benchmarked esp-dsp backend. An optimized backend can be introduced after demonstrating parity.

The deployed header includes the complete label map, frontend version, normalization, and INT8 tensor quantization information. The current model crops the target region before its convolutions so an earlier clap elsewhere in the 500 ms history cannot validate a later noise event. Reverberation inside the cropped region still requires hard-negative testing. Binary noise/clap experiments cannot be installed as clap-and-snap models. Representative quantization examples must cover all classes and realistic signal levels. Full integer quantization also needs explicit integer input/output settings and supported operators; requesting optimization alone can leave float tensors. See the official [integer quantization guide](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant).

## Continuous capture and cooperation between cores

The two cores should cooperate through small queues and immutable candidate snapshots. Their shared RAM does not increase when another core is used, and a single inference will not automatically become twice as fast. Espressif documents shared memory, affinity, scheduling, and the need for proper cross-core locking in its [FreeRTOS guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/freertos_idf.html).

| Work | Placement and responsibility |
| --- | --- |
| I2S capture and sample-based DSP/pairing | Core 1; audio task/Arduino loop at priority 4. Keep processing every input sample. |
| Optional feature extraction and TFLM inference | Core 1, lower priority 2; runs while capture waits for I2S data. Never holds a shared lock during inference. |
| Tuya, local HTTP/WebSocket service, serial logs | Core 0 application tasks, alongside networking; use bounded queues and avoid blocking audio. |

This scheduling is implemented in the capture/worker integration. `CLAP_ENABLE_AI` gates the worker, and `CLAP_AI_SHADOW` separates observation from control. The public model was tested in shadow mode (`1/1`). The newer room-adapted model has passed recorded pipeline checks and is the local active-veto test configuration (`1/0`); live gestures and sustained negatives remain required before a dependable-release claim. AI can reject a DSP proposal, while the existing first-impulse, slam and echo protections remain in force. Build/runtime outcomes belong in `TEST_RESULTS.md`; task placement alone does not establish timing or capture continuity.

The portable detector runs sample by sample, including during network activity. Candidates carry their original onset and decision sample positions. The pairer measures onset spacing, preserves the first onset across echoes, applies a refractory interval after a completed pair, and resets after an audio discontinuity. Async verification results must be consumed in onset order. Expiring a pending pair from inference completion time would make fast doubles unreliable.

The AI capture path uses fixed history and three owned candidate slots. Version-3 models crop frames 0–31 before any convolution, so firmware stores only the required 80 ms before and 100 ms after the onset (2,880 samples). It computes frames 32–47, fills ignored tensor inputs with quantized zero, and retains the logical `[64,48]` contract. Host tests verify that every retained input exactly matches the full frontend. This saves 40 KiB of PCM storage and reduces FFT work from 48 frames to 16. A candidate becomes ready when its trailing samples have arrived; capture never waits for inference. Full slots, queue overflow, missing results, invalid metadata, or inference errors reject/reset the relevant AI sequence and increment telemetry. Disabling AI selects DSP-only operation; shadow mode reports scores while DSP decides lamp toggles. Arduino-ESP32 3.3.11 bundles the modern `esp-tflite-micro` runtime; the old TensorFlowLite_ESP32 library is unnecessary.

### Memory budget before enabling the worker

| Allocation | Raw size/configuration |
| --- | ---: |
| Three PCM16 target-region windows | 17,280 bytes |
| Pre-trigger history | 2,560 bytes |
| FFT frontend scratch | Approximately 5,124 bytes |
| INT8 feature tensor | 3,072 bytes, normally inside the tensor arena |
| Adaptive TFLM arena | Try 32, 48, 64, then 96 KiB; 96 KiB maximum |

Candidate PCM, history, and frontend scratch total 24,964 raw bytes, before ownership metadata. Adding the arena produces roughly 56–120 KiB before task stacks, queues, I2S/DMA, Lab PCM, networking, and interpreter overhead. Allocation preserves a 32 KiB internal-heap reserve plus the 8 KiB worker stack and interpreter budget. If no arena fits or tensors fail validation, AI reports unavailable. Actual selected/used arena, heap, timing, and continuity observations belong in `TEST_RESULTS.md`. TFLM stores tensors and scratch in a shared arena; see its [memory management notes](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md).

## Recording and trustworthy labels

1. Run `lab/START.bat`, connect to the ESP32 with the serial Lab password, and wait for a live waveform. Restart the Lab server after upgrading its files.
2. Use the Clap and Finger snap paddles for separate classes. Listen and file playback must stay muted while recording. Browser listen gain changes monitoring only; new firmware recordings use raw PCM.
3. Start with at least three genuinely separate recording groups per class so train, validation, and test can each contain that class. This is a minimum for a valid split, not enough evidence for a reliable product. Collect roughly 100 or more examples of each positive class across several takes and conditions, then expand based on observed failures.
4. Include near, medium, and intended across-room positions; different directions, hand strengths, and background sound. Distant snaps are quieter than claps and may fall below the microphone/proposal noise floor. Test that limit directly.
5. First record isolated impulses with quiet gaps. Record intended double claps, double snaps, mixed target pairs, and incorrect timing in additional takes for end-to-end evaluation.
6. Record Tongue click, Door slam, and Room noise/other negatives: bags, keys, dishes, typing, speech, coughing, music, hand rubbing, thigh slaps, and uninterrupted normal room use. Include single impulses and pairs of negative impulses.
7. Press Save and wait for completion. The Lab serializes uploads, verifies sample offsets, preserves device-timed event metadata, and drains pending writes before closing the WAV. Missing packet metadata, gaps, dropped PCM, or upload failures mark a take review-only. Legacy gain-dependent recordings are excluded from normal preparation.
8. Review positive recordings. Detector events are predictions, not ground truth. Verified onset indices can be recorded as `event_samples` in `session.json`; `split_group` keeps related takes from one capture run together. Automatically proposed positive onsets remain weak labels until reviewed. A human Mark miss click is a review aid, not a precise acoustic onset.

External data can bootstrap a three-class model while room data is being collected. ESC-50 applause is weak supervision for isolated claps; preserve its official source-aware folds. The preparation default treats every non-clapping ESC class as noise and retains broad negatives. The [ESC-50 release](https://github.com/karolpiczak/ESC-50) specifies the official folds and source metadata.

The current model uses 388 FSD50K recordings: public claps/snaps and hard negatives, selected from the [official release](https://zenodo.org/records/4060432) with its authoritative metadata. Selected clips are CC0/CC-BY; IDs, uploaders, license/attribution records, source URLs, CRC32, and SHA256 checks are preserved. Clips are isolated across splits by uploader/source and deduplicated against ESC. Clip labels can describe several sounds; they do not establish the exact event onset. See `MODEL_STATUS.md` for provenance and the remaining weak-label limitations. A real public-data baseline still requires local testing on this microphone.

## Preserved public model and current room experiment

The preserved public artifact is a genuine three-class INT8 model with 6,139 parameters, convolution widths 8/16/32, and a 12,752-byte TFLite file. Its eight supported operators are Conv2D, MaxPool2D, AveragePool2D, Reshape, Concatenation, FullyConnected, Softmax, and StridedSlice. Float parity and INT8 conversion checks passed; baseline artifacts, hashes, calibration, and results remain in `MODEL_STATUS.md`.

The baseline's held-out INT8 test accuracy is 78.72%, but its calibrated gate accepts only 14.11% of clap windows and 30.90% of snap windows, while accepting 1.35% of noise windows. FSD-only hard negatives had a higher 9.52% float false-accept rate. These are weakly labelled public event windows, not intentional gestures or false toggles per hour. That baseline supports shadow observation only.

The current room experiment has 13,527 parameters, widths 12/24/48 and a 20,872-byte INT8 model. It uses raw room clap, snap and close bag takes with controlled training-only augmentation. Complete recording replay eliminates 13 false doubles from the close bag training take at the recording arm threshold of 281,840. Repeating with the newer boot arm threshold of 181,296 eliminates seven false doubles; both replays reject every candidate in the two held-out bag takes and preserve all existing DSP clap/snap pairs. The model and AI threshold remain fixed between these checks. Positive takes were used in training, so this is sufficient to try an active veto locally but not to claim independent gesture accuracy. Full provenance, experiment settings, public-data tradeoffs and limits are in [ROOM_MODEL_STATUS.md](ROOM_MODEL_STATUS.md).

## Training and deployment sequence

Room adaptation uses equal class/domain/source sampling (`--balance-domains`) so microphone takes receive useful weight beside public recordings. The selected experiment assigns 70% domain probability to the room within each class (`--room-weight 0.7`). Noise budgets normally apply separately to each validation domain; the room experiment explicitly selects Lab for threshold calibration (`--calibration-noise-domains lab`) and retains the other domains' results. Public-noise acceptance is higher at this room threshold, so the AI veto retains the DSP guards. `--lab-max-windows-per-file` retains more room negatives while preserving the public extraction limit. Warm starts verify the original dataset/checkpoint hashes and reject moved source/group splits; learned normalization is retained. Explicit channel widening preserves the original model's initial inference function and has a prediction/gradient test.

Training generates different views on every pass. Baseline defaults support −18 to +6 dB gain, ±20 ms jitter and four-band masks; the selected room experiment uses gentler −6 to +6 dB gain, ±10 ms jitter, two-band masks, and bass/treble shelves ±2 dB. Optional quiet training-only background spectral mixing uses 20–35 dB SNR. Spectral mixing approximates uncorrelated energy; it is not exact waveform mixing. These transforms apply equally across classes and never to validation/test. Views remain in their original source/group and do not increase the independent-recording count. Spectrum augmentation is supported by [SpecAugment](https://arxiv.org/abs/1904.08779); this task's useful policy still requires validation.

At the owner's request, the last three seconds of the close bag take are removed in an auditable prefix derivative; the complete original is preserved and excluded from training to prevent duplicate use. The current room experiment places the close bag derivative and the single clap/snap takes in training, with the other two bag recordings in validation/test. Room gesture recall on those training takes is a fit diagnostic, not independent recognition evidence. New live gestures after upload remain necessary to verify control behavior.

INT8 conversion includes the small room domain in representative inputs when domain balancing is enabled. It calibrates the deployed threshold on INT8 validation scores, enforces the 0.85 firmware minimum, and checks gate recall changes as well as argmax-class recall. Public held-out and room bag test recordings do not tune this threshold.

Preparation and training live in this repository (`hybrid/`, `train_custom.py`). The baseline uses 8/16/32 channel blocks, the room model 12/24/48, both with target-region cropping and mean/max pooling. Package setup is recorded in [MODEL_STATUS.md](MODEL_STATUS.md); room reproduction commands are in [ROOM_MODEL_STATUS.md](ROOM_MODEL_STATUS.md).

```powershell
.\.venv-hybrid\Scripts\python.exe -m hybrid.download_fsd50k --output data/fsd50k --positive-count 170 --negative-count 50
.\.venv-hybrid\Scripts\python.exe -m hybrid.prepare --fsd50k data/fsd50k --max-windows-per-file 6
.\.venv-hybrid\Scripts\python.exe train_custom.py --epochs 50 --patience 10 --threads 2
.\.venv-hybrid\Scripts\python.exe -m hybrid.export
.\.venv-hybrid\Scripts\python.exe -m pytest tests
node --test tests/lab_recording.test.js
```

Preparation can also use `--local-only`, a specific `--esc50` path, or an explicit experimental binary class selection. Review `data/hybrid/meta.json` before training. Regenerate old datasets: the trainer requires the versioned schema and rejects unverifiable splits and frontend contracts. Missing classes in any split stop training instead of creating a misleading model. Legacy sessions are permitted only through an explicit exploratory option and must not become deployment evidence. Conversion has been exercised successfully in the recorded environment; package or architecture changes require fresh export checks.

For each experiment:

1. Split by original file/source and capture group. Keep ESC official folds; retain public dataset evaluation boundaries. Deduplicate cross-dataset source IDs and related recordings where metadata permits.
2. Fit normalization and any augmentation policy using training data only. Tune checkpoints and the accept threshold using validation only. Augmentations must preserve the label: attenuation/background mixing can help, while changing a snap into another transient cannot.
3. Evaluate the untouched test set with confusion matrices, per-class recall, noise false accepts, class counts, and dataset-domain breakdowns. Save dataset hashes, seeds, labels, frontend metadata, and checkpoint settings.
4. Export a fully INT8 model through a verified conversion path. Compare original, converted float, and INT8 outputs on the same held-out examples, including near-threshold cases. Check operators, tensor shape/layout, quantization, normalization placement, and probability interpretation.
5. Build the optional TFLM firmware and measure frontend time, inference time, end-to-end delay, peak RAM, stack use, capture stalls, and dropped queues/PCM. Test while HTTP, WebSocket audio, and Tuya traffic are active.
6. Begin with AI shadow mode. Replay independent recorded gestures through the same detector and then test real sounds. Enable AI lamp decisions only when it improves the complete behavior within the measured resource budget.

Model window accuracy is insufficient. Measure **candidate recall**, true double-gesture detection by class/distance, false toggles per hour, and lamp-response latency. Report missed examples and the amount of test exposure; zero observed false toggles over a finite test does not prove a zero underlying rate. Thresholds that eliminate observed noise by rejecting every gesture are not useful.

## Current status and remaining gates

- Continuous portable DSP and onset-based pairing, versioned raw Lab capture, ordered file uploads, source-disjoint training checks, and a matching C++ frontend are present in this revision.
- The capture/frontend/TFLM worker uses adaptive arena allocation and the bundled modern runtime. The room-adapted active veto is a local test configuration; a dependable-release claim still requires live gesture and sustained-noise evidence.
- Public and room-adapted three-class INT8 models have verified conversion/provenance checks. The public model remains preserved for shadow comparison; `MODEL_STATUS.md` and `ROOM_MODEL_STATUS.md` distinguish their evidence.
- Independently recorded room claps/snaps and sustained room-noise testing remain necessary to substantiate dependable recognition on this ESP32. No result here establishes zero false positives, zero misses, across-room snap sensitivity, or a sub-200 ms reaction time.
- Concrete host, compile, upload, and device observations are recorded separately in `TEST_RESULTS.md` by the hardware-testing pass. Do not infer a hardware pass from source inspection or host tests.
- The repository distributes source, the approved deployable INT8 model, configuration examples, and aggregate results. Recordings, other trained artifacts, session metadata, credentials, and diagnostics remain local; private repository visibility does not replace these exclusions.
