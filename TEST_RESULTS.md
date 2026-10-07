# Validation record — 6–7 October 2026

Development and validation used the `hybrid-ai` branch. The connected board is an ESP32-D0WD-V3 rev 3.1, dual-core 240 MHz. Builds use Arduino CLI 1.5.1 and Arduino-ESP32 3.3.11, the normal ESP32 Dev Module flash layout, and the bundled modern TFLM runtime. No PSRAM is required. Raw evidence, credentials, and recordings remain in ignored local files; this document reports anonymized measurements.

## Host checks

- Final Python/C++/privacy suite: **123 passed**, no skips (142.77 seconds). This includes real TensorFlow INT8 conversion, float parity, source/group isolation, actual C++ detector/capture execution, C++/Python frontend parity, domain sampling/calibration, provenance-checked fine-tuning, function-preserving channel widening, actual pairer replay with AI vetoes, 25 protocol/recovery cases, and 48 publication-privacy cases. One TensorFlow converter warning is present; integer tensor/operator checks still pass. The earlier 50-test run took 44.42 seconds before recovery and privacy tests were added.
- Final recording/server suite: **10 passed**, including simultaneous upload/close ordering, failed saves, sample-offset integrity, device timestamps, raw recording independent of listening gain, synthetic nine-character passwords, loopback-only default binding, and explicit LAN opt-in with a recording-access notice.
- Frontend golden cases: silence, impulse, pure tone, random PCM and clipping; float tolerance 0.003 and INT8 tolerance one bin. Optimized target-region inputs are exactly equal to full C++ inputs for all retained frames. Unused frames contain quantized zero.
- The same C++ executable independently decodes batched WebSocket metadata/audio frames, checks payload lengths across 125/126 bytes, validates the unmasked server format, and tests capacity/oversize rejection.
- New native tests extract and execute the actual WebSocket reader, lamp rediscovery, and Wi-Fi event-handler function bodies with deterministic mocks. They cover fragmented/coalesced masked pings, invalid lengths, failed writes and close cleanup, shared 400 ms handshake deadlines, clock wrap, saved-address-first retries, dead sockets, and Wi-Fi loss/reconnection. These verify the logic, not the real driver or access-point behavior.
- Privacy cases inspect staged Git blobs, reject private recordings/configuration/paths/identifiers and binary disguises, and pin the single approved deployment header and reviewed fonts. Decoded model metadata is checked for private text. This is a publication safeguard, not a formal privacy proof for learned parameters.
- The preserved public model has 6,139 parameters, 12,752 bytes, eight integer operators, and SHA256 `7cb2729f62cfc1ae2b102283baefb4a5a8b8b3816491befab045321b28a803ea`. See [MODEL_STATUS.md](MODEL_STATUS.md) for its public-data results. The current source header contains the room model described below; that does not by itself establish an upload.

## DSP upload and device check

The improved DSP firmware was built, uploaded to COM3 with flash verification, and tested through authenticated HTTP and WebSocket audio. It used 1,027,744 flash bytes and 65,744 static RAM bytes.

- 20-second audio capture: **317,696 samples in 1,664 packets**, zero sample-position gaps, zero packet-metadata errors, zero additional dropped samples.
- Wi-Fi, Tuya and I2S stayed connected. Heap was 157,148 bytes before streaming and 154,448 in the last stream status; minimum heap was 153,204 bytes.
- The test ran in a quiet room and produced no gesture events. This establishes streaming continuity, not clap/snap recall or false toggles per hour.
- The first checker included an extra 30-second WebSocket close wait in its elapsed value. Sample collection itself was 20 seconds; the checker now reports capture time separately and bounds close time.

## AI device check

The password-enabled AI-shadow build was uploaded and hash-verified after a USB reconnect resolved Windows error 31 on COM3. The later streaming-fix build was also uploaded and verified: **1,150,012 program bytes (87%)** and **65,840 static RAM bytes**. It fits the normal flash layout and links the bundled optimized ESP-NN convolution/softmax functions.

- Actual allocation: 20,072-byte capture structure, a **32,768-byte selected arena**, and **12,732 bytes used** inside that arena.
- Device silence selftest: frontend **18 ms**, inference **66 ms**, probabilities **0.965 / 0.016 / 0.020**. Device dipole selftest: frontend **18 ms**, inference **65 ms**, probabilities **0.008 / 0.020 / 0.973**. Both match the TensorFlow INT8 golden outputs at the displayed precision.
- Before the streaming fix, two viewers caused substantial PCM FIFO loss even though AI drops/errors stayed zero. Tiny audio packets and several blocking network writes per packet limited throughput. Audio now batches 512 samples (32 ms), sends separate metadata/audio WebSocket frames in one bounded nonblocking TCP write, and drops a slow viewer instead of blocking every viewer.
- After the fix: **20.02 seconds, 622 packets, 318,464 samples, zero sample gaps, zero metadata errors and zero additional dropped samples**. The final status had two listeners, Wi-Fi/Tuya/I2S live, AI shadow mode, zero AI drops/errors and inference time 65 ms. Heap was 87,144 before streaming and 84,188 in the last status. This is a short streaming smoke test, not sustained recognition evidence.
- The first new bag-only take saved **25.664 seconds / 410,624 raw samples** with no drops, gaps, metadata errors or clipped samples. The user confirmed its contents. It contains 25 AI candidate decisions, all rejected by the public-data gate; DSP still controls the lamp in shadow mode. A single take cannot establish general bag rejection or gesture recall.

The AI capture allocation now stores three 2,880-sample windows and 1,280 history samples. This saves **40,960 bytes of PCM storage** versus three full 8,000-sample windows plus 6,400 history samples. It calculates 16 FFT frames rather than 48; the model crops ignored frames before convolution. The tensor input remains `[1,64,48,1]`.

## Room model and recorded pipeline

The new model has **13,527 parameters**, widths 12/24/48, a **20,872-byte** fully INT8 file and SHA256 `9e9f9dd8c0efee615e9a5d51fc2b6071bb01814d0b3c8e914c6be95b4dba9b94`. It retains the public checkpoint's learned normalization and verified source boundaries. Training-only bass/treble variation, gain, timing, masking and background mixing expand views without creating fake independent recordings. The selected checkpoint is epoch 12 of 27; two failed earlier adaptations were not installed for control.

- Float export parity: maximum error **0.000000306**. Validation INT8 probability change: maximum **0.0754**; clap gate recall drop **0.91 percentage points**, snap gate recall unchanged. All supported operators and tensors are integer.
- Room threshold **0.85**, calibrated on Lab validation only. Room validation/test have **0 / 77** and **0 / 92** accepted noise windows. Combined INT8 test noise acceptance is **164 / 2,456 = 6.68%**; public-only noise acceptance is **164 / 2,364 = 6.94%**. Public clap-window acceptance is **60.58%**, snap **57.99%**. These weakly labelled public windows do not establish false lamp toggles; their higher noise acceptance is a reason to retain DSP protections.
- Actual C++ detector + INT8 + actual pairer replay at arm 281,840: the close bag prefix produces **13 DSP doubles → 0 after the veto**, with **0 / 63** candidates approved. The other two bag recordings produce zero AI approvals out of 25 and 14 candidates.
- Positive replay: AI approves **12 / 12** reviewed clap proposals and **20 / 20** snaps. The conservative DSP policy still admits 10 claps and 19 snaps; its **2 clap pairs and 9 snap pairs are preserved** after the veto.
- The positive recordings and close bag prefix were used for training. The two held-out room takes are bag-only. These are fit/replay results, not independent positive recognition evidence. Bag exposure totals 94.28 seconds, including 64.192 held-out seconds. No sustained false-toggles/hour claim follows.
- The full close bag recording is preserved. A derived prefix omits the user-identified three-second speech tail; its original source group/hash and exact sample removal are recorded. Original/derivative duplicate training is prevented.

Artifacts, reproduction commands and split limits are in [ROOM_MODEL_STATUS.md](ROOM_MODEL_STATUS.md). The source default is an experimental active AI veto; explicit shadow/DSP modes remain available.

## Room model upload and device runtime

The active room build was compiled and uploaded to COM3, with flash hash verification and a successful reboot. It uses **1,158,704 program bytes (88%)** and **65,840 static RAM bytes**. Firmware binary SHA256: `f3a70899530872c0ec3f5c44b3f8562839a13f2fbdb31ffda32d092ff95ae84c`. The boot log identifies the exact room model hash and **active** mode. The canonical firmware and matching private header were copied and hash-checked across all 12 files in `Documents/Arduino/clap_double`, with the prior sketch preserved in a timestamped backup.

- Actual arena: **32,768 bytes allocated, 18,076 used**. Capture remains 20,072 bytes. This requires no PSRAM and retains the 40,960-byte PCM-storage saving.
- Silence and dipole device goldens both pass: scores **0.996 / 0.004 / 0.000** and **0.637 / 0.016 / 0.348**, matching the expected TensorFlow INT8 outputs within the checker tolerance (one output bin plus display rounding).
- Frontend **18 ms**; selftest inference **115 ms**, with actual transient inference **113–114 ms** in the stream check. This is worker computation time, not measured lamp latency. Capture continues while the worker runs.
- New quiet calibration: floor **22,662**, arm **181,296**. A second replay at this arm level approves **0 / 43**, **0 / 27**, and **0 / 63** candidates across the three bag takes. The close bag's **7 DSP doubles become 0**; the **2 clap pairs and 9 snap pairs remain**. This checks the changed detector setting on the same recordings; it is not another independent dataset.
- Active-mode stream check: **20.02 seconds, 622 packets, 318,464 samples**, **zero sample gaps, metadata errors or additional drops**, with two listeners in the final status. The cumulative PCM-drop counter was 240 before collection and stayed 240; the result does not claim zero startup loss.
- Wi-Fi, I2S and Tuya stayed live; AI drops/errors and command drops stayed **zero**. Heap was **84,088 bytes** initially and **80,372** in the final status; initial recorded minimum heap was **73,360**.
- A new room transient was rejected in active mode. This quiet/brief check does not establish live clap/snap sensitivity or sustained close-bag rejection.

A subsequent live trial asked for double claps, double finger snaps, and close bag handling. The user reported “that is so good.” This is qualitative confirmation of improved behavior, without per-gesture counts or a measured observation duration. Hardware check artifacts are preserved under ignored `.build/device_*room-v3.*` files.

## Restart and streaming recovery — 7 October

The earlier long stream failed its WebSocket keepalive timeout. A raw masked-ping
probe reproduced missing pong responses despite audio delivery. HTTP upgrade
parsing left a line-feed byte unread, shifting the first control frame. The
new reader consumes complete CRLF lines, caps each line at 512 characters, and
shares a 400 ms deadline across the handshake. Invalid frames and failed pong
writes now stop parsing immediately after client cleanup.

Startup now checks queue and task creation, makes three bounded microphone
initialization attempts, and restarts after microphone or AI initialization
failure. Lamp recovery detects dead sockets, retries the last saved address
before scanning, and restores discovery after Wi-Fi loss/reconnection. The
network recovery functions and parser are covered by actual-body native tests;
allocation-failure and real driver behavior are not fault-injected on hardware.

The final active firmware **compiled and uploaded successfully**, with flash
hash verification: **1,161,072 program bytes
(88%)**, **65,912 static RAM bytes**. Its binary SHA256 is
`bd4c2721c7848f6a87f3ff37597b2760c67c7e6e477db6d99b6dd6f0ce326273`.
All 13 canonical sketch files were copied to the Arduino sketch directory and
hash-verified, preserving the prior files in a local backup.

Three consecutive checked **EN resets** restored active AI, microphone capture,
Wi-Fi, and lamp connectivity. Every boot identified the expected room model and
passed its silence/dipole selftests; the first check also compared outputs against
desktop TensorFlow INT8 goldens. Arena allocation remains 32,768 bytes, with
18,076 used; capture occupies 20,072 bytes. Frontend time is 18 ms and selftest
inference is 115 ms.

| Reset check | Arm after calibration | Stream time | Packets / samples | New gaps / metadata errors / drops | Cumulative drops before collection |
|---|---:|---:|---:|---:|---:|
| 1 | 335,072 | 20.00 s | 621 / 317,952 | 0 / 0 / 0 | 960 |
| 2 | 334,544 | 20.02 s | 620 / 317,440 | 0 / 0 / 0 | 0 |
| 3 | 351,568 | 20.02 s | 620 / 317,440 | 0 / 0 / 0 | 240 |

AI drop/error and lamp-command drop counters remained zero. The changing arm
values reflect fresh ambient calibration; the recorded replay at arm 181,296
above does not independently evaluate these newer thresholds. These quiet
checks establish startup and audio continuity, not live gesture recall.

A raw authenticated WebSocket upgrade followed by a masked `ping` now receives
the matching pong. The previously missing-pong failure is resolved in this
device check. Two controlled viewers then ran concurrently for **55.00 seconds**
each, with default client keepalive enabled:

| Viewer | Packets | Samples | New gaps / metadata errors / drops |
|---|---:|---:|---:|
| 1 | 1,715 | 878,080 | 0 / 0 / 0 |
| 2 | 1,717 | 879,104 | 0 / 0 / 0 |

The final status confirms two listeners, active AI, live microphone/lamp links,
and zero AI/command drop/error counters. Heap changed from 85,748 to 82,112 bytes;
the initially reported minimum heap was 78,876 bytes. Inference time was 114 ms.
The cumulative PCM-drop counter remained 240 throughout; this is not a claim of
zero loss since startup. Fifty-five seconds covers the old keepalive failure,
but does not establish hours of unattended reliability.

The successful three reset checks do not remove power from the external
microphone. In response to a requested five-second USB unplug/reconnect and
double-clap/double-snap trial, the user reported **“it works perfectly.”** This
is positive qualitative feedback; physical power removal was not instrumented,
and no per-gesture success counts or observation duration were supplied. Real
access-point/lamp outages and allocation failures are not fault-injected. Raw
upload/boot/stream evidence stays in ignored `.build/` files.

A final check **without resetting the board** after that feedback collected
**20.01 seconds, 622 packets, and 318,464 samples**, with zero new gaps,
metadata errors, or drops. Active AI and microphone/lamp connectivity remained
live, with AI/command error/drop counters at zero. The freshly calibrated arm
was 325,728 and the cumulative PCM-drop counter stayed 480 during collection.
This confirms operating continuity after the reported trial without replacing
it with another tool-triggered reset.

The PC recording server is separately verified listening on **127.0.0.1**.
Other LAN devices cannot access its recordings in default mode. LAN recording
access is a deliberate opt-in; local users/apps still have access.

## Recognition limits and next measurements

The preserved public model's gate accepted only 14.1% of weakly labelled clap windows and 30.9% of snaps, and accepted 1.35% of noise windows; it was tested in shadow mode. Room adaptation substantially improves the supplied-recording fit, but independent positive recordings and other room negatives remain untested. More epochs alone do not repair weak labels or a lack of varied conditions.

Fresh live gestures, success by class/distance, sustained false toggles and lamp latency still need measurement. Quiet streaming and synthetic inference are not substitutes. Legacy gain-adjusted recordings remain preserved and excluded from normal training.

Room adaptation supports domain/source sampling, explicitly selected validation noise budgets and warm starts that retain normalization and verify dataset/checkpoint hashes and split boundaries. Quantization calibration includes the room domain; INT8 thresholds use validation scores. Device boot now identifies the exact model hash, and the checker can compare silence/dipole inference with TensorFlow goldens.

## Run the tested paths

Open [Clap Lab](http://127.0.0.1:8788/), or run `lab/START.bat`. Use the connected ESP32 address shown at boot. New recordings must finish Save successfully and have `trainingReady: true` before training; verify their labels separately.

```powershell
# Build or build-and-upload the experimental comparison firmware.
./tools/build_firmware.ps1 -Mode shadow
./tools/build_firmware.ps1 -Mode shadow -Upload -Port COM3

# Build the recorded-pipeline-checked room veto for local testing.
./tools/build_firmware.ps1 -Mode active

# Copy canonical firmware tabs and headers to Documents/Arduino, with backup.
./tools/sync_firmware.ps1

# Re-run the host and authorized hardware checks.
./.venv-hybrid/Scripts/python.exe -m pytest tests -q
node --test tests/lab_recording.test.js
./.venv-hybrid/Scripts/python.exe tools/check_device.py --reset --port COM3 --seconds 20 --expect-ai-mode active --expect-export data/hybrid/room_v3/export
```

The canonical sketch is `clap_double/clap_double.ino`. The legacy root `.ino` gives an explicit redirect so it cannot silently upload stale code. Private passwords and boot/stream diagnostics remain in ignored local files.
