# Put Clap Lights on your ESP32

This guide uses the included 20,872-byte INT8 model for personal home use or noncommercial study under its [model license](MODEL_LICENSE.md). You do not need to train a model, install Python, or obtain the project's private recordings. The tested board is a classic dual-core ESP32 at 240 MHz without PSRAM; other ESP32 families have not been validated by this project.

The model was adapted to one microphone and room. Start in shadow mode, test your own conditions, then enable its veto if the results are useful. The published experiment does not establish reliable recognition in every home.

## 1. Prepare the hardware

Use an ESP32 development board, an INMP441 microphone module, a USB data cable, and a compatible Tuya 3.5 lamp on your local network. The current lamp implementation uses data points 20 (on/off), 21 (white mode), 22 (brightness), and 23 (white temperature). A lamp with different data points or protocol needs a code change; this is not a universal Tuya driver.

Wire the microphone before powering the board:

| INMP441 module pin | ESP32 connection |
|---|---|
| VDD / VCC | 3.3 V |
| GND | GND |
| SCK / BCLK | GPIO 18 |
| WS / LRCLK | GPIO 15 |
| SD | GPIO 19 |
| L/R | GND for the left channel |

Use the module's pin labels rather than assuming its physical pin order. The [manufacturer's INMP441 datasheet](https://product.tdk.com/system/files/dam/doc/product/sw_piezo/mic/mems-mic/data_sheet/inmp441.pdf) specifies the supply and channel selection. Put the microphone where you intend to use it, with its sound port unobstructed.

## 2. Install the firmware toolchain

Install Arduino IDE and follow [Espressif's Boards Manager installation guide](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html). Add the stable package URL from that guide, then install **esp32 by Espressif Systems, version 3.3.11**. The tested target selection is **ESP32 Dev Module** with the standard flash layout. The build script uses Arduino CLI 1.5.1; it can find the CLI bundled with a standard Windows Arduino IDE installation or a CLI on your PATH.

This board package provides the modern TensorFlow Lite Micro/ESP-NN runtime. Do not add the old `TensorFlowLite_ESP32` library to this build. The canonical sketch is [clap_double/clap_double.ino](../clap_double/clap_double.ino); the root `.ino` is a redirect to avoid uploading a stale copy.

## 3. Supply your own local values

Copy [clap_local.example.h](../clap_double/clap_local.example.h) to `clap_double/clap_local.h` and edit that local copy:

| Setting | Your value |
|---|---|
| `CLAP_WIFI_SSID` | Your 2.4 GHz Wi-Fi network name |
| `CLAP_WIFI_PASSWORD` | That network's password |
| `CLAP_TUYA_DEVICE_ID` | Your lamp's device identifier |
| `CLAP_TUYA_LOCAL_KEY` | Your lamp's exact 16-character Tuya 3.5 local key |
| `CLAP_TUYA_IP` | Four comma-separated address octets, or `0,0,0,0` for discovery |
| `CLAP_LAB_PASSWORD` | Your own 6–32-character Lab password, or empty to retain/generate one |

Obtain the local key and device ID for your own lamp through your existing Tuya setup. This repository provides neither account credentials nor another person's device values. A factory reset or re-pairing can invalidate a previous local key.

The successful lamp address and Lab password are stored on the board. Reserving the lamp address in your router's DHCP settings can make later discovery simpler. Keep `clap_local.h` private: it is ignored by Git. Keep the ESP32, lamp, and recording computer on a local network that permits communication between them.

## 4. Upload shadow mode

Close the Arduino Serial Monitor before uploading so it does not hold the serial port open. From the repository directory in PowerShell, replace `<SERIAL_PORT>` with the board's port:

```powershell
./tools/build_firmware.ps1 -Mode shadow -Upload -Port <SERIAL_PORT>
```

The script compiles before it uploads. Shadow mode runs the included classifier and exposes its scores; the original DSP still decides lamp toggles. For an Arduino IDE upload, set `CLAP_AI_SHADOW` to `1` in `clap_config.h` first, open the canonical sketch, and select your board and port.

After upload, inspect the Serial Monitor at **115200 baud**. Keep the room quiet during the printed one-second noise-floor calibration. Confirm microphone startup, the expected model identity, `ready`, and Wi-Fi/lamp connectivity. The supplied model hash begins `9e9f9dd8`; the complete hash is in [the model status](../ROOM_MODEL_STATUS.md). The serial log also prints your board's Lab address/password; treat that log as private.

## 5. Observe and test

The recording interface needs Node.js; its current checks use Node 22. On Windows, run `lab/START.bat`; elsewhere run `node lab/server.js`. Open the local address printed by the server on that computer, enter the board address and your Lab password, and connect. `clap.local` may resolve through mDNS; use the printed device address if it does not.

The PC recording server is private to the computer by default. A phone can use the board's own password-protected page at `http://clap.local/` or the printed board address. Setting `CLAP_LAB_HOST=0.0.0.0` deliberately shares the PC Lab on the LAN; its saved-recording routes have no authentication, so use that option only on a trusted network when you intend to share access. The board password does not secure PC recording files.

Test separate double claps and double finger snaps from your intended position. Two accepted impulses must start **150–800 ms apart**; mixed clap/snap pairs are currently accepted too. Try ordinary room noises, especially close bag handling, keys, dishes, speech, and single impulses. Count successes, misses, and false toggles rather than relying only on the waveform or a high score. Observe classification scores and loss/error counters in the Lab.

Shadow mode intentionally does not veto noise, so the old detector can still toggle the lamp during this comparison. The important comparison is whether target candidates receive suitable scores and the unwanted candidates receive noise scores. Quiet or distant sounds can fail the DSP proposal gate before AI sees them.

## 6. Enable the veto and check restarts

If the comparison is useful, upload active mode:

```powershell
./tools/build_firmware.ps1 -Mode active -Upload -Port <SERIAL_PORT>
```

For the IDE route, set `CLAP_AI_SHADOW` to `0` and upload again. Repeat your counted gestures and negative tests. Active mode adds the AI decision to the existing first-impulse, slam, echo, timing, and refractory protections.

Once firmware is uploaded, recognition and lamp commands run on the ESP32. The computer, browser, and Lab do not need to remain open. Turn the ESP32 off and on, allow calibration/network reconnection to finish, and repeat a double gesture. Also test restoration after your lamp or Wi-Fi becomes temporarily unavailable. Hardware power cycles, software resets, and network recovery exercise different paths; the project's current measured evidence is recorded in [TEST_RESULTS.md](../TEST_RESULTS.md).

To compare with the original detector or disable AI allocations:

```powershell
./tools/build_firmware.ps1 -Mode dsp -Upload -Port <SERIAL_PORT>
```

## If it needs adjustment

| Symptom | First checks |
|---|---|
| Upload cannot open the port | Close other serial tools; verify the USB data cable and selected port |
| Microphone does not start or waveform is silent | Recheck power, common ground, pin labels, and L/R channel selection |
| Wi-Fi is unavailable | Verify local values, 2.4 GHz access, signal, and network isolation |
| Scores look correct but the lamp does not respond | Verify Tuya protocol, local key, lamp address/discovery, and data-point mapping |
| DSP works but active mode misses gestures | Compare scores in shadow mode; the room model may need your own recordings |
| Bag or other noise still triggers | Record separate hard-negative takes and evaluate without weakening all safeguards |
| Close sounds work but distant snaps do not | Check microphone placement and proposal sensitivity; AI cannot recover proposals it never receives |

Keep original captures, collect independent positive and negative takes, and follow [the training plan](../HYBRID_AI_PLAN.md) if you retrain. New recordings and augmentation do not replace untouched evaluation takes. The selected model's public held-out noise acceptance is **6.94%** at its room threshold (**6.68%** when held-out room noise is included); bag replay fit alone is insufficient to claim general noise rejection.
