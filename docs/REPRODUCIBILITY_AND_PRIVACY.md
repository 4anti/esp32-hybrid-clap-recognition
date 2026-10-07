# Reproducibility and privacy

The checkout supplies the selected deployable model so someone can build an ESP32 gesture controller without receiving the original training audio. It preserves the method, source, experiment settings, aggregate outcomes, and model identity. It deliberately does not provide the complete private experiment archive.

## Included and omitted artifacts

| Included in the repository | Kept outside version control |
|---|---|
| Firmware, portable detector, frontend, and Lab source | Wi-Fi, lamp, and Lab credentials |
| Room-v3 INT8 weights and normalization in `clap_double/model_data.h` | Raw audio, recording directories, and session metadata |
| Configuration template with placeholders | Device addresses, identifiers, and private diagnostic logs |
| Training, conversion, replay, and integrity-check tools | Feature archives, checkpoints, and other generated exports |
| Aggregate model results and experiment history | Personal photos, videos, and unrelated project files |
| Synthetic test fixtures and third-party notices | Deployment binaries containing local configuration |

The model is a classifier represented by weights, normalization, and tensor metadata. It does not contain WAV/PCM files and cannot be played as a recording. It was learned from local recordings, however; excluding the raw files is not a formal guarantee against information inferred from learned parameters. Distribution of this one model was explicitly chosen. Other generated models remain excluded. Its model hash and [license](MODEL_LICENSE.md) identify the approved artifact.

## What can be reproduced

- **Firmware behavior:** build the included model with your own ignored configuration, execute the portable tests, and check the model identity on your board.
- **Pipeline method:** acquire permitted public datasets, regenerate features, train, convert, and evaluate using the published code and settings. Keep source and uploader groups across splits.
- **The selected room run:** its architecture, seed, settings, model identity, and aggregate results are documented. Exactly repeating the training or the private recording replays requires the omitted private data and checkpoints. A new dataset produces a new experiment, not an independent replication of the reported private-recording results.

The deployment guide does not require Python, TensorFlow, or access to the private data. Training/export dependencies are only needed to run the research workflow.

## Local capture and publication checks

The recording Lab writes captures locally under `data/sessions` by default and binds the PC server to loopback, keeping its routes on that computer. Setting `CLAP_LAB_HOST=0.0.0.0` is an explicit LAN-sharing opt-in. The PC server's saved-recording routes have no authentication; the board password protects the board service, not these routes. Use LAN sharing only when you intend that access on a trusted network. A phone can instead open the board's own password-protected page.

Browser monitoring gain does not change recorded PCM. Originals are preserved; trimming and augmentation use derived material grouped with the original source. Captures with packet gaps, incomplete writes, or failed closure stay review-only.

Real deployment values belong in the ignored `clap_double/clap_local.h`. Diagnostics and builds belong under ignored local directories. Do not paste credentials into source, screenshots, issue reports, or example commands.

Before publishing, stage only intended files and run:

```powershell
python tools/audit_publish.py
```

The audit inspects the exact staged Git blobs. It checks an allowed file set, private paths/media, credential patterns and locally known credential values, personal paths, network identifiers, recording identifiers, and approved binary assets. Only the reviewed current model header is accepted; a replacement requires an intentional review and audit update. Findings print file names and categories without printing matching secret values. A passing scan supports the manual staged-file review; it cannot establish the absence of every possible form of private information.

Private repository visibility is an access setting, separate from excluding sensitive files. Making the repository public later should be a deliberate action after another staged/history review, model-license review, and publication audit.
