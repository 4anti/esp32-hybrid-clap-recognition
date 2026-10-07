# Included deployment model

The single included `clap_double/model_data.h` contains the approved room-v3
INT8 network and feature normalization. It contains learned parameters rather
than playable audio. Its model SHA256 is
`9e9f9dd8c0efee615e9a5d51fc2b6071bb01814d0b3c8e914c6be95b4dba9b94`.
Recordings, feature archives, session metadata, and other checkpoints are not
distributed.

The included weights and normalization are offered under
[Creative Commons Attribution–NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Attribute **Clap Lights contributors**, link to this repository and its model
history, retain these notices, and identify modifications. This permits personal
home deployment and noncommercial study. The source code outside this generated
model header is licensed under the repository's MIT license.

Training combined a public-data baseline with permitted local gesture and bag
recordings. The baseline used selected FSD50K CC0/CC-BY clips and ESC-50. ESC-50's
dataset license includes a noncommercial restriction, which this distribution
retains for its included model. The model is not offered for commercial use.
Training code can be used with independently licensed data to produce a
separately licensed model.

- **FSD50K:** Fonseca et al., *FSD50K: an open dataset of human-labeled sound
  events*, IEEE/ACM TASLP 30 (2022), 829–852.
  [Official dataset](https://zenodo.org/records/4060432),
  [selected-clip attribution](FSD50K_ATTRIBUTION.txt).
- **ESC-50:** Karol J. Piczak, *ESC: Dataset for Environmental Sound
  Classification*, ACM Multimedia (2015).
  [Official dataset](https://github.com/karolpiczak/ESC-50),
  [dataset license](ESC50_DATA_LICENSE.txt).
- **Local adaptation:** deliberately labelled claps, finger snaps, and bag
  handling. Original recordings remain local; only anonymous aggregate results
  appear in the experiment record.

The current model is experimental. Its recorded room fit and public-noise
limitations are documented in [ROOM_MODEL_STATUS.md](../ROOM_MODEL_STATUS.md).
Distribution of a compact classifier does not establish a formal privacy
guarantee or reliable operation in every home.
