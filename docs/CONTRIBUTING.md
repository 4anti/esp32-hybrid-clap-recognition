# Contributing

Use the canonical sketch in `clap_double/`. Keep local credentials in the ignored
configuration header and keep datasets, generated exports, recordings, and diagnostic logs
outside Git. The single reviewed deployment header is included deliberately;
replacing it requires a model/license review and a publication-audit update.
Start model changes in shadow mode and compare them against the
existing detector and pairing policy.

Run the Python/C++ suite and Node recording tests described in the README. Tests
should establish behavior rather than repeat implementation details. Hardware
claims need the board, mode, exact model identity, measurement duration, and
counter baseline. Do not describe short quiet streaming as gesture accuracy.

Keep original captures intact. Apply trimming and augmentation to derivatives;
group derivatives with their original take across splits. Fit normalization and
quantization calibration on training data, select thresholds on validation data,
and reserve test data for reporting.

Before committing, stage only intended source and documentation, review the diff,
and run `python tools/audit_publish.py`. Use synthetic fixtures in reports and
tests. Never attach personal audio or private deployment logs to a contribution.
