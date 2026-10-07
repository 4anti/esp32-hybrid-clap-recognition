# Security and privacy

Keep Wi-Fi credentials, Tuya keys, access passwords, device addresses, and raw
audio on your own computer. Use the ignored `clap_double/clap_local.h` for
deployment settings. The maintainer has approved the included compact deployment
model. Other private checkpoints and models remain excluded along with their
training audio.

The PC recording service binds to loopback by default, so other LAN devices
cannot fetch its recordings. Local users and software on that computer can
still access them. Explicit `CLAP_LAB_HOST=0.0.0.0` enables LAN access; its
recording routes have no authentication and should only be enabled deliberately
on a trusted network. The board's stream password does not protect those PC
routes or provide transport encryption. Do not expose either service to the
internet. Avoid including recordings, serial logs, or device identifiers in
issue reports. Provide a synthetic reproduction and anonymized counters instead.

Before any commit or push, run `python tools/audit_publish.py` against the exact
staged index. The allowlist and secret checks are an additional safeguard;
review changes as well. Private repository visibility is not a substitute for
keeping sensitive files out of Git history.

Report vulnerabilities through GitHub's private vulnerability reporting when
available. If unavailable, contact the maintainer privately without attaching
audio or credentials.
