"""Inspect saved raw takes; predictions are reported separately from labels."""
from pathlib import Path
import argparse
import collections
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hybrid.prepare import read_wav, transient_samples


def inspect(directory):
    meta = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if not meta.get("ended"):
        return {"id": directory.name, "state": "recording; wait for save"}, None
    audio, rate = read_wav(directory / "audio.wav")
    events_file = directory / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()] if events_file.exists() else []
    predictions = [event for event in events if event.get("kind") == "ai"]
    onsets = transient_samples(audio)
    integrity = meta.get("integrity", {})
    raw = meta.get("capture", {}).get("raw") and meta.get("capture", {}).get("pcmVersion") == 2
    contiguous = meta.get("capture", {}).get("deviceSampleEnd", 0) - meta.get("capture", {}).get("deviceSampleStart", 0) == len(audio)
    valid = bool(meta.get("trainingReady") and raw and contiguous and rate == 16000 and
                 integrity.get("samples") == len(audio) and not integrity.get("reasons"))
    report = {"id": directory.name, "label": meta["label"], "note": meta.get("note", ""),
              "seconds": len(audio) / rate, "samples": len(audio), "captureValid": valid,
              "clippedFraction": float((np.abs(audio) >= 32760 / 32768).mean()),
              "peak": float(np.abs(audio).max()), "rms": float(np.sqrt(np.mean(audio * audio))),
              "events": dict(collections.Counter(event["kind"] for event in events)),
              "aiAccepted": sum(bool(event.get("accepted")) for event in predictions),
              "aiCandidates": len(predictions),
              "proposedOnsetsNotGroundTruth": onsets,
              "integrity": integrity}
    return report, audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-10-06")
    parser.add_argument("--out", type=Path, default=ROOT / ".build/take_quality.json")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    reports, signals = [], []
    for directory in sorted((ROOT / "data/sessions").iterdir()):
        if directory.is_dir() and directory.name >= args.since and (directory / "session.json").exists():
            report, audio = inspect(directory)
            reports.append(report)
            if audio is not None:
                signals.append((report, audio))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: value for key, value in report.items() if key not in ("proposedOnsetsNotGroundTruth", "integrity")}
                      for report in reports], indent=2))
    if args.plot and signals:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(signals), 1, figsize=(12, 2.1 * len(signals)), squeeze=False)
        for axis, (report, audio) in zip(axes[:, 0], signals):
            # Min/max envelope retains brief impulses that simple decimation loses.
            padded = np.pad(audio, (0, (-len(audio)) % 160)).reshape(-1, 160)
            clock = np.arange(len(padded)) / 100
            axis.fill_between(clock, padded.min(1), padded.max(1), color="#177e89")
            axis.vlines(np.array(report["proposedOnsetsNotGroundTruth"]) / 16000, -1, 1,
                        colors="#db6443", alpha=0.3, linewidth=0.7)
            axis.set(title=report["id"], xlabel="seconds", ylabel="raw amplitude", ylim=(-1, 1))
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
