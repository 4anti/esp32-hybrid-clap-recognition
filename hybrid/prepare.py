"""Prepare trigger-aligned audio with auditable, source-disjoint splits.

ESC labels describe whole clips: automatic positive onsets remain weak labels.
Lab sessions can provide exact ``event_samples`` (16 kHz sample indices),
``split_group`` (same person/room/recording run), and an optional ``split``.
Never use the existing FSM's accepted events as ground-truth annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly

from hybrid.frontend import (
    FEATURE_SHAPE, LABELS, SAMPLE_RATE, WINDOW_SAMPLES, contract, event_window, log_mel,
)

NOISE_ESC = {
    21: "sneezing", 23: "breathing", 24: "coughing", 25: "footsteps",
    28: "snoring", 30: "door_wood_knock", 31: "mouse_click",
    32: "keyboard_typing", 34: "can_opening", 39: "glass_breaking",
}
CLAP_ESC = {22: "clapping"}
LOCAL = {
    "door_slam": "noise", "tongue_click": "noise", "other": "noise",
    "clap_far": "clap", "clap": "clap", "finger_snap": "finger_snap",
}
PEAK_FLOOR = 0.002
SPLITS = ("train", "val", "test")
ESC_CANDIDATES = (Path("data/esc50"),)


def find_esc50(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else list(ESC_CANDIDATES)
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "meta" / "esc50.csv").is_file():
            return candidate
        if candidate.is_dir():
            for child in sorted(candidate.iterdir()):
                if (child / "meta" / "esc50.csv").is_file():
                    return child
    raise SystemExit("ESC-50 not found. Pass --esc50 with meta/esc50.csv, or --local-only.")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Normalize before mixing channels; unsigned PCM has a midpoint offset."""
    rate, raw = wavfile.read(path)
    raw = np.asarray(raw)
    if raw.ndim not in (1, 2) or raw.size == 0:
        raise ValueError(f"empty or unsupported WAV shape {raw.shape}: {path}")
    if np.issubdtype(raw.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(raw.dtype).max + 1) / 2
        data = (raw.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(raw.dtype, np.signedinteger):
        data = raw.astype(np.float32) / float(-int(np.iinfo(raw.dtype).min))
    elif np.issubdtype(raw.dtype, np.floating):
        data = raw.astype(np.float32)
    else:
        raise ValueError(f"unsupported PCM type {raw.dtype}: {path}")
    if raw.ndim == 2:
        data = data.mean(axis=1)
    if rate <= 0 or not np.isfinite(data).all():
        raise ValueError(f"invalid sample rate or nonfinite audio: {path}")
    return np.clip(data, -1, 1).astype(np.float32), int(rate)


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if rate <= 0 or data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("bad sample rate or audio")
    if rate == SAMPLE_RATE:
        return data
    g = int(np.gcd(rate, SAMPLE_RATE))
    return np.clip(resample_poly(data, SAMPLE_RATE // g, rate // g), -1, 1).astype(np.float32)


def transient_samples(audio: np.ndarray, peak_floor: float = PEAK_FLOOR) -> list[int]:
    """Propose separated peaks of sample difference, like the DSP trigger.

    This is a proposal extractor, not an exact replay of the adaptive detector
    and not a substitute for human annotation.
    """
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim != 1 or not np.isfinite(data).all() or peak_floor <= 0:
        raise ValueError("invalid audio or peak floor")
    if data.size < 2:
        return []
    delta = np.abs(np.diff(data, prepend=data[0]))
    block = SAMPLE_RATE // 200
    padded = np.pad(delta, (0, (-len(delta)) % block))
    envelope = padded.reshape(-1, block).max(axis=1)
    floor = max(float(peak_floor), float(np.median(envelope)) * 6)
    peaks, _ = find_peaks(np.pad(envelope, (1, 1)), height=floor,
                         prominence=max(peak_floor / 2, floor / 4), distance=32)
    result = []
    for peak in peaks - 1:
        start = int(peak) * block
        section = delta[start : start + block]
        if section.size:
            result.append(start + int(section.argmax()))
    return result


def slices(audio: np.ndarray) -> list[np.ndarray]:
    """Background examples, including silence and the last partial window."""
    data = np.asarray(audio, dtype=np.float32)
    if not len(data):
        return []
    return [np.pad(data[start : start + WINDOW_SAMPLES],
                   (0, max(0, start + WINDOW_SAMPLES - len(data))))
            for start in range(0, len(data), WINDOW_SAMPLES)]


def assign_splits(files: dict[str, int]) -> dict[str, str]:
    """Stratify independent Lab groups, never individual windows."""
    by_label = defaultdict(list)
    for name, label in files.items():
        by_label[label].append(name)
    result = {}
    for label, names in sorted(by_label.items()):
        names = sorted(names)
        random.Random(42 + label).shuffle(names)
        n = len(names)
        n_test = max(1, round(n * 0.15)) if n >= 3 else 0
        n_val = max(1, round(n * 0.15)) if n >= 3 else 0
        for index, name in enumerate(names):
            result[name] = "test" if index < n_test else "val" if index < n_test + n_val else "train"
    return result


def build_dataset(esc50: Path | None, sessions: Path, labels=LABELS,
                  noise_policy: str = "all", max_windows_per_file: int = 8,
                  peak_floor: float = PEAK_FLOOR,
                  allow_legacy_sessions: bool = False,
                  fsd50k: Path | None = None,
                  lab_max_windows_per_file: int | None = None) -> tuple[dict, dict]:
    labels = tuple(labels)
    if not labels or labels[0] != "noise" or len(set(labels)) != len(labels) or any(x not in LABELS for x in labels):
        raise ValueError("classes must start with noise and use distinct known labels")
    if max_windows_per_file <= 0 or peak_floor <= 0 or noise_policy not in ("all", "curated"):
        raise ValueError("invalid preparation settings")
    if lab_max_windows_per_file is not None and lab_max_windows_per_file <= 0:
        raise ValueError("Lab window limit must be positive")
    features, records = [], []
    fixed_groups, local_groups = {}, {}
    notices = []
    source_provenance = []
    fsd_entries = []
    if fsd50k is not None:
        manifest = json.loads((fsd50k / "manifest.json").read_text(encoding="utf-8"))
        fsd_entries = manifest if isinstance(manifest, list) else manifest["records"]
    fsd_origins = {str(row["source_id"]).rsplit(":", 1)[-1] for row in fsd_entries}

    def add(audio, label, source, group, domain, category, onsets=None, split=None):
        if label not in labels:
            return  # Binary baselines exclude snaps, never silently relabel them.
        if split is not None:
            if split not in SPLITS:
                raise ValueError(f"invalid split {split} for {source}")
            if group in fixed_groups and fixed_groups[group] != split:
                raise ValueError(f"source group spans splits: {group}")
            fixed_groups[group] = split
        index = labels.index(label)
        # A mixed-label capture group must stay together; stratify it as noise.
        local_groups[group] = index if group not in local_groups else min(local_groups[group], index)
        if label != "noise":
            points = transient_samples(audio, peak_floor) if onsets is None else onsets
            windows = [(event_window(audio, int(point)), int(point), "weak_onset" if onsets is None else "annotated")
                       for point in points]
        else:
            points = transient_samples(audio, peak_floor) if onsets is None else onsets
            candidates = [(event_window(audio, point), point, "negative_onset") for point in points]
            candidates += [(window, start * WINDOW_SAMPLES, "background") for start, window in enumerate(slices(audio))]
            limit = lab_max_windows_per_file if domain == "lab" and lab_max_windows_per_file is not None else max_windows_per_file
            if len(candidates) > limit:
                indices = np.linspace(0, len(candidates) - 1, limit, dtype=int)
                candidates = [candidates[i] for i in indices]
            windows = candidates
        if not windows:
            notices.append(f"No usable {label} onsets: {source}")
        for window, point, quality in windows:
            features.append(log_mel(window))
            records.append((index, source, group, domain, category, point, quality))

    if esc50 is not None:
        with (esc50 / "meta" / "esc50.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        origin_splits = defaultdict(set)
        for row in rows:
            fold = int(row["fold"])
            origin_splits[row["src_file"]].add("train" if fold <= 3 else "val" if fold == 4 else "test")
        conflicts = {origin for origin, names in origin_splits.items() if len(names) > 1}
        if conflicts:
            notices.append(f"Excluded ESC original sources appearing across official splits: {sorted(conflicts)}")
        for row in rows:
            target = int(row["target"])
            if noise_policy == "curated" and target not in {**NOISE_ESC, **CLAP_ESC}:
                continue
            if row["src_file"] in conflicts or row["src_file"] in fsd_origins:
                # ESC and FSD both derive from Freesound: different dataset
                # prefixes do not make the same original recording independent.
                continue
            expected = {**NOISE_ESC, **CLAP_ESC}.get(target)
            if expected is not None and row["category"] != expected:
                raise ValueError(f"wrong ESC category for target {target}: {row['category']}")
            fold = int(row["fold"])
            if fold not in range(1, 6) or not row.get("src_file"):
                raise ValueError("ESC metadata needs official folds and original src_file IDs")
            split = "train" if fold <= 3 else "val" if fold == 4 else "test"
            audio, rate = read_wav(esc50 / "audio" / row["filename"])
            add(to_16k(audio, rate), "clap" if target == 22 else "noise",
                f"esc50:{row['filename']}", f"esc50:{row['src_file']}", "esc50",
                row["category"], split=split)
        notices.append("ESC clap onsets are weak labels; applause and isolated room claps differ.")

    if sessions.is_dir():
        for session_json in sorted(sessions.glob("*/session.json")):
            meta = json.loads(session_json.read_text(encoding="utf-8"))
            if meta.get("exclude_from_training"):
                notices.append(f"Explicitly excluded session: {session_json.parent.name}")
                continue
            category = str(meta.get("label") or "")
            if category not in LOCAL:
                notices.append(f"Skipped unknown label {category}: {session_json.parent.name}")
                continue
            label = LOCAL[category]
            if label not in labels:
                continue
            capture = meta.get("capture") or {}
            integrity = meta.get("integrity") or {}
            ready = (meta.get("trainingReady") is True and capture.get("pcmVersion") == 2
                     and capture.get("raw") is True and capture.get("sampleRate") == SAMPLE_RATE
                     and not any(integrity.get(key, 0) for key in
                                 ("uploadErrors", "pcmDropped", "discontinuities", "packetMetadataMissing")))
            if not ready:
                if not allow_legacy_sessions or capture.get("pcmVersion") == 2:
                    notices.append(f"Skipped unverified/discontinuous PCM session: {session_json.parent.name}")
                    continue
                notices.append(f"Experimental legacy PCM, gain/clipping/timing unknown: {session_json.parent.name}")
            wav = session_json.parent / "audio.wav"
            if not wav.is_file():
                notices.append(f"Skipped session without audio.wav: {session_json.parent.name}")
                continue
            audio, rate = read_wav(wav)
            if ready and (rate != SAMPLE_RATE or ("samples" in integrity and integrity["samples"] != len(audio))):
                raise ValueError(f"WAV does not match verified raw capture metadata: {session_json.parent.name}")
            audio = to_16k(audio, rate)
            group = f"lab:{meta.get('split_group') or session_json.parent.name}"
            if not meta.get("split_group"):
                notices.append(f"Lab group defaults to session; set split_group for shared capture runs: {session_json.parent.name}")
            onsets = meta.get("event_samples")
            if onsets is not None and (not isinstance(onsets, list) or any(
                    not isinstance(point, int) or isinstance(point, bool) or not 0 <= point < len(audio)
                    for point in onsets)):
                raise ValueError(f"event_samples must be valid 16 kHz sample indices: {session_json}")
            if label != "noise" and onsets is None:
                notices.append(f"Positive session uses weak onset proposals; review or add event_samples: {session_json.parent.name}")
            clipped = float((np.abs(audio) >= 0.999).mean())
            if clipped > 0.001:
                notices.append(f"Clipped PCM ({clipped:.1%} samples): {session_json.parent.name}; review microphone gain.")
            add(audio, label, f"lab:{session_json.parent.name}", group, "lab", category,
                onsets=onsets, split=meta.get("split"))

    if fsd50k is not None:
        root = fsd50k.resolve()
        file_hash_splits = {}
        for row in fsd_entries:
            label = row["label"]
            if label not in LABELS:
                raise ValueError(f"unknown FSD class: {label}")
            path = (root / row["filename"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"missing or unsafe FSD audio path: {path}")
            if not row.get("source_id") or not row.get("uploader") or not row.get("license") or not row.get("source_url"):
                raise ValueError("FSD recordings require source, uploader, license and source URL provenance")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if row.get("sha256") and digest != row["sha256"]:
                raise ValueError(f"FSD audio no longer matches its provenance SHA256: {path}")
            if digest in file_hash_splits and file_hash_splits[digest] != row["split"]:
                raise ValueError(f"duplicate FSD audio spans splits: {path}")
            file_hash_splits[digest] = row["split"]
            if {"Clapping", "Finger_snapping"}.issubset(row.get("ground_truth_labels", [])):
                raise ValueError(f"ambiguous clap/snap label for FSD source {row['source_id']}")
            source_provenance.append(row)
            audio, rate = read_wav(path)
            # Uploader is stricter than filename/clip grouping: repeated capture
            # setup from one contributor cannot appear in held-out splits.
            add(to_16k(audio, rate), label, str(row["source_id"]),
                f"fsd50k:uploader:{row['uploader']}", "fsd50k", str(row.get("category") or label),
                split=row["split"])
        notices.append("FSD gesture onsets are clip-level weak labels; review clips and validate raw room recordings before enabling the AI gate.")

    if not features:
        raise ValueError("no usable audio examples")
    assigned = assign_splits({group: label for group, label in local_groups.items() if group not in fixed_groups})
    assigned.update(fixed_groups)
    y = np.asarray([record[0] for record in records], dtype=np.int64)
    split = np.asarray([assigned[record[2]] for record in records])
    arrays = {
        "schema_version": np.asarray(2), "x": np.stack(features).astype(np.float32),
        "y": y, "split": split, "labels": np.asarray(labels),
        "frontend_json": np.asarray(json.dumps(contract(), sort_keys=True)),
        "source_provenance_json": np.asarray(json.dumps(source_provenance, sort_keys=True)),
    }
    for index, name in enumerate(("sources", "groups", "domains", "categories", "trigger_samples", "label_quality"), 1):
        arrays[name] = np.asarray([record[index] for record in records])
    counts = lambda mask: {name: int(((y == i) & mask).sum()) for i, name in enumerate(labels)}
    meta = {
        "schema_version": 2, "labels": list(labels), "frontend": contract(),
        "sample_rate": SAMPLE_RATE, "window_samples": WINDOW_SAMPLES,
        "feature_shape": list(FEATURE_SHAPE), "peak_floor": peak_floor,
        "noise_policy": noise_policy, "max_noise_windows_per_file": max_windows_per_file,
        "lab_max_noise_windows_per_file": lab_max_windows_per_file,
        "chunks": counts(np.ones(len(y), dtype=bool)),
        "splits": {name: counts(split == name) for name in SPLITS},
        "domains": {name: counts(arrays["domains"] == name) for name in np.unique(arrays["domains"])},
        "groups": {name: len(set(arrays["groups"][split == name])) for name in SPLITS},
        "warnings": notices, "esc50": str(esc50) if esc50 is not None else None,
        "fsd50k": str(fsd50k) if fsd50k is not None else None,
    }
    return arrays, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esc50", type=Path)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--fsd50k", type=Path, help="downloaded FSD50K folder containing manifest.json")
    parser.add_argument("--sessions", type=Path, default=Path("data/sessions"))
    parser.add_argument("--out", type=Path, default=Path("data/hybrid"))
    parser.add_argument("--classes", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--noise-policy", choices=("all", "curated"), default="all")
    parser.add_argument("--max-windows-per-file", type=int, default=8)
    parser.add_argument("--lab-max-windows-per-file", type=int,
                        help="Retain more actual-room hard negatives without expanding the public dataset")
    parser.add_argument("--peak-floor", type=float, default=PEAK_FLOOR)
    parser.add_argument("--allow-legacy-sessions", action="store_true")
    args = parser.parse_args()
    try:
        arrays, meta = build_dataset(None if args.local_only else find_esc50(args.esc50),
                                    args.sessions, args.classes, args.noise_policy,
                                    args.max_windows_per_file, args.peak_floor, args.allow_legacy_sessions, args.fsd50k,
                                    args.lab_max_windows_per_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "features.npz", **arrays)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chunks": meta["chunks"], "splits": meta["splits"]}, indent=2), flush=True)
    for notice in meta["warnings"]:
        warnings.warn(notice, stacklevel=1)
    missing = [name for name, count in meta["chunks"].items() if not count]
    if missing:
        print(f"Missing {missing}; full gesture training will refuse. Record independent Lab sessions first.")


if __name__ == "__main__":
    main()
