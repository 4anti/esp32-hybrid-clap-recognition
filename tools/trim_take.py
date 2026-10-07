"""Make an auditable PCM prefix crop while preserving the complete source take."""
from pathlib import Path
import argparse
import hashlib
import json
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def trim_take(source: Path, seconds: float) -> Path:
    source = source.resolve()
    if not source.is_relative_to((ROOT / "data/sessions").resolve()):
        raise ValueError("take must be inside this project's data/sessions")
    meta_file = source / "session.json"
    original_json = meta_file.read_text(encoding="utf-8")
    meta = json.loads(original_json)
    if not meta.get("ended") or not meta.get("trainingReady") or not meta.get("capture", {}).get("raw"):
        raise ValueError("save a verified raw take before trimming")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("trim duration must be positive")
    with wave.open(str(source / "audio.wav"), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("expected mono PCM16 at 16000 Hz")
        pcm = wav.readframes(wav.getnframes())
    if (source / "audio.pcm").read_bytes() != pcm:
        raise ValueError("source WAV/PCM mismatch")
    removed = round(seconds * 16000)
    kept = len(pcm) // 2 - removed
    if kept <= 0 or meta["integrity"]["samples"] != len(pcm) // 2:
        raise ValueError("trim would remove the entire take or source count is invalid")
    destination = source.with_name(source.name + f"-trimmed-{seconds:g}s")
    if destination.exists():
        raise ValueError("trimmed take already exists; source remains preserved")
    destination.mkdir()
    prefix = pcm[:kept * 2]
    (destination / "audio.pcm").write_bytes(prefix)
    with wave.open(str(destination / "audio.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(prefix)
    derived = json.loads(original_json)
    derived.update(id=destination.name, seconds=kept / 16000, bytes=len(prefix),
                   note=f"Bag-only prefix; last {seconds:g}s removed at the user's request. Original preserved.",
                   split_group=meta.get("split_group") or source.name,
                   noise_kind="paper_bag", label_verified=True,
                   derived_from={"session": source.name, "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                                 "trim_end_samples": removed, "operation": "unaltered PCM prefix"})
    derived.pop("exclude_from_training", None)
    derived["capture"]["deviceSampleEnd"] = derived["capture"]["deviceSampleStart"] + kept
    derived["integrity"]["samples"] = kept
    values = np.frombuffer(prefix, "<i2").astype(np.int32)
    derived["integrity"]["clippedSamples"] = int((np.abs(values) >= 32760).sum())
    if "event_samples" in derived:
        derived["event_samples"] = [sample for sample in derived["event_samples"] if sample < kept]
    (destination / "session.json").write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
    retained_events = []
    for line in (source / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("sessionSample", kept) >= kept or event.get("sessionOnsetSample", kept) >= kept:
            continue
        event.update(session=destination.name, source_session=source.name)
        retained_events.append(json.dumps(event))
    (destination / "events.jsonl").write_text("\n".join(retained_events) + "\n", encoding="utf-8")
    backup = source / "session.before-trim.json"
    if not backup.exists():
        backup.write_text(original_json, encoding="utf-8")
    meta.update(exclude_from_training=True, training_exclusion_reason=f"Use preserved prefix derivative {destination.name}")
    meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    print(trim_take(ROOT / "data/sessions" / args.session, args.seconds))
