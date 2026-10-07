"""Replay intact raw recordings through the actual detector, INT8 AI and pairer.

Positive recordings used for training measure fit, not independent generalization.
This host replay does not simulate real-time worker queues or radio/lamp latency.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hybrid.frontend import event_window, log_mel, LABELS
from hybrid.export import quantize_input


def compiler_command():
    spec = importlib.util.spec_from_file_location("detector_test_support", ROOT / "tests/test_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compiler_command()


def replay(binary, pcm, arm, votes=None):
    command = [str(binary), str(pcm), "256", str(arm), "--events"]
    if votes is not None:
        command.extend(["--votes", str(votes)])
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in completed.stdout.splitlines()]
    return rows[:-1], rows[-1]


def evaluate(export_dir, sessions, out, arm=281840):
    import tensorflow as tf
    import torch

    report = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    model_bytes = (export_dir / "model.tflite").read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != report["model_sha256"]:
        raise ValueError("model/report hash mismatch")
    checkpoint_path = export_dir.parent / "run/best.pt"
    if report.get("checkpoint_sha256") and hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("checkpoint/report hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mean, std = np.asarray(checkpoint["mean"], np.float32), np.asarray(checkpoint["std"], np.float32)
    interpreter = tf.lite.Interpreter(model_content=model_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    input_info, output_info = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    threshold = max(.85, report["decision"]["positive_threshold"])
    results = []
    with tempfile.TemporaryDirectory(prefix="clap-room-replay-") as temporary:
        binary = Path(temporary) / ("replay.exe" if os.name == "nt" else "replay")
        subprocess.run(compiler_command() + ["-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
            str(ROOT / "tests/test_detector_replay.cpp"), "-o", str(binary)], check=True, capture_output=True)
        for directory in sorted(sessions.glob("*/session.json")):
            metadata = json.loads(directory.read_text(encoding="utf-8"))
            if metadata.get("exclude_from_training") or not metadata.get("label_verified"):
                continue
            capture = metadata.get("capture", {})
            if not metadata.get("trainingReady") or capture.get("pcmVersion") != 2 or not capture.get("raw"):
                raise ValueError(f"unverified raw recording: {directory.parent.name}")
            pcm = directory.parent / "audio.pcm"
            values = np.frombuffer(pcm.read_bytes(), dtype="<i2")
            if len(values) != metadata["integrity"]["samples"] or len(values) != capture["deviceSampleEnd"] - capture["deviceSampleStart"]:
                raise ValueError(f"raw length mismatch: {directory.parent.name}")
            audio = values.astype(np.float32) / 32768
            events, baseline = replay(binary, pcm, arm)
            predictions, votes = [], []
            for event in events:
                features = log_mel(event_window(audio, event["onset"]))
                normalized = ((features - mean[:, None]) / std[:, None])[None, ..., None]
                normalized[:, :, :checkpoint["target_frame_start"], :] = 0
                interpreter.set_tensor(input_info["index"], quantize_input(normalized, *input_info["quantization"]))
                interpreter.invoke()
                raw = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
                scores = (raw - output_info["quantization"][1]) * output_info["quantization"][0]
                target, noise = 1 - float(scores[0]), float(scores[0])
                accepted = not event["timedOut"] and target >= threshold and noise <= .15 and \
                           target - noise >= .35 and (scores[1] > scores[0] or scores[2] > scores[0])
                votes.append(f"{event['onset']} {int(accepted)}\n")
                predictions.append({**event, "scores": dict(zip(LABELS, scores.tolist())), "ai_accept": bool(accepted)})
            votes_path = Path(temporary) / "votes.tsv"
            votes_path.write_text("".join(votes), encoding="ascii")
            _, gated = replay(binary, pcm, arm, votes_path)
            label = metadata["label"]
            reviewed = metadata.get("event_samples", []) if label in ("clap_far", "clap_near", "finger_snap") else []
            # A peak annotation can lag the triggering high-pass onset. Use a
            # 50 ms tolerance and one-to-one matches, never prediction labels.
            used = set()
            matched = accepted_targets = 0
            for onset in reviewed:
                choices = [(abs(event["onset"] - onset), index) for index, event in enumerate(predictions)
                           if index not in used and abs(event["onset"] - onset) <= 800]
                if choices:
                    _, index = min(choices)
                    used.add(index)
                    matched += 1
                    accepted_targets += int(predictions[index]["ai_accept"])
            results.append({"session": directory.parent.name, "label": label, "split": metadata["split"],
                "seconds": len(values) / 16000, "pcm_sha256": hashlib.sha256(pcm.read_bytes()).hexdigest(),
                "baseline_dsp": baseline, "ai_veto_pipeline": gated,
                "ai_candidate_accepts": sum(event["ai_accept"] for event in predictions),
                "reviewed_gestures": len(reviewed), "detector_matches": matched,
                "ai_accepted_matches": accepted_targets, "predictions": predictions})
    result = {"model_sha256": report["model_sha256"], "arm_threshold": arm, "positive_threshold": threshold,
              "sessions": results, "limitations": [
                "Raw PCM16 reconstructs raw24 with eight low bits lost; this is not a bit-exact microphone replay.",
                "Training takes measure fit only; the independent room validation/test takes contain bag noise only.",
                "Host replay does not establish target-device inference throughput or live lamp behavior."]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": report["model_sha256"], "sessions": [
        {key: value for key, value in row.items() if key not in ("predictions", "pcm_sha256")} for row in results]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, default=ROOT / "data/sessions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", type=int, default=281840)
    args = parser.parse_args()
    evaluate(args.export, args.sessions, args.out, args.arm)
