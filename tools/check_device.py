"""Authorized local hardware checks; credentials stay in ignored .build files."""
from pathlib import Path
import argparse
import asyncio
import json
import re
import time
import urllib.request
import urllib.parse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def expected_selftests(export_dir):
    import hashlib
    import numpy as np
    import tensorflow as tf
    import torch
    from hybrid.frontend import log_mel
    from hybrid.export import quantize_input
    report = json.loads((export_dir / "export.json").read_text())
    binary = (export_dir / "model.tflite").read_bytes()
    if hashlib.sha256(binary).hexdigest() != report["model_sha256"]:
        raise RuntimeError("Expected model/export hash mismatch")
    saved = torch.load(export_dir.parent / "run/best.pt", weights_only=True)
    mean, std = np.asarray(saved["mean"], np.float32), np.asarray(saved["std"], np.float32)
    interpreter = tf.lite.Interpreter(model_content=binary,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    input_info, output_info = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    golden = {}
    pcm = np.zeros(8000, np.int16)
    for name in ("silence", "impulse"):
        if name == "impulse":
            pcm[6400:6402] = [24000, -24000]
        normalized = ((log_mel(pcm.astype(np.float32) / 32768) - mean[:, None]) / std[:, None])[None, ..., None]
        normalized[:, :, :saved["target_frame_start"], :] = 0
        interpreter.set_tensor(input_info["index"], quantize_input(normalized, *input_info["quantization"]))
        interpreter.invoke()
        scores = interpreter.get_tensor(output_info["index"])[0].astype(np.float32)
        golden[name] = ((scores - output_info["quantization"][1]) * output_info["quantization"][0]).tolist()
    return report["model_sha256"], golden


def read_boot(port, duration=18, expected_export=None):
    import serial
    device = serial.Serial(port=None, baudrate=115200, timeout=0.2)
    device.dtr = False
    device.rts = False
    device.port = port
    device.open()
    device.rts = True
    time.sleep(0.15)
    device.rts = False
    end = time.monotonic() + duration
    data = bytearray()
    while time.monotonic() < end:
        data.extend(device.read(max(1, device.in_waiting)))
    device.close()
    log = data.decode("utf-8", errors="replace")
    build = ROOT / ".build"
    build.mkdir(exist_ok=True)
    (build / "device_serial.log").write_text(log, encoding="utf-8")
    passwords = re.findall(r"(?:lab pass=|pass=)([A-Za-z0-9]{6,32})(?:\s|$)", log)
    addresses = re.findall(r"wifi ok (\d+\.\d+\.\d+\.\d+)", log)
    if not passwords or not addresses:
        raise RuntimeError("No complete device boot status; inspect ignored .build/device_serial.log")
    details = {"ip": addresses[-1], "password": passwords[-1]}
    (build / "device_state.json").write_text(json.dumps(details), encoding="utf-8")
    # Deliberately redact any boot line containing credentials.
    summary = [line for line in log.splitlines() if any(s in line for s in
               ("reset=", "floor=", "arm=", "ready", "heap=", "AI ", "ai ", "FAIL", "Error", "assert", "panic"))]
    print(json.dumps({"boot": summary, "ip": details["ip"]}))
    if expected_export is not None:
        expected_hash, goldens = expected_selftests(expected_export)
        if f"AI model={expected_hash}" not in log:
            raise RuntimeError("Device boot did not identify the expected trained model")
        observed = {}
        for name, ok, values in re.findall(r"AI selftest (silence|impulse) ok=(\d) .*?scores=([\d./]+)", log):
            observed[name] = [float(value) for value in values.split("/")] if ok == "1" else []
        for name, expected in goldens.items():
            if len(observed.get(name, [])) != 3 or max(abs(a - b) for a, b in zip(observed[name], expected)) > 1 / 256 + .00051:
                raise RuntimeError(f"Device {name} inference did not match its INT8 golden")
        (build / "device_model_check.json").write_text(json.dumps(
            {"model_sha256": expected_hash, "expected": goldens, "observed": observed,
             "tolerance": 1 / 256 + .00051}, indent=2) + "\n")
        print(json.dumps({"model_sha256": expected_hash, "selftest_parity": "passed"}))
    return details


async def check_stream(details, duration, expected_ai_mode=None):
    import websockets
    key = urllib.parse.quote(details["password"])
    url = f"http://{details['ip']}/api/status?k={key}"
    status = json.load(urllib.request.urlopen(url, timeout=8))
    header, expected, baseline_drop = None, None, None
    packets = samples = gaps = missing = drop_delta = 0
    events, stats = [], []
    started = time.monotonic()
    async with websockets.connect(f"ws://{details['ip']}:81/?k={key}", open_timeout=8, close_timeout=1) as ws:
        while time.monotonic() - started < duration:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=3)
            except asyncio.TimeoutError:
                raise RuntimeError("PCM stalled for three seconds")
            if isinstance(message, str):
                payload = json.loads(message)
                if payload.get("t") == "pcm":
                    header = payload
                elif payload.get("t") in ("ev", "ai"):
                    events.append(payload)
                elif payload.get("t") == "stat":
                    stats.append(payload)
                continue
            length = len(message) // 2
            if header is None or len(message) % 2 or header.get("samples") != length:
                missing += 1
            else:
                if expected is not None and header["sample"] != expected:
                    gaps += 1
                expected = header["sample"] + length
                if baseline_drop is None:
                    baseline_drop = header["pcmDropped"]
                drop_delta = header["pcmDropped"] - baseline_drop
            header = None
            packets += 1
            samples += length
        elapsed = time.monotonic() - started
    report = {"seconds": round(elapsed, 2), "packets": packets,
              "samples": samples, "gaps": gaps, "metadataErrors": missing,
              "droppedDuringCheck": drop_delta, "initialStatus": status,
              "lastStat": stats[-1] if stats else None, "events": events}
    (ROOT / ".build/device_stream_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if gaps or missing or drop_delta or status.get("i2s") != 1:
        raise RuntimeError("Audio integrity check failed")
    if expected_ai_mode is not None:
        if not stats or status.get("aiMode") != expected_ai_mode or stats[-1].get("aiMode") != expected_ai_mode:
            raise RuntimeError("Device did not maintain the expected AI mode")
        for key in ("aiDropped", "aiErrors", "commandDrops"):
            if status.get(key) != 0 or stats[-1].get(key) != 0:
                raise RuntimeError(f"Nonzero device {key} during the check")
        if stats[-1].get("i2s") != 1 or stats[-1].get("tuya") != 1:
            raise RuntimeError("Microphone or lamp link did not remain connected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--reset", action="store_true", help="User-authorized board restart")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--expect-ai-mode", choices=("active", "shadow", "dsp"))
    parser.add_argument("--expect-export", type=Path, help="With --reset, verify boot model hash and silence/impulse INT8 goldens")
    args = parser.parse_args()
    if args.expect_export and not args.reset:
        parser.error("--expect-export requires --reset for a complete boot log")
    details = read_boot(args.port, expected_export=args.expect_export) if args.reset else json.loads((ROOT / ".build/device_state.json").read_text())
    asyncio.run(check_stream(details, args.seconds, args.expect_ai_mode))


if __name__ == "__main__":
    main()
