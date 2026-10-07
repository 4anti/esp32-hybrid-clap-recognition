"""Run the actual C++ frontend against the Python training frontend."""
from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import pytest

from hybrid.frontend import log_mel
from hybrid.export import quantize_input

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frontend_exe(tmp_path_factory):
    compiler = os.environ.get("CLAP_CXX") or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        cached = Path(os.environ.get("TEMP", "/tmp")) / "clap-lights-test-tools/zig-windows-x86_64-0.13.0/zig.exe"
        if cached.exists():
            compiler = str(cached)
    if not compiler:
        pytest.skip("C++ compiler unavailable; firmware frontend parity was not verified")
    exe = tmp_path_factory.mktemp("frontend") / "frontend.exe"
    command = [compiler]
    if Path(compiler).stem == "zig":
        command.append("c++")
    command += ["-std=c++17", "-O2", "-I", str(ROOT / "clap_double"), str(ROOT / "tests/frontend_host.cpp"), "-o", str(exe)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    return exe


@pytest.mark.parametrize("kind", ["silence", "impulse", "tone", "random", "clipped"])
def test_cpp_python_frontend_parity(frontend_exe, tmp_path, kind):
    rng = np.random.default_rng(41)
    pcm = np.zeros(8000, dtype=np.int16)
    if kind == "impulse":
        pcm[6400] = 28000
    elif kind == "tone":
        pcm = (22000 * np.sin(2 * np.pi * 1875 * np.arange(8000) / 16000)).astype(np.int16)
    elif kind == "random":
        pcm = rng.integers(-22000, 22000, 8000, dtype=np.int16)
    elif kind == "clipped":
        pcm[6380:6540:2] = 32767
        pcm[6381:6540:2] = -32768
    source, result, quantized = tmp_path / "pcm.bin", tmp_path / "features.bin", tmp_path / "int8.bin"
    pcm.astype("<i2").tofile(source)
    subprocess.run([str(frontend_exe), str(source), str(result), str(quantized)], check=True, timeout=30)
    actual = np.fromfile(result, dtype="<f4").reshape(64, 48)
    expected = log_mel(pcm.astype(np.float32) / 32768)
    # Low-energy FFT cancellation near a pure tone magnifies float roundoff
    # through log(). Typical sound/impulse error is much smaller than 0.003.
    np.testing.assert_allclose(actual, expected, atol=0.003, rtol=0)
    mean = -3 + np.arange(64, dtype=np.float32) / 64
    std = 1 + np.arange(64, dtype=np.float32) / 128
    reference = quantize_input((expected - mean[:, None]) / std[:, None], 0.075, -3)
    observed = np.fromfile(quantized, dtype=np.int8).reshape(64, 48)
    # Float FFT roundoff can cross one INT8 bin at a quantization boundary.
    assert np.abs(observed.astype(int) - reference.astype(int)).max() <= 1
