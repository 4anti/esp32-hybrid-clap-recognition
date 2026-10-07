"""Compile and execute the firmware's portable detector on the host."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
import json
import struct


ROOT = Path(__file__).resolve().parents[1]


def compiler_command() -> list[str]:
    supplied = os.environ.get("CLAP_CXX") or os.environ.get("CXX")
    if supplied:
        if Path(supplied).is_file():
            return [supplied, "c++"] if Path(supplied).stem == "zig" else [supplied]
        return shlex.split(supplied, posix=os.name != "nt")
    for name in ("g++", "clang++", "zig"):
        found = shutil.which(name)
        if found:
            return [found, "c++"] if name == "zig" else [found]
    portable = Path(tempfile.gettempdir()) / "clap-lights-test-tools" / \
        "zig-windows-x86_64-0.13.0" / "zig.exe"
    if portable.is_file():
        return [str(portable), "c++"]
    raise RuntimeError("No C++ compiler available. Set CLAP_CXX to g++, clang++, or zig.")


class DetectorTests(unittest.TestCase):
    def test_replay_uses_raw_scale_and_ai_veto_before_pairing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="clap-replay-test-") as directory:
            binary = Path(directory) / ("replay.exe" if os.name == "nt" else "replay")
            subprocess.run(compiler_command() + ["-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                           str(ROOT / "tests/test_detector_replay.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, cwd=ROOT)
            pcm = Path(directory) / "raw.pcm"
            samples = [0] * 16000
            samples[4000] = samples[9000] = 4000
            pcm.write_bytes(struct.pack("<16000h", *samples))
            command = [str(binary), str(pcm), "256", "281840", "--events"]
            rows = [json.loads(row) for row in subprocess.run(command, check=True, capture_output=True,
                                                            text=True).stdout.splitlines()]
            self.assertEqual(rows[-1]["scale"], 256)
            self.assertEqual(rows[-1]["legacy_doubles"], 1)
            self.assertEqual([row["onset"] for row in rows[:-1]], [4000, 9000])
            votes = Path(directory) / "votes.tsv"
            votes.write_text("4000 0\n9000 1\n", encoding="ascii")
            gated = subprocess.run(command + ["--votes", str(votes)], check=True, capture_output=True, text=True)
            result = json.loads(gated.stdout.splitlines()[-1])
            self.assertEqual(result["legacy_doubles"], 0)
            self.assertEqual(result["legacy_accepted"], 1)

    def test_portable_detector(self) -> None:
        with tempfile.TemporaryDirectory(prefix="clap-detector-test-") as directory:
            binary = Path(directory) / ("detector.exe" if os.name == "nt" else "detector")
            compilation = subprocess.run(
                compiler_command() + ["-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                                      str(ROOT / "tests/test_detector.cpp"), "-o", str(binary)],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(compilation.returncode, 0, compilation.stderr)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
            self.assertIn("tests passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
