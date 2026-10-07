"""Execute the exact capture ownership implementation with native C++ threads."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_detector import ROOT, compiler_command


class CaptureTests(unittest.TestCase):
    def test_portable_capture(self) -> None:
        with tempfile.TemporaryDirectory(prefix="clap-capture-test-") as directory:
            binary = Path(directory) / ("capture.exe" if os.name == "nt" else "capture")
            flags = [] if os.name == "nt" else ["-pthread"]
            compilation = subprocess.run(
                compiler_command() + flags + ["-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                                             str(ROOT / "tests/test_capture.cpp"), "-o", str(binary)],
                capture_output=True, text=True, cwd=ROOT,
            )
            self.assertEqual(compilation.returncode, 0, compilation.stderr)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
            self.assertIn("tests passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
