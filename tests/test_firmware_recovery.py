"""Exercise actual ESP32 protocol/recovery bodies against deterministic mocks.

These tests use synthetic bytes and no device credentials or network. They
verify ordering and bounded parsing, not Wi-Fi drivers or physical cold boot.
"""
from pathlib import Path
import re
import subprocess

import pytest

from test_detector import ROOT, compiler_command


def function_body(source: str, name: str) -> str:
    """Extract a complete named function without copying its implementation.

    String/comment braces are skipped so a later diagnostic message cannot
    accidentally change which body the test compiles.
    """
    match = re.search(r"(?m)^(?:static\s+)?void\s+" + re.escape(name) + r"\([^)]*\)\s*\{", source)
    if match is None:
        raise AssertionError(f"Firmware function missing: {name}")
    position = match.end()
    depth = 1
    while depth and position < len(source):
        if source.startswith("//", position):
            end = source.find("\n", position)
            position = len(source) if end < 0 else end + 1
        elif source.startswith("/*", position):
            end = source.find("*/", position + 2)
            if end < 0:
                raise AssertionError("Unclosed firmware comment")
            position = end + 2
        elif source[position] in "\"'":
            quote = source[position]
            position += 1
            while position < len(source):
                if source[position] == "\\":
                    position += 2
                elif source[position] == quote:
                    position += 1
                    break
                else:
                    position += 1
        else:
            if source[position] == "{":
                depth += 1
            elif source[position] == "}":
                depth -= 1
            position += 1
    if depth:
        raise AssertionError(f"Unclosed firmware function: {name}")
    return source[match.start():position] + "\n"


@pytest.fixture(scope="module")
def recovery_exe(tmp_path_factory):
    build = tmp_path_factory.mktemp("firmware-recovery")
    lab = (ROOT / "clap_double/clap_lab.ino").read_text(encoding="utf-8")
    sketch = (ROOT / "clap_double/clap_double.ino").read_text(encoding="utf-8")
    (build / "ws_under_test.inc").write_text(function_body(lab, "labReadWs"), encoding="utf-8")
    (build / "recovery_under_test.inc").write_text(
        function_body(sketch, "idleHuntTick") + function_body(sketch, "handleWifiEvents"),
        encoding="utf-8",
    )
    binary = build / "recovery.exe"
    compilation = subprocess.run(
        compiler_command() + ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                              "-I", str(build), "-I", str(ROOT / "clap_double"),
                              str(ROOT / "tests/recovery_host.cpp"), "-o", str(binary)],
        capture_output=True, text=True, timeout=180,
    )
    assert compilation.returncode == 0, compilation.stderr
    return binary


@pytest.mark.parametrize("case", [
    "http_boundary", "http_limits", "http_shared_deadline", "http_clock_wrap",
    "ws_fragmented_ping", "ws_coalesced_ping_text", "ws_split_length_mask",
    "ws_invalid_after_text", "ws_oversized_after_text", "ws_failed_pong_followed_text",
    "ws_close_followed_text", "ws_empty_ping", "ws_disconnected",
    "recovery_dead_socket", "recovery_dead_socket_no_udp", "recovery_heartbeat_disconnect",
    "recovery_saved_first", "recovery_rate_limit",
    "recovery_discovery_no_session", "recovery_query_drops", "recovery_scan_fallback",
    "recovery_wifi_lost", "recovery_wifi_reconnected", "recovery_stale_loss_event",
    "recovery_coalesced_wifi_events",
])
def test_firmware_protocol_and_recovery(recovery_exe, case):
    result = subprocess.run([str(recovery_exe), case], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{case} passed"
