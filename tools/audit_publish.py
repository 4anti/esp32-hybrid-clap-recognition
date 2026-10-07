"""Fail-closed privacy preflight for the exact Git index being published.

Run from any directory inside the repository. Findings contain file names and
categories only; never print staged contents, credentials, or matching values.
Local private configuration is read only to recognize accidentally copied
credentials. No files or Git state are changed by this command.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


TOP_FILES = {
    ".gitignore", "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING",
    "MODEL_LICENSE.md", "FSD50K_ATTRIBUTION.txt", "ESC50_DATA_LICENSE.txt",
    "HYBRID_AI_PLAN.md", "MODEL_STATUS.md",
    "ROOM_MODEL_STATUS.md", "TEST_RESULTS.md", "train_custom.py",
    "requirements_hybrid.txt", "requirements_hybrid_export.txt", "clap_double.ino",
}
ALLOWED_EXTENSIONS = {
    "clap_double": {".ino", ".h"},
    "hybrid": {".py"},
    "lab": {".html", ".js", ".css", ".bat", ".md", ".txt"},
    "tests": {".py", ".cpp", ".c", ".h", ".js"},
    "tools": {".py", ".ps1", ".bat", ".sh"},
    "docs": {".md", ".txt"},
    ".github": {".md", ".yml", ".yaml"},
}
FONT_EXTENSIONS = {".woff2", ".woff", ".ttf", ".otf"}
# Fonts are the only published binary assets. Pin the reviewed files, rather
# than accepting any blob whose first four bytes resemble a font signature.
PUBLIC_FONT_SHA256 = {
    "lab/fonts/azeret-mono-400.woff2": "88f7411ee2a8993b1242db2ee403e918eec5e7878b6f84cfd9049ad157f55de3",
    "lab/fonts/azeret-mono-500.woff2": "36c3d87ca8625e5009b6d9eeef4aacf735badb0eebc813521d40c8c154db97ba",
    "lab/fonts/chivo-latin-ext.woff2": "737fd7c57fc4dd52cf5738f1c0a6726a54c5860dfbc0560bce72299b8dda32e5",
    "lab/fonts/chivo-latin.woff2": "499c79a165e69a65cd2e1823c1a1ae822d7d5a09d2b10b5a8b831fdb15f2ff45",
    "lab/fonts/public-sans-400.woff2": "36274b5787b4f03b27e65ae971d6c808a96838ccd60c9dabaee154889b6bba82",
    "lab/fonts/public-sans-600.woff2": "c7842f97346df1b3b470e9d71acde90f132dd858508ec5ae6606401e5d0938f7",
    "lab/fonts/public-sans-700.woff2": "dace741613696827f61dbee2d984e8685f14a2846136783f3ad1c8950914bf1d",
}
# The owner approved only this deployable export, not the training recordings,
# checkpoints, or other generated models. Normalize line endings for Git on
# Windows while pinning every byte of code, weights, and frontend metadata.
APPROVED_MODEL_PATH = "clap_double/model_data.h"
APPROVED_MODEL_HEADER_SHA256 = "b0edc512a9ea00a232407bfcf4e832213b4c3e23cb984fb1f56355bf233ffc34"
APPROVED_MODEL_SHA256 = "9e9f9dd8c0efee615e9a5d51fc2b6071bb01814d0b3c8e914c6be95b4dba9b94"
APPROVED_MODEL_BYTES = 20872
PRIVATE_DIRECTORIES = {
    "data", ".build", "build", ".deps", ".venv", ".venv-hybrid", "venv",
    "node_modules", "__pycache__", ".pytest_cache", ".cursor", ".impeccable",
    "_photo_preview", "dashboard", "ir_blaster", ".git",
}
PRIVATE_FILES = {
    "model_data.h", "clap_local.h", "state.json", "PLAN.md", "PRODUCT.md",
    "SOLDERING.md", "IR_SHOPPING.md", "SHOPPING_CART.md", ".env",
}
PRIVATE_EXTENSIONS = {
    ".wav", ".pcm", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff",
    ".mp4", ".mov", ".mkv", ".avi", ".jpg", ".jpeg", ".png", ".gif",
    ".webp", ".heic", ".svg", ".pt", ".pth", ".npz", ".npy", ".tflite",
    ".onnx", ".bin", ".elf", ".hex", ".log", ".zip", ".7z", ".tar", ".gz",
}
REQUIRED_FILES = {
    ".gitignore", "README.md", "docs/MODEL_LICENSE.md", "docs/FSD50K_ATTRIBUTION.txt", "docs/ESC50_DATA_LICENSE.txt",
    "requirements_hybrid.txt", "requirements_hybrid_export.txt",
    "train_custom.py", "tools/audit_publish.py", "clap_double/clap_double.ino",
    "clap_double/clap_detector.h", "clap_double/clap_config.h",
    "clap_double/clap_capture.h", "clap_double/clap_ai.ino",
    "clap_double/clap_frontend.h", "clap_double/clap_frontend_tables.h",
    "clap_double/clap_pcm.h", "clap_double/clap_ws.h", "clap_double/clap_lab.ino",
    APPROVED_MODEL_PATH,
    "hybrid/__init__.py", "hybrid/prepare.py", "hybrid/frontend.py", "hybrid/cnn.py",
    "hybrid/export.py", "lab/app.js", "lab/server.js", "lab/index.html", "lab/styles.css",
}
CREDENTIAL_NAME = r"(?:CLAP_WIFI_SSID|CLAP_WIFI_PASSWORD|CLAP_TUYA_DEVICE_ID|CLAP_TUYA_LOCAL_KEY|WIFI_SSID|WIFI_PASS(?:WORD)?|TUYA_ID|TUYA_KEY|CLAP_LAB_PASSWORD|SSID|LOCAL_KEY|DEVICE_ID|API_KEY|ACCESS_TOKEN|SECRET_KEY)"
CREDENTIAL_DEFINITION = re.compile(
    rf"\b{CREDENTIAL_NAME}\b\s*(?:\[\s*\])?\s*(?:=\s*|[ \t]+)[\"']([^\"'\r\n]*)[\"']",
    re.IGNORECASE,
)
CREDENTIAL_FIELD = re.compile(
    rf"[\"'](?:{CREDENTIAL_NAME}|password|passwd|wifi_password|wifi_pass|local_key|api_key|access_token|secret_key)[\"']\s*:\s*[\"']([^\"'\r\n]+)[\"']",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(
    r"^(?:YOUR_|EXAMPLE_|TEST_ONLY_|SYNTHETIC_|CHANGE_ME|REPLACE_ME|PLACEHOLDER|<[^>]+>|\$\{[^}]+\})",
    re.IGNORECASE,
)
PERSONAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/" + r"(?:Users|home)/)", re.IGNORECASE)
MAC_ADDRESS = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])")
SAFE_MACS = {"00:00:00:00:00:00", "FF:FF:FF:FF:FF:FF", "AA:BB:CC:DD:EE:FF"}
PRIVATE_IP = re.compile(
    r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
PRIVATE_IP_CONSTRUCTOR = re.compile(r"\bIPAddress\s*\(\s*(?:192\s*,\s*168|10\s*,|172\s*,\s*(?:1[6-9]|2\d|3[01]))")
PRIVATE_IP_DEFINE = re.compile(r"\bCLAP_TUYA_IP\s+(?:192\s*,\s*168|10\s*,|172\s*,\s*(?:1[6-9]|2\d|3[01]))")
RECORDING_IDENTIFIER = re.compile(r"\b20\d{2}-\d{2}-\d{2}T\d{2}[-:]\d{2}[-:]\d{2}(?:[-.]\d{3})?Z?(?:[-_][0-9a-f]{6,32})?[-_](?:clap(?:_far)?|snap|finger(?:_snap)?|door|tongue|other|noise)\b", re.IGNORECASE)


def git_bytes(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        # Git diagnostics may include paths or blob content; keep them private.
        raise RuntimeError("Git index could not be inspected")
    return result.stdout


def path_category(name: str) -> str | None:
    path = PurePosixPath(name)
    parts = path.parts
    if not parts or path.is_absolute() or ".." in parts or "\\" in name:
        return "unsafe-path"
    if name == APPROVED_MODEL_PATH:
        return None  # The exact staged blob is checked below.
    if any(part.casefold() in {value.casefold() for value in PRIVATE_DIRECTORIES} or part.casefold().startswith(".venv") for part in parts):
        return "private-or-unrelated-directory"
    if path.name.casefold() in {value.casefold() for value in PRIVATE_FILES} or path.name.casefold().startswith(".env."):
        return "private-file"
    if path.suffix.lower() in PRIVATE_EXTENSIONS:
        return "recording-media-model-or-build-artifact"
    if name in TOP_FILES:
        return None
    if name in PUBLIC_FONT_SHA256:
        return None
    if len(parts) >= 2 and parts[0] in ALLOWED_EXTENSIONS and path.suffix.lower() in ALLOWED_EXTENSIONS[parts[0]]:
        return None
    return "outside-publish-allowlist"


def configured_secrets(root: Path) -> set[str]:
    """Recognize local credentials without copying them into this script/output."""
    values: set[str] = set()
    for relative in ("clap_double/clap_local.h", "clap_double/clap_double.ino", "ir_blaster/ir_ac/ir_ac.ino"):
        candidate = root / relative
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        for expression in (CREDENTIAL_DEFINITION, CREDENTIAL_FIELD):
            for value in expression.findall(text):
                if len(value) >= 4 and not PLACEHOLDER.match(value):
                    values.add(value)
    return values


def text_categories(text: str, secrets: set[str], owner: str | None) -> set[str]:
    categories: set[str] = set()
    if any(secret in text for secret in secrets):
        categories.add("configured-private-credential")
    for expression in (CREDENTIAL_DEFINITION, CREDENTIAL_FIELD):
        if any(value and not PLACEHOLDER.match(value) for value in expression.findall(text)):
            categories.add("literal-credential-definition")
    if PERSONAL_PATH.search(text):
        categories.add("personal-absolute-path")
    if owner and re.search(rf"(?<![\w]){re.escape(owner)}(?![\w])", text, re.IGNORECASE):
        categories.add("local-owner-identifier")
    if any(match.group().upper().replace("-", ":") not in SAFE_MACS for match in MAC_ADDRESS.finditer(text)):
        categories.add("device-mac-address")
    if PRIVATE_IP.search(text) or PRIVATE_IP_CONSTRUCTOR.search(text) or PRIVATE_IP_DEFINE.search(text):
        categories.add("specific-private-network-address")
    if RECORDING_IDENTIFIER.search(text):
        categories.add("private-recording-identifier")
    if re.search(r"namespace\s+clap_model\b", text) and re.search(r"(?:unsigned\s+char|uint8_t)\s+data\s*\[[^]]*\]\s*=\s*\{\s*0x[0-9a-fA-F]{2}", text):
        categories.add("embedded-trained-model")
    return categories


def deployment_model_bytes(blob: bytes) -> bytes | None:
    try:
        text = blob.decode("utf-8")
    except UnicodeError:
        return None
    match = re.search(r"unsigned\s+char\s+data\[\]\s*=\s*\{([^}]+)\}", text, re.DOTALL)
    if not match:
        return None
    return bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})\b", match.group(1)))


def approved_model(blob: bytes) -> bool:
    canonical = blob.replace(b"\r\n", b"\n")
    if hashlib.sha256(canonical).hexdigest() != APPROVED_MODEL_HEADER_SHA256:
        return False
    # Pin both the entire export and its actual FlatBuffer, so a stale metadata
    # hash cannot authorize changed model bytes.
    values = deployment_model_bytes(canonical)
    if values is None:
        return False
    return len(values) == APPROVED_MODEL_BYTES and hashlib.sha256(values).hexdigest() == APPROVED_MODEL_SHA256


def model_text_categories(values: bytes, secrets: set[str], owner: str | None) -> set[str]:
    # A C hex array can conceal metadata strings from a source-text scan.
    # Latin-1 preserves bytes without joining strings across invalid UTF-8.
    categories = text_categories(values.decode("latin-1"), set(), owner)
    if any(secret.encode("utf-8") in values for secret in secrets):
        categories.add("configured-private-credential")
    return {"deployment-model-" + category for category in categories}


def inspect_index(root: Path) -> tuple[list[tuple[str, str]], int]:
    findings: list[tuple[str, str]] = []
    entries = git_bytes(root, "ls-files", "--stage", "-z").split(b"\0")
    names: set[str] = set()
    secrets = configured_secrets(root)
    owner = os.environ.get("USERNAME") or Path.home().name
    # Generic environment accounts are not identifying owner names.
    if owner.lower() in {"user", "root", "runner", "admin", "administrator"}:
        owner = None
    for entry in entries:
        if not entry:
            continue
        metadata, encoded_name = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        name = encoded_name.decode("utf-8", errors="strict")
        names.add(name)
        category = path_category(name)
        if category:
            findings.append((name, category))
            continue
        if mode not in {"100644", "100755"} or stage != "0":
            findings.append((name, "symlink-submodule-or-unmerged-index-entry"))
            continue
        blob = git_bytes(root, "cat-file", "blob", object_id)
        if PurePosixPath(name).suffix.lower() in FONT_EXTENSIONS:
            signature = PurePosixPath(name).suffix.lower()
            magic = {".woff2": (b"wOF2",), ".woff": (b"wOFF",), ".ttf": (b"\x00\x01\x00\x00", b"true"), ".otf": (b"OTTO",)}
            if not blob.startswith(magic[signature]):
                findings.append((name, "invalid-public-font-format"))
            if hashlib.sha256(blob).hexdigest() != PUBLIC_FONT_SHA256[name]:
                findings.append((name, "unreviewed-public-font-content"))
            continue  # Only allowlisted public font paths may contain binary data.
        try:
            text = blob.decode("utf-8-sig")
        except UnicodeError:
            findings.append((name, "unexpected-binary-data"))
            continue
        if "\0" in text:
            findings.append((name, "unexpected-binary-data"))
            continue
        categories = text_categories(text, secrets, owner)
        if name == APPROVED_MODEL_PATH:
            if approved_model(blob):
                categories.discard("embedded-trained-model")
                categories.update(model_text_categories(deployment_model_bytes(blob), secrets, owner))
            else:
                categories.add("unapproved-deployment-model")
        for category in sorted(categories):
            findings.append((name, category))
    for missing in sorted(REQUIRED_FILES - names):
        findings.append((missing, "required-publish-file-not-staged"))
    if not names.intersection({"LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"}):
        findings.append(("LICENSE", "required-license-not-staged"))
    groups = {
        "documentation": lambda name: name in {"HYBRID_AI_PLAN.md", "MODEL_STATUS.md", "ROOM_MODEL_STATUS.md", "TEST_RESULTS.md"} or name.startswith("docs/"),
        "tests": lambda name: name.startswith("tests/"),
        "public-fonts": lambda name: name.startswith("lab/fonts/") and PurePosixPath(name).suffix.lower() in FONT_EXTENSIONS,
        "github-templates": lambda name: name.startswith(".github/"),
    }
    for label, predicate in groups.items():
        if not any(predicate(name) for name in names):
            findings.append((label, "required-publish-group-not-staged"))
    return sorted(set(findings)), len(names)


def main() -> int:
    try:
        root = Path(git_bytes(Path.cwd(), "rev-parse", "--show-toplevel").decode("utf-8").strip())
        findings, count = inspect_index(root)
    except (OSError, RuntimeError, UnicodeError, ValueError):
        print("Privacy preflight BLOCKED: index inspection unavailable.", file=sys.stderr)
        return 2
    if findings:
        print(f"Privacy preflight BLOCKED ({count} staged files).")
        for name, category in findings:
            print(f"{name}: {category}")
        return 1
    print(f"Privacy preflight PASSED ({count} staged files; allowlist and credential/path checks).")
    print("This checks the staged index only. Publish this reviewed index without adding other files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
