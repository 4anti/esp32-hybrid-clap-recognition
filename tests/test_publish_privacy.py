"""Publication checks use disposable Git indexes and synthetic private values."""

from pathlib import Path
import subprocess
import sys

import pytest

from tools import audit_publish as audit


SOURCE_ROOT = Path(__file__).resolve().parents[1]
CREDENTIAL_NAMES = (
    "CLAP_WIFI_SSID", "CLAP_WIFI_PASSWORD", "CLAP_TUYA_DEVICE_ID",
    "CLAP_TUYA_LOCAL_KEY", "CLAP_LAB_PASSWORD",
)
PUBLISHED_MODEL_NOTICES = (
    "docs/MODEL_LICENSE.md", "docs/FSD50K_ATTRIBUTION.txt", "docs/ESC50_DATA_LICENSE.txt",
)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True).stdout


def write(root, name, content):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


@pytest.fixture
def publish_index(tmp_path, monkeypatch):
    """Make a complete harmless index, without touching the project index."""
    root = tmp_path / "publish"
    root.mkdir()
    git(root, "init", "--quiet")
    monkeypatch.setenv("USERNAME", "runner")
    for name in audit.REQUIRED_FILES:
        # Model notices use realistic publication paths independently of the
        # scanner constants, so an incorrect required path cannot pass itself.
        if Path(name).name in {Path(notice).name for notice in PUBLISHED_MODEL_NOTICES}:
            continue
        write(root, name, "// synthetic public fixture\n")
    for name in PUBLISHED_MODEL_NOTICES:
        write(root, name, "Synthetic published model notice\n")
    write(root, audit.APPROVED_MODEL_PATH,
          (SOURCE_ROOT / audit.APPROVED_MODEL_PATH).read_bytes())
    for name, content in {
        "LICENSE": "Synthetic test license\n",
        "docs/HISTORY.md": "Synthetic documentation\n",
        "tests/check.py": "# synthetic test\n",
        ".github/workflows/check.yml": "name: Synthetic check\n",
    }.items():
        write(root, name, content)
    font = next(iter(audit.PUBLIC_FONT_SHA256))
    write(root, font, (SOURCE_ROOT / font).read_bytes())
    git(root, "add", "--all")
    return root


def findings_for(root, name):
    findings, _ = audit.inspect_index(root)
    return {category for path, category in findings if path == name}


def test_complete_index_with_approved_model_and_font_passes(publish_index):
    findings, count = audit.inspect_index(publish_index)
    assert not findings
    assert count >= len(audit.REQUIRED_FILES)


def test_model_notices_are_required_at_the_actual_documentation_paths():
    assert set(PUBLISHED_MODEL_NOTICES) <= audit.REQUIRED_FILES
    assert not {Path(name).name for name in PUBLISHED_MODEL_NOTICES} & audit.REQUIRED_FILES


@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
@pytest.mark.parametrize("style", ["macro", "assignment", "json"])
def test_all_private_macros_and_fields_are_detected(name, style):
    value = "fixture" + "-private-value"
    content = {
        "macro": f'#define {name} "{value}"',
        "assignment": f'const char *{name} = "{value}";',
        "json": f'{{"{name}": "{value}"}}',
    }[style]
    assert "literal-credential-definition" in audit.text_categories(content, set(), None)


@pytest.mark.parametrize("name", CREDENTIAL_NAMES)
def test_placeholders_and_empty_optional_credentials_pass(name):
    for value in (f"YOUR_{name}", "SYNTHETIC_FIXTURE", ""):
        assert not audit.text_categories(f'#define {name} "{value}"', set(), None)


def test_ignored_local_credentials_are_recognized_when_copied_elsewhere(publish_index):
    value = "fixture" + "-private-local-key"
    write(publish_index, "clap_double/clap_local.h",
          f'#define {CREDENTIAL_NAMES[3]} "{value}"\n')
    write(publish_index, "README.md", f"Accidentally copied {value}\n")
    git(publish_index, "add", "README.md")
    assert "configured-private-credential" in findings_for(publish_index, "README.md")


def test_exact_index_is_inspected_even_when_worktree_is_cleaned(publish_index):
    value = "fixture" + "-staged-private-value"
    name = "lab/app.js"
    write(publish_index, name, f'const {CREDENTIAL_NAMES[1]} = "{value}";\n')
    git(publish_index, "add", name)
    write(publish_index, name, "// clean unstaged replacement\n")
    assert "literal-credential-definition" in findings_for(publish_index, name)


def test_unstaged_private_text_is_not_mistaken_for_published_blob(publish_index):
    write(publish_index, "README.md", "Safe staged content\n")
    git(publish_index, "add", "README.md")
    write(publish_index, "README.md", "\0private unstaged bytes\n")
    assert not audit.inspect_index(publish_index)[0]


@pytest.mark.parametrize("suffix", ["other", "clap_far", "finger_snap"])
def test_timestamp_hex_recording_ids_are_detected(suffix):
    recording_id = "2099-01-01" + "T01-02-03-004Z-abcdef-" + suffix
    assert "private-recording-identifier" in audit.text_categories(recording_id, set(), None)


def test_network_identifiers_and_personal_paths_are_detected():
    address = ".".join(["192", "168", "42", "12"])
    mac = ":".join(["12", "34", "56", "78", "9A", "BC"])
    path = "/".join(["C:", "Users", "fixture-owner", "private"])
    for content, category in (
        (address, "specific-private-network-address"),
        ("IPAddress(" + address.replace(".", ", ") + ")", "specific-private-network-address"),
        ("#define CLAP_TUYA_IP " + address.replace(".", ", "), "specific-private-network-address"),
        (mac, "device-mac-address"),
        (path, "personal-absolute-path"),
    ):
        assert category in audit.text_categories(content, set(), None)


@pytest.mark.parametrize("name", [
    "data/session/audio.wav", "clap_double/clap_local.h", "docs/private.pcm",
    "hybrid/other/model_data.h", "docs/checkpoint.pt", "AGENT_AI_NOTES.md",
])
def test_private_and_unrelated_files_are_rejected(publish_index, name):
    write(publish_index, name, "synthetic private fixture\n")
    git(publish_index, "add", name)
    assert findings_for(publish_index, name)


def test_approved_export_is_independently_verified():
    blob = (SOURCE_ROOT / audit.APPROVED_MODEL_PATH).read_bytes()
    assert audit.approved_model(blob)
    assert audit.approved_model(blob.replace(b"\r\n", b"\n"))
    # Changing weights keeps the model's declared metadata hash untouched.
    changed = blob.replace(b"0x20", b"0x21", 1)
    assert not audit.approved_model(changed)
    assert not audit.approved_model(blob + b"\n// extra unapproved content\n")


def test_scanner_and_export_template_are_safe_source_text():
    # Recognition expressions and generated-header templates are source code,
    # rather than concrete owner paths or serialized trained-model arrays.
    for name in ("tools/audit_publish.py", "hybrid/export.py"):
        text = (SOURCE_ROOT / name).read_text(encoding="utf-8")
        assert not audit.text_categories(text, set(), None)


def test_approved_flatbuffer_has_no_private_metadata_strings():
    blob = (SOURCE_ROOT / audit.APPROVED_MODEL_PATH).read_bytes()
    values = audit.deployment_model_bytes(blob)
    assert values is not None and len(values) == audit.APPROVED_MODEL_BYTES
    assert not audit.model_text_categories(values, set(), None)


def test_decoded_model_metadata_is_inspected_including_utf8_credentials():
    value = "fixture" + "-private-\u79d8\u5bc6"
    path = "/".join(["C:", "Users", "fixture-owner", "private"])
    recording_id = "2099-01-01" + "T01-02-03-004Z-abcdef-other"
    values = b"\0".join(text.encode("utf-8") for text in (value, path, recording_id))
    categories = audit.model_text_categories(values, {value}, None)
    assert "deployment-model-configured-private-credential" in categories
    assert "deployment-model-personal-absolute-path" in categories
    assert "deployment-model-private-recording-identifier" in categories


def test_changed_model_is_rejected_in_exact_staged_blob(publish_index):
    name = audit.APPROVED_MODEL_PATH
    blob = (SOURCE_ROOT / name).read_bytes()
    write(publish_index, name, blob.replace(b"0x20", b"0x21", 1))
    git(publish_index, "add", name)
    write(publish_index, name, blob)  # Clean worktree does not repair the index.
    categories = findings_for(publish_index, name)
    assert "unapproved-deployment-model" in categories
    assert "embedded-trained-model" in categories


def test_font_signature_does_not_allow_arbitrary_binary_content(publish_index):
    name = next(iter(audit.PUBLIC_FONT_SHA256))
    write(publish_index, name, b"wOF2\0unreviewed private binary payload")
    git(publish_index, "add", name)
    assert "unreviewed-public-font-content" in findings_for(publish_index, name)


def test_unreviewed_font_filename_is_rejected(publish_index):
    name = "lab/fonts/unreviewed.woff2"
    write(publish_index, name, b"wOF2\0synthetic fixture")
    git(publish_index, "add", name)
    assert "outside-publish-allowlist" in findings_for(publish_index, name)


def test_binary_disguised_as_source_is_rejected(publish_index):
    name = "tools/hidden.py"
    write(publish_index, name, b"\xff\xfe\0private fixture")
    git(publish_index, "add", name)
    assert "unexpected-binary-data" in findings_for(publish_index, name)


def test_missing_required_file_blocks_publication(publish_index):
    name = "README.md"
    git(publish_index, "rm", "--cached", name)
    assert "required-publish-file-not-staged" in findings_for(publish_index, name)


def test_source_gitignore_allows_only_the_approved_deployment_header(tmp_path):
    root = tmp_path / "ignore-check"
    root.mkdir()
    git(root, "init", "--quiet")
    write(root, ".gitignore", (SOURCE_ROOT / ".gitignore").read_bytes())
    private_names = (
        "clap_double/clap_local.h", "data/sessions/private/audio.wav",
        "data/hybrid/export/model_data.h", "hybrid/old/model_data.h",
        "data/hybrid/best.pt", ".build/clap-double.bin", "lab/private.mp3",
    )
    for name in (*private_names, audit.APPROVED_MODEL_PATH):
        write(root, name, "synthetic fixture\n")
    for name in private_names:
        result = subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", name])
        assert result.returncode == 0
    result = subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", audit.APPROVED_MODEL_PATH])
    assert result.returncode == 1


def test_symlink_index_entry_is_rejected(publish_index):
    object_id = git(publish_index, "hash-object", "README.md").decode().strip()
    git(publish_index, "update-index", "--cacheinfo", f"120000,{object_id},README.md")
    assert "symlink-submodule-or-unmerged-index-entry" in findings_for(publish_index, "README.md")


def test_cli_fails_closed_without_git_index(tmp_path):
    result = subprocess.run([sys.executable, str(SOURCE_ROOT / "tools/audit_publish.py")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "BLOCKED" in result.stderr


def test_cli_reports_category_without_disclosing_secret(publish_index, monkeypatch, capsys):
    value = "fixture" + "-never-print-this-value"
    write(publish_index, "README.md", f'#define {CREDENTIAL_NAMES[1]} "{value}"\n')
    git(publish_index, "add", "README.md")
    monkeypatch.chdir(publish_index)
    assert audit.main() == 1
    output = capsys.readouterr().out
    assert "README.md: literal-credential-definition" in output
    assert value not in output
