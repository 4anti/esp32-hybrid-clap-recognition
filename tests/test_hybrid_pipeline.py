import json

import numpy as np
import pytest
from scipy.io import wavfile

from hybrid.frontend import FEATURE_SHAPE, PRE_TRIGGER_SAMPLES, WINDOW_SAMPLES, contract, event_window, log_mel
from hybrid.prepare import assign_splits, build_dataset, read_wav, slices, to_16k, transient_samples
from train_custom import classification_metrics, load_dataset, select_threshold, train
from train_custom import sampling_weights, warm_start_model


def impulse(length=16000, points=(3000, 12000)):
    audio = np.zeros(length, np.float32)
    for point in points:
        audio[point] = 0.5
    return audio


def verified_session(root, name, label, split=None, group=None, ready=True):
    directory = root / name
    directory.mkdir(parents=True)
    meta = {"label": label, "trainingReady": ready,
            "capture": {"raw": True, "pcmVersion": 2, "sampleRate": 16000},
            "event_samples": [3000, 12000]}
    if split:
        meta["split"] = split
    if group:
        meta["split_group"] = group
    (directory / "session.json").write_text(json.dumps(meta), encoding="utf-8")
    wavfile.write(directory / "audio.wav", 16000, (impulse() * 32768).astype(np.int16))


def valid_blob():
    rng = np.random.default_rng(5)
    y = np.tile(np.arange(3), 6).astype(np.int64)
    split = np.repeat(["train", "val", "test"], 6)
    return {"schema_version": np.asarray(2), "x": rng.normal(size=(18, *FEATURE_SHAPE)).astype(np.float32),
            "y": y, "split": split, "labels": np.asarray(["noise", "clap", "finger_snap"]),
            "sources": np.asarray([f"recording-{i}" for i in range(18)]),
            "groups": np.asarray([f"capture-{i}" for i in range(18)]),
            "domains": np.asarray(["lab"] * 18),
            "frontend_json": np.asarray(json.dumps(contract()))}


def test_stereo_pcm_is_scaled_before_channel_average(tmp_path):
    raw = np.array([[16384, 0], [-16384, 0], [-32768, -32768]], np.int16)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 16000, raw)
    audio, rate = read_wav(path)
    assert rate == 16000
    np.testing.assert_array_equal(audio, [0.25, -0.25, -1])


def test_unsigned_pcm_silence_has_zero_offset(tmp_path):
    path = tmp_path / "unsigned.wav"
    wavfile.write(path, 16000, np.array([0, 128, 255], np.uint8))
    audio, _ = read_wav(path)
    np.testing.assert_array_equal(audio, [-1, 0, 127 / 128])


def test_nonfinite_wav_and_bad_resampling_rejected(tmp_path):
    path = tmp_path / "nan.wav"
    wavfile.write(path, 16000, np.array([np.nan], np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        read_wav(path)
    with pytest.raises(ValueError):
        to_16k(np.zeros(10), 0)


@pytest.mark.parametrize("point", [1, 5000, 9999])
def test_trigger_is_at_same_feature_position_including_file_edges(point):
    audio = impulse(10000, (point,))
    window = event_window(audio, point)
    assert window.shape == (WINDOW_SAMPLES,)
    assert window.argmax() == PRE_TRIGGER_SAMPLES
    assert log_mel(window).shape == FEATURE_SHAPE


def test_frontend_silence_and_bad_inputs():
    np.testing.assert_allclose(log_mel(np.zeros(WINDOW_SAMPLES)), np.log(0.001), atol=1e-6)
    with pytest.raises(ValueError, match="expected"):
        log_mel(np.zeros(12))
    with pytest.raises(ValueError, match="NaN"):
        log_mel(np.full(WINDOW_SAMPLES, np.nan))


def test_positive_candidates_omit_long_silent_gaps():
    assert transient_samples(impulse(48000, (1000, 47000))) == [1000, 47000]
    assert transient_samples(np.zeros(16000)) == []
    assert len(slices(np.zeros(WINDOW_SAMPLES + 1))) == 2


def test_grouped_splits_are_deterministic_and_complete():
    first = assign_splits({f"label-{label}-{i}": label for label in range(3) for i in range(3)})
    second = assign_splits(dict(reversed(list({f"label-{label}-{i}": label for label in range(3) for i in range(3)}.items()))))
    assert first == second
    for label in range(3):
        assert {value for key, value in first.items() if key.startswith(f"label-{label}-")} == {"train", "val", "test"}


def test_positive_lab_windows_and_binary_exclusion(tmp_path):
    verified_session(tmp_path, "claps", "clap", split="train")
    verified_session(tmp_path, "snaps", "finger_snap", split="test")
    verified_session(tmp_path, "gappy", "clap", ready=False)
    arrays, meta = build_dataset(None, tmp_path, labels=("noise", "clap"))
    assert arrays["y"].tolist() == [1, 1]
    assert set(arrays["label_quality"]) == {"annotated"}
    assert set(arrays["sources"]) == {"lab:claps"}
    assert any("gappy" in text for text in meta["warnings"])


def test_conflicting_manual_capture_splits_fail(tmp_path):
    verified_session(tmp_path, "a", "clap", split="train", group="same-room-run")
    verified_session(tmp_path, "b", "finger_snap", split="test", group="same-room-run")
    with pytest.raises(ValueError, match="spans splits"):
        build_dataset(None, tmp_path)


def test_legacy_session_requires_explicit_opt_in(tmp_path):
    verified_session(tmp_path, "old", "clap")
    meta_path = tmp_path / "old" / "session.json"
    meta = json.loads(meta_path.read_text())
    del meta["capture"]
    del meta["trainingReady"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="no usable"):
        build_dataset(None, tmp_path)
    arrays, meta = build_dataset(None, tmp_path, allow_legacy_sessions=True)
    assert len(arrays["y"]) == 2
    assert any("legacy" in text for text in meta["warnings"])


def test_room_hard_negative_limit_is_independent_of_public_limit(tmp_path):
    verified_session(tmp_path, "bag", "other")
    limited, _ = build_dataset(None, tmp_path, max_windows_per_file=2)
    expanded, _ = build_dataset(None, tmp_path, max_windows_per_file=2, lab_max_windows_per_file=20)
    assert len(limited["y"]) == 2
    assert len(expanded["y"]) == 4
    assert set(expanded["y"]) == {0}


@pytest.mark.parametrize("change,message", [
    ("old", "Old/unverifiable"), ("leak", "leakage"), ("nan", "nonfinite"),
    ("missing", "test split"), ("contract", "frontend differs"), ("fractional", "invalid labels"),
])
def test_training_rejects_bad_datasets(tmp_path, change, message):
    blob = valid_blob()
    if change == "old":
        del blob["groups"]
    elif change == "leak":
        blob["groups"][6] = blob["groups"][0]
    elif change == "nan":
        blob["x"][0, 0, 0] = np.nan
    elif change == "missing":
        blob["y"][blob["split"] == "test"] = 0
    elif change == "contract":
        blob["frontend_json"] = np.asarray("{}")
    else:
        blob["y"] = blob["y"].astype(float) + 0.5
    path = tmp_path / "features.npz"
    np.savez(path, **blob)
    with pytest.raises(ValueError, match=message):
        load_dataset(path)


def test_threshold_rejects_noise_and_preserves_more_confident_gesture():
    y = np.array([0, 0, 1, 2])
    probs = np.array([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.03, 0.02, 0.95]], np.float32)
    threshold = select_threshold(y, probs, 0)
    metrics = classification_metrics(y, probs, ("noise", "clap", "finger_snap"), threshold)
    assert metrics["gate"]["noise_false_accepts"] == 0
    assert metrics["gate"]["gesture_recall"] == {"clap": 1, "finger_snap": 1}
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_training_smoke_writes_heldout_metrics_and_safe_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    blob = valid_blob()
    path = tmp_path / "features.npz"
    np.savez(path, **blob)
    report = train(epochs=1, patience=1, batch_size=6, data_path=path, out=tmp_path / "run", threads=1)
    assert report["splits"]["test"]["samples"] == 6
    assert report["labels"] == ["noise", "clap", "finger_snap"]
    assert (tmp_path / "run" / "metrics.json").is_file()
    checkpoint = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert checkpoint["frontend"] == contract()
    np.testing.assert_allclose(checkpoint["mean"], blob["x"][:6, :, 32:].mean(axis=(0, 2)), atol=1e-6)
    adapted = dict(blob)
    adapted["x"] = blob["x"] + 2
    adapted_path = tmp_path / "adapted.npz"
    np.savez(adapted_path, **adapted)
    adapted_report = train(epochs=1, patience=1, batch_size=6, data_path=adapted_path,
                           out=tmp_path / "adapted", threads=1, balance_domains=True,
                           init_checkpoint=tmp_path / "run/best.pt", init_data=path)
    adapted_checkpoint = torch.load(tmp_path / "adapted/best.pt", weights_only=True)
    np.testing.assert_array_equal(adapted_checkpoint["mean"], checkpoint["mean"])
    assert adapted_report["decision"]["noise_budget_scope"] == "each validation domain"
    assert adapted_report["initialization"]["original_data_sha256"] == checkpoint["data_sha256"]
    leaked = load_dataset(adapted_path)
    leaked["sources"][[0, 12]] = leaked["sources"][[12, 0]]
    with pytest.raises(ValueError, match="moved between splits"):
        warm_start_model(tmp_path / "run/best.pt", path, leaked, (8, 16, 32))
    with pytest.raises(ValueError, match="provenance"):
        warm_start_model(tmp_path / "run/best.pt", adapted_path, load_dataset(adapted_path), (8, 16, 32))
    with pytest.raises(ValueError, match="architecture"):
        warm_start_model(tmp_path / "run/best.pt", path, load_dataset(adapted_path), (12, 24, 48))
    wider, provenance = warm_start_model(tmp_path / "run/best.pt", path, load_dataset(adapted_path),
                                         (12, 24, 48), allow_widen=True)
    assert wider.widths == (12, 24, 48) and provenance["widened"]


def test_widening_preserves_predictions_and_leaves_new_channels_trainable():
    torch = pytest.importorskip("torch")
    from hybrid.cnn import ResearchCNN, widen_model
    torch.manual_seed(7)
    original = ResearchCNN(np.linspace(-3, 2, 64), np.linspace(.5, 2, 64)).eval()
    # Exercise non-default learned BN state, not only fresh identity statistics.
    for index in range(3):
        bn = original.features[index * 4 + 1]
        bn.running_mean.copy_(torch.randn_like(bn.running_mean))
        bn.running_var.copy_(torch.rand_like(bn.running_var) + .2)
    wider = widen_model(original, (12, 24, 48)).eval()
    inputs = torch.randn(9, 1, 64, 48)
    torch.testing.assert_close(wider(inputs), original(inputs), rtol=1e-5, atol=1e-6)
    wider(inputs).square().sum().backward()
    assert wider.classifier.weight.grad[:, 32:48].abs().sum() > 0
    assert wider.classifier.weight.grad[:, 80:96].abs().sum() > 0
    with pytest.raises(ValueError, match="remove"):
        widen_model(original, (4, 16, 32))


def test_augmentation_can_disable_gain_jitter_and_masks():
    torch = pytest.importorskip("torch")
    from train_custom import augment_features
    inputs = torch.rand(6, 1, 64, 48) * 7 - 6
    actual = augment_features(inputs, torch.zeros(64), gain_db=(0, 0),
                              jitter_frames=0, frequency_mask_bands=0)
    torch.testing.assert_close(actual, inputs, rtol=0, atol=1e-6)


def test_prior_clap_history_cannot_validate_current_noise():
    torch = pytest.importorskip("torch")
    from hybrid.cnn import ResearchCNN
    model = ResearchCNN(np.zeros(64), np.ones(64)).eval()
    current_noise = torch.zeros(1, 1, 64, 48)
    with_prior_clap = current_noise.clone()
    with_prior_clap[:, :, :, :32] = torch.randn(1, 1, 64, 32) * 100
    with torch.inference_mode():
        torch.testing.assert_close(model(current_noise), model(with_prior_clap), rtol=0, atol=0)


def fsd_fixture(root, source="123", uploader="person", split="train"):
    directory = root / "audio"
    directory.mkdir(parents=True, exist_ok=True)
    wavfile.write(directory / f"{source}.wav", 16000, (impulse() * 32768).astype(np.int16))
    rows = [{"filename": f"audio/{source}.wav", "source_id": f"fsd50k:{source}", "uploader": uploader,
             "split": split, "label": "clap", "category": "Clapping",
             "license": "https://creativecommons.org/publicdomain/zero/1.0/", "source_url": f"https://freesound.org/s/{source}/"}]
    (root / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    return rows


def esc_fixture(root, rows):
    (root / "meta").mkdir(parents=True)
    (root / "audio").mkdir()
    import csv
    with (root / "meta" / "esc50.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "fold", "target", "category", "src_file"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        wavfile.write(root / "audio" / row["filename"], 16000, (impulse() * 32768).astype(np.int16))


def test_cross_dataset_freesound_origin_is_not_duplicated(tmp_path):
    fsd_fixture(tmp_path / "fsd", source="123", split="train")
    esc_fixture(tmp_path / "esc", [{"filename": "5-123-A-22.wav", "fold": 5, "target": 22, "category": "clapping", "src_file": "123"}])
    arrays, _ = build_dataset(tmp_path / "esc", tmp_path / "sessions", fsd50k=tmp_path / "fsd")
    assert set(arrays["sources"]) == {"fsd50k:123"}
    assert set(arrays["split"]) == {"train"}


def test_official_esc_original_source_split_conflicts_are_excluded(tmp_path):
    rows = [{"filename": "4-123-A-37.wav", "fold": 4, "target": 37, "category": "clock_alarm", "src_file": "123"},
            {"filename": "5-123-A-38.wav", "fold": 5, "target": 38, "category": "clock_tick", "src_file": "123"},
            {"filename": "1-456-A-30.wav", "fold": 1, "target": 30, "category": "door_wood_knock", "src_file": "456"}]
    esc_fixture(tmp_path / "esc", rows)
    arrays, meta = build_dataset(tmp_path / "esc", tmp_path / "sessions")
    assert set(arrays["groups"]) == {"esc50:456"}
    assert any("123" in notice and "official splits" in notice for notice in meta["warnings"])


def test_fsd_sha256_tampering_is_rejected(tmp_path):
    rows = fsd_fixture(tmp_path / "fsd")
    rows[0]["sha256"] = "bad"
    (tmp_path / "fsd" / "manifest.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="SHA256"):
        build_dataset(None, tmp_path / "sessions", fsd50k=tmp_path / "fsd")


def test_sampling_keeps_long_recordings_from_dominating_small_sources():
    pytest.importorskip("torch")
    from train_custom import _loader
    y = np.array([0] * 20 + [1] * 10 + [1] + [2])
    sources = np.array(["noise-a"] * 20 + ["dense-applause"] * 10 + ["single-clap", "single-snap"])
    loader = _loader(np.zeros((len(y), 64, 48), np.float32), y, 8, True, 42, sources=sources, balance=True)
    weights = loader.sampler.weights.numpy()
    np.testing.assert_allclose([weights[y == i].sum() for i in range(3)], [1, 1, 1])
    assert weights[sources == "dense-applause"].sum() == weights[sources == "single-clap"].sum()


def test_room_bag_recording_has_equal_domain_weight_to_large_public_dataset():
    y = np.array([0] * 102 + [1, 1, 2])
    sources = np.array([f"public-{i}" for i in range(100)] + ["room-bag"] * 2 + ["public-clap", "room-clap", "room-snap"])
    domains = np.array(["public"] * 100 + ["lab"] * 2 + ["public", "lab", "lab"])
    weights = sampling_weights(y, sources, domains)
    np.testing.assert_allclose([weights[y == i].sum() for i in range(3)], [1, 1, 1])
    assert weights[:100].sum() == pytest.approx(weights[100:102].sum())
    assert weights[102] == weights[103]
    weighted = sampling_weights(y, sources, domains, room_weight=0.7)
    assert weighted[:100].sum() == pytest.approx(0.3)
    assert weighted[100:102].sum() == pytest.approx(0.7)
    np.testing.assert_allclose([weighted[y == i].sum() for i in range(3)], [1, 1, 1])


def test_room_noise_failure_cannot_hide_inside_public_validation_noise():
    y = np.array([0] * 101 + [1, 2])
    probs = np.array([[0.99, 0.005, 0.005]] * 100 + [[0.05, 0.9, 0.05], [0.02, 0.97, 0.01], [0.02, 0.01, 0.97]], np.float32)
    domains = np.array(["public"] * 100 + ["lab"] * 3)
    pooled = select_threshold(y, probs, 0.01)
    room_safe = select_threshold(y, probs, 0.01, domains)
    assert pooled < 0.95 < room_safe
    gate = classification_metrics(y[-3:], probs[-3:], ("noise", "clap", "finger_snap"), room_safe)["gate"]
    assert gate["noise_false_accepts"] == 0
    assert gate["gesture_recall"] == {"clap": 1, "finger_snap": 1}


def test_explicit_room_calibration_reports_selected_noise_scope():
    y = np.array([0, 0, 1, 2])
    probabilities = np.array([[0.02, 0.97, 0.01], [0.8, 0.1, 0.1], [0.1, 0.85, 0.05], [0.08, 0.02, 0.9]], np.float32)
    domains = np.array(["public", "lab", "public", "public"])
    threshold = select_threshold(y, probabilities, 0, domains, noise_domains=["lab"])
    assert threshold == pytest.approx(0.85)
    assert select_threshold(y, probabilities, 0, domains) > 0.98
    with pytest.raises(ValueError, match="validation noise"):
        select_threshold(y, probabilities, 0, domains, noise_domains=["unrecorded-room"])


def test_threshold_matches_esp32_minimum_even_with_easy_validation():
    y = np.array([0, 1, 2])
    probs = np.array([[0.99, 0.005, 0.005], [0.2, 0.79, 0.01], [0.05, 0.01, 0.94]], np.float32)
    assert select_threshold(y, probs, 0) >= 0.85


def test_eq_and_background_augmentation_preserves_silence_and_source_features():
    torch = pytest.importorskip("torch")
    from train_custom import augment_features
    torch.manual_seed(6)
    floor = float(np.log(0.001))
    silent = torch.full((16, 1, 64, 48), floor)
    unchanged = silent.clone()
    background = torch.zeros(5, 1, 64, 48)
    augmented = augment_features(silent, torch.full((64,), floor), eq_db_max=4, background=background)
    torch.testing.assert_close(silent, unchanged, rtol=0, atol=0)
    torch.testing.assert_close(augmented, unchanged, rtol=0, atol=1e-6)
    active = torch.zeros(16, 1, 64, 48)
    modified = augment_features(active, torch.zeros(64), eq_db_max=4, background=background)
    assert torch.isfinite(modified).all()
    assert not torch.equal(active, modified)
    assert torch.unique(modified[:, :, :, 32:].reshape(16, -1), dim=0).shape[0] == 16


def test_excluded_full_take_cannot_duplicate_its_trimmed_derivative(tmp_path):
    verified_session(tmp_path, "full", "other")
    verified_session(tmp_path, "trimmed", "other", group="full")
    path = tmp_path / "full/session.json"
    metadata = json.loads(path.read_text())
    metadata["exclude_from_training"] = True
    path.write_text(json.dumps(metadata))
    arrays, meta = build_dataset(None, tmp_path)
    assert set(arrays["sources"]) == {"lab:trimmed"}
    assert any("Explicitly excluded" in warning for warning in meta["warnings"])


def test_training_ready_flag_cannot_hide_capture_mismatch(tmp_path):
    verified_session(tmp_path, "mismatch", "clap")
    path = tmp_path / "mismatch" / "session.json"
    meta = json.loads(path.read_text())
    meta["integrity"] = {"samples": 42}
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="capture metadata"):
        build_dataset(None, tmp_path)
