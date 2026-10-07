import numpy as np
import pytest

from hybrid.export import calibration_indices, fused_conv_weights, header_text, quantize_input


def test_calibration_is_class_complete_and_training_only():
    y = np.tile(np.arange(3), 10)
    split = np.repeat(["train", "val", "test"], 10)
    selected = calibration_indices(y, split, samples=6)
    assert set(y[selected]) == {0, 1, 2}
    assert set(split[selected]) == {"train"}
    np.testing.assert_array_equal(selected, calibration_indices(y, split, samples=6))
    with pytest.raises(ValueError, match="every class"):
        calibration_indices(y, split, samples=2)


def test_quantization_calibration_covers_small_room_domain():
    y = np.array([0] * 100 + [1] * 100 + [2] * 100 + [0, 1, 2] + [0, 1, 2])
    split = np.array(["train"] * 303 + ["val"] * 3)
    domains = np.array(["public"] * 300 + ["lab"] * 6)
    selected = calibration_indices(y, split, samples=30, domains=domains)
    assert {(int(y[i]), domains[i]) for i in selected} == {(label, domain) for label in range(3) for domain in ("public", "lab")}
    assert set(split[selected]) == {"train"}
    with pytest.raises(ValueError, match="class/domain"):
        calibration_indices(y, split, samples=3, domains=domains)


def test_int8_quantization_uses_cpp_rounding_and_saturates():
    values = np.array([-100, -0.75, -0.25, 0.25, 0.75, 100], np.float32)
    np.testing.assert_array_equal(quantize_input(values, 0.5, 0), [-128, -2, -1, 1, 2, 127])
    with pytest.raises(ValueError):
        quantize_input(values, 0, 0)


def test_header_carries_real_model_and_frontend_contract():
    header = header_text(b"\x00\x11\xff", np.zeros(64), np.ones(64), (0.1, -4), (1 / 256, -128), 0.85)
    assert "namespace clap_model" in header
    assert "0x00, 0x11, 0xff" in header
    assert '"noise", "clap", "finger_snap"' in header
    assert "input_zero_point = -4" in header
    assert "positive_threshold = 0.85f" in header
    assert "target_frame_start = 32" in header
    with pytest.raises(ValueError, match="firmware header"):
        header_text(b"bytes", np.zeros(64), np.ones(64), (0.1, 0), (0.1, 0), 0.8, labels=("noise", "clap"))


def test_folded_conv_batchnorm_matches_torch_eval():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    conv = torch.nn.Conv2d(2, 3, 3, padding=1, bias=False)
    bn = torch.nn.BatchNorm2d(3)
    bn.running_mean = torch.randn(3)
    bn.running_var = torch.rand(3) + 0.5
    bn.weight.data = torch.rand(3)
    bn.bias.data = torch.randn(3)
    bn.eval()
    kernel, bias = fused_conv_weights(conv, bn)
    features = torch.randn(1, 2, 8, 6)
    expected = bn(conv(features))
    actual = torch.nn.functional.conv2d(features, torch.from_numpy(kernel.transpose(3, 2, 0, 1)),
                                        torch.from_numpy(bias), padding=1)
    torch.testing.assert_close(actual, expected)


def test_keras_float_parity_when_tensorflow_is_available():
    tf = pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")
    from hybrid.cnn import ResearchCNN
    from hybrid.export import _keras_model

    torch.set_num_threads(1)
    torch.manual_seed(3)
    mean, std = np.linspace(-3, 2, 64).astype(np.float32), np.linspace(0.5, 3, 64).astype(np.float32)
    model = ResearchCNN(mean, std, widths=(8, 16, 32)).eval()
    features = np.random.default_rng(3).normal(size=(1, 64, 48)).astype(np.float32)
    normalized = ((features - mean[None, :, None]) / std[None, :, None])[..., None]
    keras = _keras_model(tf, model)
    with torch.inference_mode():
        expected = model(torch.from_numpy(features).unsqueeze(1)).softmax(1).numpy()
    np.testing.assert_allclose(keras(normalized).numpy(), expected, atol=1e-6)


def test_full_int8_export_runtime_on_temporary_fixture(tmp_path):
    pytest.importorskip("tensorflow")
    torch = pytest.importorskip("torch")
    import hashlib
    import json
    from hybrid.cnn import ResearchCNN
    from hybrid.export import export
    from hybrid.frontend import contract

    torch.manual_seed(7)
    features = np.random.default_rng(7).normal(size=(18, 64, 48)).astype(np.float32)
    path = tmp_path / "features.npz"
    np.savez(path, schema_version=np.asarray(2), x=features, y=np.tile(np.arange(3), 6),
             labels=np.asarray(["noise", "clap", "finger_snap"]), split=np.repeat(["train", "val", "test"], 6),
             sources=np.asarray([str(i) for i in range(18)]), groups=np.asarray([str(i) for i in range(18)]),
             domains=np.asarray(["test_fixture"] * 18), frontend_json=np.asarray(json.dumps(contract())))
    model = ResearchCNN(np.zeros(64), np.ones(64), widths=(8, 16, 32)).eval()
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"state_dict": model.state_dict(), "widths": [8, 16, 32], "target_frame_start": 32, "frontend": contract(),
                "labels": ["noise", "clap", "finger_snap"], "mean": [0.] * 64, "std": [1.] * 64,
                "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "decision": {"positive_threshold": 0.8}}, checkpoint)
    report = export(checkpoint, path, tmp_path / "export", samples=6,
                    max_probability_error=0.2, max_accuracy_drop=1.0)
    assert report["float_parity_max_error"] < 1e-5
    assert report["model_bytes"] > 1000
    assert set(report["operators"]).issubset({"CONV_2D", "MAX_POOL_2D", "AVERAGE_POOL_2D", "RESHAPE",
                                             "CONCATENATION", "FULLY_CONNECTED", "SOFTMAX", "STRIDED_SLICE"})
    assert (tmp_path / "export" / "model_data.h").is_file()
