"""Optional full-INT8 export with float parity and held-out quantization checks.

Requires TensorFlow and PyTorch in the selected Python environment. No guessed
or synthetic model is emitted when dependencies, class coverage or checks fail.
The ESP32 input is NORMALIZED log-mel, NHWC [1,64,48,1]; normalization constants
and tensor quantization are included in model_data.h. This exporter must itself
be exercised with TensorFlow before its output is considered deployable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hybrid.frontend import FEATURE_SHAPE, LABELS, contract
from train_custom import classification_metrics, load_dataset, select_threshold

MICRO_OPS = {"CONV_2D", "MAX_POOL_2D", "AVERAGE_POOL_2D", "RESHAPE",
             "CONCATENATION", "FULLY_CONNECTED", "SOFTMAX", "STRIDED_SLICE"}


def calibration_indices(y, split, samples: int = 300, seed: int = 42, domains=None) -> np.ndarray:
    """Use all active classes, exclusively training examples, reproducibly."""
    y, split = np.asarray(y), np.asarray(split)
    classes = np.unique(y)
    if samples < len(classes):
        raise ValueError("representative sample budget must cover every class")
    domains = np.asarray(domains) if domains is not None else np.full(len(y), "all")
    if domains.shape != y.shape:
        raise ValueError("calibration requires one domain per example")
    pairs = [(label, domain) for label in classes for domain in np.unique(domains[(y == label) & (split == "train")])]
    if len(pairs) > samples:
        raise ValueError("representative sample budget must cover every class/domain")
    generator = np.random.default_rng(seed)
    indices = []
    for label in classes:
        pool = np.flatnonzero((y == label) & (split == "train"))
        if not pool.size:
            raise ValueError(f"no training calibration examples for class {label}")
        active_domains = np.unique(domains[pool])
        for domain in active_domains:
            domain_pool = pool[domains[pool] == domain].copy()
            generator.shuffle(domain_pool)
            indices.extend(domain_pool[:max(1, samples // len(classes) // len(active_domains))])
    return np.asarray(indices, dtype=np.int64)


def quantize_input(features: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0 or not -128 <= zero_point <= 127:
        raise ValueError("invalid INT8 quantization")
    # C++ lroundf rounds ties away from zero; avoid NumPy's bankers' rounding.
    values = np.asarray(features, dtype=np.float32) / np.float32(scale)
    rounded = np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))
    return np.clip(rounded + zero_point, -128, 127).astype(np.int8)


def fused_conv_weights(conv, batch_norm) -> tuple[np.ndarray, np.ndarray]:
    """Fold inference BatchNorm into Conv, then convert OIHW to HWIO."""
    weights = conv.weight.detach().cpu().numpy()
    gamma = batch_norm.weight.detach().cpu().numpy()
    beta = batch_norm.bias.detach().cpu().numpy()
    variance = batch_norm.running_var.detach().cpu().numpy()
    running_mean = batch_norm.running_mean.detach().cpu().numpy()
    multiplier = gamma / np.sqrt(variance + batch_norm.eps)
    bias = np.zeros_like(multiplier) if conv.bias is None else conv.bias.detach().cpu().numpy()
    return ((weights * multiplier[:, None, None, None]).transpose(2, 3, 1, 0).astype(np.float32),
            ((bias - running_mean) * multiplier + beta).astype(np.float32))


def _keras_model(tf, torch_model):
    layers = tf.keras.layers
    inputs = tf.keras.Input(batch_shape=(1, *FEATURE_SHAPE, 1), dtype=tf.float32, name="normalized_log_mel")
    x = layers.Cropping2D(cropping=((0, 0), (torch_model.target_frame_start, 0)))(inputs) if torch_model.target_frame_start else inputs
    convolutions = []
    for index, width in enumerate(torch_model.widths):
        conv = layers.Conv2D(width, 3, padding="same", activation="relu", use_bias=True, name=f"conv_{index}")
        x = layers.MaxPooling2D(2)(conv(x))
        convolutions.append(conv)
    # Builtin pooling avoids dynamic reduction operators in Micro.
    pool_size = (FEATURE_SHAPE[0] // 8, (FEATURE_SHAPE[1] - torch_model.target_frame_start) // 8)
    average = layers.Flatten()(layers.AveragePooling2D(pool_size)(x))
    maximum = layers.Flatten()(layers.MaxPooling2D(pool_size)(x))
    combined = layers.Concatenate()([average, maximum])
    dense = layers.Dense(torch_model.classifier.out_features, activation="softmax", name="probabilities")
    model = tf.keras.Model(inputs, dense(combined))
    for index, layer in enumerate(convolutions):
        layer.set_weights(list(fused_conv_weights(torch_model.features[index * 4], torch_model.features[index * 4 + 1])))
    dense.set_weights([torch_model.classifier.weight.detach().cpu().numpy().T,
                       torch_model.classifier.bias.detach().cpu().numpy()])
    return model


def header_text(model: bytes, mean, std, input_quantization, output_quantization,
                threshold: float, labels=LABELS, target_frame_start: int = 32) -> str:
    if tuple(labels) != LABELS:
        raise ValueError("firmware header requires the noise/clap/finger_snap contract")
    mean, std = np.asarray(mean), np.asarray(std)
    if mean.shape != (64,) or std.shape != (64,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid normalization arrays")
    if not model:
        raise ValueError("empty model")

    def number(value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("nonfinite header constant")
        rendered = f"{value:.9g}"
        if "." not in rendered and "e" not in rendered:
            rendered += ".0"
        return rendered + "f"

    byte_lines = ["  " + ", ".join(f"0x{byte:02x}" for byte in model[start : start + 16])
                  for start in range(0, len(model), 16)]
    return "\n".join([
        "// Generated by python -m hybrid.export after parity and INT8 checks.",
        "#pragma once", "#include <stdint.h>", "#include <stddef.h>",
        "namespace clap_model {", "alignas(16) static const unsigned char data[] = {",
        ",\n".join(byte_lines), "};", "static constexpr size_t data_len = sizeof(data);",
        f"static constexpr int class_count = {len(labels)};",
        "static constexpr const char* labels[] = {" + ", ".join(json.dumps(label) for label in labels) + "};",
        "static constexpr int model_version = 3;",
        f"static constexpr const char* sha256 = {json.dumps(hashlib.sha256(model).hexdigest())};",
        f"static constexpr int target_frame_start = {int(target_frame_start)};",
        "static constexpr float mean[64] = {" + ", ".join(number(value) for value in mean) + "};",
        "static constexpr float std[64] = {" + ", ".join(number(value) for value in std) + "};",
        f"static constexpr float input_scale = {number(input_quantization[0])};",
        f"static constexpr int input_zero_point = {int(input_quantization[1])};",
        f"static constexpr float output_scale = {number(output_quantization[0])};",
        f"static constexpr int output_zero_point = {int(output_quantization[1])};",
        f"static constexpr float positive_threshold = {number(threshold)};",
        f"static constexpr const char* frontend_version = {json.dumps(contract()['version'])};",
        "}  // namespace clap_model", "",
    ])


def export(checkpoint: Path, data_path: Path, out: Path, samples: int = 300,
           max_probability_error: float = 0.10, max_accuracy_drop: float = 0.02,
           max_class_recall_drop: float = 0.05) -> dict:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ValueError("TensorFlow is unavailable. Use an environment with TensorFlow for optional INT8 export; no model was generated.") from exc
    import torch
    from hybrid.cnn import ResearchCNN

    if not 0 <= max_accuracy_drop <= 1 or not 0 < max_probability_error <= 1 or not 0 <= max_class_recall_drop <= 1:
        raise ValueError("invalid quantization error budget")
    torch.set_num_threads(2)
    data = load_dataset(data_path)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tuple(saved.get("labels", ())) != LABELS or tuple(data["labels"]) != LABELS:
        raise ValueError("Deployment requires a genuinely trained three-class noise/clap/finger_snap model")
    if saved.get("frontend") != contract() or saved.get("data_sha256") != hashlib.sha256(data_path.read_bytes()).hexdigest():
        raise ValueError("Checkpoint and dataset/frontend mismatch")
    mean, std = np.asarray(saved["mean"], np.float32), np.asarray(saved["std"], np.float32)
    model = ResearchCNN(mean, std, len(LABELS), widths=saved.get("widths", (16, 32, 64)),
                        target_frame_start=saved.get("target_frame_start", 0))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    keras = _keras_model(tf, model)
    x = data["x"]
    normalized = ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)[..., None]
    normalized[:, :, :model.target_frame_start, :] = 0
    room_balanced = saved["decision"].get("balance_domains") or saved["decision"].get("noise_budget_scope") == "each validation domain"
    indices = calibration_indices(data["y"], data["split"], samples,
                                  domains=data["domains"] if room_balanced else None)
    parity_indices = indices[:min(24, len(indices))]
    with torch.inference_mode():
        torch_probs = model(torch.from_numpy(x[parity_indices]).unsqueeze(1)).softmax(1).numpy()
    keras_probs = np.concatenate([keras(normalized[index : index + 1], training=False).numpy() for index in parity_indices])
    float_error = float(np.max(np.abs(torch_probs - keras_probs)))
    if float_error > 1e-5:
        raise ValueError(f"PyTorch/Keras float parity failed: max probability error {float_error}")

    def representative():
        for index in indices:
            yield [normalized[index : index + 1]]

    # Freeze variables before conversion. TensorFlow 2.16/Keras 3's direct
    # from_keras_model path can abort the process on ReadVariableOp (TF #63987).
    # A constant concrete graph also makes the Micro operator audit explicit.
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    concrete = tf.function(lambda value: keras(value, training=False),
                           input_signature=[tf.TensorSpec((1, *FEATURE_SHAPE, 1), tf.float32)]).get_concrete_function()
    frozen = convert_variables_to_constants_v2(concrete)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    binary = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=binary,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    input_info, output_info = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
        raise ValueError("Export did not produce INT8 input/output")
    if tuple(input_info["shape"]) != (1, *FEATURE_SHAPE, 1) or tuple(output_info["shape"]) != (1, 3):
        raise ValueError("Exported tensor shape violates firmware contract")
    operators = sorted({entry["op_name"] for entry in interpreter._get_ops_details()})
    if set(operators) - MICRO_OPS or any(info["dtype"] == np.float32 for info in interpreter.get_tensor_details()):
        raise ValueError(f"Unsupported Micro operators or float tensors: {operators}")
    input_quant, output_quant = input_info["quantization"], output_info["quantization"]
    if input_quant[0] <= 0 or output_quant[0] <= 0:
        raise ValueError("missing tensor quantization")
    threshold = max(0.85, float(saved["decision"]["positive_threshold"]))
    if threshold > 1:
        raise ValueError("Validation gate rejects every gesture; improve data/model before deploying")

    def predict(mask):
        with torch.inference_mode():
            float_probs = np.concatenate([model(torch.from_numpy(batch).unsqueeze(1)).softmax(1).numpy()
                for batch in np.array_split(x[mask], max(1, int(mask.sum()) // 32))])
        int_probs = []
        for value in normalized[mask]:
            interpreter.set_tensor(input_info["index"], quantize_input(value[None], *input_quant))
            interpreter.invoke()
            output = interpreter.get_tensor(output_info["index"])[0]
            int_probs.append((output.astype(np.float32) - output_quant[1]) * output_quant[0])
        return float_probs, np.asarray(int_probs)

    # Quantization acceptance is decided on validation, never by optimizing
    # against held-out test performance. Noise-heavy global accuracy alone can
    # hide a substantial loss of snap/clap recall.
    validation_mask = data["split"] == "val"
    val_actual = data["y"][validation_mask]
    val_float, val_int = predict(validation_mask)
    float_threshold = threshold
    if "max_validation_noise_false_accept" in saved["decision"]:
        threshold = select_threshold(val_actual, val_int, saved["decision"]["max_validation_noise_false_accept"],
                    data["domains"][validation_mask] if room_balanced else None,
                    noise_domains=saved["decision"].get("calibration_noise_domains"))
        if threshold > 1:
            raise ValueError("INT8 validation gate rejects every gesture; improve the model before deploying")
    decision = {**saved["decision"], "positive_threshold": threshold,
                "float_positive_threshold": float_threshold, "threshold_calibration": "INT8 validation"}
    val_float_metrics = classification_metrics(val_actual, val_float, LABELS, float_threshold)
    val_int_metrics = classification_metrics(val_actual, val_int, LABELS, threshold)
    probability_error = float(np.max(np.abs(val_float - val_int)))
    accuracy_drop = val_float_metrics["accuracy"] - val_int_metrics["accuracy"]
    recall_drops = {label: val_float_metrics["per_class"][label]["recall"] - val_int_metrics["per_class"][label]["recall"] for label in LABELS}
    gate_drops = {label: val_float_metrics["gate"]["gesture_recall"][label] - val_int_metrics["gate"]["gesture_recall"][label]
                  for label in LABELS[1:]}
    if probability_error > max_probability_error or accuracy_drop > max_accuracy_drop or \
            max(recall_drops.values()) > max_class_recall_drop or max(gate_drops.values()) > max_class_recall_drop:
        raise ValueError(f"INT8 validation failed: max error {probability_error:.4f}, accuracy drop {accuracy_drop:.4f}, class recall drops {recall_drops}")
    test_mask = data["split"] == "test"
    actual = data["y"][test_mask]
    float_probs, int_probs = predict(test_mask)
    report = {"frontend": contract(), "labels": list(LABELS), "widths": list(model.widths),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "data_sha256": saved["data_sha256"],
              "target_frame_start": model.target_frame_start, "model_bytes": len(binary),
              "model_sha256": hashlib.sha256(binary).hexdigest(), "operators": operators,
              "input_shape": input_info["shape"].tolist(), "input_quantization": list(input_quant),
              "output_quantization": list(output_quant), "representative_samples": len(indices),
              "representative_domains": {str(domain): int((data["domains"][indices] == domain).sum())
                                         for domain in np.unique(data["domains"][indices])},
              "representative_split": "train", "float_parity_max_error": float_error,
              "int8_max_probability_error": probability_error,
              "quantization_acceptance_split": "val", "validation_class_recall_drops": recall_drops,
              "validation_gate_recall_drops": gate_drops,
              "float_validation": val_float_metrics, "int8_validation": val_int_metrics,
              "float_test": classification_metrics(actual, float_probs, LABELS, float_threshold),
              "int8_test": classification_metrics(actual, int_probs, LABELS, threshold),
              "decision": decision, "domains": {},
              "limitations": ["ESP32 arena usage and inference latency still require target-device measurement.",
                              "Quantized model probabilities can change decisions near the validation threshold."]}
    for split_name, mask, actual_labels, float_values, int_values in (
            ("val", validation_mask, val_actual, val_float, val_int),
            ("test", test_mask, actual, float_probs, int_probs)):
        for domain in np.unique(data["domains"][mask]):
            local = data["domains"][mask] == domain
            report["domains"][f"{split_name}:{domain}"] = {
                "float": classification_metrics(actual_labels[local], float_values[local], LABELS, float_threshold),
                "int8": classification_metrics(actual_labels[local], int_values[local], LABELS, threshold)}
    # Nothing deployable is written until every check above succeeds.
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.tflite").write_bytes(binary)
    (out / "model_data.h").write_text(header_text(binary, mean, std, input_quant, output_quant, threshold,
                                                target_frame_start=model.target_frame_start), encoding="utf-8")
    (out / "export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if "source_provenance_json" in data:
        (out / "sources.json").write_text(str(data["source_provenance_json"]) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/hybrid/run/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/hybrid/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("data/hybrid/export"))
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()
    try:
        report = export(args.checkpoint, args.data, args.out, args.samples)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
