"""Train and evaluate a source-disjoint clap/snap confirmation model.

The default prepared dataset requires noise, clap AND finger_snap examples in
every split. ``hybrid.prepare --classes noise clap`` creates an explicitly
experimental binary baseline; it cannot be deployed as a snap recognizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from hybrid.frontend import FEATURE_SHAPE, LABELS, contract

DATA = Path("data/hybrid/features.npz")
OUT = Path("data/hybrid/run")


def _counts(y: np.ndarray, labels=LABELS) -> dict[str, int]:
    return {name: int((y == i).sum()) for i, name in enumerate(labels)}


def _refuse(counts: dict[str, int]) -> None:
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Missing classes {missing}. Record independent raw Lab sessions, rerun python -m hybrid.prepare.")


def load_dataset(path: Path) -> dict:
    """Reject stale frontend contracts and source leakage before any training."""
    with np.load(path, allow_pickle=False) as blob:
        required = {"schema_version", "x", "y", "split", "labels", "sources", "groups", "domains", "frontend_json"}
        if not required.issubset(blob.files) or int(blob["schema_version"]) != 2:
            raise ValueError("Old/unverifiable dataset. Rerun python -m hybrid.prepare before training.")
        data = {name: np.asarray(blob[name]) for name in blob.files}
    labels = tuple(str(name) for name in data["labels"])
    if len(labels) < 2 or labels[0] != "noise" or len(set(labels)) != len(labels) or any(name not in LABELS for name in labels):
        raise ValueError("invalid class mapping")
    if json.loads(str(data["frontend_json"])) != contract():
        raise ValueError("Dataset frontend differs from this code; regenerate features and retrain.")
    x, y, split = data["x"], data["y"], data["split"]
    if x.ndim != 3 or tuple(x.shape[1:]) != FEATURE_SHAPE or not np.isfinite(x).all() or not len(x):
        raise ValueError("invalid feature shape or nonfinite features")
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= len(labels)):
        raise ValueError("invalid labels")
    for key in ("split", "sources", "groups", "domains"):
        if data[key].shape != (len(x),):
            raise ValueError(f"invalid {key} array")
    if not set(split).issubset({"train", "val", "test"}):
        raise ValueError("unknown split")
    for key in ("sources", "groups"):
        for value in np.unique(data[key]):
            if len(set(split[data[key] == value])) != 1:
                raise ValueError(f"{key} leakage between splits: {value}")
    for name in ("train", "val", "test"):
        try:
            _refuse(_counts(y[split == name], labels))
        except ValueError as exc:
            raise ValueError(f"{name} split: {exc}") from exc
    data.update(x=x.astype(np.float32), y=y.astype(np.int64), labels=labels)
    return data


def sampling_weights(y, sources, domains=None, room_weight=None):
    """Equal class/domain/recording mass, regardless of each take's length."""
    from collections import Counter

    domains = np.asarray(domains) if domains is not None else np.full(len(y), "all")
    sources = np.asarray(sources)
    if sources.shape != y.shape or domains.shape != y.shape:
        raise ValueError("sampling requires one source and domain per example")
    keys = list(zip(y.tolist(), domains.tolist(), sources.tolist()))
    events = Counter(keys)
    recordings = Counter((label, domain) for label, domain, source in events)
    domain_count = Counter(label for label, domain in recordings)
    def mass(label, domain):
        if room_weight is None or domain_count[label] == 1 or (label, "lab") not in recordings:
            return 1 / domain_count[label]
        return room_weight if domain == "lab" else (1 - room_weight) / (domain_count[label] - 1)
    return np.array([mass(label, domain) / (recordings[label, domain] * events[label, domain, source])
                     for label, domain, source in keys], dtype=np.float64)


def _loader(x, y, batch_size: int, shuffle: bool, seed: int, sources=None, balance: bool = False, domains=None, room_weight=None):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balance:
        weights = sampling_weights(y, sources, domains, room_weight)
        counts = np.bincount(y)
        # Broad negatives need exposure across epochs, while minority gestures
        # need frequent gradients. Cap repeated-positive oversampling per epoch.
        samples = min(len(y), 3 * int(counts[1:].max()))
        sampler = WeightedRandomSampler(weights, num_samples=samples, replacement=True, generator=generator)
    return DataLoader(TensorDataset(torch.from_numpy(x).unsqueeze(1), torch.from_numpy(y)),
                      batch_size=batch_size, num_workers=0, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, generator=generator)


def augment_features(features, band_mean, eq_db_max: float = 0, background=None,
                     gain_db=(-18, 6), jitter_frames=2, frequency_mask_bands=4):
    """Train-only gain variation, small timing jitter and narrow-band masking.

    Gain is applied to magnitude before restoring the frontend's log offset,
    preserving silence instead of merely shifting every log-mel value.
    """
    import torch

    log_gain = (torch.rand(len(features), 1, 1, 1) * (gain_db[1] - gain_db[0]) + gain_db[0]) * (np.log(10) / 20)
    if eq_db_max:
        # A mild low/high shelf at mel-band centers approximates microphone and
        # room coloration. Apply the same policy to every class, including bags.
        mel_low, mel_high = 2595 * np.log10(1 + np.array([125., 7500.]) / 700)
        hz = 700 * (10 ** (np.linspace(mel_low, mel_high, 66)[1:-1] / 2595) - 1)
        low_shelf = torch.from_numpy((1 / (1 + (hz / 800) ** 2)).astype(np.float32))[None, None, :, None]
        bass = (torch.rand(len(features), 1, 1, 1) * 2 - 1) * eq_db_max
        treble = (torch.rand(len(features), 1, 1, 1) * 2 - 1) * eq_db_max
        log_gain = log_gain + (bass * low_shelf + treble * (1 - low_shelf)) * (np.log(10) / 20)
    magnitude = (features.exp() - 0.001).clamp_min(0)
    magnitude = magnitude * log_gain.exp()
    if background is not None and len(background):
        # Feature-space mixing uses expected uncorrelated spectral energy;
        # it approximates quiet background addition, not exact waveform mixing.
        noise = (background[torch.randint(len(background), (len(features),))].exp() - 0.001).clamp_min(0)
        signal_energy = magnitude[..., 32:].square().mean(dim=(1, 2, 3), keepdim=True)
        noise_energy = noise[..., 32:].square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        snr = 20 + 15 * torch.rand(len(features), 1, 1, 1)
        noise_gain_squared = signal_energy / (noise_energy * (10 ** (snr / 10)))
        selected = (torch.rand(len(features), 1, 1, 1) < 0.25)
        magnitude = (magnitude.square() + noise.square() * noise_gain_squared * selected).sqrt()
    result = (magnitude + 0.001).log()
    for index in range(len(result)):
        shift = int(torch.randint(-jitter_frames, jitter_frames + 1, ()).item())
        if shift:
            result[index] = torch.roll(result[index], shift, dims=-1)
            if shift > 0:
                result[index, :, :, :shift] = float(np.log(0.001))
            else:
                result[index, :, :, shift:] = float(np.log(0.001))
        if frequency_mask_bands and torch.rand(()) < 0.25:
            width = int(torch.randint(1, frequency_mask_bands + 1, ()).item())
            start = int(torch.randint(0, FEATURE_SHAPE[0] - width + 1, ()).item())
            result[index, :, start : start + width, :] = band_mean[None, start : start + width, None]
    return result


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, labels,
                           threshold: float | None = None) -> dict:
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    support = confusion.sum(axis=1)
    recalled = np.divide(confusion.diagonal(), support, out=np.zeros(len(labels)), where=support > 0)
    precision = np.divide(confusion.diagonal(), confusion.sum(axis=0), out=np.zeros(len(labels)),
                          where=confusion.sum(axis=0) > 0)
    result = {
        "samples": len(y), "accuracy": float((predicted == y).mean()) if len(y) else None,
        "balanced_accuracy": float(recalled[support > 0].mean()) if support.any() else None,
        "confusion_matrix": confusion.tolist(), "confusion_order": list(labels),
        "per_class": {name: {"support": int(support[i]), "recall": float(recalled[i]) if support[i] else None,
                             "precision": float(precision[i])} for i, name in enumerate(labels)},
    }
    if threshold is not None:
        accepted = (predicted != 0) & ((1 - probabilities[:, 0]) >= threshold)
        noise = y == 0
        result["gate"] = {
            "threshold": threshold,
            "noise_false_accepts": int((accepted & noise).sum()), "noise_trials": int(noise.sum()),
            "noise_false_accept_rate": float(accepted[noise].mean()) if noise.any() else None,
            "gesture_recall": {name: float(accepted[y == i].mean()) if (y == i).any() else None
                               for i, name in enumerate(labels) if i},
        }
    return result


def select_threshold(y: np.ndarray, probabilities: np.ndarray, max_noise_false_accept: float,
                     domains=None, minimum: float = 0.85, noise_domains=None) -> float:
    """Validation only: enforce the noise budget separately in each domain.

    The minimum matches the deployed ESP32 gate. A few room/bag errors must
    never disappear inside thousands of easier public noise examples.
    """
    if not 0 <= max_noise_false_accept <= 1 or not (y == 0).any():
        raise ValueError("threshold calibration needs noise and a valid false-accept target")
    score = 1 - probabilities[:, 0]
    possible = probabilities.argmax(axis=1) != 0
    # Advance by a float32 ULP: the deployed threshold itself is float32.
    if not 0 <= minimum <= 1:
        raise ValueError("minimum threshold must be [0,1]")
    domains = np.asarray(domains) if domains is not None else np.full(len(y), "all")
    if domains.shape != y.shape:
        raise ValueError("calibration requires one domain per example")
    active_domains = np.unique(domains[y == 0])
    if noise_domains is not None:
        if not set(noise_domains).issubset(active_domains) or not noise_domains:
            raise ValueError("selected noise domains need validation noise examples")
        active_domains = noise_domains
    noise_masks = [(y == 0) & (domains == domain) for domain in active_domains]
    thresholds = np.unique(np.concatenate(([minimum, 1.000001], np.nextafter(score.astype(np.float32), np.float32(np.inf)))))
    best_threshold, best_recall = 1.000001, -1.0
    for threshold in thresholds:
        if threshold < minimum:
            continue
        accepted = possible & (score >= threshold)
        rate = max(float(accepted[mask].mean()) for mask in noise_masks)
        recall = float(accepted[y != 0].mean()) if (y != 0).any() else 0.0
        if rate <= max_noise_false_accept and recall > best_recall:
            best_threshold, best_recall = float(threshold), recall
    return best_threshold


def domain_source_balanced_gate(y, probabilities, sources, domains, labels, threshold):
    by_domain = {str(domain): source_balanced_gate(y[domains == domain], probabilities[domains == domain],
                 sources[domains == domain], labels, threshold) for domain in np.unique(domains)}
    def average(name):
        values = [entry["noise_false_accept_rate"] if name == "noise" else entry["gesture_recall"][name]
                  for entry in by_domain.values()]
        values = [value for value in values if value is not None]
        return float(np.mean(values)) if values else None
    return {"noise_false_accept_rate": average("noise"),
            "gesture_recall": {name: average(name) for name in labels[1:]}, "by_domain": by_domain}


def warm_start_model(checkpoint, original_data, data, widths, allow_widen=False):
    """Retain learned normalization, and refuse moved/leaked original sources."""
    import torch
    from hybrid.cnn import ResearchCNN, TARGET_FRAME_START, widen_model

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("data_sha256") != hashlib.sha256(original_data.read_bytes()).hexdigest():
        raise ValueError("warm-start dataset does not match checkpoint provenance")
    original = load_dataset(original_data)
    old_widths = tuple(saved.get("widths", ()))
    changed_widths = old_widths != tuple(widths)
    if tuple(saved.get("labels", ())) != data["labels"] or saved.get("frontend") != contract() or \
            (changed_widths and not allow_widen) or saved.get("target_frame_start") != TARGET_FRAME_START:
        raise ValueError("warm-start model contract/architecture differs")
    for key in ("sources", "groups"):
        old_split = {str(value): str(split) for value, split in zip(original[key], original["split"])}
        for value, split in zip(data[key], data["split"]):
            if str(value) in old_split and old_split[str(value)] != str(split):
                raise ValueError(f"warm-start {key} moved between splits: {value}")
    model = ResearchCNN(saved["mean"], saved["std"], len(data["labels"]), widths=old_widths)
    model.load_state_dict(saved["state_dict"])
    if changed_widths:
        model = widen_model(model, widths)
    return model, {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "original_data_sha256": saved["data_sha256"],
                   "original_widths": list(old_widths), "widened": changed_widths,
                   "normalization": "retained from the original training split"}


def source_balanced_gate(y, probabilities, sources, labels, threshold: float) -> dict:
    """Give each independent recording equal weight within a class."""
    accepted = (probabilities.argmax(1) != 0) & ((1 - probabilities[:, 0]) >= threshold)
    rates = {label: [] for label in labels}
    for source in np.unique(sources):
        source_mask = sources == source
        for index, label in enumerate(labels):
            class_mask = source_mask & (y == index)
            if class_mask.any():
                rates[label].append(float(accepted[class_mask].mean()))
    return {
        "noise_false_accept_rate": float(np.mean(rates["noise"])) if rates["noise"] else None,
        "gesture_recall": {label: float(np.mean(values)) if values else None
                           for label, values in rates.items() if label != "noise"},
        "independent_sources": {label: len(values) for label, values in rates.items()},
    }


def _evaluate(model, loader, class_weights):
    import torch
    import torch.nn.functional as functional

    losses, weight_sum, predictions, targets = 0.0, 0.0, [], []
    model.eval()
    with torch.inference_mode():
        for features, y in loader:
            logits = model(features)
            # A global weighted mean avoids batch-composition-dependent loss.
            losses += float(functional.cross_entropy(logits, y, weight=class_weights, reduction="sum"))
            weight_sum += float(class_weights[y].sum())
            predictions.append(logits.softmax(1).numpy())
            targets.append(y.numpy())
    return losses / weight_sum, np.concatenate(targets), np.concatenate(predictions)


def train(epochs: int = 50, patience: int = 8, batch_size: int = 32, seed: int = 42,
          data_path: Path = DATA, out: Path = OUT, max_noise_false_accept: float = 0.01,
          threads: int = 2, augment: bool = True, widths=(8, 16, 32), balance_domains: bool = False,
          init_checkpoint: Path | None = None, init_data: Path | None = None, learning_rate: float = 0.001,
          eq_db: float = 0, mix_background: bool = False, room_weight: float | None = None,
          calibration_noise_domains=None, min_room_fit: float = 0, allow_widen: bool = False,
          gain_db=(-18, 6), jitter_frames: int = 2, frequency_mask_bands: int = 4) -> dict:
    import torch
    import torch.nn.functional as functional
    from hybrid.cnn import ResearchCNN, TARGET_FRAME_START

    if min(epochs, patience, batch_size, threads) <= 0 or not 0 <= max_noise_false_accept <= 1:
        raise ValueError("epochs/patience/batch-size/threads must be positive; false accept target must be [0,1]")
    if (init_checkpoint is None) != (init_data is None):
        raise ValueError("warm start requires both --init-checkpoint and --init-data")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not np.isfinite(eq_db) or not 0 <= eq_db <= 6:
        raise ValueError("EQ range must be between 0 and 6 dB to preserve transient labels")
    if room_weight is not None and (not balance_domains or not 0 < room_weight < 1):
        raise ValueError("room weight requires domain balancing and must be between 0 and 1")
    if not 0 <= min_room_fit <= 1 or (calibration_noise_domains is not None and not balance_domains):
        raise ValueError("invalid room fit requirement or calibration without domain balancing")
    if len(gain_db) != 2 or not np.isfinite(gain_db).all() or not -18 <= gain_db[0] <= gain_db[1] <= 6 or \
            not 0 <= jitter_frames <= 2 or not 0 <= frequency_mask_bands <= 4:
        raise ValueError("augmentation exceeds supported gain/timing/masking bounds")
    if allow_widen and init_checkpoint is None:
        raise ValueError("widening requires a verified warm-start checkpoint")
    data = load_dataset(data_path)
    x, y, split, labels = (data[key] for key in ("x", "y", "split", "labels"))
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_mask = split == "train"
    train_x = x[train_mask]
    target_features = train_x[:, :, TARGET_FRAME_START:]
    mean = target_features.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(target_features.std(axis=(0, 2), dtype=np.float64).astype(np.float32), 1e-6)
    model = ResearchCNN(mean, std, len(labels), widths=widths)
    initialization = None
    if init_checkpoint is not None:
        model, initialization = warm_start_model(init_checkpoint, init_data, data, widths, allow_widen)
        mean = model.mean.detach().numpy().reshape(64).copy()
        std = model.std.detach().numpy().reshape(64).copy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    band_mean = torch.from_numpy(mean)
    background = None
    if mix_background and augment:
        if "label_quality" not in data:
            raise ValueError("background mixing requires prepared background-window labels")
        candidates = np.flatnonzero(train_mask & (y == 0) & (data["label_quality"] == "background"))
        if not len(candidates):
            raise ValueError("no training-only background windows for augmentation")
        generator = np.random.default_rng(seed)
        chosen = []
        for domain in np.unique(data["domains"][candidates]):
            pool = candidates[data["domains"][candidates] == domain].copy()
            generator.shuffle(pool)
            chosen.extend(pool[:128])
        background = torch.from_numpy(x[chosen]).unsqueeze(1)
    counts = np.bincount(y[train_mask], minlength=len(labels)).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (len(labels) * counts))
    train_loader = _loader(train_x, y[train_mask], batch_size, True, seed,
                           sources=data["sources"][train_mask], balance=True,
                           domains=data["domains"][train_mask] if balance_domains else None, room_weight=room_weight)
    val_loader = _loader(x[split == "val"], y[split == "val"], batch_size, False, seed)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "best.pt"
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    best_loss, best_score, stale, history = float("inf"), -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, denominator = 0.0, 0.0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if augment:
                features = augment_features(features, band_mean, eq_db, background,
                                            gain_db, jitter_frames, frequency_mask_bands)
            logits = model(features)
            # The source/class-balanced sampler already corrects imbalance;
            # applying inverse-frequency loss again would double compensate.
            loss_sum = functional.cross_entropy(logits, targets, reduction="sum")
            weight_sum = torch.tensor(float(len(targets)))
            (loss_sum / weight_sum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss_sum.detach())
            denominator += float(weight_sum)
        val_loss, val_y, val_probs = _evaluate(model, val_loader, class_weights)
        scheduler.step(val_loss)
        metrics = classification_metrics(val_y, val_probs, labels)
        epoch_threshold = select_threshold(val_y, val_probs, max_noise_false_accept,
                            data["domains"][split == "val"] if balance_domains else None,
                            noise_domains=calibration_noise_domains)
        gate = classification_metrics(val_y, val_probs, labels, epoch_threshold)["gate"]
        source_gate = source_balanced_gate(val_y, val_probs, data["sources"][split == "val"], labels, epoch_threshold)
        selection_gate = domain_source_balanced_gate(val_y, val_probs, data["sources"][split == "val"],
                            data["domains"][split == "val"], labels, epoch_threshold) if balance_domains else source_gate
        gate_score = float(np.mean([1 - selection_gate["noise_false_accept_rate"], *selection_gate["gesture_recall"].values()]))
        entry = {"epoch": epoch, "training_loss": total / denominator, "validation_loss": val_loss,
                 "validation_accuracy": metrics["accuracy"], "validation_balanced_accuracy": metrics["balanced_accuracy"],
                 "validation_gate_macro_recall": gate_score, "validation_gate": gate,
                 "validation_source_balanced_gate": source_gate, "validation_selection_gate": selection_gate}
        history.append(entry)
        eligible = True
        if min_room_fit:
            room_mask = train_mask & (data["domains"] == "lab")
            if not all((y[room_mask] == label).any() for label in range(len(labels))):
                raise ValueError("room fit requirement needs every class in room training data")
            room_loader = _loader(x[room_mask], y[room_mask], batch_size, False, seed)
            _, room_y, room_probs = _evaluate(model, room_loader, class_weights)
            room_gate = classification_metrics(room_y, room_probs, labels, epoch_threshold)["gate"]
            entry["room_training_fit_diagnostic"] = room_gate
            eligible = min(room_gate["gesture_recall"].values()) >= min_room_fit and \
                       room_gate["noise_false_accept_rate"] <= max_noise_false_accept
            entry["eligible_fit"] = eligible
        print(f"epoch {epoch}/{epochs} loss {entry['training_loss']:.4f} val {val_loss:.4f} "
              f"balanced_acc {metrics['balanced_accuracy']:.4f} gate_recall {gate_score:.4f}", file=sys.stderr, flush=True)
        improved = gate_score > best_score + 1e-5 or (abs(gate_score - best_score) <= 1e-5 and val_loss < best_loss - 1e-5)
        if improved and eligible:
            best_loss, best_score, stale = val_loss, gate_score, 0
            torch.save({"state_dict": model.state_dict(), "labels": list(labels), "frontend": contract(),
                        "mean": mean.tolist(), "std": std.tolist(), "epoch": epoch,
                        "data_sha256": data_sha256, "model_architecture": "research_cnn",
                        "widths": list(model.widths), "initialization": initialization,
                        "selection": "validation_domain_source_balanced_gate_macro_recall" if balance_domains else
                                     "validation_source_balanced_gate_macro_recall",
                        "target_frame_start": model.target_frame_start,
                        "selection_score": best_score}, checkpoint)
        elif best_score >= 0:
            stale += 1
        if stale >= patience:
            break
    if best_score < 0:
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        raise ValueError("No checkpoint met the room fit requirement; no model is ready for activation")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    _, val_y, val_probs = _evaluate(model, val_loader, class_weights)
    threshold = select_threshold(val_y, val_probs, max_noise_false_accept,
                                 data["domains"][split == "val"] if balance_domains else None,
                                 noise_domains=calibration_noise_domains)
    saved["decision"] = {"positive_threshold": threshold, "max_validation_noise_false_accept": max_noise_false_accept,
                         "accept_rule": "argmax != noise and 1 - p_noise >= positive_threshold",
                         "minimum_deployed_threshold": 0.85,
                         "balance_domains": balance_domains,
                         "calibration_noise_domains": calibration_noise_domains,
                         "noise_budget_scope": "selected validation domains" if calibration_noise_domains else
                                               "each validation domain" if balance_domains else "pooled validation"}
    torch.save(saved, checkpoint)
    report = {"labels": list(labels), "frontend": contract(), "seed": seed,
              "data_sha256": data_sha256, "best_epoch": saved["epoch"],
              "parameters": sum(p.numel() for p in model.parameters()),
              "widths": list(model.widths), "target_frame_start": model.target_frame_start,
              "selection": saved["selection"],
              "training": {"augment": augment, "learning_rate": learning_rate,
                           "room_domain_probability": room_weight, "minimum_room_training_fit": min_room_fit,
                           "gain_db": list(gain_db) if augment else None,
                           "timing_jitter_frames": jitter_frames if augment else 0,
                           "max_frequency_mask_bands": frequency_mask_bands if augment else 0,
                           "eq_shelf_db": [-eq_db, eq_db] if augment and eq_db else None,
                           "background_mix": {"mode": "expected spectral-energy mixing", "snr_db": [20, 35],
                                              "probability": 0.25, "split": "train",
                                              "pool_size": len(background)} if background is not None else None,
                           "sampling": "equal class, domain and independent-source probability" if balance_domains else
                                       "equal class and independent-source probability",
                           "samples_per_epoch": len(train_loader.sampler)},
              "initialization": initialization,
              "decision": saved["decision"], "splits": {}, "domains": {}, "source_balanced_gate": {}}
    for name in ("train", "val", "test"):
        mask = split == name
        loader = _loader(x[mask], y[mask], batch_size, False, seed)
        loss, actual, probs = _evaluate(model, loader, class_weights)
        report["splits"][name] = {"loss": loss, **classification_metrics(actual, probs, labels, threshold)}
        report["source_balanced_gate"][name] = source_balanced_gate(actual, probs, data["sources"][mask], labels, threshold)
        for domain in np.unique(data["domains"][mask]):
            domain_mask = data["domains"][mask] == domain
            report["domains"][f"{name}:{domain}"] = classification_metrics(actual[domain_mask], probs[domain_mask], labels, threshold)
    report["limitations"] = [
        "Scores are model probabilities, not calibrated guarantees.",
        "Gate threshold uses validation only; test results are held out and must not tune the threshold.",
        "Window-level false acceptance is not false toggles/hour or full double-gesture accuracy.",
        "ESC applause is weakly labelled; deployment needs isolated claps/snaps and hard negatives from the actual microphone.",
        "No INT8/ESP32 performance or RAM claim is established by this training run.",
    ]
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {checkpoint} and {out / 'metrics.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-noise-false-accept", type=float, default=0.01)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--widths", type=int, nargs=3, default=[8, 16, 32])
    parser.add_argument("--balance-domains", action="store_true",
                        help="Give room recordings equal domain weight; enforce noise budgets per validation domain")
    parser.add_argument("--init-checkpoint", type=Path, help="Fine-tune a compatible, provenance-verified checkpoint")
    parser.add_argument("--init-data", type=Path, help="Original dataset used by the initial checkpoint")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eq-db", type=float, default=0, help="Random bass/treble shelf magnitude, at most 6 dB")
    parser.add_argument("--mix-background", action="store_true", help="Add quiet training-only background spectra")
    parser.add_argument("--room-weight", type=float, help="Room fraction per class; remaining probability goes to public domains")
    parser.add_argument("--calibration-noise-domains", nargs="+", help="Restrict noise calibration to named validation domains; report other domains separately")
    parser.add_argument("--min-room-fit", type=float, default=0, help="Require this training gesture recall and noise fit before checkpoint eligibility; not held-out evidence")
    parser.add_argument("--allow-widen", action="store_true", help="Expand a verified checkpoint's channels while preserving its initial inference function")
    parser.add_argument("--gain-db", type=float, nargs=2, default=[-18, 6])
    parser.add_argument("--jitter-frames", type=int, default=2)
    parser.add_argument("--frequency-mask-bands", type=int, default=4)
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Missing {args.data}. Run python -m hybrid.prepare first.")
    try:
        train(args.epochs, args.patience, args.batch_size, args.seed, args.data, args.out,
              args.max_noise_false_accept, args.threads, not args.no_augment, args.widths,
              args.balance_domains, args.init_checkpoint, args.init_data, args.learning_rate, args.eq_db, args.mix_background,
              args.room_weight, args.calibration_noise_domains, args.min_room_fit, args.allow_widen,
              args.gain_db, args.jitter_frames, args.frequency_mask_bands)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
