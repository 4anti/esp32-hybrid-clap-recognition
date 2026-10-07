"""Compact ResearchCNN with explicit widths for constrained ESP32 deployment."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

TARGET_FRAME_START = 32


class ResearchCNN(nn.Module):
    """Log-mel in, training-set mean/std baked in as buffers."""

    def __init__(self, mean, std, classes: int = 3, widths=(8, 16, 32), target_frame_start: int = TARGET_FRAME_START):
        super().__init__()
        if classes < 2:
            raise ValueError("a detector needs noise and at least one gesture class")
        self.widths = tuple(int(width) for width in widths)
        if len(self.widths) != 3 or any(width <= 0 for width in self.widths):
            raise ValueError("widths must contain three positive channel counts")
        self.target_frame_start = int(target_frame_start)
        if not 0 <= self.target_frame_start <= 40 or (48 - self.target_frame_start) % 8:
            raise ValueError("target crop must leave a positive multiple of eight time frames")
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, 64, 1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, 64, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("normalization must be finite with positive std")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))
        blocks = []
        channels = 1
        for width in self.widths:
            blocks.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            channels = width
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(2 * self.widths[-1], classes)

    def forward(self, audio_features):
        # Classification is about THIS trigger. A first clap 150-400 ms earlier
        # cannot validate a second noise event merely by being in the history.
        normalized = (audio_features[:, :, :, self.target_frame_start:] - self.mean) / self.std
        x = self.features(normalized)
        pooled = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        return self.classifier(self.dropout(pooled))


def widen_model(original: ResearchCNN, widths) -> ResearchCNN:
    """Add trainable channels while preserving the original inference path.

    Legacy outputs have zero weights from new inputs. New outputs keep their
    random initialization, but enter the classifier with zero initial weight.
    Average and maximum pooling occupy separate channel blocks in that layer.
    """
    widths = tuple(int(value) for value in widths)
    if len(widths) != 3 or any(new < old for new, old in zip(widths, original.widths)):
        raise ValueError("widening cannot remove existing channels")
    wider = ResearchCNN(original.mean, original.std, original.classifier.out_features,
                        widths, original.target_frame_start)
    with torch.no_grad():
        for index in range(3):
            old_conv, old_bn = original.features[index * 4:index * 4 + 2]
            new_conv, new_bn = wider.features[index * 4:index * 4 + 2]
            outputs, inputs = old_conv.weight.shape[:2]
            new_conv.weight[:outputs].zero_()
            new_conv.weight[:outputs, :inputs].copy_(old_conv.weight)
            for name in ("weight", "bias", "running_mean", "running_var"):
                getattr(new_bn, name)[:outputs].copy_(getattr(old_bn, name))
            new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
        old_last, new_last = original.widths[-1], wider.widths[-1]
        wider.classifier.weight.zero_()
        wider.classifier.weight[:, :old_last].copy_(original.classifier.weight[:, :old_last])
        wider.classifier.weight[:, new_last:new_last + old_last].copy_(original.classifier.weight[:, old_last:])
        wider.classifier.bias.copy_(original.classifier.bias)
    return wider


def _demo() -> None:
    model = ResearchCNN(np.zeros(64, np.float32), np.ones(64, np.float32), 3)
    logits = model(torch.zeros(2, 1, 64, 48))
    assert tuple(logits.shape) == (2, 3)
    print("cnn ok", tuple(logits.shape), "params", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    _demo()
