"""The audio/frontend contract shared by preparation, training and deployment.

Magnitude (not power) spectra, natural log, no centering/padding and a periodic
Hann window are deliberate. Changing any of these requires retraining a model.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 8_000
PRE_TRIGGER_SAMPLES = 6_400
POST_TRIGGER_SAMPLES = WINDOW_SAMPLES - PRE_TRIGGER_SAMPLES
FRAME_SAMPLES = 400
HOP_SAMPLES = 160
FFT_SAMPLES = 512
FEATURE_SHAPE = (64, 48)
LABELS = ("noise", "clap", "finger_snap")
FRONTEND_VERSION = "logmag_htk64_16k_500ms_v1"


def contract() -> dict:
    """Serializable specification; checkpoints must carry this with their labels."""
    return {
        "version": FRONTEND_VERSION,
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "pre_trigger_samples": PRE_TRIGGER_SAMPLES,
        "post_trigger_samples": POST_TRIGGER_SAMPLES,
        "frame_samples": FRAME_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "fft_samples": FFT_SAMPLES,
        "feature_shape": list(FEATURE_SHAPE),
        "mel_low_hz": 125.0,
        "mel_high_hz": 7500.0,
        "log_offset": 0.001,
        "spectrum": "magnitude",
        "window": "hann_periodic",
        "layout": "mel_time",
    }


def event_window(audio, trigger_sample: int) -> np.ndarray:
    """400 ms history plus 100 ms after a trigger, zero-padded at file edges."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("audio must be a finite mono vector")
    if not 0 <= trigger_sample < len(data):
        raise ValueError("trigger_sample outside audio")
    start = int(trigger_sample) - PRE_TRIGGER_SAMPLES
    left, right = max(0, start), min(len(data), start + WINDOW_SAMPLES)
    result = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    result[left - start : right - start] = data[left:right]
    return result


def _mel_filter() -> np.ndarray:
    # HTK mel, 125–7500 Hz, 64 bands, FFT 512. Matches SafeHear hearsafe.audio._mel_filter.
    low, high = 2595 * np.log10(1 + np.array([125.0, 7500.0]) / 700)
    points = 700 * (10 ** (np.linspace(low, high, 66) / 2595) - 1)
    frequencies = np.fft.rfftfreq(FFT_SAMPLES, 1 / SAMPLE_RATE)
    lower = (frequencies[:, None] - points[:-2]) / (points[1:-1] - points[:-2])
    upper = (points[2:] - frequencies[:, None]) / (points[2:] - points[1:-1])
    return np.maximum(0, np.minimum(lower, upper)).astype(np.float32)


_MEL = _mel_filter()
_HANN = hann(FRAME_SAMPLES, sym=False).astype(np.float32)


def log_mel(audio) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.shape != (WINDOW_SAMPLES,):
        raise ValueError(f"expected {(WINDOW_SAMPLES,)} float samples, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("audio contains NaN or inf")
    frames = np.lib.stride_tricks.sliding_window_view(data, FRAME_SAMPLES)[::HOP_SAMPLES]
    spectrum = np.abs(np.fft.rfft(frames * _HANN, n=FFT_SAMPLES, axis=1)).astype(np.float32)
    feat = np.log(spectrum @ _MEL + 0.001).T.astype(np.float32)
    if feat.shape != FEATURE_SHAPE:
        raise ValueError(f"mel shape {feat.shape}, expected {FEATURE_SHAPE}")
    return feat


def _demo() -> None:
    impulse = np.zeros(WINDOW_SAMPLES, np.float32)
    impulse[1000:1080] = 0.8
    feat = log_mel(impulse)
    assert feat.shape == FEATURE_SHAPE and np.isfinite(feat).all()
    assert float(feat.std()) > 0
    assert log_mel(np.zeros(WINDOW_SAMPLES, np.float32)).shape == FEATURE_SHAPE
    print("frontend ok", feat.shape)


if __name__ == "__main__":
    _demo()
