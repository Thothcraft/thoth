"""Lightweight signal features for the live view.

Computes compact, JSON-serializable features from the streaming radar and CSI
samples so the live view can show *signal quality and motion*, not just model
predictions:

  radar  -> per-frame SNR (dB) and a range-FFT mean-power trace.
  csi    -> 52-subcarrier amplitude matrix, its rolling variance (motion
            indicator), and a short-time Fourier transform (STFT) magnitude
            spectrogram of the mean-amplitude series. Low-frequency energy
            (~0.1-5 Hz) corresponds to breathing / walking; high-frequency
            energy is dominated by noise.

Everything is numpy + scipy only (torch-free) and reuses the E2 CSI amplitude
parser from ``model_e2`` so the live view sees the same subcarrier selection
the occupancy models were trained on.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

try:
    from .model_e2 import _e2_parse_csi_amplitude, _e2_read_uint12, E2_SUBCARRIERS
except ImportError:  # direct/script import
    from model_e2 import _e2_parse_csi_amplitude, _e2_read_uint12, E2_SUBCARRIERS

# Keep payloads small: cap the spectrogram dimensions sent to the UI.
_STFT_MAX_FREQ_BINS = 48
_STFT_MAX_TIME_BINS = 60
_ROLLING_VAR_WINDOW = 20


# ── CSI ─────────────────────────────────────────────────────────────────
def csi_amplitude_matrix(lines: Sequence[str]) -> np.ndarray:
    """Parse CSI_DATA lines -> (N, 52) amplitude matrix (bad lines dropped)."""
    rows = []
    for line in lines:
        amp = _e2_parse_csi_amplitude(str(line))
        if amp is not None:
            rows.append(amp)
    if not rows:
        return np.zeros((0, E2_SUBCARRIERS), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def rolling_variance(arr: np.ndarray, window: int = _ROLLING_VAR_WINDOW) -> np.ndarray:
    """O(N) rolling variance per column (whispy-style cumulative sums)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n = arr.shape[0]
    if n == 0:
        return arr
    window = max(1, int(window))
    c1 = np.concatenate(([np.zeros(arr.shape[1])], np.cumsum(arr, axis=0)))
    c2 = np.concatenate(([np.zeros(arr.shape[1])], np.cumsum(arr * arr, axis=0)))
    out = np.zeros_like(arr)
    for i in range(n):
        lo = max(0, i - window + 1)
        cnt = i - lo + 1
        s1 = c1[i + 1] - c1[lo]
        s2 = c2[i + 1] - c2[lo]
        mean = s1 / cnt
        var = s2 / cnt - mean * mean
        out[i] = np.maximum(var, 0.0)
    return out


def stft_magnitude(series: np.ndarray, nperseg: int = 64) -> np.ndarray:
    """Magnitude spectrogram of a 1-D series -> (freq, time) float array.

    Uses scipy.signal.stft when available; falls back to a manual
    Hann-windowed FFT so the feature still works without scipy.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    if series.size < 4:
        return np.zeros((0, 0), dtype=np.float32)
    series = series - series.mean()
    try:
        from scipy import signal as ssignal
        nperseg = int(min(nperseg, series.size))
        _, _, z = ssignal.stft(series, nperseg=nperseg, noverlap=nperseg // 2)
        mag = np.abs(z)
    except Exception:
        nperseg = int(min(nperseg, series.size))
        hop = max(1, nperseg // 2)
        win = np.hanning(nperseg)
        cols = []
        for start in range(0, series.size - nperseg + 1, hop):
            seg = series[start:start + nperseg] * win
            cols.append(np.abs(np.fft.rfft(seg)))
        mag = np.stack(cols, axis=1) if cols else np.zeros((nperseg // 2 + 1, 0))
    return mag.astype(np.float32)


def _downsample2d(mag: np.ndarray, max_f: int, max_t: int) -> np.ndarray:
    """Block-average a (F,T) spectrogram to fit the UI payload budget."""
    if mag.size == 0:
        return mag
    f, t = mag.shape
    ff = min(f, max_f)
    tt = min(t, max_t)
    # Trim to an integer multiple then block-average.
    f_trim = (f // ff) * ff if ff else f
    t_trim = (t // tt) * tt if tt else t
    mag = mag[:f_trim, :t_trim]
    if f_trim == 0 or t_trim == 0:
        return mag[:ff, :tt]
    mag = mag.reshape(ff, f_trim // ff, tt, t_trim // tt).mean(axis=(1, 3))
    return mag.astype(np.float32)


# ── radar ───────────────────────────────────────────────────────────────
def _radar_payload(frame: bytes) -> np.ndarray:
    """Strip the 12-byte wire header and decode uint12 -> float32 samples."""
    payload = bytes(frame[12:]) if len(frame) >= 12 else bytes(frame)
    if len(payload) % 3:
        payload = payload[: len(payload) - (len(payload) % 3)]
    if not payload:
        return np.zeros(0, dtype=np.float32)
    return _e2_read_uint12(payload)


def radar_snr_db(frames: Sequence[bytes], max_frames: int = 64) -> list[float]:
    """Per-frame SNR estimate (dB): peak range-bin power vs median noise floor.

    A Hann-windowed range FFT is taken per frame; the peak bin is treated as
    the reflection and the median bin power as the noise floor.
    """
    out: list[float] = []
    for frame in list(frames)[-max_frames:]:
        adc = _radar_payload(frame)
        if adc.size < 16:
            continue
        adc = adc - adc.mean()
        spectrum = np.abs(np.fft.rfft(adc * np.hanning(adc.size)))
        if spectrum.size < 2:
            continue
        peak = float(spectrum.max())
        noise = float(np.median(spectrum)) or 1e-9
        out.append(round(20.0 * math.log10((peak + 1e-9) / noise), 2))
    return out


# ── top-level feature bundle ────────────────────────────────────────────
def compute_live_features(
    radar_frames: Sequence[bytes] | None = None,
    csi_lines: Sequence[str] | None = None,
    csi_sample_rate_hz: float = 100.0,
) -> dict[str, Any]:
    """Build the compact ``features`` dict attached to a live-chunk payload.

    Returns only keys that could be computed — an empty dict when neither
    modality produced data, so callers can omit the field entirely.
    """
    features: dict[str, Any] = {}

    snr = radar_snr_db(radar_frames or [])
    if snr:
        features["radar_snr_db"] = snr
        features["radar_snr_mean_db"] = round(float(np.mean(snr)), 2)

    amp = csi_amplitude_matrix(csi_lines or [])
    if amp.shape[0] >= 2:
        mean_amp = amp.mean(axis=1)  # (N,) motion proxy
        var = rolling_variance(amp, _ROLLING_VAR_WINDOW).mean(axis=1)
        features["csi_amplitude_mean"] = [round(float(v), 3) for v in mean_amp[-_STFT_MAX_TIME_BINS:]]
        features["csi_rolling_var"] = [round(float(v), 4) for v in var[-_STFT_MAX_TIME_BINS:]]
        spec = _downsample2d(stft_magnitude(mean_amp), _STFT_MAX_FREQ_BINS, _STFT_MAX_TIME_BINS)
        if spec.size:
            features["csi_stft"] = {
                "shape": [int(spec.shape[0]), int(spec.shape[1])],
                "sample_rate_hz": csi_sample_rate_hz,
                "magnitude": np.round(spec, 3).tolist(),
            }
    return features
