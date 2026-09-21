"""E2 occupancy preprocessing ("e2_maps" / "e2_grid" representations).

Ported from the radar repo's E2/preprocess.py so the exported occupancy
models receive exactly the tensors they were trained on:
  radar: uint12 payload -> Hann-windowed range FFT -> range-Doppler and
         range-azimuth maps -> log1p -> bilinear resize to 24x24 ->
         non-overlapping 50-frame windows -> per-channel z-score.
  csi:   CSI_DATA lines -> 52-subcarrier amplitude -> per-window linear
         interpolation onto a 128-step grid -> log1p -> per-subcarrier
         z-score; windows with <2 samples stay zeroed.
Normalization statistics come from the archive's embedded meta.json.

This module is intentionally free of torch and registry dependencies so it can
be imported cheaply and reused by live-view feature extraction.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np

E2_WINDOW_FRAMES = 50
E2_MAP_SIZE = 24
E2_CSI_STEPS = 128
E2_SUBCARRIERS = 52
E2_RANGE_BINS = 64
E2_AZ_BINS = 16
E2_FRAME_PAYLOAD_BYTES = 64 * 128 * 3 * 12 // 8  # 36864
_E2_HANN_R = np.hanning(128).astype(np.float32)
_E2_HANN_D = np.hanning(64).astype(np.float32)
_E2_CSI_MASK = np.array(
    [False] * 6 + [True] * 26 + [False] + [True] * 26 + [False] * 5,
    dtype=bool,
)
_E2_CSI_RE = re.compile(r"\[([^\]]+)\]")


def _e2_read_uint12(blob: bytes) -> np.ndarray:
    """Vectorized uint12 decode: packed bytes -> float32 samples."""
    data = np.frombuffer(blob, dtype=np.uint8)
    triplets = data.reshape(-1, 3).astype(np.uint16)
    a, b, c = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    out = np.empty(a.shape[0] * 2, dtype=np.float32)
    out[0::2] = (a << 4) + (b >> 4)
    out[1::2] = ((b % 16) << 8) + c
    return out


def _e2_maps_batch(payloads: Sequence[bytes]) -> np.ndarray:
    """(N) raw 36864-byte payloads -> (N, 2, 24, 24) log1p maps."""
    from scipy import fft as sfft
    from scipy.ndimage import zoom

    n = len(payloads)
    adc = _e2_read_uint12(b"".join(payloads)).reshape(n, 64, 128, 3)
    adc *= _E2_HANN_R[None, None, :, None] * _E2_HANN_D[None, :, None, None]
    R = sfft.fft(adc, axis=2, workers=-1)                            # range
    RD = sfft.fftshift(sfft.fft(R, axis=1, workers=-1), axes=1)      # doppler
    rd = np.abs(RD[:, :, :E2_RANGE_BINS, :]).mean(axis=3)            # (N,64,64)
    RA = sfft.fft(R[:, :, :E2_RANGE_BINS, :], n=E2_AZ_BINS, axis=3, workers=-1)
    ra = np.abs(RA).mean(axis=1).transpose(0, 2, 1)                  # (N,16,64)
    s = E2_MAP_SIZE
    rd = zoom(np.log1p(rd), (1, s / rd.shape[1], s / rd.shape[2]), order=1)
    ra = zoom(np.log1p(ra), (1, s / ra.shape[1], s / ra.shape[2]), order=1)
    return np.stack([rd, ra], axis=1).astype(np.float32)


# Per-frame map memoization. Partial-minute inference re-runs every time a new
# 50-frame window completes, so without a cache the FFT/zoom preprocessing is
# quadratic over the minute. A frame's map depends only on its payload bytes,
# which makes the entries safe to reuse across partial and final runs.
_E2_MAP_CACHE: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
_E2_MAP_CACHE_LIMIT = 4096
_E2_MAP_CACHE_LOCK = threading.Lock()


def _e2_frames_to_maps(payloads: Sequence[bytes]) -> np.ndarray:
    """(N) raw 36864-byte payloads -> (N, 2, 24, 24) log1p maps (cached)."""
    results: list[np.ndarray | None] = [None] * len(payloads)
    missing: list[bytes] = []
    missing_index: list[int] = []
    with _E2_MAP_CACHE_LOCK:
        for index, payload in enumerate(payloads):
            cached = _E2_MAP_CACHE.get(payload)
            if cached is not None:
                _E2_MAP_CACHE.move_to_end(payload)
                results[index] = cached
            else:
                missing.append(payload)
                missing_index.append(index)
    if missing:
        computed = _e2_maps_batch(missing)
        with _E2_MAP_CACHE_LOCK:
            for position, (payload, index) in enumerate(zip(missing, missing_index)):
                maps = computed[position]
                _E2_MAP_CACHE[payload] = maps
                _E2_MAP_CACHE.move_to_end(payload)
                results[index] = maps
            while len(_E2_MAP_CACHE) > _E2_MAP_CACHE_LIMIT:
                _E2_MAP_CACHE.popitem(last=False)
    return np.stack(results, axis=0)


def e2_radar_windows(
    frames: Sequence[bytes],
    times: Sequence[float],
    norm: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Raw wire packets -> normalized (nW, 50, 2, 24, 24) windows.

    Returns (windows, win_t0, win_t1, win_max_gap); windows is None when
    fewer than E2_WINDOW_FRAMES valid frames exist. Malformed packets are
    dropped before windowing so window boundaries stay aligned with
    `times`. win_max_gap is the largest inter-frame timestamp gap inside
    each window — large gaps mark FIFO-overflow holes whose Doppler
    content is corrupted.
    """
    payloads: list[bytes] = []
    kept_times: list[float] = []
    for index, packet in enumerate(frames):
        payload = bytes(packet[12:]) if len(packet) >= 12 else bytes(packet)
        if len(payload) != E2_FRAME_PAYLOAD_BYTES:
            continue
        payloads.append(payload)
        kept_times.append(float(times[index]) if index < len(times) else 0.0)
    n_win = len(payloads) // E2_WINDOW_FRAMES
    if n_win == 0:
        return None, np.empty(0), np.empty(0), np.empty(0)
    maps = _e2_frames_to_maps(payloads[: n_win * E2_WINDOW_FRAMES])
    windows = maps.reshape(n_win, E2_WINDOW_FRAMES, 2, E2_MAP_SIZE, E2_MAP_SIZE)
    mean = np.asarray(norm.get("radar_mean") or [0.0, 0.0], dtype=np.float32)
    std = np.asarray(norm.get("radar_std") or [1.0, 1.0], dtype=np.float32)
    windows = (windows - mean.reshape(1, 1, 2, 1, 1)) / std.reshape(1, 1, 2, 1, 1)
    kept = np.asarray(kept_times[: n_win * E2_WINDOW_FRAMES], dtype=np.float64)
    kept = kept.reshape(n_win, E2_WINDOW_FRAMES)
    win_t0 = kept[:, 0]
    win_t1 = kept[:, -1]
    win_max_gap = np.diff(kept, axis=1).max(axis=1)
    return windows.astype(np.float32), win_t0, win_t1, win_max_gap


# CSI lines are re-parsed on every partial-minute run; memoize per line so the
# serial log is only tokenized once per minute instead of once per window.
_E2_CSI_CACHE: "OrderedDict[str, np.ndarray | None]" = OrderedDict()
_E2_CSI_CACHE_LIMIT = 8192


def _e2_parse_csi_amplitude_uncached(line: str) -> np.ndarray | None:
    """One CSI_DATA line -> float32 amplitude vector of 52 subcarriers."""
    match = _E2_CSI_RE.search(line)
    if not match:
        return None
    tokens = [t for t in match.group(1).split(",") if t.strip()]
    if len(tokens) != 128:
        return None
    try:
        values = np.asarray(tokens, dtype=np.float64)
    except ValueError:
        return None
    imag = values[0::2][_E2_CSI_MASK]
    real = values[1::2][_E2_CSI_MASK]
    return np.hypot(real, imag).astype(np.float32)


def _e2_parse_csi_amplitude(line: str) -> np.ndarray | None:
    with _E2_MAP_CACHE_LOCK:
        if line in _E2_CSI_CACHE:
            cached = _E2_CSI_CACHE[line]
            _E2_CSI_CACHE.move_to_end(line)
            return cached
    parsed = _e2_parse_csi_amplitude_uncached(line)
    with _E2_MAP_CACHE_LOCK:
        _E2_CSI_CACHE[line] = parsed
        _E2_CSI_CACHE.move_to_end(line)
        while len(_E2_CSI_CACHE) > _E2_CSI_CACHE_LIMIT:
            _E2_CSI_CACHE.popitem(last=False)
    return parsed


def e2_csi_windows(
    csi_samples: Sequence[tuple[int, float, str]],
    win_t0: np.ndarray,
    win_t1: np.ndarray,
    norm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """(receiver, t, line) samples -> normalized (nW, 128, 52) grids.

    The receiver with the most samples is used (E2 rule). Returns
    (windows, valid); windows with <2 in-window samples stay zeroed.
    """
    n_win = len(win_t0)
    out = np.zeros((n_win, E2_CSI_STEPS, E2_SUBCARRIERS), dtype=np.float32)
    valid = np.zeros(n_win, dtype=bool)
    by_receiver: dict[int, list[tuple[float, str]]] = {}
    for receiver, t, line in csi_samples:
        by_receiver.setdefault(int(receiver), []).append((float(t), line))
    if by_receiver:
        best = max(by_receiver, key=lambda r: len(by_receiver[r]))
        parsed = [(t, _e2_parse_csi_amplitude(line)) for t, line in by_receiver[best]]
        parsed = [(t, v) for t, v in parsed if v is not None]
        if parsed:
            ts = np.asarray([t for t, _ in parsed], dtype=np.float64)
            amp = np.stack([v for _, v in parsed])
            order = np.argsort(ts, kind="stable")
            ts, amp = ts[order], amp[order]
            for k in range(n_win):
                if win_t1[k] <= win_t0[k]:
                    continue
                mask = (ts >= win_t0[k]) & (ts <= win_t1[k])
                count = int(mask.sum())
                if count == 0:
                    continue
                if count == 1:
                    out[k] = np.repeat(amp[mask], E2_CSI_STEPS, axis=0)
                else:
                    grid = np.linspace(win_t0[k], win_t1[k], E2_CSI_STEPS)
                    for j in range(E2_SUBCARRIERS):
                        out[k, :, j] = np.interp(grid, ts[mask], amp[mask, j])
                valid[k] = count >= 2
    np.log1p(out, out=out)
    mean = np.asarray(norm.get("csi_mean") or [0.0] * E2_SUBCARRIERS, dtype=np.float32)
    std = np.asarray(norm.get("csi_std") or [1.0] * E2_SUBCARRIERS, dtype=np.float32)
    out = (out - mean) / std
    out[~valid] = 0.0
    return out.astype(np.float32), valid
