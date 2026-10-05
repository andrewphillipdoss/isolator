"""Combining the same stem from several separation models.

`avg_wave` is the default because it is the only method that is linear in
the inputs. Every model's output is phase-locked to the mix, so the average
is too, and the layer kit keeps summing cleanly. The spectral methods
(`max_spec` for fullness, `min_spec` for less bleed) are what the stem
separation community uses to trade fullness against bleed. They borrow
phase from the average.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

_NPERSEG = 4096
_NOVERLAP = 3072


def avg_wave(stems: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    w = np.asarray(weights or [1.0] * len(stems), dtype=np.float64)
    w = w / w.sum()
    out = np.zeros_like(stems[0], dtype=np.float64)
    for s, wi in zip(stems, w):
        out += wi * s
    return out.astype(np.float32)


def median_wave(stems: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(stems), axis=0).astype(np.float32)


def _spec(x: np.ndarray, sr: int) -> np.ndarray:
    return stft(x, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)[2]


def _ispec(Z: np.ndarray, sr: int, n: int) -> np.ndarray:
    y = istft(Z, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)[1]
    if y.shape[-1] < n:
        y = np.pad(y, ((0, 0), (0, n - y.shape[-1])))
    return y[..., :n].astype(np.float32)


def _spectral(stems: list[np.ndarray], sr: int, pick) -> np.ndarray:
    n = stems[0].shape[-1]
    specs = np.stack([_spec(s, sr) for s in stems])
    mag = pick(np.abs(specs), axis=0)
    phase = np.angle(specs.mean(axis=0))
    return _ispec(mag * np.exp(1j * phase), sr, n)


def max_spec(stems: list[np.ndarray], sr: int) -> np.ndarray:
    """Loudest bin wins: fuller stems, more bleed."""
    return _spectral(stems, sr, np.max)


def min_spec(stems: list[np.ndarray], sr: int) -> np.ndarray:
    """Quietest bin wins: least bleed, can sound thinner."""
    return _spectral(stems, sr, np.min)


def combine(stems: list[np.ndarray], sr: int, method: str = "avg_wave", weights: list[float] | None = None) -> np.ndarray:
    if len(stems) == 1:
        return stems[0].astype(np.float32)
    if method == "avg_wave":
        return avg_wave(stems, weights)
    if method == "median_wave":
        return median_wave(stems)
    if method == "max_spec":
        return max_spec(stems, sr)
    if method == "min_spec":
        return min_spec(stems, sr)
    raise ValueError(f"unknown ensemble method {method!r}")


def split_by_guides(source: np.ndarray, guide_a: np.ndarray, guide_b: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Split `source` into (a, b) using the per-bin energy ratio of two guide signals.

    The second part is the exact remainder, so a + b == source and nothing is lost.
    Iso uses this to split the 5-stem model's cymbals into ride and crash,
    guided by the 6-stem model's ride and crash.
    """
    n = source.shape[-1]
    A, B, S = _spec(guide_a, sr), _spec(guide_b, sr), _spec(source, sr)
    pa, pb = np.abs(A) ** 2, np.abs(B) ** 2
    mask = pa / (pa + pb + 1e-12)
    a = _ispec(mask * S, sr, n)
    return a, (source - a).astype(np.float32)
