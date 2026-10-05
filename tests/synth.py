"""Synthetic drum hits with exactly known attack positions, for tests."""

from __future__ import annotations

import numpy as np

SR = 44100


def kick(sr: int = SR, seed: int = 0, f0: float = 120.0, f1: float = 48.0, decay: float = 0.18) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.5 * sr)) / sr
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    phase = 2 * np.pi * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / decay)
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.002) * 0.3
    return (body + click).astype(np.float32)


def snare(sr: int = SR, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.4 * sr)) / sr
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    noise = rng.standard_normal(len(t)) * np.exp(-t / 0.09) * 0.6
    return (tone + noise).astype(np.float32)


def place(n: int, shot: np.ndarray, starts: list[int], gains: list[float] | None = None) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    gains = gains or [1.0] * len(starts)
    for s, g in zip(starts, gains):
        m = min(len(shot), n - s)
        out[s : s + m] += g * shot[:m]
    return out


def stereo(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x]).astype(np.float32)
