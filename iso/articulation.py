"""Articulations the separated stems can tell us: open vs closed hi-hat.

A closed hat chokes in a few tens of milliseconds; an open hat keeps
ringing. So we look at how far each hat hit has decayed when its window
ends (150 ms, or just before the next hat hit if sooner).
"""

from __future__ import annotations

import numpy as np

from .onsets import Hit


def _rms(x: np.ndarray, sr: int, win_ms: float = 5.0) -> np.ndarray:
    m = x.mean(axis=0) if x.ndim == 2 else x
    w = max(1, int(win_ms * sr / 1000))
    c = np.concatenate([[0.0], np.cumsum(m.astype(np.float64) ** 2)])
    r = np.sqrt(np.maximum(c[w:] - c[:-w], 0.0) / w)
    return np.concatenate([r, np.zeros(w - 1)])


def hat_open(stem: np.ndarray, sr: int, hits: list[Hit], open_above_db: float = -14.0, min_window_ms: float = 60.0) -> list[bool]:
    """True for hits that are still ringing within `open_above_db` of their peak at the window's end."""
    env = _rms(stem, sr)
    n = len(env)
    out = []
    for i, h in enumerate(hits):
        nxt = hits[i + 1].sample if i + 1 < len(hits) else n
        end = min(h.sample + int(0.150 * sr), nxt - int(0.003 * sr), n)
        if end - h.sample < int(min_window_ms * sr / 1000):
            out.append(False)  # too short to tell; closed is the safe default
            continue
        peak = env[h.sample : h.sample + int(0.015 * sr)].max(initial=0.0)
        tail = env[end - int(0.015 * sr) : end].mean() if end - int(0.015 * sr) > h.sample else 0.0
        out.append(bool(peak > 0 and 20 * np.log10(tail / peak + 1e-12) > open_above_db))
    return out
