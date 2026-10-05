"""Hit-keyed gating: the last step that makes a piece stem "as clean as possible".

A separated tom or kick stem still carries faint leakage between hits
(cymbal wash, guitar ghosts). We already know where every hit is, so we
open the gate a few milliseconds *before* each attack (no clipped
transients, unlike a level-triggered gate) and close it once the hit has
decayed. Between hits, the stem drops to `floor_db`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import db_to_gain


@dataclass(frozen=True)
class GateParams:
    lookahead_ms: float = 4.0  # open this long before the attack start
    attack_ms: float = 2.0  # fade-in length (fits inside the lookahead)
    min_hold_ms: float = 60.0
    max_hold_ms: float = 450.0
    decay_db: float = 36.0  # close once the hit has decayed this far below its peak
    release_ms: float = 50.0
    floor_db: float = -60.0  # level between hits; -inf for digital silence


PIECE_GATES: dict[str, GateParams] = {
    "kick": GateParams(min_hold_ms=80, max_hold_ms=400, decay_db=36),
    "snare": GateParams(min_hold_ms=90, max_hold_ms=450, decay_db=40),
    "toms": GateParams(min_hold_ms=150, max_hold_ms=900, decay_db=40, release_ms=80),
}
# Hats, ride and crash ring continuously; gating them sounds choppy, so they
# aren't gated by default.


def _envelope(x: np.ndarray, sr: int, win_ms: float = 5.0) -> np.ndarray:
    m = x.mean(axis=0) if x.ndim == 2 else x
    w = max(1, int(win_ms * sr / 1000))
    c = np.concatenate([[0.0], np.cumsum(m.astype(np.float64) ** 2)])
    rms = np.sqrt(np.maximum(c[w:] - c[:-w], 0.0) / w)
    return np.concatenate([rms, np.full(w - 1, rms[-1] if rms.size else 0.0)])


def _cosine_ramp(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 1) / n)


def gate_curve(x: np.ndarray, sr: int, hit_samples: list[int], p: GateParams) -> np.ndarray:
    """Gain curve in [0, 1] (before the floor is applied), one value per sample."""
    n = x.shape[-1]
    env = _envelope(x, sr)
    g = np.zeros(n)
    la = int(p.lookahead_ms * sr / 1000)
    att = max(1, min(int(p.attack_ms * sr / 1000), la))
    rel = max(1, int(p.release_ms * sr / 1000))
    hmin = int(p.min_hold_ms * sr / 1000)
    hmax = int(p.max_hold_ms * sr / 1000)
    pk_win = int(0.02 * sr)
    decay = 10.0 ** (-p.decay_db / 20.0)
    for h in sorted(hit_samples):
        if h >= n:
            continue
        a0 = max(0, h - la)
        a1 = min(n, a0 + att)
        peak = env[h : min(n, h + pk_win)].max(initial=0.0)
        start = min(n, h + hmin)
        stop = min(n, h + hmax)
        quiet = np.nonzero(env[start:stop] < peak * decay)[0]
        r0 = max(a1, start + int(quiet[0]) if quiet.size else stop)
        r1 = min(n, r0 + rel)
        w = np.zeros(r1 - a0)
        w[: a1 - a0] = _cosine_ramp(a1 - a0)
        w[a1 - a0 : r0 - a0] = 1.0
        w[r0 - a0 :] = _cosine_ramp(r1 - r0)[::-1]
        np.maximum(g[a0:r1], w, out=g[a0:r1])
    return g


def hit_gate(x: np.ndarray, sr: int, hit_samples: list[int], p: GateParams | None = None) -> np.ndarray:
    p = p or GateParams()
    g = gate_curve(x, sr, hit_samples, p)
    floor = db_to_gain(p.floor_db)
    gain = floor + (1.0 - floor) * g
    return (x * gain.astype(np.float32)).astype(np.float32)
