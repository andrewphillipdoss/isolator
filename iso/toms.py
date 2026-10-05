"""Telling the toms apart.

The separators give one "toms" stem. A fill goes rack -> rack -> floor,
and a trigger layer or MIDI part must follow that. Each tom hit gets a
pitch estimate. The song's hits are clustered into as many toms as needed,
and clusters are matched by rank (highest -> highest), because the
recorded toms and the sample kit's toms are never tuned the same.
"""

from __future__ import annotations

import numpy as np

from .onsets import Hit

TOM_NOTES_BY_COUNT = {
    1: ["tom_mid"],
    2: ["tom_high", "tom_floor"],
    3: ["tom_high", "tom_mid", "tom_floor"],
}


def hit_pitch(x: np.ndarray, start: int, sr: int, lo_hz: float = 50.0, hi_hz: float = 500.0) -> float:
    """Strongest partial between lo_hz and hi_hz in the 5-150 ms after the attack."""
    m = x.mean(axis=0) if x.ndim == 2 else x
    seg = m[start + int(0.005 * sr) : start + int(0.150 * sr)].astype(np.float64)
    if seg.size < 256:
        return float("nan")
    n = 1 << 15
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size), n))
    f = np.fft.rfftfreq(n, 1 / sr)
    band = (f >= lo_hz) & (f <= hi_hz)
    if not band.any() or spec[band].max() <= 0:
        return float("nan")
    return float(f[band][np.argmax(spec[band])])


def cluster_pitches(pitches: list[float], k: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """1-D k-means on log pitch. Returns (labels, centers), cluster 0 = highest pitch."""
    p = np.log(np.asarray(pitches, dtype=np.float64))
    ok = np.isfinite(p)
    labels = np.zeros(len(p), dtype=int)
    if ok.sum() == 0:
        return labels, np.zeros(1)
    vals = p[ok]
    k = max(1, min(k, len(np.unique(np.round(vals, 2)))))
    centers = np.quantile(vals, np.linspace(0, 1, k + 2)[1:-1]) if k > 1 else np.array([vals.mean()])
    for _ in range(iters):
        lab = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1)
        new = np.array([vals[lab == i].mean() if np.any(lab == i) else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    order = np.argsort(-centers)  # highest first
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    lab = np.argmin(np.abs(vals[:, None] - centers[None, :]), axis=1)
    labels[ok] = rank[lab]
    # Hits with no usable pitch go to the most common tom.
    if (~ok).any():
        labels[~ok] = np.bincount(labels[ok]).argmax()
    return labels, np.exp(centers[order])


def split_toms(stem: np.ndarray, sr: int, hits: list[Hit], max_toms: int = 3, min_gap_semitones: float = 2.0) -> list[int]:
    """Tom index per hit (0 = highest). Merges clusters closer than `min_gap_semitones`."""
    if not hits:
        return []
    pitches = [hit_pitch(stem, h.sample, sr) for h in hits]
    for k in range(max_toms, 0, -1):
        labels, centers = cluster_pitches(pitches, k)
        if len(centers) <= 1:
            return list(labels)
        gaps = 12 * np.log2(centers[:-1] / centers[1:])
        if np.all(gaps >= min_gap_semitones):
            return list(labels)
    return [0] * len(hits)
