"""Hit detection on separated drum-piece stems.

The piece stems are already sorted by drum, so this module doesn't classify
anything. It finds where each hit starts, to the sample, and how loud it
was. Those two numbers drive the MIDI velocities, the hit-keyed gates and
the trigger alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d


@dataclass(frozen=True)
class Hit:
    sample: int  # attack start, in samples from the top of the song
    level_db: float  # peak level of the hit on its own stem, dBFS
    velocity: int = 100  # 1..127


@dataclass(frozen=True)
class DetectParams:
    min_interval_ms: float = 40.0  # two hits closer than this count as one
    delta: float = 0.07  # peak must exceed the local mean by this much (envelope normalized to 1)
    rel_db: float = 30.0  # hits further than this below the stem's typical loud hit are dropped as leakage
    velocity_range_db: float = 36.0  # level span mapped onto velocities 127 -> velocity_floor
    velocity_floor: int = 8
    fine: bool = True  # snap to the stick/beater transient (see fine_onset)


# Per-piece defaults. Ghost notes live 20-30 dB under backbeats, so snare and
# hats keep a wider window than kick and crash.
PIECE_PARAMS: dict[str, DetectParams] = {
    "kick": DetectParams(min_interval_ms=45, rel_db=30),
    "snare": DetectParams(min_interval_ms=40, rel_db=36),
    # In fast fills the fine pass can jump to the next tom's transient.
    "toms": DetectParams(min_interval_ms=50, rel_db=30, fine=False),
    "hihat": DetectParams(min_interval_ms=35, rel_db=36),
    "ride": DetectParams(min_interval_ms=60, rel_db=36),
    "crash": DetectParams(min_interval_ms=90, rel_db=30),
}

_HOP = 128
_NFFT = 1024


def onset_strength(x: np.ndarray, sr: int, hop: int = _HOP, n_fft: int = _NFFT) -> np.ndarray:
    """Log-magnitude spectral flux, one value per hop. Computed in blocks to keep memory flat."""
    x = x.mean(axis=0) if x.ndim == 2 else x
    pad = n_fft // 2
    xp = np.pad(x.astype(np.float32), (pad, pad))
    n_frames = 1 + (len(xp) - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop][:n_frames]
    flux = np.zeros(n_frames, dtype=np.float32)
    prev = None
    block = 4096
    for i in range(0, n_frames, block):
        spec = np.log1p(1000.0 * np.abs(np.fft.rfft(frames[i : i + block] * win, axis=1)))
        first = spec[:1] if prev is None else prev
        d = np.diff(np.concatenate([first, spec]), axis=0)
        flux[i : i + len(spec)] = np.maximum(d, 0.0).sum(axis=1)
        prev = spec[-1:]
    return flux


def pick_peaks(env: np.ndarray, hop: int, sr: int, params: DetectParams) -> np.ndarray:
    """Return frame indices of onset peaks."""
    if env.size == 0 or env.max() <= 0:
        return np.zeros(0, dtype=np.int64)
    e = env / np.percentile(env[env > 0], 99.5)
    frames_per_ms = sr / hop / 1000.0
    wait = max(1, int(round(params.min_interval_ms * frames_per_ms)))
    local_max = maximum_filter1d(e, size=2 * max(1, int(3 * frames_per_ms)) + 1)
    local_mean = uniform_filter1d(e, size=2 * max(1, int(50 * frames_per_ms)) + 1)
    cand = np.nonzero((e == local_max) & (e >= local_mean + params.delta))[0]
    peaks: list[int] = []
    for c in cand:
        if peaks and c - peaks[-1] < wait:
            if e[c] > e[peaks[-1]]:
                peaks[-1] = int(c)
            continue
        peaks.append(int(c))
    return np.asarray(peaks, dtype=np.int64)


def _ratio_peak(x: np.ndarray, lo: int, hi: int, w: int) -> int | None:
    """Sample in [lo, hi) where energy(next w) / energy(previous w) peaks.

    The signal is treated as silent outside its bounds, so an attack at
    sample 0 (typical for one-shot samples) is found too.
    """
    n = len(x)
    lo, hi = max(0, lo), min(n, hi)
    if hi - lo < 1:
        return None
    a, b = lo - w, hi + w
    seg = np.zeros(b - a)
    sa, sb = max(0, a), min(n, b)
    seg[sa - a : sb - a] = x[sa:sb]
    c = np.concatenate([[0.0], np.cumsum(seg * seg)])
    i = np.arange(w, w + hi - lo)
    post = c[i + w] - c[i]
    pre = c[i] - c[i - w]
    eps = 1e-6 * post.max() + 1e-20
    return lo + int(np.argmax((post + eps) / (pre + eps)))


def refine_onset(
    x: np.ndarray,
    approx: int,
    sr: int,
    back_ms: float = 15.0,
    fwd_ms: float = 25.0,
    preemphasis: float = 0.0,
) -> int:
    """Move a frame-level onset to the sample where the attack starts.

    At each candidate sample it compares the energy in the next 10 ms with
    the energy in the previous 10 ms; the attack start is where that ratio
    peaks. 10 ms is about half a period of a kick's fundamental, so the
    energy of a ringing low tail stays nearly constant from window to window
    and its zero crossings don't fake an attack. On clean attacks this lands
    within a couple of samples.

    `preemphasis` (e.g. 0.95) tilts the signal toward highs first. That
    helps find a soft hit buried in a louder hit's low, beating tail.
    """
    x = x.mean(axis=0) if x.ndim == 2 else x
    if preemphasis:
        x = np.concatenate([x[:1], x[1:] - preemphasis * x[:-1]])
    c = _ratio_peak(x, approx - int(back_ms * sr / 1000), approx + int(fwd_ms * sr / 1000), int(0.010 * sr))
    return int(approx) if c is None else c


def fine_onset(x: np.ndarray, coarse: int, sr: int, back_ms: float = 2.0, fwd_ms: float = 12.0) -> int:
    """Snap a coarse onset to the stick or beater transient.

    Separation models smear attacks backwards a little (masking in the
    frequency domain leaks energy into the frames before a hit), so the
    10 ms coarse detector lands up to ~8 ms early on separated stems. The
    real attack is a sudden burst of treble, so look just after the coarse
    estimate for the sharpest 1 ms jump in a treble-tilted copy. On Iso's
    test song this took separated-snare timing from -5.4 ms median
    (spread 7.7 ms) to 0.0 ms (spread 0.9 ms).
    """
    x = x.mean(axis=0) if x.ndim == 2 else x
    lo, hi = coarse - int(back_ms * sr / 1000), coarse + int(fwd_ms * sr / 1000)
    a, b = max(0, lo - int(0.02 * sr)), min(len(x), hi + int(0.02 * sr))
    seg = x[a:b].astype(np.float64)
    if seg.size < 8:
        return coarse
    y = np.concatenate([seg[:1], seg[1:] - 0.95 * seg[:-1]])
    r = _ratio_peak(y, lo - a, hi - a, max(4, int(0.001 * sr)))
    return coarse if r is None else a + r


def is_new_hit(x: np.ndarray, start: int, sr: int, min_rise_db: float = 3.0) -> bool:
    """True if the signal jumps up at `start` instead of just continuing a tail."""
    x = x.mean(axis=0) if x.ndim == 2 else x
    pre = x[max(0, start - int(0.015 * sr)) : max(0, start - int(0.001 * sr))]
    post = x[start : start + int(0.02 * sr)]
    if post.size == 0:
        return False
    if pre.size == 0:
        return True
    return np.max(np.abs(post)) >= np.max(np.abs(pre)) * 10 ** (min_rise_db / 20)


def hit_level_db(x: np.ndarray, start: int, sr: int, win_ms: float = 20.0) -> float:
    """How hard this hit was, in dB, not counting what was already ringing.

    Peak of the hit's first `win_ms`, minus the energy of the ring that was
    already there (the previous tom in a fill, the ride's wash). Without
    the subtraction, the second hit of a fast double reads loud just
    because the first is still sounding.
    """
    x = x.mean(axis=0) if x.ndim == 2 else x
    w = max(1, int(win_ms * sr / 1000))
    seg = x[start : start + w].astype(np.float64)
    if seg.size == 0:
        return -120.0
    peak2 = float(np.max(seg**2))
    pre = x[max(0, start - w) : start].astype(np.float64)
    ring2 = float(np.mean(pre**2)) * 2.0 if pre.size else 0.0  # sine peak^2 = 2 x mean square
    new2 = max(peak2 - ring2, 0.05 * peak2)
    return float(10.0 * np.log10(max(new2, 1e-18)))


def reference_level_db(levels_db: np.ndarray) -> float:
    """The stem's 'typical loud hit': 90th percentile, robust to one freak spike."""
    return float(np.percentile(levels_db, 90)) if levels_db.size else -120.0


def to_velocity(level_db: float, ref_db: float, params: DetectParams) -> int:
    v = 127.0 + (level_db - ref_db) * (127.0 - params.velocity_floor) / params.velocity_range_db
    return int(np.clip(round(v), 1, 127))


def detect_hits(x: np.ndarray, sr: int, params: DetectParams | None = None) -> list[Hit]:
    """Find hits on one separated piece stem (kick, snare, ...)."""
    params = params or DetectParams()
    mono = x.mean(axis=0) if x.ndim == 2 else x
    frames = pick_peaks(onset_strength(mono, sr), _HOP, sr, params)
    starts: set[int] = set()
    # A spectral-flux frame peaks as the attack enters the analysis window,
    # so the true attack sits a few ms after the frame centre, not before it.
    back, fwd = 6.0, 20.0
    for f in frames:
        s = refine_onset(mono, int(f * _HOP), sr, back, fwd)
        if not is_new_hit(mono, s, sr):
            # Probably a hit inside a louder or still-ringing tail (fast tom
            # fills, ghost notes). The tail has little treble left, while a new
            # stick strike brings a fresh burst of it, so look again with the
            # signal tilted toward the attack.
            # Also require the overall level to still be rising (>= 1 dB), so
            # noise in a decaying tail doesn't count as a hit.
            tilted = np.concatenate([mono[:1], mono[1:] - 0.95 * mono[:-1]])
            s = refine_onset(tilted, int(f * _HOP), sr, back, fwd)
            if not (is_new_hit(tilted, s, sr, min_rise_db=6.0) and is_new_hit(mono, s, sr, min_rise_db=1.0)):
                continue
        if params.fine:
            s = fine_onset(mono, s, sr)
        starts.add(s)
    if not starts:
        return []
    ordered = sorted(starts)
    min_gap = int(params.min_interval_ms * sr / 1000)
    merged = [ordered[0]]
    for s in ordered[1:]:
        if s - merged[-1] >= min_gap:
            merged.append(s)
    levels = np.array([hit_level_db(mono, s, sr) for s in merged])
    ref = reference_level_db(levels)
    return [
        Hit(sample=int(s), level_db=float(lv), velocity=to_velocity(lv, ref, params))
        for s, lv in zip(merged, levels)
        if lv >= ref - params.rel_db
    ]


def drop_cross_leakage(
    hits: dict[str, list[Hit]],
    sr: int,
    tol_ms: float = 12.0,
    leak_db: float = 20.0,
) -> dict[str, list[Hit]]:
    """Drop hits that are really another drum leaking into this stem.

    A hit on stem A is treated as leakage when another stem B has a hit within
    `tol_ms`, and A's hit sits more than `leak_db` further below A's typical
    loud hit than B's hit sits below B's typical loud hit. A real kick and
    crash landing together are both near their own reference level, so both
    survive. A snare ghost showing up faintly in the kick stem does not.
    """
    tol = int(tol_ms * sr / 1000)
    refs = {p: reference_level_db(np.array([h.level_db for h in hs])) for p, hs in hits.items() if hs}
    out: dict[str, list[Hit]] = {}
    for p, hs in hits.items():
        keep = []
        for h in hs:
            rel_a = h.level_db - refs.get(p, h.level_db)
            leaked = False
            for q, qs in hits.items():
                if q == p or not qs:
                    continue
                idx = int(np.searchsorted([g.sample for g in qs], h.sample))
                for j in (idx - 1, idx):
                    if 0 <= j < len(qs) and abs(qs[j].sample - h.sample) <= tol:
                        if (qs[j].level_db - refs[q]) - rel_a > leak_db:
                            leaked = True
                if leaked:
                    break
            if not leaked:
                keep.append(h)
        out[p] = keep
    return out


def drop_buried(hits: dict[str, list[Hit]], kit: np.ndarray, sr: int, floor_db: float = 30.0) -> dict[str, list[Hit]]:
    """Drop hits that sit more than `floor_db` under the whole kit at that instant.

    A piece that wasn't played (no ride in the song, say) still comes out of
    the separator as a faint stem full of blips: leakage 40-50 dB under the
    kit. A real hit, even a quiet hi-hat tick, is a meaningful part of the
    kit sound at its moment. Comparing locally rather than to the song's
    loudest hit also keeps quiet passages and fade-outs intact.
    """
    m = kit.mean(axis=0) if kit.ndim == 2 else kit
    return {
        p: [h for h in hs if h.level_db >= hit_level_db(m, h.sample, sr) - floor_db]
        for p, hs in hits.items()
    }
