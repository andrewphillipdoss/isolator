"""Trigger layers: drum samples placed on every detected hit, sample-accurately.

Layering a sample under a real kick or snare only sounds bigger if the two
waveforms line up. A 1 ms offset already hollows out the low end of a
kick. So each sample is placed in three steps:

1. Line up the sample's attack start with the hit's attack start.
2. Slide it up to +/- `max_shift_ms`, picking the offset with the highest
   normalized cross-correlation against the separated stem. The comparison
   runs in the band that matters for that drum (lows for kick).
3. Choose one polarity for the whole song, by a weighted vote over all hits.
   Flipping single hits back and forth would sound worse than either choice.

Loudness follows the original performance (`dynamics`=1) or flattens it
toward consistent hits (`dynamics`=0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .audio import load, mono
from .onsets import Hit, refine_onset, reference_level_db

AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg"}


@dataclass
class OneShot:
    data: np.ndarray  # (2, n) float32, peak-normalized to 1.0
    attack: int  # attack start within `data`
    level_db: float  # peak level before normalization; sorts velocity layers
    name: str = ""


@dataclass
class KitPiece:
    name: str
    layers: list[list[OneShot]]  # soft -> hard; each layer holds its round robins
    # Separate drums under one piece, e.g. three toms, highest pitch first.
    variants: list["KitPiece"] = field(default_factory=list)

    def pick(self, velocity: int, rr: int) -> tuple[OneShot, int]:
        """Layer by MIDI velocity (even split), round robin `rr`."""
        i = min(len(self.layers) - 1, int((velocity - 1) / 127 * len(self.layers)))
        layer = self.layers[i]
        return layer[rr % len(layer)], i

    def pick_by_level(self, rel_db: float, counters: dict[int, int]) -> tuple[OneShot, int]:
        """Layer whose recorded level, relative to the loudest layer, is nearest `rel_db`.

        A hit 12 dB under the drummer's loud hits gets the sample that was
        played about 12 dB softer, so the timbre follows the dynamics.
        Round robins advance per layer, so an accent pattern that keeps
        returning to one layer still cycles through its samples instead of
        machine-gunning one.
        """
        top = self.layers[-1][0].level_db
        rel = np.array([layer[0].level_db - top for layer in self.layers])
        i = int(np.argmin(np.abs(rel - min(rel_db, 0.0))))
        k = counters.get(i, 0)
        counters[i] = k + 1
        layer = self.layers[i]
        return layer[k % len(layer)], i


@dataclass
class Kit:
    name: str
    sr: int
    pieces: dict[str, KitPiece] = field(default_factory=dict)


def _layers_by_loudness(shots: list[OneShot], same_layer_db: float = 1.5) -> list[list[OneShot]]:
    """Samples within `same_layer_db` of each other are round robins of one velocity layer."""
    shots = sorted(shots, key=lambda s: s.level_db)
    layers: list[list[OneShot]] = []
    for s in shots:
        if layers and s.level_db - layers[-1][0].level_db <= same_layer_db:
            layers[-1].append(s)
        else:
            layers.append([s])
    return layers


def oneshot_attack(data: np.ndarray, sr: int) -> int:
    """Attack start of a one-shot sample.

    Nothing comes before the hit in a sample, so the first point where the
    signal reaches 10% of its peak is a safe anchor even for cymbals, whose
    peak can arrive 100 ms after the stick. The ratio detector then refines
    it within a few milliseconds.
    """
    m = np.abs(mono(data))
    if m.max() <= 0:
        return 0
    first = int(np.argmax(m >= 0.1 * m.max()))
    return refine_onset(data, first, sr, back_ms=5, fwd_ms=3)


def load_oneshot(path: Path, sr: int) -> OneShot:
    a = load(path, sr=sr, stereo=True)
    peak = float(np.max(np.abs(a.data))) or 1.0
    data = (a.data / peak).astype(np.float32)
    return OneShot(data=data, attack=oneshot_attack(data, sr), level_db=20 * np.log10(peak), name=path.name)


def load_kit_folder(path: str | Path, sr: int) -> Kit:
    """A kit is a folder with one sub-folder of one-shots per piece.

        my_kit/kick/*.wav  my_kit/snare/*.wav  my_kit/toms/*.wav ...

    Velocity layers are worked out from the files' own loudness. Files of
    nearly equal level become round robins, so any pile of one-shots works.

    Files named `<drum>__<anything>.wav` (as `iso kit import` writes them)
    are kept apart as separate drums of one piece, e.g. `tom_1__*.wav`,
    `floor_tom__*.wav`. Toms are ordered by pitch, so a song's tom hits can
    be matched high-to-high.
    """
    from .toms import hit_pitch

    path = Path(path)
    kit = Kit(name=path.name, sr=sr)
    for d in sorted(p for p in path.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        if not files:
            continue
        name = d.name.lower()
        shots = {f: load_oneshot(f, sr) for f in files}
        groups: dict[str, list[OneShot]] = {}
        for f, shot in shots.items():
            groups.setdefault(f.name.split("__", 1)[0] if "__" in f.name else "", []).append(shot)
        piece = KitPiece(name, _layers_by_loudness(list(shots.values())))
        if len(groups) > 1:
            variants = [KitPiece(f"{name}:{g}", _layers_by_loudness(v)) for g, v in groups.items()]
            if name == "toms":
                def pitch(kp: KitPiece) -> float:
                    loud = kp.layers[-1][0]
                    return hit_pitch(loud.data, loud.attack, sr)
                variants.sort(key=lambda kp: -np.nan_to_num(pitch(kp), nan=0.0))
            piece.variants = variants
        kit.pieces[name] = piece
    return kit


@dataclass(frozen=True)
class AlignParams:
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    max_shift_ms: float = 2.0
    corr_ms: float = 25.0
    min_corr: float = 0.15


PIECE_ALIGN: dict[str, AlignParams] = {
    "kick": AlignParams(lowpass_hz=200.0, corr_ms=30.0),
    "snare": AlignParams(highpass_hz=120.0, lowpass_hz=2500.0),
    "toms": AlignParams(lowpass_hz=600.0, corr_ms=30.0),
}


def _band(x: np.ndarray, sr: int, hp: float | None, lp: float | None) -> np.ndarray:
    """Zero-phase band-limit, so filtering never shifts the alignment itself."""
    y = x.astype(np.float64)
    if hp:
        y = sosfiltfilt(butter(4, hp, "highpass", fs=sr, output="sos"), y, padlen=min(len(y) - 1, 300))
    if lp:
        y = sosfiltfilt(butter(4, lp, "lowpass", fs=sr, output="sos"), y, padlen=min(len(y) - 1, 300))
    return y


def ncc_scan(ref: np.ndarray, template: np.ndarray, center: int, max_shift: int) -> np.ndarray:
    """Normalized cross-correlation of `template` against `ref` starting at center+s, for s in [-S, S]."""
    L = len(template)
    lo = center - max_shift
    seg = np.zeros(L + 2 * max_shift)
    a, b = max(0, lo), min(len(ref), lo + len(seg))
    if b > a:
        seg[a - lo : b - lo] = ref[a:b]
    num = np.correlate(seg, template, mode="valid")
    c = np.concatenate([[0.0], np.cumsum(seg**2)])
    energy = c[L:] - c[:-L]
    den = np.sqrt(np.maximum(energy, 0.0) * float(np.dot(template, template))) + 1e-12
    return num / den


@dataclass
class TriggerReport:
    polarity: int
    hits: list[dict]
    polarity_confidence: float = 1.0  # 0..1; under ~0.3 the sample barely resembles the drum, check by ear

    @property
    def median_corr(self) -> float:
        cs = [h["corr"] for h in self.hits if h["aligned"]]
        return float(np.median(cs)) if cs else 0.0


def render_triggers(
    piece_stem: np.ndarray,
    sr: int,
    hits: list[Hit],
    kit_piece: KitPiece,
    *,
    params: AlignParams | None = None,
    align: bool = True,
    polarity: int | None = None,
    dynamics: float = 1.0,
    variant_of_hit: list[int] | None = None,
) -> tuple[np.ndarray, TriggerReport]:
    """Render one trigger track the same length as `piece_stem`.

    `polarity`: None decides it from the material; +1 / -1 forces it.
    `variant_of_hit`: per hit, which of `kit_piece.variants` to play (toms).

    The track is level-matched to the stem: a hit lands at the same peak as
    the hit it reinforces (dynamics=1), or at the stem's typical loud-hit
    level (dynamics=0). A trigger fader at -8 dB therefore really means
    8 dB under the piece.
    """
    p = params or AlignParams()
    n = piece_stem.shape[-1]
    out = np.zeros((2, n), dtype=np.float32)
    if not hits:
        return out, TriggerReport(polarity=1, hits=[])

    ref = _band(mono(piece_stem), sr, p.highpass_hz, p.lowpass_hz)
    S = int(p.max_shift_ms * sr / 1000)
    L = int(p.corr_ms * sr / 1000)
    ref_db = reference_level_db(np.array([h.level_db for h in hits]))

    plan = []
    banded: dict[int, np.ndarray] = {}
    n_var = len(kit_piece.variants)
    n_song = (max(variant_of_hit) + 1) if variant_of_hit else 1
    counters: dict[int, dict[int, int]] = {}
    for i, h in enumerate(hits):
        kp = kit_piece
        if n_var and variant_of_hit:
            v = variant_of_hit[i]
            kp = kit_piece.variants[round(v * (n_var - 1) / (n_song - 1)) if n_song > 1 else (n_var - 1) // 2]
        shot, layer = kp.pick_by_level(h.level_db - ref_db, counters.setdefault(id(kp), {}))
        scan = None
        if align:
            key = id(shot)
            if key not in banded:
                banded[key] = _band(mono(shot.data), sr, p.highpass_hz, p.lowpass_hz)
            tmpl = banded[key][shot.attack : shot.attack + L]
            if len(tmpl) >= 16 and np.any(tmpl):
                scan = ncc_scan(ref, tmpl, h.sample, S)
        plan.append((h, shot, layer, scan))

    vote, weight = 0.0, 0.0
    for _, _, _, scan in plan:
        if scan is not None:
            k = int(np.argmax(np.abs(scan)))
            vote += scan[k] * abs(scan[k])
            weight += scan[k] ** 2
    confidence = abs(vote) / weight if weight > 0 else 0.0
    if polarity is None:
        polarity = -1 if vote < 0 else 1

    rows = []
    for h, shot, layer, scan in plan:
        shift, corr, aligned = 0, 0.0, False
        if scan is not None:
            signed = polarity * scan
            k = int(np.argmax(signed))
            if signed[k] >= p.min_corr:
                shift, corr, aligned = k - S, float(signed[k]), True
        gain = 10.0 ** ((ref_db + dynamics * (h.level_db - ref_db)) / 20.0)
        start = h.sample - shot.attack + shift
        a, b = max(0, start), min(n, start + shot.data.shape[1])
        if b > a:
            out[:, a:b] += polarity * gain * shot.data[:, a - start : b - start]
        rows.append(
            {
                "time_s": h.sample / sr,
                "velocity": h.velocity,
                "layer": layer,
                "sample": shot.name,
                "shift_samples": shift,
                "corr": round(corr, 4),
                "aligned": aligned,
            }
        )
    return out, TriggerReport(polarity=polarity, hits=rows, polarity_confidence=round(confidence, 3))


