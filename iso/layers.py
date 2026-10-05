"""The layer kit: the stems the producer mixes, with starting levels.

Every stem comes from the same source signal and none of them adds
latency, so they sum without comb filtering. That holds for pieces vs full
kit, full kit vs room (room is defined as full - dry), and triggers vs
pieces (aligned per hit). Real multi-mic recordings don't get this for
free: overheads arrive later than close mics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import Audio, db_to_gain, load, save


def lag_of(ref: np.ndarray, est: np.ndarray, max_lag: int = 64) -> int:
    """Samples `est` is late (+) or early (-) relative to `ref`, by cross-correlation (FFT)."""
    a = ref.mean(axis=0) if ref.ndim == 2 else ref
    b = est.mean(axis=0) if est.ndim == 2 else est
    # First differences whiten the signals, so even a low kick gives a peak
    # about one sample wide instead of a broad hump.
    a, b = np.diff(a.astype(np.float64)), np.diff(b.astype(np.float64))
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    lags = np.concatenate([c[: max_lag + 1], c[-max_lag:]])
    k = int(np.argmax(lags))
    norm = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-20
    # Only move when the evidence is strong and clearly beats zero lag; a
    # near-empty stem (an unplayed ride, say) correlates by noise alone.
    if lags[k] / norm < 0.3 or lags[k] < 1.2 * max(lags[0], 0.0):
        return 0
    return k if k <= max_lag else k - len(lags)


def realign(ref: np.ndarray, est: np.ndarray, max_lag: int = 64) -> tuple[np.ndarray, int]:
    """Shift `est` so it lines up with `ref` to the sample.

    Every model Iso ships is sample-aligned, so this should always find 0.
    It's a guard: a single sample of offset between dry and full would
    smear every transient into the room stem.
    """
    lag = lag_of(ref, est, max_lag)
    if lag == 0:
        return est, 0
    out = np.zeros_like(est)
    if lag > 0:
        out[:, :-lag] = est[:, lag:]
    else:
        out[:, -lag:] = est[:, :lag]
    return out, lag


def split_room(full: np.ndarray, dry: np.ndarray) -> np.ndarray:
    """Room is whatever the dereverb took out, so dry + room == full exactly."""
    if full.shape != dry.shape:
        raise ValueError(f"full {full.shape} and dry {dry.shape} must match")
    return (full - dry).astype(np.float32)


@dataclass
class LayerTrack:
    name: str
    file: str  # path relative to the kit folder
    group: str  # pieces | triggers | kit | room | song
    gain_db: float
    muted: bool = False


# Starting points only; every song and every kit differs. Pieces are the core
# at unity, triggers sit under them for consistency and punch, the full dry
# kit glues it together, and a little room goes on top.
DEFAULT_LEVELS_DB: dict[str, float] = {
    "kick": 0.0,
    "snare": 0.0,
    "toms": -2.0,
    "hihat": -4.0,
    "ride": -4.0,
    "crash": -4.0,
    # Trigger stems are level-matched to their piece, so these are "this far under".
    "trig_kick": -8.0,  # -10 natural ... -6 punchy
    "trig_snare": -10.0,  # -12 ... -8
    "trig_toms": -12.0,
    "drums_dry": -8.0,
    "drums_full": -10.0,
    "room": -12.0,  # uncompressed; if you crush it, start 3-6 dB lower
    "no_drums": -3.0,
}

GROUP_OF = {
    "kick": "pieces",
    "snare": "pieces",
    "toms": "pieces",
    "hihat": "pieces",
    "ride": "pieces",
    "crash": "pieces",
    "trig_kick": "triggers",
    "trig_snare": "triggers",
    "trig_toms": "triggers",
    "drums_dry": "kit",
    "drums_full": "kit",
    "room": "room",
    "no_drums": "song",
}

# drums_full already contains drums_dry + room, so using both would count
# the room twice. The default stack uses dry + room; full is there for A/B.
DEFAULT_MUTED = {"drums_full"}


def default_session(stem_files: dict[str, str]) -> list[LayerTrack]:
    order = list(DEFAULT_LEVELS_DB)
    names = sorted(stem_files, key=lambda k: order.index(k) if k in order else len(order))
    return [
        LayerTrack(
            name=k,
            file=stem_files[k],
            group=GROUP_OF.get(k, "other"),
            gain_db=DEFAULT_LEVELS_DB.get(k, -6.0),
            muted=k in DEFAULT_MUTED,
        )
        for k in names
    ]


def save_session(path: str | Path, tracks: list[LayerTrack], sr: int) -> Path:
    path = Path(path)
    path.write_text(json.dumps({"sr": sr, "tracks": [asdict(t) for t in tracks]}, indent=2))
    return path


def load_session(path: str | Path) -> tuple[list[LayerTrack], int]:
    d = json.loads(Path(path).read_text())
    return [LayerTrack(**t) for t in d["tracks"]], d["sr"]


def mixdown(kit_dir: str | Path, tracks: list[LayerTrack], include_song: bool = False) -> Audio:
    """Bounce the layer stack (drums only unless `include_song`)."""
    kit_dir = Path(kit_dir)
    out = None
    sr = 0
    for t in tracks:
        if t.muted or (t.group == "song" and not include_song):
            continue
        a = load(kit_dir / t.file)
        sr = a.sr
        y = a.data * db_to_gain(t.gain_db)
        if out is None:
            out = y
        else:
            n = max(out.shape[1], y.shape[1])
            out = np.pad(out, ((0, 0), (0, n - out.shape[1]))) + np.pad(y, ((0, 0), (0, n - y.shape[1])))
    if out is None:
        raise ValueError("nothing to mix: every track is muted")
    return Audio(out.astype(np.float32), sr)


def write_reaper_project(path: str | Path, tracks: list[LayerTrack], sr: int, length_s: float) -> Path:
    """A REAPER project with every layer on its own track, grouped in folders, at the starting levels."""
    path = Path(path)
    groups: dict[str, list[LayerTrack]] = {}
    for t in tracks:
        groups.setdefault(t.group, []).append(t)
    lines = ['<REAPER_PROJECT 0.1 "7.0" 0', f"  SAMPLERATE {sr} 0 0"]
    for g, ts in groups.items():
        lines += ["  <TRACK", f'    NAME "{g.upper()}"', "    ISBUS 1 1", "  >"]
        for i, t in enumerate(ts):
            last = i == len(ts) - 1
            lines += [
                "  <TRACK",
                f'    NAME "{t.name}"',
                f"    VOLPAN {db_to_gain(t.gain_db):.6f} 0 -1 -1 1",
                f"    MUTESOLO {1 if t.muted else 0} 0 0",
                f"    ISBUS {2 if last else 0} {-1 if last else 0}",
                "    <ITEM",
                "      POSITION 0",
                f"      LENGTH {length_s:.6f}",
                f'      NAME "{Path(t.file).name}"',
                "      <SOURCE WAVE",
                f'        FILE "{t.file}"',
                "      >",
                "    >",
                "  >",
            ]
    lines.append(">")
    path.write_text("\n".join(lines) + "\n")
    return path


def save_stem(kit_dir: Path, name: str, data: np.ndarray, sr: int) -> str:
    rel = f"{name}.wav"
    save(kit_dir / rel, Audio(data.astype(np.float32), sr))
    return rel
