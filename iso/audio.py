"""Audio I/O.

Every signal inside Iso is a float32 array shaped (channels, samples), the
same layout the separation models use. Every stem in a layer kit has the
same sample rate and the same length as the source, and starts at sample 0,
so dropping the files into a DAW at bar 1 lines them up exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


@dataclass
class Audio:
    data: np.ndarray  # float32, (channels, samples)
    sr: int

    @property
    def channels(self) -> int:
        return self.data.shape[0]

    @property
    def samples(self) -> int:
        return self.data.shape[1]

    @property
    def seconds(self) -> float:
        return self.samples / self.sr


def load(path: str | Path, sr: int | None = None, stereo: bool = True) -> Audio:
    """Read any format libsndfile understands (WAV, FLAC, AIFF, OGG, MP3)."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = np.ascontiguousarray(data.T)
    if sr is not None and sr != file_sr:
        x = resample(x, file_sr, sr)
        file_sr = sr
    if stereo:
        x = to_stereo(x)
    return Audio(x, file_sr)


def save(path: str | Path, audio: Audio, subtype: str = "FLOAT") -> Path:
    """Write a WAV. 32-bit float by default so summed or boosted stems never clip on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.data.T, audio.sr, subtype=subtype)
    return path


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    y = soxr.resample(x.T, sr_from, sr_to, quality="VHQ")
    return np.ascontiguousarray(y.T, dtype=np.float32)


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 2:
        return x
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0)
    # Downmix anything wider than stereo by averaging into L/R.
    return np.stack([x[0::2].mean(axis=0), x[1::2].mean(axis=0)]).astype(np.float32)


def mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Pad with silence or trim so a stem matches the source length exactly."""
    if x.shape[1] == n:
        return x
    if x.shape[1] > n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[1])))


def db_to_gain(db: float) -> float:
    return 0.0 if db == -np.inf else float(10.0 ** (db / 20.0))


def gain_to_db(g: float, floor_db: float = -120.0) -> float:
    return max(20.0 * np.log10(max(abs(g), 1e-12)), floor_db)
