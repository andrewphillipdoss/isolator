"""Running separation models.

The default backend is MSST (ZFTurbo's Music-Source-Separation-Training,
MIT, `pip install msst`). It separates in-memory arrays, never rescales
individual stems, runs on CUDA, Apple MPS or CPU, and loads every
architecture Iso uses: BS-RoFormer, SCNet, MDX23C and HTDemucs. Checkpoints
come from the registry in `iso.models`.

`AudioSeparatorBackend` (python-audio-separator) is kept as an
alternative. It peak-normalizes the input and each stem separately, so we
feed it a copy scaled to -12 dBFS and undo the scaling afterwards.

`CachedBackend` stores results by (audio content, model, settings).
Re-running with a different trigger kit or gate setting never re-runs the
slow models.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import Audio, fit_length, load, save

MODEL_SR = 44100
_HEADROOM_PEAK = 0.25  # feed models at <= -12 dBFS so nothing hits the 0.9 normalizer


class Backend(Protocol):
    def separate(self, x: np.ndarray, sr: int, model: str) -> dict[str, np.ndarray]:
        """Return {stem name (lower case): (2, n) array}, same length as x."""
        ...


@dataclass
class MSSTBackend:
    model_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "iso" / "models")
    tta: bool = False  # polarity + channel-swap test-time augmentation (about 3x slower)
    overlap: int | None = None  # None = each model's own default
    force_cpu: bool = False
    keep_loaded: int = 1  # models held in memory at once; 1 keeps 8-16 GB machines comfortable
    _seps: dict = field(default_factory=dict, repr=False)

    def _free(self) -> None:
        while len(self._seps) >= max(1, self.keep_loaded):
            old = self._seps.pop(next(iter(self._seps)))
            try:
                old.close()
            except Exception:
                pass
            del old
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _separator(self, name: str):
        if name not in self._seps:
            import msst

            self._free()

            from .models import config_with_overlap, ensure_model

            spec, cfg, ckpt = ensure_model(name, self.model_dir)
            self._seps[name] = msst.Separator(
                config_path=str(config_with_overlap(cfg, self.overlap)),
                checkpoint_path=str(ckpt),
                model_type=spec.model_type,
                force_cpu=self.force_cpu,
                use_tta=self.tta,
                detailed_progress=False,
            )
        return self._seps[name]

    def separate(self, x: np.ndarray, sr: int, model: str) -> dict[str, np.ndarray]:
        if sr != MODEL_SR:
            raise ValueError(f"separation runs at {MODEL_SR} Hz, got {sr}")
        out = self._separator(model).separate(np.ascontiguousarray(x, dtype=np.float32), sample_rate=sr)
        n = x.shape[-1]
        stems = {}
        for k, v in out.items():
            v = np.asarray(v, dtype=np.float32)
            if v.ndim == 1:
                v = np.stack([v, v])
            stems[k.strip().lower()] = fit_length(v, n)
        return stems

    @property
    def device(self) -> str:
        return next((str(s.device) for s in self._seps.values()), "not loaded")


@dataclass
class AudioSeparatorBackend:
    model_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "iso" / "models")
    overlap: int | None = None  # MDXC/RoFormer overlap factor; None = the model's default
    batch_size: int | None = None
    log_level: int = logging.WARNING
    _seps: dict = field(default_factory=dict, repr=False)

    def _separator(self, model: str, out_dir: str):
        from audio_separator.separator import Separator

        key = model
        if key not in self._seps:
            sep = Separator(
                log_level=self.log_level,
                model_file_dir=str(self.model_dir),
                output_dir=out_dir,
                output_format="WAV",
                use_soundfile=True,
                sample_rate=MODEL_SR,
                normalization_threshold=0.9,
                amplification_threshold=0.0,
                mdxc_params={
                    "segment_size": 256,
                    "override_model_segment_size": False,
                    "batch_size": self.batch_size,
                    "overlap": self.overlap,
                    "pitch_shift": 0,
                },
            )
            sep.load_model(model_filename=model)
            self._seps[key] = sep
        sep = self._seps[key]
        sep.output_dir = out_dir
        if getattr(sep, "model_instance", None) is not None:
            sep.model_instance.output_dir = out_dir
        return sep

    def separate(self, x: np.ndarray, sr: int, model: str) -> dict[str, np.ndarray]:
        if sr != MODEL_SR:
            raise ValueError(f"separation runs at {MODEL_SR} Hz, got {sr}")
        n = x.shape[-1]
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        scale = _HEADROOM_PEAK / peak if peak > _HEADROOM_PEAK else 1.0
        with tempfile.TemporaryDirectory(prefix="iso-sep-") as tmp:
            src = Path(tmp) / "input.wav"
            save(src, Audio((x * scale).astype(np.float32), sr), subtype="FLOAT")
            sep = self._separator(model, tmp)
            files = sep.separate(str(src))
            stems: dict[str, np.ndarray] = {}
            for f in files:
                p = Path(f) if Path(f).is_absolute() else Path(tmp) / f
                m = re.search(r"_\(([^)]+)\)", p.name)
                name = (m.group(1) if m else p.stem).strip().lower()
                a = load(p, sr=sr)
                if np.max(np.abs(a.data)) >= 0.899:
                    logging.getLogger("iso").warning(
                        "%s: stem %r touched the normalizer; its level may be off", model, name
                    )
                stems[name] = fit_length(a.data / scale, n)
        return stems


@dataclass
class CachedBackend:
    inner: Backend
    cache_dir: Path
    settings: dict = field(default_factory=dict)  # anything that changes model output

    def separate(self, x: np.ndarray, sr: int, model: str) -> dict[str, np.ndarray]:
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        h.update(f"{sr}|{model}|{json.dumps(self.settings, sort_keys=True)}".encode())
        d = self.cache_dir / h.hexdigest()[:20]
        index = d / "stems.json"
        if index.exists():
            names = json.loads(index.read_text())
            return {k: load(d / f"{i}.wav", sr=sr).data for i, k in enumerate(names)}
        stems = self.inner.separate(x, sr, model)
        d.mkdir(parents=True, exist_ok=True)
        names = list(stems)
        for i, k in enumerate(names):
            save(d / f"{i}.wav", Audio(stems[k], sr))
        index.write_text(json.dumps(names))
        return stems


def pick_stem(stems: dict[str, np.ndarray], wanted: str, model: str) -> np.ndarray:
    """Find a stem by name, tolerating the naming differences between model configs."""
    wanted = wanted.lower()
    aliases = {
        "drums": ["drums", "drum"],
        "dry": ["dry", "noreverb", "no reverb", "no_reverb", "reverb-free"],
        "kick": ["kick", "bass drum", "kickdrum"],
        "snare": ["snare"],
        "toms": ["toms", "tom"],
        "hihat": ["hh", "hihat", "hi-hat", "hihats", "hi-hats", "hat"],
        "ride": ["ride"],
        "crash": ["crash"],
        "cymbals": ["cymbals", "cymbal"],
    }.get(wanted, [wanted])
    for a in aliases:
        if a in stems:
            return stems[a]
    raise KeyError(f"model {model!r} has no {wanted!r} stem; it produced {sorted(stems)}")
