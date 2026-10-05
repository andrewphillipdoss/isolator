"""Model registry: which checkpoints Iso uses and where to fetch them.

Every entry points at a GitHub-hosted mirror (GitHub release or LFS). The
original hosts of several community checkpoints have disappeared:
jarredou's GitHub and Hugging Face accounts are gone. Weights download to
~/.cache/iso/models on first use and are never committed to this repo.

Licenses: the community checkpoints (BS-RoFormer SW, the DrumSep models,
the dereverb model) don't state one, and several were trained on
non-commercial datasets. Iso is for personal use; don't redistribute the
weights.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

NOMAD = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"
MSST_REL = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download"
MSST_RAW = "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main"
DRUM2MIDI = "EverlastEngineering/DrumToMIDI/main/mdx_models"


@dataclass(frozen=True)
class ModelSpec:
    model_type: str  # MSST model_type
    config_url: str
    ckpt_url: str
    stems: tuple[str, ...]
    note: str = ""
    ckpt_size: int | None = None  # bytes, to catch truncated downloads
    sha256: str | None = None
    extra: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    # Mix -> drums
    "bs_roformer_sw": ModelSpec(
        "bs_roformer",
        f"{NOMAD}/BS-Roformer-SW.yaml",
        f"{NOMAD}/BS-Roformer-SW.ckpt",
        ("bass", "drums", "other", "vocals", "guitar", "piano"),
        "Best public drums model (~14.1 dB SDR, MVSep Multisong). Has a guitar stem, which helps on rock.",
        ckpt_size=699412152,
    ),
    "scnet_xl_ihf": ModelSpec(
        "scnet",
        f"{MSST_REL}/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml",
        f"{MSST_REL}/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt",
        ("drums", "bass", "other", "vocals"),
        "Drums 11.6 dB (Multisong). The SCNet-XL family keeps attacks sharper than RoFormers; ensemble partner.",
        ckpt_size=214063778,
    ),
    "htdemucs_ft_drums": ModelSpec(
        "htdemucs",
        f"{MSST_RAW}/configs/config_musdb18_htdemucs.yaml",
        f"{NOMAD}/f7e0c4bc-ba3fe64a.th",
        ("drums", "bass", "other", "vocals"),
        "Demucs v4 drums specialist (MIT). Fast fallback for CPU-only machines, ~3 dB below SW.",
        ckpt_size=84141271,
    ),
    # Drums -> dry + reverb
    "dereverb_mdx23c": ModelSpec(
        "mdx23c",
        f"{NOMAD}/config_dereverb_mdx23c.yaml",
        f"{NOMAD}/MDX23C-De-Reverb-aufr33-jarredou.ckpt",
        ("dry", "no dry"),
        "General-purpose (not vocal-only) dereverb by aufr33 & jarredou.",
    ),
    # Drums -> pieces
    "drumsep_5": ModelSpec(
        "mdx23c",
        f"https://raw.githubusercontent.com/{DRUM2MIDI}/config_mdx23c.yaml",
        f"https://media.githubusercontent.com/media/{DRUM2MIDI}/drumsep_5stems_mdx23c_jarredou.ckpt",
        ("kick", "snare", "toms", "hh", "cymbals"),
        "jarredou 5-stem DrumSep: best public piece SDR (kick 16.7, snare 11.5, toms 12.3).",
    ),
    "drumsep_6": ModelSpec(
        "mdx23c",
        f"{NOMAD}/config_drumsep_mdx23c.yaml",
        f"{NOMAD}/MDX23C-DrumSep-aufr33-jarredou.ckpt",
        ("kick", "snare", "toms", "hh", "ride", "crash"),
        "aufr33 & jarredou 6-stem DrumSep. Iso uses it mainly to split cymbals into ride and crash.",
        ckpt_size=437652699,
    ),
}


def _download(url: str, dest: Path, expect_size: int | None = None) -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "iso/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty():
                sys.stderr.write(f"\r  downloading {dest.name}: {done / total:5.1%}")
        if total and sys.stderr.isatty():
            sys.stderr.write("\n")
    if expect_size and tmp.stat().st_size != expect_size:
        tmp.unlink()
        raise IOError(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {expect_size}")
    shutil.move(tmp, dest)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(name: str, model_dir: Path) -> tuple[ModelSpec, Path, Path]:
    """Download (once) and return (spec, config_path, checkpoint_path)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    spec = MODELS[name]
    d = Path(model_dir) / name
    cfg = d / Path(spec.config_url).name
    ckpt = d / Path(spec.ckpt_url).name
    if not cfg.exists():
        _download(spec.config_url, cfg)
    if not ckpt.exists():
        _download(spec.ckpt_url, ckpt, spec.ckpt_size)
        if spec.sha256 and sha256(ckpt) != spec.sha256:
            ckpt.unlink()
            raise IOError(f"{ckpt.name}: checksum mismatch")
    return spec, cfg, ckpt


def config_with_overlap(cfg: Path, overlap: int | None) -> Path:
    """Copy of a model config with inference.num_overlap replaced (text edit, keeps custom YAML tags)."""
    if overlap is None:
        return cfg
    text = cfg.read_text()
    lines, hit = [], False
    for line in text.splitlines():
        if line.strip().startswith("num_overlap:"):
            indent = line[: len(line) - len(line.lstrip())]
            line, hit = f"{indent}num_overlap: {int(overlap)}", True
        lines.append(line)
    if not hit:
        return cfg
    out = cfg.with_name(f"{cfg.stem}.overlap{int(overlap)}{cfg.suffix}")
    out.write_text("\n".join(lines) + "\n")
    return out
