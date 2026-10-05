"""Importing free sample kits into Iso's trigger-kit layout.

Recommended free kits for indie-rock triggers (use the CLOSE-mic SFZ of
each piece; Iso's own room stem supplies the room):

* Wilkinson Naked Drums (CC BY 4.0): github.com/sfzinstruments/WilkinsonAudio.NakedDrums
* DrumGizmo DRSKit (CC BY 4.0): github.com/sfzinstruments/DrumGizmo.DRSKit
* Karoryfer Big Rusty Drums (CC0, dry and characterful): github.com/sfzinstruments/karoryfer.big-rusty-drums
* Hydrogen GMRockKit (GPL, tiny; good for testing): inside github.com/hydrogen-music/hydrogen

Iso's layout is one folder per piece, any number of one-shots inside:

    my_kit/kick/*.wav  my_kit/snare/*.wav  my_kit/toms/*.wav  my_kit/hihat/*.wav ...

Hydrogen drumkits (drumkit.xml + samples) are supported directly. Hundreds
are free, and two ship inside Hydrogen's own repository.
"""

from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Hydrogen instrument names -> Iso piece folders. First match wins.
_PIECE_PATTERNS = [
    ("kick", r"kick|bass ?drum|bd\b"),
    ("snare", r"snare|sd\b"),
    ("toms", r"tom"),
    ("hihat", r"hat|hh\b"),
    ("ride", r"ride"),
    ("crash", r"crash|china|splash"),
]


def piece_for(name: str) -> str | None:
    n = name.lower()
    if re.search(r"side ?stick|rim|clap|cowbell|bell\b|tamb|shaker|cabasa|clave", n):
        return None  # not a kit piece Iso layers
    for piece, pat in _PIECE_PATTERNS:
        if re.search(pat, n):
            return piece
    return None


def _strip_ns(root: ET.Element) -> None:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def import_hydrogen(kit_dir: str | Path, out_dir: str | Path) -> dict[str, list[str]]:
    """Copy a Hydrogen kit's samples into Iso's per-piece folders.

    Only the main articulation of each piece is used: closed hats for
    hihat, the ride bow rather than the bell. Those are what you layer
    under a separated stem.
    """
    kit_dir, out_dir = Path(kit_dir), Path(out_dir)
    root = ET.parse(kit_dir / "drumkit.xml").getroot()
    _strip_ns(root)
    copied: dict[str, list[str]] = {}
    for inst in root.iter("instrument"):
        name = (inst.findtext("name") or "").strip()
        piece = piece_for(name)
        if piece is None:
            continue
        low = name.lower()
        if piece == "hihat":
            if re.search(r"open", low) and not re.search(r"semi|half", low):
                piece = "hihat_open"  # kept apart; used for the test song, not layered by default
            elif not re.search(r"clos", low):
                continue
        if piece == "ride" and "bell" in low:
            continue
        files = [el.text.strip() for el in inst.iter("filename") if el.text]
        dest = out_dir / piece
        dest.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"[^\w]+", "_", name).strip("_").lower()
        for f in files:
            src = kit_dir / f
            if src.exists():
                target = dest / f"{tag}__{Path(f).name}"
                shutil.copy2(src, target)
                copied.setdefault(piece, []).append(target.name)
    return copied


AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff", ".ogg"}


def assemble(out_dir: str | Path, folders: dict[str, list[str | Path]]) -> dict[str, list[str]]:
    """Build a trigger kit from plain folders of one-shots, e.g. a sample pack's close-mic folders.

    `folders` maps a piece to one or more folders. Several tom folders
    (high to low) become separate toms; files are copied as-is (FLAC is
    fine) and velocity layers are worked out at load time from loudness.
    """
    out_dir = Path(out_dir)
    copied: dict[str, list[str]] = {}
    for piece, dirs in folders.items():
        for i, d in enumerate(dirs):
            d = Path(d)
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO_EXTS)
            if not files:
                raise ValueError(f"{d} has no audio files")
            tag = f"tom_{i + 1}" if piece == "toms" else (d.name if len(dirs) > 1 else piece)
            dest = out_dir / piece
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = dest / f"{tag}__{f.name}"
                shutil.copy2(f, target)
                copied.setdefault(piece, []).append(target.name)
    return copied


# General MIDI drum keys -> Iso pieces. Several keys can map to toms; each
# becomes its own drum (variant) under the toms piece.
GM_KEY_PIECE = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare",
    41: "toms", 43: "toms", 45: "toms", 47: "toms", 48: "toms", 50: "toms",
    42: "hihat", 46: "hihat_open",
    51: "ride",
    49: "crash", 57: "crash",
}


def import_sfz(sfz: str | Path, out_dir: str | Path, keys: dict[int, str] | None = None, sr: int = 44100) -> dict[str, list[str]]:
    """Render every velocity layer x round robin of an SFZ kit to one-shots.

    Uses sfizz (pysfizz, BSD-2), so ARIA extensions, $defines, #includes
    and round-robin sequencing all resolve exactly as in a sampler. Pass
    `keys` ({midi_key: piece}) for kits that don't follow the General MIDI
    map, or to import only some keys.
    """
    import hashlib

    import numpy as np
    import pysfizz
    import soundfile as sf

    keys = keys or GM_KEY_PIECE
    synth = pysfizz.Synth(sample_rate=sr, block_size=256)
    if not synth.load_sfz_file(str(sfz)):
        raise ValueError(f"sfizz could not load {sfz}")
    out_dir = Path(out_dir)
    copied: dict[str, list[str]] = {}
    toms_seen = sorted(k for k, p in keys.items() if p == "toms" and synth.get_note_info(k))
    for key, piece in keys.items():
        try:
            regions = synth.get_note_info(key)
        except Exception:
            continue
        if not regions:
            continue
        bands = sorted({(r["lovel"], r["hivel"]) for r in regions})
        per_band = {b: sum(1 for r in regions if (r["lovel"], r["hivel"]) == b) for b in bands}
        dest = out_dir / piece
        dest.mkdir(parents=True, exist_ok=True)
        tag = f"tom_{toms_seen.index(key) + 1}" if piece == "toms" else f"key{key}"
        seen: set[str] = set()
        for (lo, hi), n in per_band.items():
            for _ in range(2 * n):  # round robins cycle inside sfizz; render each one at least once
                y = np.asarray(synth.render_note(key, hi, 0.05, 4.0), dtype=np.float32)
                audible = np.nonzero(np.abs(y).max(axis=0) > 1e-5)[0]
                if audible.size == 0:
                    continue
                y = y[:, : audible[-1] + 1]
                h = hashlib.md5(np.round(y, 5).tobytes()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                name = f"{tag}__v{hi:03d}_{len(seen):02d}.wav"
                sf.write(dest / name, y.T, sr, subtype="FLOAT")
                copied.setdefault(piece, []).append(name)
    return copied
