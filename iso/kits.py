"""Importing free sample kits into Iso's trigger-kit layout.

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
