"""Multitrack mode: de-bleed real drum mics.

If you recorded the kit with several mics, each close mic hears the whole
kit: the snare in the kick mic, hats in the snare mic, and so on. The
kit-piece models already know how to pull one drum out of a kit sound, so
for each close mic we split what it hears and keep only its own drum.
Then we gate it using that drum's hits. Overheads are split into a
cymbals-only version (shells removed) and the shells they carried, so you
choose how much kit image to keep. Room mics are passed through.

Every output stays sample-aligned with its source mic, so your mic
time-alignment and phase relationships are untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import Audio, load, resample, save
from .gate import PIECE_GATES, hit_gate
from .midi import write_midi
from .onsets import PIECE_PARAMS, DetectParams, detect_hits, drop_buried
from .separation import MODEL_SR, Backend, pick_stem
from .toms import TOM_NOTES_BY_COUNT
from .triggers import PIECE_ALIGN, load_kit_folder, render_triggers

ROLE_PIECE = {"kick": "kick", "snare": "snare", "tom": "toms", "hihat": "hihat", "ride": "ride"}


@dataclass
class Mic:
    role: str  # kick | snare | tom | hihat | ride | oh | room
    path: Path
    name: str = ""


@dataclass
class MultitrackConfig:
    piece_model: str = "drumsep_6"
    gate: bool = True
    kit: str | None = None
    trigger_roles: tuple[str, ...] = ("kick", "snare", "tom")
    dynamics: float = 1.0
    bpm: float = 120.0
    extra: dict = field(default_factory=dict)


def run_multitrack(mics: list[Mic], out_dir: str | Path, backend: Backend, cfg: MultitrackConfig | None = None) -> dict:
    cfg = cfg or MultitrackConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    srcs = [load(m.path) for m in mics]
    lengths = {a.samples for a in srcs}
    rates = {a.sr for a in srcs}
    if len(rates) != 1 or max(lengths) - min(lengths) > 1:
        raise ValueError("all mic files must come from one session: same sample rate and length")
    src_sr = rates.pop()
    sr = MODEL_SR
    report: dict = {"mics": [], "midi": None}
    midi_hits: dict[str, list] = {}
    kit = load_kit_folder(cfg.kit, sr) if cfg.kit else None
    tom_mics = [m for m in mics if m.role == "tom"]
    tom_names = TOM_NOTES_BY_COUNT.get(len(tom_mics), ["toms"] * len(tom_mics))

    for m, a in zip(mics, srcs):
        label = m.name or f"{m.role}_{Path(m.path).stem}"
        x = resample(a.data, a.sr, sr)
        n_src = a.samples

        def put(name: str, y: np.ndarray) -> str:
            rel = f"{label}_{name}.wav"
            save(out / rel, Audio(resample(y, sr, src_sr)[:, :n_src], src_sr))
            return rel

        entry = {"role": m.role, "source": str(m.path), "files": {}}
        if m.role == "room":
            entry["files"]["as_recorded"] = put("room", x)
        elif m.role == "oh":
            stems = backend.separate(x, sr, cfg.piece_model)
            cym = sum(pick_stem(stems, p, cfg.piece_model) for p in ("hihat", "ride", "crash"))
            entry["files"]["cymbals"] = put("cymbals", cym)
            entry["files"]["shells"] = put("shells", x - cym)
        elif m.role in ROLE_PIECE:
            piece = ROLE_PIECE[m.role]
            stems = backend.separate(x, sr, cfg.piece_model)
            own = pick_stem(stems, piece, cfg.piece_model)
            hits = detect_hits(own, sr, PIECE_PARAMS.get(piece, DetectParams()))
            hits = drop_buried({piece: hits}, x, sr)[piece]
            entry["hits"] = len(hits)
            entry["files"]["raw"] = put("debled_raw", own)
            clean = hit_gate(own, sr, [h.sample for h in hits], PIECE_GATES[piece]) if cfg.gate and piece in PIECE_GATES else own
            entry["files"]["clean"] = put("clean", clean)
            entry["files"]["bleed_removed"] = put("bleed_removed", x - own)
            midi_key = tom_names[tom_mics.index(m)] if m.role == "tom" else piece
            midi_hits.setdefault(midi_key, []).extend(hits)
            if kit and m.role in cfg.trigger_roles and piece in kit.pieces and hits:
                trig, rep = render_triggers(own, sr, hits, kit.pieces[piece], params=PIECE_ALIGN.get(piece), dynamics=cfg.dynamics)
                entry["files"]["trigger"] = put("trigger", trig)
                entry["trigger"] = {"polarity": rep.polarity, "aligned": sum(h["aligned"] for h in rep.hits), "hits": len(rep.hits)}
        else:
            raise ValueError(f"unknown mic role {m.role!r}")
        report["mics"].append(entry)

    for p in midi_hits:
        midi_hits[p].sort(key=lambda h: h.sample)
    if midi_hits:
        name = f"drums_{cfg.bpm:g}bpm.mid"
        write_midi(out / name, midi_hits, sr, bpm=cfg.bpm)
        report["midi"] = name
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
