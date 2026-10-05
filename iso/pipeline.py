"""The full chain: song in, layer kit out.

    mix --(ensemble)--> drums_full ----------------------> no_drums = mix - drums_full
                           |
                           +--(dereverb)--> drums_dry --> room = drums_full - drums_dry
                                               |
                                               +--(kit-piece split)--> kick snare toms hihat ride crash
                                                                         |
                                                    hits (sample-accurate) + cross-leakage filter
                                                         |            |              |
                                                   hit-keyed gates   MIDI   aligned trigger layers
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .articulation import hat_open
from .audio import Audio, load, resample, save
from .ensemble import combine, split_by_guides
from .gate import PIECE_GATES, hit_gate
from .layers import default_session, realign, save_session, split_room, write_reaper_project
from .midi import write_midi
from .onsets import PIECE_PARAMS, DetectParams, Hit, detect_hits, drop_buried, drop_cross_leakage
from .separation import MODEL_SR, Backend, pick_stem
from .toms import TOM_NOTES_BY_COUNT, split_toms
from .triggers import PIECE_ALIGN, load_kit_folder, render_triggers

@dataclass
class Config:
    drum_models: list[str] = field(default_factory=lambda: ["bs_roformer_sw"])
    drum_weights: list[float] | None = None  # avg_wave weights for the drum ensemble
    dereverb_model: str | None = "dereverb_mdx23c"
    piece_models: list[str] = field(default_factory=lambda: ["drumsep_5", "drumsep_6"])
    piece_blend: float = 0.0  # weight of the second piece model in kick/snare/toms/hh (0 = first model only)
    pieces_from: str = "dry"  # "dry" or "full"
    tta: bool = False  # polarity + channel-swap test-time augmentation, about 3x slower
    overlap: int | None = None  # separation overlap; None = each model's default
    gate: bool = True
    leakage_filter: bool = True
    buried_db: float = 30.0  # a hit this far under the whole kit at that instant is leakage, not playing
    kit: str | None = None  # folder of one-shots for trigger layers
    trigger_pieces: tuple[str, ...] = ("kick", "snare", "toms")
    dynamics: float = 1.0
    bpm: float | None = None  # None = estimate from the hits
    output_sr: int | None = None  # None = the source file's rate


PRESETS: dict[str, Config] = {
    # One model per stage, no TTA. Good for auditioning on a laptop CPU.
    "fast": Config(dereverb_model="dereverb_mdx23c", piece_models=["drumsep_6"]),
    # The default: best single drum model with TTA, plus both piece models.
    "best": Config(tta=True),
    # Adds SCNet-XL (sharper attacks) to the drum ensemble and blends both piece models.
    "max": Config(
        drum_models=["bs_roformer_sw", "scnet_xl_ihf"],
        drum_weights=[3.0, 1.0],
        piece_blend=0.3,
        tta=True,
        overlap=4,
    ),
}

# File names sort into the order you'd lay them out in a DAW.
FILE_NAMES = {
    "kick": "01_kick",
    "snare": "02_snare",
    "toms": "03_toms",
    "hihat": "04_hihat",
    "ride": "05_ride",
    "crash": "06_crash",
    "trig_kick": "10_trig_kick",
    "trig_snare": "11_trig_snare",
    "trig_toms": "12_trig_toms",
    "drums_dry": "20_drums_dry",
    "room": "21_room",
    "drums_full": "22_drums_full",
    "no_drums": "30_no_drums",
}


def estimate_bpm(hit_times_s: list[float], lo: float = 70.0, hi: float = 190.0) -> float:
    """Tempo from the autocorrelation of the combined hit train. Only sets the MIDI file's grid."""
    if len(hit_times_s) < 8:
        return 120.0
    fps = 200
    n = int(max(hit_times_s) * fps) + fps
    train = np.zeros(n)
    for t in hit_times_s:
        train[int(t * fps)] += 1.0
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    train = np.convolve(train, k, mode="same")
    ac = np.correlate(train, train, mode="full")[n - 1 :]
    lags = np.arange(int(fps * 60 / hi), int(fps * 60 / lo) + 1)
    lags = lags[lags < len(ac)]
    if lags.size == 0:
        return 120.0
    best = lags[int(np.argmax(ac[lags]))]
    return round(60.0 * fps / best, 1)


def _separate_pieces(backend: Backend, x: np.ndarray, cfg: Config, sr: int) -> dict[str, np.ndarray]:
    """Kick/snare/toms/hihat/ride/crash from a drum stem.

    The first piece model leads. When it lumps ride and crash together as
    "cymbals" (the 5-stem model), the second model's ride/crash balance
    splits them, and ride + crash still equals the cymbals exactly.
    """
    runs = [(m, backend.separate(x, sr, m)) for m in cfg.piece_models]
    m0, first = runs[0]
    out: dict[str, np.ndarray] = {}
    for p in ("kick", "snare", "toms", "hihat", "ride", "crash", "cymbals"):
        try:
            out[p] = pick_stem(first, p, m0)
        except KeyError:
            pass
    if len(runs) > 1:
        m1, second = runs[1]
        if cfg.piece_blend > 0:
            for p in ("kick", "snare", "toms", "hihat"):
                if p in out:
                    try:
                        out[p] = combine([out[p], pick_stem(second, p, m1)], sr, "avg_wave", [1 - cfg.piece_blend, cfg.piece_blend])
                    except KeyError:
                        pass
        if "cymbals" in out and "ride" not in out:
            try:
                ride_g, crash_g = pick_stem(second, "ride", m1), pick_stem(second, "crash", m1)
                out["ride"], out["crash"] = split_by_guides(out.pop("cymbals"), ride_g, crash_g, sr)
            except KeyError:
                pass
    return out


def _guard(ref: np.ndarray, est: np.ndarray, name: str, lags: dict) -> tuple[np.ndarray, dict]:
    est, lag = realign(ref, est)
    if lag:
        lags = {**lags, name: lag}
    return est, lags


def run(
    input_path: str | Path,
    out_dir: str | Path,
    backend: Backend,
    config: Config | None = None,
    progress: Callable[[str, float], None] = lambda msg, frac: None,
) -> dict:
    cfg = config or Config()
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "extras").mkdir(exist_ok=True)

    src = load(input_path)
    sr = MODEL_SR
    mix = Audio(resample(src.data, src.sr, sr), sr)
    out_sr = cfg.output_sr or src.sr
    n = mix.samples
    stems: dict[str, np.ndarray] = {}

    progress("Isolating drums", 0.05)
    drums_full = combine(
        [pick_stem(backend.separate(mix.data, sr, m), "drums", m) for m in cfg.drum_models],
        sr,
        "avg_wave",
        cfg.drum_weights,
    )
    drums_full, first_lag = _guard(mix.data, drums_full, "drums_full", {})
    stems["drums_full"] = drums_full
    stems["no_drums"] = (mix.data - drums_full).astype(np.float32)

    lags: dict = dict(first_lag)
    if cfg.dereverb_model:
        progress("Splitting dry kit and room", 0.35)
        dry = pick_stem(backend.separate(drums_full, sr, cfg.dereverb_model), "dry", cfg.dereverb_model)
        dry, lags = _guard(drums_full, dry, "drums_dry", lags)
        stems["drums_dry"] = dry
        stems["room"] = split_room(drums_full, dry)
    else:
        dry = drums_full

    progress("Splitting kit pieces", 0.55)
    piece_src = dry if cfg.pieces_from == "dry" else drums_full
    raw_pieces = _separate_pieces(backend, piece_src, cfg, sr)
    for p in list(raw_pieces):
        raw_pieces[p], lags = _guard(piece_src, raw_pieces[p], p, lags)

    progress("Finding hits", 0.75)
    hits: dict[str, list[Hit]] = {
        p: detect_hits(x, sr, PIECE_PARAMS.get(p, DetectParams())) for p, x in raw_pieces.items()
    }
    hits = drop_buried(hits, piece_src, sr, cfg.buried_db)
    if cfg.leakage_filter:
        hits = drop_cross_leakage(hits, sr)

    for p, x in raw_pieces.items():
        save(out / "extras" / f"raw_{p}.wav", Audio(resample(x, sr, out_sr), out_sr))
        if cfg.gate and p in PIECE_GATES:
            stems[p] = hit_gate(x, sr, [h.sample for h in hits[p]], PIECE_GATES[p])
        else:
            stems[p] = x
    residual = piece_src - sum(raw_pieces.values())
    save(out / "extras" / "pieces_residual.wav", Audio(resample(residual, sr, out_sr), out_sr))

    tom_idx = split_toms(raw_pieces["toms"], sr, hits["toms"]) if hits.get("toms") else []
    trigger_reports = {}
    if cfg.kit:
        progress("Rendering trigger layers", 0.85)
        kit = load_kit_folder(cfg.kit, sr)
        for p in cfg.trigger_pieces:
            if p in kit.pieces and p in raw_pieces and hits.get(p):
                track, rep = render_triggers(
                    raw_pieces[p],
                    sr,
                    hits[p],
                    kit.pieces[p],
                    params=PIECE_ALIGN.get(p),
                    dynamics=cfg.dynamics,
                    variant_of_hit=tom_idx if p == "toms" else None,
                )
                stems[f"trig_{p}"] = track
                trigger_reports[p] = {
                    "polarity": rep.polarity,
                    "median_corr": round(rep.median_corr, 3),
                    "aligned": sum(h["aligned"] for h in rep.hits),
                    "hits": len(rep.hits),
                }

    progress("Writing the layer kit", 0.92)
    files: dict[str, str] = {}
    for name, x in stems.items():
        rel = f"{FILE_NAMES.get(name, name)}.wav"
        save(out / rel, Audio(resample(x, sr, out_sr), out_sr))
        files[name] = rel

    all_times = sorted(h.sample / sr for p in ("kick", "snare", "hihat") for h in hits.get(p, []))
    bpm = cfg.bpm or estimate_bpm(all_times)
    midi_name = f"drums_{bpm:g}bpm.mid"
    midi_hits = {p: hs for p, hs in hits.items() if p not in ("toms", "hihat")}
    if hits.get("hihat"):
        opened = hat_open(raw_pieces["hihat"], sr, hits["hihat"])
        midi_hits["hihat"] = [h for h, o in zip(hits["hihat"], opened) if not o]
        midi_hits["hihat_open"] = [h for h, o in zip(hits["hihat"], opened) if o]
    if tom_idx:
        names = TOM_NOTES_BY_COUNT[max(tom_idx) + 1]
        for h, i in zip(hits["toms"], tom_idx):
            midi_hits.setdefault(names[i], []).append(h)
    write_midi(out / midi_name, midi_hits, sr, bpm=bpm)

    session = default_session(files)
    save_session(out / "session.json", session, out_sr)
    write_reaper_project(out / "layer_kit.rpp", session, out_sr, n / sr)

    report = {
        "iso_version": __version__,
        "input": str(input_path),
        "seconds": round(n / sr, 2),
        "output_sr": out_sr,
        "config": asdict(cfg),
        "bpm": bpm,
        "midi": midi_name,
        "hits": {p: len(h) for p, h in hits.items()},
        "toms_found": (max(tom_idx) + 1) if tom_idx else 0,
        "hihat_open": len(midi_hits.get("hihat_open", [])),
        "realigned_samples": lags,  # should stay empty; non-zero means a model output was shifted
        "triggers": trigger_reports,
        "files": files,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    progress("Done", 1.0)
    return report
