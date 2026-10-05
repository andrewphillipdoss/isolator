"""Command line: `iso split song.mp3`, `iso serve`, `iso bounce kit_dir`, `iso doctor`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

DEFAULT_OUT = Path("iso_out")
DEFAULT_MODELS = Path.home() / ".cache" / "iso" / "models"


def make_backend(models_dir: str | Path, tta: bool, overlap: int | None, cpu: bool = False):
    from .separation import CachedBackend, MSSTBackend

    inner = MSSTBackend(model_dir=Path(models_dir), tta=tta, overlap=overlap, force_cpu=cpu)
    return CachedBackend(inner, Path(models_dir).parent / "cache", {"tta": tta, "overlap": overlap, "backend": "msst"})


def cmd_split(args) -> int:
    from .pipeline import PRESETS, run

    cfg = PRESETS[args.preset]
    cfg = replace(
        cfg,
        kit=args.kit,
        gate=not args.no_gate,
        bpm=args.bpm,
        dynamics=args.dynamics,
        pieces_from=args.pieces_from or cfg.pieces_from,
        overlap=args.overlap if args.overlap is not None else cfg.overlap,
        tta=cfg.tta and not args.no_tta,
    )
    backend = make_backend(args.models_dir, cfg.tta, cfg.overlap, args.cpu)
    for song in args.songs:
        song = Path(song)
        out = Path(args.out) / song.stem
        print(f"\n{song.name} -> {out}/")
        report = run(song, out, backend, cfg, progress=lambda m, f: print(f"  [{f:4.0%}] {m}", flush=True))
        hits = ", ".join(f"{k} {v}" for k, v in report["hits"].items())
        print(f"  hits: {hits}")
        print(f"  MIDI: {report['midi']} (set your DAW to {report['bpm']:g} BPM before importing)")
        for p, t in report["triggers"].items():
            note = ""
            if t["hits"] and t["aligned"] < 0.5 * t["hits"]:
                note = "  <- this sample doesn't resemble the recorded drum much; check polarity by ear"
            print(f"  trig_{p}: {t['aligned']}/{t['hits']} hits phase-aligned, polarity {'+' if t['polarity'] > 0 else '-'}{note}")
        print(f"  done in {report['elapsed_s']}s. Open layer_kit.rpp in REAPER or drag the WAVs into any DAW at bar 1.")
    return 0


def cmd_debleed(args) -> int:
    from .multitrack import Mic, MultitrackConfig, run_multitrack

    mics = []
    for role in ("kick", "snare", "tom", "hihat", "ride", "oh", "room"):
        for path in getattr(args, role) or []:
            mics.append(Mic(role, Path(path)))
    if not mics:
        print("give at least one mic, e.g. --kick kick.wav --snare snare.wav --oh oh.wav", file=sys.stderr)
        return 1
    cfg = MultitrackConfig(gate=not args.no_gate, kit=args.kit, bpm=args.bpm)
    backend = make_backend(args.models_dir, tta=not args.no_tta, overlap=args.overlap, cpu=args.cpu)
    rep = run_multitrack(mics, Path(args.out), backend, cfg)
    for m in rep["mics"]:
        hits = f", {m['hits']} hits" if "hits" in m else ""
        print(f"  {m['role']:5s} {Path(m['source']).name}{hits} -> {', '.join(m['files'].values())}")
    if rep["midi"]:
        print(f"  MIDI: {rep['midi']}")
    return 0


def cmd_bounce(args) -> int:
    from .audio import save
    from .layers import load_session, mixdown

    kit = Path(args.kit_dir)
    tracks, _ = load_session(kit / "session.json")
    a = mixdown(kit, tracks, include_song=args.with_song)
    out = save(kit / ("bounce_with_song.wav" if args.with_song else "bounce_drums.wav"), a)
    print(out)
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(
        host=args.host,
        port=args.port,
        out_root=Path(args.out),
        backend_factory=lambda cfg: make_backend(args.models_dir, cfg.tta, cfg.overlap, args.cpu),
    )
    return 0


def cmd_kit_import(args) -> int:
    from .kits import import_hydrogen

    src = Path(args.source)
    if not (src / "drumkit.xml").exists():
        print(f"{src} has no drumkit.xml (only Hydrogen kits are supported so far)", file=sys.stderr)
        return 1
    dest = Path(args.dest) if args.dest else Path(args.kits_dir) / src.name
    copied = import_hydrogen(src, dest)
    for piece, files in sorted(copied.items()):
        print(f"  {piece:6s} {len(files)} samples")
    print(f"Imported to {dest}. Use it with: iso split song.mp3 --kit {dest}")
    return 0


def cmd_eval_make(args) -> int:
    from .evaluate import make_song

    song = make_song(args.kit, args.out, bars=args.bars, bpm=args.bpm, seed=args.seed)
    print(f"{song}\nground truth in {Path(args.out) / 'truth'}")
    return 0


def cmd_eval_score(args) -> int:
    from .evaluate import format_score, score

    res = score(args.kit_dir, args.truth)
    (Path(args.kit_dir) / "score.json").write_text(json.dumps(res, indent=2))
    print(format_score(res))
    return 0


def cmd_doctor(args) -> int:
    info = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as e:  # torch missing or broken
        info["torch"] = f"unavailable ({e.__class__.__name__}: {e})"
    try:
        import msst

        info["msst"] = msst.__version__
    except Exception as e:
        info["msst"] = f"unavailable ({e})"
    info["models_dir"] = str(args.models_dir)
    print(json.dumps(info, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="iso", description="Drum isolation and layer kits for mixing.")
    ap.add_argument("--models-dir", default=str(DEFAULT_MODELS))
    ap.add_argument("--overlap", type=int, default=None, help="separation overlap (higher = slower, smoother)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="make a layer kit from one or more songs")
    s.add_argument("songs", nargs="+")
    s.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    s.add_argument("--preset", choices=["fast", "best", "max"], default="best")
    s.add_argument("--no-tta", action="store_true", help="skip test-time augmentation (3x faster, slightly worse)")
    s.add_argument("--kit", help="folder of one-shots for trigger layers (kit/kick/*.wav, kit/snare/*.wav, ...)")
    s.add_argument("--no-gate", action="store_true", help="skip hit-keyed gating of kick/snare/toms")
    s.add_argument("--pieces-from", choices=["dry", "full"], help="split pieces from the dry kit or the full kit")
    s.add_argument("--bpm", type=float, help="tempo for the MIDI file (default: estimated)")
    s.add_argument("--dynamics", type=float, default=1.0, help="trigger dynamics: 1 = follow the drummer, 0 = even")
    s.set_defaults(fn=cmd_split)

    m = sub.add_parser("debleed", help="multitrack mode: remove bleed from real drum mic tracks")
    m.add_argument("--kick", action="append", help="kick mic (repeat for in/out)")
    m.add_argument("--snare", action="append", help="snare mic (repeat for top/bottom)")
    m.add_argument("--tom", action="append", help="tom mic; give them high to low")
    m.add_argument("--hihat", action="append")
    m.add_argument("--ride", action="append")
    m.add_argument("--oh", action="append", help="overhead (split into cymbals and shells)")
    m.add_argument("--room", action="append", help="room mic (passed through, aligned)")
    m.add_argument("-o", "--out", default=str(DEFAULT_OUT / "multitrack"))
    m.add_argument("--kit", help="trigger kit folder")
    m.add_argument("--bpm", type=float, default=120.0, help="tempo for the MIDI file")
    m.add_argument("--no-gate", action="store_true")
    m.add_argument("--no-tta", action="store_true")
    m.set_defaults(fn=cmd_debleed)

    b = sub.add_parser("bounce", help="mix a layer kit down using its session.json levels")
    b.add_argument("kit_dir")
    b.add_argument("--with-song", action="store_true")
    b.set_defaults(fn=cmd_bounce)

    v = sub.add_parser("serve", help="open the browser UI")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("-o", "--out", default=str(DEFAULT_OUT))
    v.set_defaults(fn=cmd_serve)

    k = sub.add_parser("kit", help="manage trigger kits")
    ksub = k.add_subparsers(dest="kit_cmd", required=True)
    ki = ksub.add_parser("import", help="import a Hydrogen drumkit folder (drumkit.xml + samples)")
    ki.add_argument("source")
    ki.add_argument("--dest", help="output folder (default: ~/.cache/iso/kits/<name>, where the UI looks)")
    ki.add_argument("--kits-dir", default=str(Path.home() / ".cache" / "iso" / "kits"))
    ki.set_defaults(fn=cmd_kit_import)

    e = sub.add_parser("eval", help="measure quality on a song with known ground truth")
    esub = e.add_subparsers(dest="eval_cmd", required=True)
    em = esub.add_parser("make", help="render a test song + ground truth from a sample kit")
    em.add_argument("--kit", required=True, help="Iso kit folder (e.g. from `iso kit import`)")
    em.add_argument("-o", "--out", default="iso_eval")
    em.add_argument("--bars", type=int, default=16)
    em.add_argument("--bpm", type=float, default=124.0)
    em.add_argument("--seed", type=int, default=0)
    em.set_defaults(fn=cmd_eval_make)
    es = esub.add_parser("score", help="score a layer kit against the truth folder")
    es.add_argument("kit_dir")
    es.add_argument("truth")
    es.set_defaults(fn=cmd_eval_score)

    d = sub.add_parser("doctor", help="check PyTorch / GPU / model setup")
    d.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
