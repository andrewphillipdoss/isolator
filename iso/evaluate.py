"""Measuring Iso on a song where the right answer is known.

`make_song` renders an indie-rock style track from a real multi-velocity
acoustic kit and writes the ground truth next to it: every kit piece, the
dry kit, the room, the accompaniment and every hit with its velocity. The
kit is detuned and coloured so it doesn't match the trigger kit. The song
has 8th-note hats, ghost notes, tom fills into crashes, a ride section, a
room, and distorted double-tracked guitars and bass.

`score` compares a layer kit to that truth: SDR per stem, hit F1, onset
timing error, velocity correlation and tom grouping accuracy.

This is the regression yardstick. Any change to models or settings should
keep or improve these numbers. With your own multitracks (drum mics +
band), the same scoring works on real recordings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from .audio import Audio, load, resample, save
from .triggers import load_kit_folder

SR = 44100
PIECES = ("kick", "snare", "toms", "hihat", "ride", "crash")


@dataclass
class Note:
    piece: str
    t: float  # seconds
    vel: int
    tom: int = 0  # 0 = highest tom
    open: bool = False  # open hi-hat


def groove(bars: int, bpm: float, rng: np.random.Generator) -> list[Note]:
    """Indie-rock groove: verse on hats, chorus on ride, fills every 4 bars."""
    beat = 60.0 / bpm
    six = beat / 4
    notes: list[Note] = []
    kick_patterns = [[0, 8, 10], [0, 6, 8], [0, 8, 11, 14], [0, 3, 8, 10]]
    for bar in range(bars):
        t0 = bar * 4 * beat
        chorus = (bar // 4) % 2 == 1
        fill = bar % 4 == 3
        if bar % 4 == 0:
            notes.append(Note("crash", t0, int(rng.integers(105, 125))))
        steps_kick = kick_patterns[int(rng.integers(len(kick_patterns)))]
        for s in steps_kick:
            notes.append(Note("kick", t0 + s * six, int(rng.integers(95, 122))))
        for s in (4, 12):
            if not (fill and s == 12):
                notes.append(Note("snare", t0 + s * six, int(rng.integers(108, 127))))
        for s in (7, 15, 9):  # ghost notes
            if rng.random() < 0.45 and not (fill and s >= 12):
                notes.append(Note("snare", t0 + s * six, int(rng.integers(22, 38))))
        cym = "ride" if chorus else "hihat"
        open_step = 14 if (cym == "hihat" and not fill and rng.random() < 0.5) else None
        for s in range(0, 16, 2):
            if fill and s >= 12:
                continue
            accent = s % 4 == 0
            vel = int(rng.integers(85, 105) if accent else rng.integers(55, 75))
            notes.append(Note(cym, t0 + s * six, vel, open=(s == open_step)))
        if fill:
            for i, s in enumerate(range(12, 16)):
                tom = min(2, i * 3 // 4)
                notes.append(Note("toms", t0 + s * six, int(rng.integers(95, 125)), tom))
                notes.append(Note("toms", t0 + (s + 0.5) * six, int(rng.integers(90, 120)), tom))
    for n in notes:  # humanise: +-4 ms, never before 0
        n.t = max(0.0, n.t + rng.normal(0, 0.002) + 0.25)
    return sorted(notes, key=lambda n: n.t)


TOM_TUNING = (1.25, 0.95, 0.80)  # rack 1 up, rack 2 a touch down, floor down: a typical 2-rack + floor spread


def _detuned_kit(kit_dir: Path, factor: float, sr: int):
    """The 'recorded' kit: the sample kit pitched down and warmed up, so it differs from the triggers.

    Toms are retuned to a realistic spread (several semitones apart). Some
    free kits ship toms tuned almost in unison, which no drummer does.
    """
    kit = load_kit_folder(kit_dir, int(round(sr * factor)))  # read at a different rate == pitch shift
    toms = kit.pieces.get("toms")
    if toms is not None:
        for v, ratio in zip(toms.variants, TOM_TUNING):
            for layer in v.layers:
                for s in layer:
                    s.data = resample(s.data, int(round(sr * ratio)), sr)
                    s.attack = int(round(s.attack / ratio))
    tilt = butter(1, 3500, "lowpass", fs=sr, output="sos")
    for kp in [kit.pieces[p] for p in kit.pieces] + [v for p in kit.pieces.values() for v in p.variants]:
        for layer in kp.layers:
            for s in layer:
                y = np.tanh(1.6 * s.data) / np.tanh(1.6)
                s.data = (0.6 * y + 0.4 * sosfilt(tilt, y, axis=-1)).astype(np.float32)
    return kit


def _room_ir(sr: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal((2, n)) * np.exp(-6.9 * t / rt60)
    ir[:, : int(0.008 * sr)] = 0.0  # pre-delay
    for d, g in ((0.011, 0.5), (0.017, 0.4), (0.023, 0.35), (0.031, 0.3), (0.042, 0.25)):
        ir[0, int(d * sr)] += g
        ir[1, int((d + 0.003) * sr)] += g
    ir = sosfilt(butter(2, 6000, "lowpass", fs=sr, output="sos"), ir, axis=-1)
    return (ir / np.sqrt(np.sum(ir**2) / 2)).astype(np.float32)


def _guitars(n: int, bpm: float, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / sr
    beat = 60.0 / bpm
    roots = [82.41, 65.41, 98.0, 73.42]  # E2 C2 G2 D2
    gtr = np.zeros((2, n))
    bass = np.zeros(n)
    eighth = beat / 2
    k = 0
    for start in np.arange(0.25, n / sr, eighth):
        root = roots[int(start // (4 * beat)) % 4]
        a, b = int(start * sr), min(n, int((start + eighth) * sr))
        tt = t[a:b] - start
        env = np.exp(-tt / 0.35) * (1 - np.exp(-tt / 0.002))
        for ch, det in ((0, 1.0), (1, 1.004)):
            sig = sum(((tt * f * det) % 1.0) * 2 - 1 for f in (root * 2, root * 3, root * 4))
            gtr[ch, a:b] += env * np.tanh(4.0 * sig)
        bass[a:b] += np.exp(-tt / 0.5) * (np.sin(2 * np.pi * root * tt) + 0.3 * (((tt * root) % 1.0) * 2 - 1))
        k += 1
    cab = butter(4, [90, 4500], "bandpass", fs=sr, output="sos")
    gtr = sosfilt(cab, gtr, axis=-1)
    bass = sosfilt(butter(4, 900, "lowpass", fs=sr, output="sos"), bass)
    return gtr.astype(np.float32), np.stack([bass, bass]).astype(np.float32)


def make_song(kit_dir: str | Path, out_dir: str | Path, bars: int = 16, bpm: float = 124.0, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    truth = out / "truth"
    truth.mkdir(parents=True, exist_ok=True)
    kit = _detuned_kit(Path(kit_dir), 0.94, SR)
    notes = groove(bars, bpm, rng)
    n = int((notes[-1].t + 2.0) * SR)
    pan = {"kick": (1, 1), "snare": (1, 0.9), "hihat": (0.7, 1), "ride": (1, 0.7), "crash": (0.85, 1), "toms": (1, 1)}
    stems = {p: np.zeros((2, n), dtype=np.float32) for p in PIECES}
    rr = 0
    for note in notes:
        kp = kit.pieces["hihat_open"] if note.open and "hihat_open" in kit.pieces else kit.pieces[note.piece]
        if note.piece == "toms" and kp.variants:
            kp = kp.variants[min(note.tom, len(kp.variants) - 1)]
        shot, _ = kp.pick(note.vel, rr)
        rr += 1
        gain = (note.vel / 127.0) ** 1.6
        start = int(note.t * SR) - shot.attack
        a, b = max(0, start), min(n, start + shot.data.shape[1])
        lr = np.array(pan[note.piece], dtype=np.float32)[:, None]
        if note.piece == "toms":
            lr = np.array([[1.0 - 0.3 * note.tom], [0.7 + 0.3 * note.tom]], dtype=np.float32)
        stems[note.piece][:, a:b] += gain * lr * shot.data[:, a - start : b - start]
    levels = {"kick": 1.0, "snare": 0.9, "toms": 0.8, "hihat": 0.35, "ride": 0.35, "crash": 0.4}
    for p in stems:
        stems[p] *= levels[p]
    dry = sum(stems.values())
    ir = _room_ir(SR, 0.7, rng)
    room = np.stack([fftconvolve(dry[c], ir[c])[:n] for c in range(2)]).astype(np.float32) * 0.2
    gtr, bass = _guitars(n, bpm, SR, rng)
    band = 0.35 * gtr + 0.45 * bass
    full = dry + room
    mix = full + band
    scale = 0.89 / float(np.max(np.abs(mix)))
    for name, x in {**stems, "drums_dry": dry, "room": room, "drums_full": full, "no_drums": band}.items():
        save(truth / f"{name}.wav", Audio((x * scale).astype(np.float32), SR))
    save(out / "song.wav", Audio((mix * scale).astype(np.float32), SR))
    notes_json = [{"piece": x.piece, "t": round(x.t, 6), "vel": x.vel, "tom": x.tom, "open": x.open} for x in notes]
    (truth / "notes.json").write_text(json.dumps({"bpm": bpm, "notes": notes_json}, indent=1))
    return out / "song.wav"


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n].astype(np.float64), est[..., :n].astype(np.float64)
    return float(10 * np.log10((np.sum(ref**2) + 1e-9) / (np.sum((ref - est) ** 2) + 1e-9)))


def match_hits(true_t: list[float], est_t: list[float], tol: float = 0.015) -> tuple[list[tuple[int, int]], float, float, float]:
    """Greedy one-to-one matching. Returns (pairs, precision, recall, f1)."""
    pairs, used = [], set()
    for i, t in enumerate(true_t):
        best, bd = None, tol
        for j, e in enumerate(est_t):
            if j not in used and abs(e - t) <= bd:
                best, bd = j, abs(e - t)
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    p = len(pairs) / len(est_t) if est_t else (1.0 if not true_t else 0.0)
    r = len(pairs) / len(true_t) if true_t else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return pairs, p, r, f


def score(kit_dir: str | Path, truth_dir: str | Path) -> dict:
    import mido

    kit_dir, truth_dir = Path(kit_dir), Path(truth_dir)
    report = json.loads((kit_dir / "report.json").read_text())
    files = report["files"]

    def est(name: str) -> np.ndarray | None:
        if name in files:
            return load(kit_dir / files[name], sr=SR).data
        raw = kit_dir / "extras" / f"raw_{name}.wav"
        return load(raw, sr=SR).data if raw.exists() else None

    def truth(name: str) -> np.ndarray:
        return load(truth_dir / f"{name}.wav", sr=SR).data

    res: dict = {"sdr": {}, "sdr_raw_pieces": {}, "hits": {}}
    for name in ("drums_full", "drums_dry", "room", "no_drums"):
        e = est(name)
        if e is not None:
            res["sdr"][name] = round(sdr(truth(name), e), 2)
    for p in PIECES:
        raw = kit_dir / "extras" / f"raw_{p}.wav"
        if raw.exists():
            res["sdr_raw_pieces"][p] = round(sdr(truth(p), load(raw, sr=SR).data), 2)
        if p in files:
            res["sdr"][p] = round(sdr(truth(p), load(kit_dir / files[p], sr=SR).data), 2)

    notes = json.loads((truth_dir / "notes.json").read_text())["notes"]
    mf = mido.MidiFile(str(kit_dir / report["midi"]))
    from .midi import GM_NOTES

    note_piece = {}
    for k, v in GM_NOTES.items():
        note_piece.setdefault(v, "toms" if k.startswith("tom") else "hihat" if k.startswith("hihat") else k)
    est_hits: dict[str, list[tuple[float, int, int]]] = {}
    t = 0.0
    for msg in mf:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0 and msg.note in note_piece:
            est_hits.setdefault(note_piece[msg.note], []).append((t, msg.velocity, msg.note))
    for p in PIECES:
        tn = [x for x in notes if x["piece"] == p]
        eh = est_hits.get(p, [])
        pairs, prec, rec, f1 = match_hits([x["t"] for x in tn], [e[0] for e in eh])
        errs = [abs(eh[j][0] - tn[i]["t"]) * 1000 for i, j in pairs]
        vel_r = float("nan")
        if len(pairs) > 3:
            tv = np.array([tn[i]["vel"] for i, _ in pairs], float)
            ev = np.array([eh[j][1] for _, j in pairs], float)
            if tv.std() > 0 and ev.std() > 0:
                vel_r = float(np.corrcoef(tv, ev)[0, 1])
        row = {
            "true": len(tn),
            "found": len(eh),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "onset_err_ms_median": round(float(np.median(errs)), 2) if errs else None,
            "velocity_r": round(vel_r, 3) if np.isfinite(vel_r) else None,
        }
        if p == "hihat" and pairs and any(x.get("open") for x in tn):
            row["open_hat_acc"] = round(sum(1 for i, j in pairs if (eh[j][2] == 46) == bool(tn[i].get("open"))) / len(pairs), 3)
            row["open_hats"] = f"{sum(1 for i, j in pairs if eh[j][2] == 46 and tn[i].get('open'))}/{sum(1 for x in tn if x.get('open'))} found"
        if p == "toms" and pairs:
            # Higher GM tom note = higher tom; compare the rank order with the truth.
            order = {48: 0, 50: 0, 45: 1, 47: 1, 41: 2, 43: 2}
            ok = sum(1 for i, j in pairs if order.get(eh[j][2], 1) == tn[i]["tom"])
            row["tom_grouping_acc"] = round(ok / len(pairs), 3)
        res["hits"][p] = row
    res["triggers"] = report.get("triggers", {})
    res["config"] = {k: report["config"][k] for k in ("drum_models", "dereverb_model", "piece_models", "pieces_from", "tta")}
    return res


def format_score(res: dict) -> str:
    lines = ["SDR (dB, higher is better)"]
    for k, v in res["sdr"].items():
        raw = res["sdr_raw_pieces"].get(k)
        lines.append(f"  {k:11s} {v:6.2f}" + (f"   (before gate {raw:6.2f})" if raw is not None and raw != v else ""))
    lines.append("Hits (F1 / onset error / velocity r)")
    for p, r in res["hits"].items():
        extra = f"  toms grouped right {r['tom_grouping_acc']:.0%}" if "tom_grouping_acc" in r else ""
        if "open_hat_acc" in r:
            extra += f"  open/closed right {r['open_hat_acc']:.0%} (open {r['open_hats']})"
        lines.append(
            f"  {p:6s} F1 {r['f1']:.2f}  P {r['precision']:.2f} R {r['recall']:.2f}  "
            f"({r['found']}/{r['true']})  err {r['onset_err_ms_median']} ms  vel r {r['velocity_r']}{extra}"
        )
    return "\n".join(lines)
