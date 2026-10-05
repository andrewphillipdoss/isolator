"""End-to-end pipeline test with a fake backend (no model weights needed).

The fake "models" know the synthetic song's true parts, so we can check
the layer-kit invariants exactly: lengths, sums, room = full - dry,
no_drums = mix - drums, MIDI timing and trigger alignment.
"""

import json

import mido
import numpy as np
import soundfile as sf

from iso.ensemble import split_by_guides
from iso.layers import load_session
from iso.pipeline import Config, estimate_bpm, run

from .synth import SR, kick, place, snare, stereo

BPM = 120
BEAT = int(SR * 60 / BPM)
N = 8 * BEAT  # two bars of 4/4
KICKS = [0 * BEAT, 2 * BEAT + BEAT // 2, 4 * BEAT, 6 * BEAT + BEAT // 2]
SNARES = [1 * BEAT, 3 * BEAT, 5 * BEAT, 7 * BEAT]
OFFSET = 2000  # start the song a little in so the first hit isn't at sample 0
KICKS = [k + OFFSET for k in KICKS]
SNARES = [s + OFFSET for s in SNARES]


def parts():
    rng = np.random.default_rng(7)
    k = stereo(place(N, kick(), KICKS))
    s = stereo(place(N, snare(), SNARES))
    hats = stereo(place(N, (rng.standard_normal(4000) * np.exp(-np.arange(4000) / 400)).astype(np.float32) * 0.2,
                        [OFFSET + i * BEAT // 2 for i in range(16)]))
    t = np.arange(N) / SR
    guitar = stereo((0.2 * np.sign(np.sin(2 * np.pi * 110 * t))).astype(np.float32))
    room = 0.3 * (np.roll(k, 900, axis=1) + np.roll(s, 900, axis=1))
    return k, s, hats, guitar, room


class FakeBackend:
    def __init__(self):
        self.k, self.s, self.h, self.g, self.room = parts()
        self.dry = self.k + self.s + self.h
        self.drums = self.dry + self.room
        self.calls = []

    def separate(self, x, sr, model):
        self.calls.append(model)
        assert sr == 44100
        if model == "bs_roformer_sw":
            return {"drums": self.drums.copy(), "guitar": (x - self.drums).astype(np.float32)}
        if model == "dereverb_mdx23c":
            return {"dry": self.dry.copy(), "no dry": self.room.copy()}
        if model == "drumsep_6":
            z = np.zeros_like(x)
            # The ride wasn't played; the model still leaks faint (-60 dB) copies of the snare into it.
            ride_leak = (1e-3 * self.s).astype(np.float32)
            return {"kick": self.k.copy(), "snare": self.s.copy(), "toms": z, "hh": self.h.copy(), "ride": ride_leak, "crash": z}
        raise KeyError(model)


def make_song(tmp_path):
    k, s, h, g, room = parts()
    mix = k + s + h + room + g
    p = tmp_path / "song.wav"
    sf.write(p, mix.T, SR, subtype="FLOAT")
    return p, mix


def make_kit(tmp_path):
    for piece, shot in (("kick", kick(seed=11)), ("snare", snare(seed=12))):
        d = tmp_path / "kit" / piece
        d.mkdir(parents=True)
        for i, g in enumerate((0.4, 1.0)):
            sf.write(d / f"{piece}_{i}.wav", np.stack([g * shot] * 2).T, SR)
    return tmp_path / "kit"


def read(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    return x.T, sr


def test_layer_kit_end_to_end(tmp_path):
    song, mix = make_song(tmp_path)
    be = FakeBackend()
    cfg = Config(piece_models=["drumsep_6"], kit=str(make_kit(tmp_path)), bpm=None)
    out = tmp_path / "out"
    rep = run(song, out, be, cfg)

    full, sr = read(out / "22_drums_full.wav")
    dry, _ = read(out / "20_drums_dry.wav")
    room, _ = read(out / "21_room.wav")
    nod, _ = read(out / "30_no_drums.wav")
    assert sr == SR and full.shape == mix.shape
    # The coherence invariants the whole layering idea rests on.
    np.testing.assert_allclose(dry + room, full, atol=1e-6)
    np.testing.assert_allclose(full + nod, mix, atol=1e-5)

    # Every stem is the source's exact length and starts at sample 0.
    for f in rep["files"].values():
        x, _ = read(out / f)
        assert x.shape == mix.shape, f

    # Hits land on the true attacks; the unplayed ride yields nothing.
    assert rep["hits"]["kick"] == len(KICKS) and rep["hits"]["snare"] == len(SNARES)
    assert rep["hits"]["ride"] == 0 and rep["hits"]["toms"] == 0

    # MIDI: right tempo guess and note times within 1 ms.
    assert abs(rep["bpm"] - BPM) < 1.0 or abs(rep["bpm"] - 2 * BPM) < 2.0
    mf = mido.MidiFile(str(out / rep["midi"]))
    t, on = 0.0, {36: [], 38: [], 42: []}
    for msg in mf:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note].append(t)
    np.testing.assert_allclose(on[36], np.array(KICKS) / SR, atol=0.001)
    np.testing.assert_allclose(on[38], np.array(SNARES) / SR, atol=0.001)

    # Triggers rendered, aligned, positive polarity.
    for p in ("kick", "snare"):
        tr = rep["triggers"][p]
        assert tr["hits"] == 4 and tr["aligned"] == 4 and tr["polarity"] == 1, tr
        x, _ = read(out / rep["files"][f"trig_{p}"])
        assert x.shape == mix.shape and np.max(np.abs(x)) > 0.1

    # Gated kick is silent (floor) well after each hit, untouched at the attack.
    gk, _ = read(out / "01_kick.wav")
    rk, _ = read(out / "extras" / "raw_kick.wav")
    np.testing.assert_allclose(gk[:, KICKS[0] : KICKS[0] + 500], rk[:, KICKS[0] : KICKS[0] + 500], atol=1e-6)

    # Session + REAPER project exist and start from the documented stack.
    tracks, ssr = load_session(out / "session.json")
    names = {t.name: t for t in tracks}
    assert ssr == SR and names["drums_full"].muted and not names["drums_dry"].muted
    assert (out / "layer_kit.rpp").read_text().count("<TRACK") >= len(tracks)
    assert json.loads((out / "report.json").read_text())["files"] == rep["files"]


def test_models_are_cached_across_runs(tmp_path):
    from iso.separation import CachedBackend

    song, _ = make_song(tmp_path)
    be = FakeBackend()
    cached = CachedBackend(be, tmp_path / "cache")
    cfg = Config(piece_models=["drumsep_6"])
    run(song, tmp_path / "a", cached, cfg)
    n = len(be.calls)
    run(song, tmp_path / "b", cached, cfg)
    assert len(be.calls) == n  # second run hit the cache for every model


def test_split_by_guides_is_lossless():
    rng = np.random.default_rng(3)
    src = rng.standard_normal((2, 30000)).astype(np.float32)
    a_guide = rng.standard_normal((2, 30000)).astype(np.float32)
    b_guide = rng.standard_normal((2, 30000)).astype(np.float32)
    a, b = split_by_guides(src, a_guide, b_guide, SR)
    np.testing.assert_allclose(a + b, src, atol=1e-6)
    # A source that matches guide A ends up (almost) entirely in A.
    a2, b2 = split_by_guides(a_guide, a_guide, 0.01 * b_guide, SR)
    assert np.sum(b2**2) < 0.01 * np.sum(a2**2)


def test_estimate_bpm():
    times = [i * 0.5 for i in range(32)]  # 120 BPM quarter notes
    assert abs(estimate_bpm(times) - 120) < 1.0
    assert estimate_bpm([0.1, 0.2]) == 120.0  # too few hits -> default
