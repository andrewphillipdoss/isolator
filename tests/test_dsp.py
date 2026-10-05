import mido
import numpy as np
import pytest

from iso.gate import GateParams, hit_gate
from iso.layers import default_session, split_room, write_reaper_project
from iso.midi import write_midi
from iso.onsets import DetectParams, Hit, detect_hits, drop_cross_leakage
from iso.triggers import KitPiece, OneShot, render_triggers

from .synth import SR, kick, place, snare, stereo

STARTS = [11025, 33075, 55125, 66150, 88200, 110250, 121275, 143325]  # deliberately uneven spacing
GAINS = [1.0, 0.5, 0.9, 0.2, 1.0, 0.7, 0.35, 0.95]
N = 4 * SR


def kick_stem(noise_db=-70.0, seed=3):
    rng = np.random.default_rng(seed)
    x = place(N, kick(), STARTS, GAINS)
    x += rng.standard_normal(N).astype(np.float32) * 10 ** (noise_db / 20)
    return stereo(x)


def test_detect_hits_finds_every_attack_to_the_sample():
    hits = detect_hits(kick_stem(), SR, DetectParams(rel_db=30))
    got = [h.sample for h in hits]
    assert len(got) == len(STARTS), got
    err_ms = np.abs(np.array(got) - np.array(STARTS)) / SR * 1000
    assert err_ms.max() <= 0.5, err_ms


def test_velocity_follows_hit_level():
    hits = detect_hits(kick_stem(), SR, DetectParams(rel_db=30))
    order_by_gain = np.argsort(GAINS)
    vels = np.array([h.velocity for h in hits])
    assert np.all(np.diff(vels[order_by_gain]) >= 0), vels
    assert vels.max() == 127


def test_quiet_leakage_below_rel_db_is_dropped():
    x = place(N, kick(), [11025, 55125], [1.0, 1.0]) + place(N, kick(), [99225], [0.005])  # -46 dB ghost
    hits = detect_hits(stereo(x), SR, DetectParams(rel_db=30))
    assert [h.sample for h in hits] == pytest.approx([11025, 55125], abs=22)


def test_cross_leakage_drops_faint_copy_but_keeps_real_coincident_hits():
    hits = {
        "kick": [Hit(1000, -6.0, 120), Hit(50000, -40.0, 20), Hit(90000, -6.0, 120)],
        "snare": [Hit(50010, -5.0, 127), Hit(90005, -7.0, 118)],
    }
    out = drop_cross_leakage(hits, SR)
    assert [h.sample for h in out["kick"]] == [1000, 90000]  # the -40 dB one was snare leakage
    assert len(out["snare"]) == 2  # a kick + snare together are both real


def test_gate_keeps_transients_and_kills_bleed_between_hits():
    rng = np.random.default_rng(0)
    hits_x = place(N, kick(), STARTS, [1.0] * len(STARTS))
    bleed = rng.standard_normal(N).astype(np.float32) * 0.01  # -40 dB wash between hits
    x = stereo(hits_x + bleed)
    p = GateParams(floor_db=-80.0)
    y = hit_gate(x, SR, STARTS, p)
    # Attack and the first 15 ms of every hit pass untouched.
    for s in STARTS:
        np.testing.assert_allclose(y[:, s : s + int(0.015 * SR)], x[:, s : s + int(0.015 * SR)], rtol=1e-5, atol=1e-6)
    # In a gap between hits, after release, bleed is down by roughly the floor.
    a, b = STARTS[-1] + int(0.6 * SR), N - 100
    reduction_db = 20 * np.log10(np.std(y[:, a:b]) / np.std(x[:, a:b]))
    assert reduction_db < -70, reduction_db


@pytest.mark.parametrize("flip", [False, True])
def test_same_drum_trigger_lands_on_the_exact_sample(flip):
    shot_x = kick(seed=5, f0=110.0, f1=52.0, decay=0.2)
    true_starts = [s + d for s, d in zip(STARTS, [0, 7, -5, 13, 2, -9, 4, 11])]
    sign = -1.0 if flip else 1.0
    stem = stereo(sign * place(N, shot_x, true_starts, GAINS))
    # Detection is 20 samples late on purpose; alignment has to fix it.
    hits = [Hit(s + 20, 20 * np.log10(g), 100) for s, g in zip(true_starts, GAINS)]
    shot = OneShot(data=stereo(shot_x), attack=0, level_db=0.0, name="k.wav")
    out, rep = render_triggers(stem, SR, hits, KitPiece("kick", [[shot]]), dynamics=1.0)
    assert rep.polarity == (-1 if flip else 1)
    placed = [round(h["time_s"] * SR) + h["shift_samples"] for h in rep.hits]
    assert placed == true_starts
    # With dynamics=1 and the same drum, the trigger reproduces the stem.
    np.testing.assert_allclose(out, stem, atol=1e-4)


def test_different_drum_trigger_reinforces_the_low_end():
    from scipy.signal import butter, sosfiltfilt

    real = kick(seed=5, f0=110.0, f1=52.0, decay=0.2)
    sample = kick(seed=9, f0=130.0, f1=46.0, decay=0.15)
    stem = stereo(place(N, real, STARTS, GAINS))
    hits = [Hit(s + 30, 20 * np.log10(g), 100) for s, g in zip(STARTS, GAINS)]
    shot = OneShot(data=stereo(sample), attack=0, level_db=0.0, name="k.wav")
    aligned, rep = render_triggers(stem, SR, hits, KitPiece("kick", [[shot]]))
    naive, _ = render_triggers(stem, SR, hits, KitPiece("kick", [[shot]]), align=False)
    assert rep.polarity == 1
    assert all(abs(h["shift_samples"]) <= int(0.002 * SR) for h in rep.hits)

    lp = butter(4, 150, "lowpass", fs=SR, output="sos")

    def low_corr(a, b):
        a, b = sosfiltfilt(lp, a[0]), sosfiltfilt(lp, b[0])
        return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))

    # Aligned triggers sit more in phase with the real kick's low end than a
    # placement 30 samples off would.
    assert low_corr(stem, aligned) > low_corr(stem, naive)


def test_room_split_is_exact():
    rng = np.random.default_rng(1)
    full = rng.standard_normal((2, 1000)).astype(np.float32)
    dry = (full * 0.7).astype(np.float32)
    room = split_room(full, dry)
    np.testing.assert_array_equal(dry + room, full)


def test_midi_round_trip_keeps_timing_and_velocity(tmp_path):
    hits = {"kick": [Hit(44100, -3.0, 110), Hit(66150, -6.0, 90)], "snare": [Hit(88200, -2.0, 127)]}
    p = write_midi(tmp_path / "d.mid", hits, SR, bpm=120, ppq=960)
    mf = mido.MidiFile(str(p))
    t, notes = 0.0, []
    for msg in mf:  # iterating a MidiFile yields delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append((round(t, 4), msg.note, msg.velocity, msg.channel))
    assert notes == [(1.0, 36, 110, 9), (1.5, 36, 90, 9), (2.0, 38, 127, 9)]


def test_reaper_project_is_balanced(tmp_path):
    tracks = default_session({"kick": "kick.wav", "snare": "snare.wav", "room": "room.wav", "drums_full": "drums_full.wav"})
    p = write_reaper_project(tmp_path / "x.rpp", tracks, SR, 10.0)
    text = p.read_text()
    assert text.count("<") == text.count("\n>") + text.count(" >")
    assert 'NAME "kick"' in text and "MUTESOLO 1" in text  # drums_full starts muted


def test_snare_detection_with_ghost_notes():
    starts = [5000, 30000, 52000, 80000]
    gains = [1.0, 0.06, 1.0, 0.08]  # two ghost notes about -24 dB
    x = stereo(place(N, snare(), starts, gains))
    hits = detect_hits(x, SR, DetectParams(rel_db=36, min_interval_ms=40))
    assert [h.sample for h in hits] == pytest.approx(starts, abs=22)
    assert hits[1].velocity < 60 and hits[0].velocity > 120


def test_oneshot_attack_at_sample_zero_is_found(tmp_path):
    import soundfile as sf

    from iso.triggers import load_oneshot

    for pre in (0, 30, 400):
        x = np.concatenate([np.zeros(pre, dtype=np.float32), kick(seed=11)])
        sf.write(tmp_path / f"k{pre}.wav", np.stack([x, x]).T, SR)
        shot = load_oneshot(tmp_path / f"k{pre}.wav", SR)
        assert abs(shot.attack - pre) <= 2, (pre, shot.attack)


def test_realign_fixes_a_shifted_stem_and_leaves_aligned_ones_alone():
    from iso.layers import realign

    x = kick_stem()
    for shift in (0, 3, -7):
        est = np.roll(x, shift, axis=1)
        fixed, lag = realign(x, est)
        assert lag == shift
        np.testing.assert_allclose(fixed[:, 100:-100], x[:, 100:-100], atol=1e-6)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(x.shape).astype(np.float32) * 1e-4
    assert realign(x, noise)[1] == 0  # unrelated content is never shifted


def test_triggers_are_level_matched_to_their_piece():
    real = kick(seed=5, f0=110.0, f1=52.0, decay=0.2)
    stem = stereo(0.3 * place(N, real, STARTS, GAINS))  # piece peaks well under 0 dBFS
    hits = detect_hits(stem, SR, DetectParams(rel_db=30))
    loud = OneShot(data=stereo(kick(seed=9)) / np.max(np.abs(kick(seed=9))), attack=0, level_db=0.0, name="k.wav")
    out, _ = render_triggers(stem, SR, hits, KitPiece("kick", [[loud]]), dynamics=1.0)
    from iso.onsets import hit_level_db

    # Soft hits right after loud ones are found on the exact sample too.
    assert [h.sample for h in hits] == STARTS
    prev = -(10**9)
    for h in hits:
        if h.sample - prev > int(0.4 * SR):  # isolated hit: no earlier ring to confuse the measurement
            trig_db = hit_level_db(out, h.sample, SR)
            assert abs(trig_db - h.level_db) < 0.5, (trig_db, h.level_db)
        prev = h.sample


def test_round_robins_cycle_within_each_layer():
    soft = [OneShot(stereo(kick(seed=s)) * 0.3, 0, -10.0, f"soft{s}") for s in range(2)]
    hard = [OneShot(stereo(kick(seed=s)), 0, 0.0, f"hard{s}") for s in range(2, 4)]
    kp = KitPiece("kick", [soft, hard])
    counters = {}
    picks = [kp.pick_by_level(rel, counters)[0].name for rel in (0, -10, 0, -10, 0, -10)]
    assert picks == ["hard2", "soft0", "hard3", "soft1", "hard2", "soft0"]


def test_velocity_curve_matches_sampler_playback():
    from iso.onsets import to_velocity

    p = DetectParams()
    assert to_velocity(-6.0, -6.0, p) == 127
    assert to_velocity(-18.0, -6.0, p) == 64  # -12 dB -> plays back 12 dB down at 40*log10(v/127)
    assert to_velocity(-2.0, -6.0, p) == 127  # louder than reference clamps
