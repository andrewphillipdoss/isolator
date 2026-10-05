import numpy as np
import soundfile as sf

from iso.kits import import_hydrogen, piece_for
from iso.onsets import Hit
from iso.toms import split_toms
from iso.triggers import load_kit_folder, render_triggers

from .synth import SR, kick, place, stereo


def tom(f0, seed=0):
    return kick(seed=seed, f0=f0 * 1.15, f1=f0, decay=0.35)


def test_three_toms_are_told_apart_by_pitch():
    starts = [i * 9000 + 1000 for i in range(12)]
    pitch = [220, 220, 150, 150, 95, 95, 220, 150, 95, 95, 150, 220]
    x = np.zeros(starts[-1] + 30000, dtype=np.float32)
    for s, f in zip(starts, pitch):
        x[s : s + 22050] += tom(f)[:22050]
    hits = [Hit(s, -6.0, 100) for s in starts]
    idx = split_toms(stereo(x), SR, hits)
    want = {220: 0, 150: 1, 95: 2}
    assert idx == [want[f] for f in pitch]


def test_single_tom_stays_one_tom():
    starts = [1000, 15000, 30000]
    x = place(60000, tom(140), starts)
    assert split_toms(stereo(x), SR, [Hit(s, -6, 100) for s in starts]) == [0, 0, 0]


def test_hydrogen_import_and_tom_variants(tmp_path):
    kit = tmp_path / "hkit"
    kit.mkdir()
    insts = []
    for name, f0 in (("Kick", None), ("Tom 1", 230), ("Floor Tom", 100), ("Hat Closed", None), ("Cowbell", None)):
        files = []
        for i, g in enumerate((0.3, 1.0)):
            fn = f"{name.replace(' ', '')}-{i}.wav"
            shot = tom(f0) if f0 else kick(seed=i)
            sf.write(kit / fn, np.stack([g * shot] * 2).T, SR)
            files.append(f"<layer><min>0</min><max>1</max><filename>{fn}</filename></layer>")
        insts.append(f"<instrument><name>{name}</name><instrumentComponent>{''.join(files)}</instrumentComponent></instrument>")
    (kit / "drumkit.xml").write_text(
        f'<drumkit_info xmlns="http://www.hydrogen-music.org/drumkit"><name>T</name><instrumentList>{"".join(insts)}</instrumentList></drumkit_info>'
    )
    copied = import_hydrogen(kit, tmp_path / "iso_kit")
    assert set(copied) == {"kick", "toms", "hihat"}  # cowbell skipped
    k = load_kit_folder(tmp_path / "iso_kit", SR)
    toms = k.pieces["toms"]
    assert [v.name for v in toms.variants] == ["toms:tom_1", "toms:floor_tom"]  # highest pitch first

    # A song with a high and a low tom triggers the matching kit toms.
    starts = [1000, 20000, 40000, 60000]
    song_pitch = [180, 80, 180, 80]
    x = np.zeros(90000, dtype=np.float32)
    for s, f in zip(starts, song_pitch):
        x[s : s + 22050] += tom(f)[:22050]
    hits = [Hit(s, -6.0, 120) for s in starts]
    idx = split_toms(stereo(x), SR, hits)
    _, rep = render_triggers(stereo(x), SR, hits, toms, variant_of_hit=idx)
    assert [r["sample"].split("__")[0] for r in rep.hits] == ["tom_1", "floor_tom", "tom_1", "floor_tom"]


def test_piece_name_mapping():
    assert piece_for("Kick") == "kick"
    assert piece_for("Snare Rimshot") is None
    assert piece_for("Floor Tom") == "toms"
    assert piece_for("Hat Closed") == "hihat"
    assert piece_for("Ride 2") == "ride"
    assert piece_for("Splash") == "crash"
    assert piece_for("Cowbell") is None


def test_sfz_import_renders_layers_and_round_robins(tmp_path):
    import pytest

    pytest.importorskip("pysfizz")
    from iso.kits import import_sfz
    from iso.toms import hit_pitch

    (tmp_path / "s").mkdir()
    lines = []
    # Each band has its own pitch, so we can tell which sample got rendered.
    band_pitch = {0: 100.0, 1: 200.0}
    for vi, (lo, hi) in enumerate(((1, 64), (65, 127))):
        for rr in (1, 2):
            fn = f"s/k_{vi}_{rr}.wav"
            sf.write(tmp_path / fn, tom(band_pitch[vi], seed=vi * 10 + rr).astype(np.float32), SR)
            lines.append(f"<region> sample={fn} key=36 lovel={lo} hivel={hi} seq_length=2 seq_position={rr}")
        fn = f"s/t_{vi}.wav"
        sf.write(tmp_path / fn, tom(140).astype(np.float32), SR)
        lines.append(f"<region> sample={fn} key=45 lovel={lo} hivel={hi}")
    (tmp_path / "kit.sfz").write_text("\n".join(lines) + "\n")
    copied = import_sfz(tmp_path / "kit.sfz", tmp_path / "out")
    assert len(copied["kick"]) == 4  # 2 layers x 2 round robins
    pitches = sorted(hit_pitch(sf.read(tmp_path / "out" / "kick" / f, always_2d=True)[0].T, 0, SR) for f in copied["kick"])
    # Two samples from the ~100 Hz band and two from the ~200 Hz band: no band lost.
    assert max(pitches[:2]) < 150 < min(pitches[2:]), pitches
    assert all(f.startswith("kit_k45__") for f in copied["toms"])

    # A second SFZ imported into the same kit adds to it instead of overwriting.
    (tmp_path / "Tom 2.sfz").write_text(f"<region> sample=s/t_0.wav key=45\n")
    import_sfz(tmp_path / "Tom 2.sfz", tmp_path / "out")
    k = load_kit_folder(tmp_path / "out", SR)
    assert len(k.pieces["toms"].variants) == 2
    assert len(k.pieces["kick"].layers) == 2 and all(len(layer) == 2 for layer in k.pieces["kick"].layers)


def test_two_different_drums_in_one_piece_do_not_alternate(tmp_path):
    for group, f0 in (("snare_a", 180.0), ("snare_b", 260.0)):
        d = tmp_path / "kit" / "snare"
        d.mkdir(parents=True, exist_ok=True)
        n = 3 if group == "snare_a" else 2
        for i in range(n):
            sf.write(d / f"{group}__{i}.wav", tom(f0, seed=i).astype(np.float32), SR)
    (tmp_path / "kit" / "kick").mkdir()
    sf.write(tmp_path / "kit" / "kick" / "silent.wav", np.zeros(4410, dtype=np.float32), SR)
    sf.write(tmp_path / "kit" / "kick" / "real.wav", kick().astype(np.float32), SR)
    k = load_kit_folder(tmp_path / "kit", SR)
    names = {s.name for layer in k.pieces["snare"].layers for s in layer}
    assert names and all(n.startswith("snare_a__") for n in names)  # one drum: the larger group
    assert [s.name for layer in k.pieces["kick"].layers for s in layer] == ["real.wav"]  # silent file skipped
