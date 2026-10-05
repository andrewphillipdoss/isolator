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

    (tmp_path / "s").mkdir()
    lines = []
    for vi, (lo, hi, g) in enumerate(((1, 64, 0.3), (65, 127, 1.0))):
        for rr in (1, 2):
            fn = f"s/k_{vi}_{rr}.wav"
            sf.write(tmp_path / fn, (g * kick(seed=vi * 10 + rr)).astype(np.float32), SR)
            lines.append(f"<region> sample={fn} key=36 lovel={lo} hivel={hi} seq_length=2 seq_position={rr}")
        fn = f"s/t_{vi}.wav"
        sf.write(tmp_path / fn, (g * tom(140)).astype(np.float32), SR)
        lines.append(f"<region> sample={fn} key=45 lovel={lo} hivel={hi}")
    (tmp_path / "kit.sfz").write_text("\n".join(lines) + "\n")
    copied = import_sfz(tmp_path / "kit.sfz", tmp_path / "out")
    assert len(copied["kick"]) == 4  # 2 layers x 2 round robins
    assert len(copied["toms"]) == 2 and all(f.startswith("tom_1__") for f in copied["toms"])
    k = load_kit_folder(tmp_path / "out", SR)
    assert len(k.pieces["kick"].layers) == 2 and all(len(layer) == 2 for layer in k.pieces["kick"].layers)
