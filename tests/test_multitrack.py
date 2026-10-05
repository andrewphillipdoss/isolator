import hashlib

import numpy as np
import soundfile as sf

from iso.multitrack import Mic, MultitrackConfig, run_multitrack

from .synth import SR, kick, place, snare, stereo

N = 3 * SR
K_STARTS = [2000, 46100, 90200]
S_STARTS = [24050, 68150, 112250]


def test_close_mics_lose_their_bleed(tmp_path):
    k = stereo(place(N, kick(), K_STARTS))
    s = stereo(place(N, snare(), S_STARTS))
    kick_mic = k + 0.25 * s  # snare bleeding into the kick mic
    snare_mic = s + 0.3 * k
    oh = 0.3 * k + 0.4 * s
    truth = {}

    def key(x):
        return hashlib.sha1(np.round(x, 5).astype(np.float32).tobytes()).hexdigest()

    for mic, (kk, ss) in ((kick_mic, (k, 0.25 * s)), (snare_mic, (0.3 * k, s)), (oh, (0.3 * k, 0.4 * s))):
        z = np.zeros_like(mic)
        truth[key(mic)] = {"kick": kk, "snare": ss, "toms": z, "hh": z, "ride": z, "crash": z}

    class Fake:
        def separate(self, x, sr, model):
            return truth[key(x)]

    paths = {}
    for name, x in (("kick", kick_mic), ("snare", snare_mic), ("oh", oh)):
        paths[name] = tmp_path / f"{name}.wav"
        sf.write(paths[name], x.T, SR, subtype="FLOAT")
    rep = run_multitrack(
        [Mic("kick", paths["kick"]), Mic("snare", paths["snare"]), Mic("oh", paths["oh"])],
        tmp_path / "out",
        Fake(),
        MultitrackConfig(),
    )
    by = {m["role"]: m for m in rep["mics"]}
    assert by["kick"]["hits"] == 3 and by["snare"]["hits"] == 3
    clean_kick, _ = sf.read(tmp_path / "out" / by["kick"]["files"]["raw"], always_2d=True)
    np.testing.assert_allclose(clean_kick.T, k, atol=1e-5)  # snare bleed gone, kick untouched
    cym, _ = sf.read(tmp_path / "out" / by["oh"]["files"]["cymbals"], always_2d=True)
    assert np.max(np.abs(cym)) < 1e-6  # no cymbals in this overhead
    assert rep["midi"]
