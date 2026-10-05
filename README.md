# Iso

Pull the drums out of a song and get a **layer kit** you can mix like a multitracked kit:

```
21_room.wav        ▁▁▁▁   a little on top, crush it              start ≈ -15 dB
20_drums_dry.wav   ▃▃▃▃   the whole kit, dry: glue               start ≈  -8 dB
01-06 pieces       █████  kick snare toms hihat ride crash: core start ≈ 0 / -3 dB
10-12 triggers     ▅▅▅▅   samples on every hit, under the core   start ≈ -6..-10 dB
30_no_drums.wav           the rest of the song, for context
drums_<bpm>bpm.mid        the performance as MIDI (velocities included)
layer_kit.rpp             REAPER session with all of the above, at those levels
```

Every file starts at sample 0 and runs the song's full length, and the
layers are **phase-coherent**: they all come from one signal with no
latency, `room = drums_full − drums_dry` exactly, and each trigger is
aligned to its hit by cross-correlation with a song-wide polarity check.
Stack them and they reinforce instead of phasing. That's easier than with
real mics, where overheads arrive late.

Personal-use tool. Plan, model choices and roadmap: [docs/GAME_PLAN.md](docs/GAME_PLAN.md).

## Install

Python 3.10–3.13.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
iso doctor        # shows whether you're on CUDA, Apple MPS or CPU
```

Models (~2 GB) download on first use from GitHub mirrors into `~/.cache/iso/models`.

## Use

**Browser UI** (runs locally, nothing is uploaded anywhere):

```bash
iso serve          # then open http://127.0.0.1:8765
```

Drop a song in, wait, then balance the layers with faders, mute and solo,
with the rest of the song playing for context. Bounce, or download the
kit, MIDI or REAPER project.

**Command line:**

```bash
iso split song.mp3                       # -> iso_out/song/
iso split *.wav --preset max --kit ~/.cache/iso/kits/GMRockKit
iso bounce iso_out/song --with-song      # mix down using the saved fader levels
```

Presets:

| preset | drums | room split | pieces | test-time augmentation (TTA) | use |
|---|---|---|---|---|---|
| `fast` | BS-RoFormer SW | yes | DrumSep 6-stem | off | quick audition, laptop CPU |
| `best` (default) | BS-RoFormer SW | yes | DrumSep 5-stem + 6-stem (ride/crash split) | on | normal |
| `max` | SW + SCNet-XL (3:1) | yes | both, blended | on, overlap 4 | final renders, GPU |

## Trigger kits

Any folder with one subfolder of one-shots per piece works
(`kit/kick/*.wav`, `kit/snare/*.wav`, `kit/toms/*.wav`, ...). Velocity
layers are worked out from loudness, so any pile of one-shots works.
Hydrogen kits import directly:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/hydrogen-music/hydrogen
cd hydrogen && git sparse-checkout set data/drumkits && cd ..
iso kit import hydrogen/data/drumkits/GMRockKit     # -> ~/.cache/iso/kits/GMRockKit
```

Toms are told apart by pitch and matched high-to-high, so a fill
rack → rack → floor triggers the kit's rack → rack → floor. If you
already own Superior Drummer, EZdrummer, SSD or similar, use the MIDI file
instead. Set the DAW to the tempo in the filename before importing.

## Checking quality

```bash
iso eval make --kit ~/.cache/iso/kits/GMRockKit -o iso_eval   # test song with known answers
iso split iso_eval/song.wav -o iso_eval/run
iso eval score iso_eval/run/song iso_eval/truth               # SDR, hit F1, timing, velocity
```

## No GPU?

Everything runs on CPU, just slowly: about 25× real time for the `fast`
preset on a 4-core laptop. Free cloud GPUs (Google Colab, Kaggle) run the
same commands much faster; see the game plan.

## Licenses

No license has been chosen for Iso's own code yet. Its dependencies are
MIT/BSD (MSST, NumPy, SciPy, FastAPI). The model weights are community
checkpoints with unstated or non-commercial terms. They're fine for
personal use, but don't redistribute them. Hydrogen kits carry their own
licenses (mostly GPL / CC).
