# Iso

Pull the drums out of a song and get a **layer kit** you can mix like a multitracked kit:

```
21_room.wav        ▁▁▁▁   a little on top, crush it              start ≈ -15 dB
20_drums_dry.wav   ▃▃▃▃   the whole kit, dry: glue               start ≈  -8 dB
01-06 pieces       █████  kick snare toms hihat ride crash: core start ≈ 0 / -3 dB
10-12 triggers     ▅▅▅▅   samples on every hit, under the core   start ≈ -6..-10 dB
30_no_drums.wav           the rest of the song, for context
drums_<bpm>bpm.mid        the performance as MIDI (velocities, open/closed hats, per-tom notes)
00_mix_ref.wav            the decoded song every stem was made from; align to this, not the MP3
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
iso split *.wav --preset max --kit ~/.cache/iso/kits/BigRusty
iso split song.mp3 --gate-floor -60     # cleaner pieces (default -30 dB keeps a missed hit faintly audible)
iso bounce iso_out/song --with-song      # mix down using the saved fader levels
```

Presets:

| preset | drums | room split | pieces | test-time augmentation (TTA) | use |
|---|---|---|---|---|---|
| `fast` | BS-RoFormer SW | yes | DrumSep 6-stem | off | quick audition, laptop CPU |
| `best` (default) | SW + SCNet-XL (2:1) | yes | DrumSep 5-stem + 6-stem (ride/crash split) | off | normal; best on a GPU |
| `max` | SW + SCNet-XL (2:1) | yes | both, blended | on, overlap 4 | final renders, GPU |

## Recorded the drums yourself? Multitrack mode

With separate mic tracks from one session, Iso cleans each close mic of
the other drums' bleed. The kick mic keeps only kick, the snare mic only
snare, and so on, gated per hit. Overheads are split into cymbals-only
and the shells they carried. Room mics are passed through. Everything
stays sample-aligned with the original mics.

```bash
iso debleed --kick kick.wav --snare snare_top.wav --tom rack.wav --tom floor.wav \
            --oh oh.wav --room room.wav --kit ~/.cache/iso/kits/BigRusty -o cleaned/
```

## Trigger kits

A trigger kit is a folder with one subfolder of one-shots per piece
(`kit/kick/*.wav`, `kit/snare/*.flac`, `kit/toms/*.wav`, ...). Velocity
layers are worked out from each sample's recorded loudness, and round
robins come from samples of near-equal level. Use **close-mic** samples;
the room comes from your song's own room stem.

**Recommended free kit for indie rock: Karoryfer Big Rusty Drums** (CC0, a
dry vintage kit, 16 MB of close mics):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/sfzinstruments/karoryfer.big-rusty-drums bigrusty
cd bigrusty && git sparse-checkout set Samples/kick_24/kick/kick Samples/snare_14/center/top \
  Samples/tom_14/center/cl Samples/tom_15/center/cl Samples/tom_18/center/cl Samples/tom_22/center/cl \
  Samples/hihat_14/cl/cl && cd ..
iso kit assemble BigRusty --kick bigrusty/Samples/kick_24/kick/kick --snare bigrusty/Samples/snare_14/center/top \
  --tom bigrusty/Samples/tom_14/center/cl --tom bigrusty/Samples/tom_15/center/cl \
  --tom bigrusty/Samples/tom_18/center/cl --tom bigrusty/Samples/tom_22/center/cl \
  --hihat bigrusty/Samples/hihat_14/cl/cl
```

Other good free kits: Wilkinson Naked Drums and DrumGizmo DRSKit (both CC BY,
on github.com/sfzinstruments). Import their close-mic SFZ files one piece at
a time:

```bash
iso kit import "Kick In.sfz" --dest ~/.cache/iso/kits/Naked     # GM key map by default; --key 36=kick for others
```

Hydrogen kits import directly (`iso kit import path/to/GMRockKit`).

Toms are told apart by pitch and matched high-to-high, so a fill
rack → rack → floor triggers the kit's rack → rack → floor. Triggers are
level-matched to their piece, and each kit's polarity is checked against
your song (kick-in mics are often inverted). If you already own Superior
Drummer, EZdrummer, SSD or similar, use the MIDI file instead. Its
velocities follow the standard sampler curve. Set the DAW to the tempo in
the filename before importing.

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
