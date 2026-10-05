# Iso: Game Plan (v2: personal use)

**Goal:** the cleanest possible drums from a finished song, delivered as a
*layer kit* an indie-rock producer can mix like a multitracked kit.

**Status (Oct 2026):** v0 is built, tested (29 unit/integration tests) and
measured on a ground-truth test song. See [§5](#5-measured-quality).

The first, commercial version of this plan is in git history (commit
`01a27c9`). Personal use changes three things: we can use the best
community model weights as they are, there is no training program, and the
budget is ~$0.

---

## 0. Decisions

| Question | Answer | Consequence |
|---|---|---|
| Commercial or personal? | Personal | Use the best community checkpoints (unstated or non-commercial licenses are fine). Don't redistribute weights. |
| Web or desktop? | "Whatever is best" | **Local web app**: browser UI, engine on your machine (§2) |
| Clean-up scope | Clean full kit, kit with room, each drum as clean as possible, layered with triggers, full kit and room | The **layer kit** (§1) |
| Multi-mic de-bleed in v2? | "Why wait?" | It doesn't: `iso debleed` is in v0 (§4) |
| Who for | Producer demoing indie rock | Acoustic kits, guitar-heavy mixes, room sound matters |
| Budget | As free as possible | $0 path end to end (§6) |

---

## 1. The layer kit

```
        what you hear                      file                      starting fader
   ┌──────────────────────┐
   │  room                │  ▁▁▁        21_room.wav                       -12 dB (crush it? -15..-18)
   ├──────────────────────┤
   │  full kit, dry       │  ▃▃▃▃       20_drums_dry.wav                   -8 dB
   ├──────────────────────┤
   │  pieces (the core)   │  █████      01_kick 02_snare 03_toms           0 / 0 / -2 dB
   │                      │             04_hihat 05_ride 06_crash          -4 dB
   ├──────────────────────┤
   │  triggers (under)    │  ▅▅▅▅       10_trig_kick 11_trig_snare         -8 / -10 dB
   │                      │             12_trig_toms                       -12 dB
   └──────────────────────┘
   plus: 22_drums_full (kit + room, for A/B), 30_no_drums (the song minus drums),
         00_mix_ref (the decoded song), drums_<bpm>bpm.mid, layer_kit.rpp, session.json
```

**Why the layers stack without phasing.** This is the part that makes the
idea work.

- Every stem is derived from **one decoded signal**, with **no latency**. All
  files start at sample 0 and run the song's full length.
- `room = drums_full − drums_dry`, so dry + room is the original kit exactly
  (checked in every run's QC).
- Pieces come from the dry kit, so they sit inside it in time. Layering them
  over the dry kit is like close mics over overheads, but with zero arrival
  delay.
- Triggers are placed per hit by band-limited cross-correlation against the
  separated piece (±2 ms). One polarity is chosen per song by weighted vote
  and reported with a confidence. Kick-in style samples are often
  polarity-inverted against a mix; Iso catches that.
- Triggers are **level-matched** to their piece, so "-8 dB" really means 8 dB
  under the real drum. Their dynamics follow the drummer, and velocity
  layers are chosen by loudness, with per-layer round robins (no
  machine-gunning).
- An alignment guard cross-correlates every model output against its source
  and corrects any whole-sample offset. On every run so far, nothing has
  needed correcting.

**Pieces "as clean as possible."** Kick, snare and toms are gated on their own
detected hits:
- The gate opens 4 ms *before* each attack, so transients are never clipped.
- It closes once the hit has decayed. Between hits the floor is −30 dB by
  default (`--gate-floor -60` for surgical).
- Two filters drop false hits:
  - leakage from other drums (a faint copy of the snare in the kick stem);
  - hits buried more than 30 dB under the whole kit at that instant (an
    unplayed ride's blips).
- Toms are split into individual drums by pitch, so MIDI and triggers know
  rack 1 from rack 2 from floor.
- Hi-hats are labelled open or closed from their decay.

---

## 2. Web or desktop? A local web app

| Option | Quality ceiling | Cost | Privacy / copyright | Effort | Verdict |
|---|---|---|---|---|---|
| Hosted web app (server GPU) | Full | Pay for GPUs | You upload songs to a server | Medium | Not needed for one person |
| Pure in-browser (WebGPU/ONNX) | 1 model, ~300 MB, slow | $0 | Private | High (port every model + STFT to JS) | Later, maybe, for quick previews |
| **Local web app** (`iso serve`) | **Full** (5–6 models, ensembles) | **$0** | **Private** | **Done** | **Chosen** |
| Native desktop app (Tauri/Electron) | Full | $0 | Private | Wraps the same thing | Optional later: an app icon and installer |

The browser UI is the accessible part. The engine runs on your machine,
using CUDA on NVIDIA, MPS on Apple Silicon, or CPU. A computer too slow for
the models has a free cloud fallback (§6). The UI never needs a GPU.

The browser mixer plays every stem from one `AudioContext` clock, sample-locked.
Separate `<audio>` players drift by milliseconds and would comb-filter
correlated drum layers.

---

## 3. The model stack (October 2026)

| Stage | Model | Why | License |
|---|---|---|---|
| Song → drums | **BS-RoFormer SW** (6-stem) | Best public drums model: 14.1 dB SDR (MVSep Multisong), ~2.5 dB above anything else downloadable. Its own guitar stem helps on rock. | unknown provenance; personal use |
| (best/max presets) | + **SCNet-XL IHF**, avg 2:1 | A CNN partner. On Iso's test song the blend beat SW alone on SDR (22.9 vs 22.3 dB) *and* attack preservation (0.40 vs 0.50 dB error). | MIT repo, MUSDB-trained |
| Drums → dry + room | **MDX23C De-Reverb** (aufr33 & jarredou) | General-purpose (not vocal-trained) dereverb; ~4 dB on MVSep's drums test. The popular RoFormer dereverbs are vocal-only and score ~0–2 dB on drums. | unstated |
| Dry kit → pieces | **DrumSep 5-stem** (jarredou) + **6-stem** (aufr33 & jarredou) | 5-stem has the best public piece SDR (kick 16.7, snare 11.5, toms 12.3). The 6-stem guides the ride/crash split of its cymbals; ride + crash = cymbals exactly. | unstated |
| CPU fallback | HTDemucs FT drums | MIT, fast, ~3 dB worse | MIT |

Engine: ZFTurbo's **MSST** (`pip install msst`, MIT). It separates in
memory, never renormalizes stems, and loads all of the above. Weights
download on first use from GitHub mirrors, with sha256 pins: the original
jarredou accounts are gone. MVSep's own best drum and kit-piece models are
server-only and not downloadable.

---

## 4. Clean-up, including multitrack de-bleed now

| Clean-up | How | Status |
|---|---|---|
| Other instruments out of the drums | BS-RoFormer SW (optionally + SCNet-XL) | v0 |
| Room out of the kit, kept as its own stem | MDX23C dereverb, room = full − dry | v0 |
| Each drum out of the kit | DrumSep 5 + 6 stem | v0 |
| Bleed between drums | Hit-keyed gates (kick/snare/toms), leakage + buried-hit filters | v0 |
| Consistency and punch | Level-matched, phase-aligned trigger samples | v0 |
| **Multi-mic de-bleed** (your own recordings) | `iso debleed`: each close mic keeps only its own drum (kit-piece split + hit gate). Overheads are split into cymbals-only and shells. Room mics pass through. | **v0**: it doesn't have to wait |

Why multi-mic was "v2" before: in the commercial plan it needed its own
trained multichannel model. For personal use, the kit-piece models already
pull one drum out of a kit sound, so each mic can be cleaned independently.
A dedicated multichannel model would still do better on heavy bleed, and
stays on the roadmap.

---

## 5. Measured quality

Harness: `iso eval make` renders an indie-rock test song from a real
multi-velocity kit (Hydrogen GMRockKit, GPL):
- the kit is detuned and coloured so it doesn't match the trigger kit, with
  toms retuned to a realistic rack/rack/floor spread;
- the playing has 8th-note hats with open hats, ghost notes, 16th/32nd tom
  fills, crashes and a ride chorus;
- there is a 0.7 s room, plus distorted double-tracked guitars and bass.

`iso eval score` then measures SDR per stem, hit F1, onset error, velocity
correlation, tom grouping and open/closed hats.

**Real models, `fast` preset, CPU (33 s song, 12 min):**

| | Result |
|---|---|
| Drums vs rest | **22.3 dB SDR** (this synthetic accompaniment is easier than real mixes) |
| Kick hits | F1 1.00, onset error 1.0 ms, velocity r 0.95 |
| Snare hits | F1 0.92, error 0.6 ms, velocity r 0.995 |
| Hi-hat hits | F1 0.94, error 1.0 ms, velocity r 0.97 |
| Ride hits | F1 0.94, error 0.2 ms, velocity r 0.90 |
| Toms | F1 0.88, grouped to the right drum 82% |
| Crash | **weak**: 1 of 4 found correctly |
| Triggers (Big Rusty kit on this song) | kick 53/53 phase-locked, polarity inverted (detected, confidence 1.0); snare 38/53 |

**Real models, `best` setup (with test-time augmentation), second test song (it has open hats), 50 min CPU:**

| | Result |
|---|---|
| Drums vs rest | 22.5 dB SDR |
| Kick | F1 1.00, onset error 1.1 ms |
| Snare | F1 0.98, error 0.3 ms, velocity r 0.996 |
| Toms | F1 0.89, **96%** grouped to the right drum |
| Hi-hat | F1 0.94, **open/closed right 97%** (4/4 open hats found) |
| Ride | F1 0.94 |
| Crash | **0/4.** Both kit-piece models file this kit's crash mostly under hi-hat; see weak spots |
| QC | drums + rest and dry + room reconstruct to −172 / −194 dB; piece residual 35 dB down; no realignment needed |

**Oracle check** (true stems in, to test Iso's own DSP): kick, snare and
hi-hat F1 1.00 with 0.2–0.9 ms error; crash 1.00; toms 0.84 (fast 32nd-note
fills); open/closed hats 100%.

Bugs this harness caught and fixed:
- onsets in the first 10 ms of a sample;
- cymbal samples whose peak arrives 100 ms late;
- soft hits inside a ringing tail;
- the ~7 ms early bias that separation pre-echo puts on snare timing
  (5.8 → 0.6 ms);
- a soft kick after a loud one landing 15 ms late;
- trigger stems 10 dB too loud for the default faders.

**Known weak spots:**
- **Crash.** On the test kit, both public kit-piece models put the crash
  mostly into the hi-hat stem (correlation with the true crash 0.53 / 0.42),
  and the 6-stem model's own crash stem is 20 dB too quiet. Until the ADTOF
  transcription pass (§7.2) lands, take cymbals from the dry-kit layer.
- The room stem is under-extracted (≈10 dB quieter than the true room on the
  test song). Neither a second dereverb pass nor WPE beat a single MDX23C
  pass.
- Hi-hat, ride and crash SDR is low for *every* public model (3–8 dB), so
  cymbals carry some guitar wash. Lean on the dry kit for cymbals.

All numbers are from a synthetic song; your real songs are the real test
(§7, step 1).

---

## 6. Budget

**The $0 path covers everything.**

| Item | Free option |
|---|---|
| Models | Community checkpoints, GitHub mirrors |
| Engine, UI, DSP | Open source (MSST MIT, FastAPI, NumPy/SciPy) |
| Trigger samples | Big Rusty Drums (CC0), Naked Drums and DRSKit (CC BY), GMRockKit (GPL) |
| GPU, if your machine is slow | Kaggle (~30 h/week T4), Google Colab free T4 (`notebooks/iso_colab.ipynb`), HF ZeroGPU (5 min/day) |
| DAW session | REAPER project file (REAPER's evaluation is free; any DAW takes the WAVs) |

**Speed on your own machine** (4-minute song, `best` preset; these are
estimates from research, so time your own):

| Machine | Time |
|---|---|
| NVIDIA, 8 GB+ | 1–3 min |
| Apple M-series Pro/Max | 3–6 min |
| Base M1/M2, 16 GB | 12–25 min |
| CPU only | 1–3 h (use `fast`, or the cloud) |

**Optional, only if you want it:**

| Item | Cost |
|---|---|
| Colab Pro (faster GPUs, longer sessions) | ~$10/month |
| Rented GPU (RunPod, Vast) for a big batch | ~$0.20–0.50/hour |
| REAPER license | $60 |
| A commercial drum library for triggers or MIDI (Superior Drummer, SSD) | $0–200; MIDI export works with what you own |

---

## 7. Roadmap

1. **Your songs** (needs you):
   - Run 3–5 of your indie-rock tracks or demos through `best`.
   - Listen to each layer soloed and in the stack.
   - Note what's wrong: cymbal wash, missing ghost notes, room too small,
     trigger polarity or tone.
   - Defaults get tuned to your ears, not to the synthetic song.
2. **Transcription pass with ADTOF** on the drum stem: the strongest
   "which drum was hit" signal (MDB F1: kick .89, hh .80, cymbals .91), with
   Iso's stem refinement for exact time and velocity. This targets the crash
   and snare false positives.
3. **Tempo map** (beat_this, MIT), so MIDI follows a band that didn't play
   to a click.
4. ~~Validate the SW + SCNet-XL ensemble~~ **Done:** it wins on both SDR
   and attacks and is now the `best` default.
5. **More room**: a FoxJoy Reverb-HQ + MDX23C dereverb ensemble, or a
   "room amount" control.
6. **Low-bitrate sources**: an optional Apollo restoration pre-pass, for
   YouTube rips and old MP3s.
7. **Cross-ducking**: remove hat fizz from the snare at hat-only hits, and
   bass leak from the kick between kicks.
8. **UI**: per-trigger polarity flip and nudge, a kit picker with audition,
   waveform zoom, a "cleanliness" slider, and a progress estimate.
9. **Packaging**: a one-click installer, or a Tauri desktop shell around the
   same UI.
10. **Optional cloud quality boost**: MVSep's server-only DrumSep models via
    their free API, run on Iso's own drum stem so everything stays aligned.
    This is a privacy trade-off, so opt-in only.
11. Kaggle runner and an MLX backend (~2× faster on Macs).

---

## 8. Risks and limits

- **Weights can vanish.** The jarredou accounts already did. Keep
  `~/.cache/iso/models`. The registry pins sha256 hashes so a re-hosted file
  is verified.
- **Cymbals stay hard.** Every public model is weak on hats, ride and crash.
  Triggers don't cover them.
- **The room stem is partial.** It also holds some drum-body decay the model
  calls reverb. It is always phase-safe to push.
- **Electronic and programmed drums confuse the kit-piece models.** They
  were trained on acoustic kits.
- **MP3 timing.** Always align to `00_mix_ref.wav`, not the MP3.

---

## Appendix: models and hashes

| Registry name | File | sha256 |
|---|---|---|
| `bs_roformer_sw` | BS-Roformer-SW.ckpt (699 MB) | `24e7d35e…916e` |
| `dereverb_mdx23c` | MDX23C-De-Reverb-aufr33-jarredou.ckpt (448 MB) | `eae2471b…e914` |
| `drumsep_5` | drumsep_5stems_mdx23c_jarredou.ckpt (438 MB) | `1f8e636f…b2e0` |
| `drumsep_6` | MDX23C-DrumSep-aufr33-jarredou.ckpt (438 MB) | `d2a4aa53…96d0` |
| `scnet_xl_ihf` | model_scnet_ep_36_sdr_10.0891.ckpt (214 MB) | `ac25975f…b74f` |
| `htdemucs_ft_drums` | f7e0c4bc-ba3fe64a.th (84 MB) | size-checked |

Full URLs are in `iso/models.py`.

**Sources:**
- MVSep leaderboards and algorithms: https://mvsep.com/en/algorithms
- MSST: https://github.com/ZFTurbo/Music-Source-Separation-Training
- Beyond SDR (2026), rhythm/transient distortion by separators: https://arxiv.org/abs/2609.04224
- MSR Challenge summary: https://arxiv.org/abs/2601.04343
- Riley & Dixon, ADT via drum stem separation: https://arxiv.org/abs/2509.24853
- ADTOF: https://github.com/MZehren/ADTOF
- beat_this: https://github.com/CPJKU/beat_this
- sfizz / pysfizz: https://github.com/sfztools/sfizz
- Kits: https://github.com/sfzinstruments
- Hydrogen: https://github.com/hydrogen-music/hydrogen
