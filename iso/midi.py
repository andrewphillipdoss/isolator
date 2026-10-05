"""MIDI export of the detected performance (General MIDI drum map, channel 10)."""

from __future__ import annotations

from pathlib import Path

import mido

from .onsets import Hit

GM_NOTES: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "toms": 45,  # low tom; per-drum tom notes come once toms are split by pitch
    "tom_high": 48,
    "tom_mid": 45,
    "tom_floor": 41,
    "hihat": 42,
    "hihat_open": 46,
    "hihat_pedal": 44,
    "ride": 51,
    "ride_bell": 53,
    "crash": 49,
    "crash2": 57,
    "china": 52,
    "splash": 55,
}

DRUM_CHANNEL = 9  # zero-based; MIDI channel 10


def write_midi(
    path: str | Path,
    hits: dict[str, list[Hit]],
    sr: int,
    *,
    bpm: float = 120.0,
    ppq: int = 960,
    note_ms: float = 40.0,
) -> Path:
    """Write one drum track. Times are exact in seconds at `bpm`.

    Set the DAW project to the same tempo before importing, or the notes will
    stretch. At 960 PPQ and 120 BPM one tick is about 0.52 ms; the rendered
    trigger WAVs are sample-accurate and are the better choice for phase-critical layers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_s = ppq * bpm / 60.0
    events: list[tuple[int, int, mido.Message]] = []
    for piece, hs in hits.items():
        note = GM_NOTES.get(piece)
        if note is None:
            continue
        for h in hs:
            on = round(h.sample / sr * ticks_per_s)
            off = on + max(1, round(note_ms / 1000 * ticks_per_s))
            # Sort key puts note-offs before note-ons at the same tick, so a
            # retrigger on the same note is never swallowed.
            events.append((on, 1, mido.Message("note_on", note=note, velocity=h.velocity, channel=DRUM_CHANNEL)))
            events.append((off, 0, mido.Message("note_off", note=note, velocity=0, channel=DRUM_CHANNEL)))
    events.sort(key=lambda e: (e[0], e[1]))

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Iso drums", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    now = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mf = mido.MidiFile(ticks_per_beat=ppq, type=0)
    mf.tracks.append(track)
    mf.save(str(path))
    return path
