# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instrument categories inferred from sample-library paths.

Sample packs encode the instrument in folder and file names ("Guitar One Shots",
"KSHMR_Snare_04"). A file is classified by its own name first, then by its
folders from the nearest outward, so "KSHMR_Violin_01.wav" inside a
"Live_Instruments" folder is a violin. Keywords must stand alone as words
(letters on either side don't count, underscores and digits do), so "strum"
does not match "instrument" and "gtr" does not match "Dugtrio".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def _words(*words: str) -> str:
    """Alternation of words that must not be glued to other letters."""
    return r"(?<![a-z])(?:" + "|".join(words) + r")(?![a-z])"


CATEGORIES = [
    ("808", r"808"),
    ("kick", _words("kicks?", "bd", "bass ?drum")),
    ("snare", _words("snares?", "rims?", "rimshot")),
    ("clap", _words("claps?", "snaps?")),
    ("hihat", _words("hi-?hats?", "hh", "hats?", "open ?hat", "closed ?hat")),
    ("cymbal", _words("cymbals?", "crash", "ride")),
    ("perc", _words("percs?", "percussion", "toms?", "shakers?", "conga", "bongo", "tamb\\w*", "cowbell")),
    ("guitar", _words("guitars?", "gtrs?", "strums?", "strummed", "nylon", "banjo", "ukulele", "sitar", "mandolin")),
    ("strings", _words("strings?", "violins?", "viola", "cello", "pizz\\w*", "orchestra\\w*", "contrabass", "double[ -]?bass")),
    ("keys", _words("piano", "keys", "rhodes", "organ", "e-?piano", "wurli\\w*", "clav\\w*")),
    ("bell", _words("bells?", "chimes?", "glock\\w*", "marimba", "mallets?", "kalimba")),
    ("pluck", _words("plucks?", "plucked")),
    ("pad", _words("pads?")),
    ("bass", _words("bass", "basses", "sub")),
    ("vocal", _words("vocals?", "vox", "voice", "choir", "adlibs?", "chants?")),
    ("brass", _words("brass", "horns?", "trumpets?", "sax\\w*", "flutes?", "trombone", "woodwinds?", "winds", "duduk")),
    ("fx", _words("fx", "risers?", "impacts?", "sweeps?", "transitions?", "textures?", "noise")),
    ("synth", _words("synths?", "leads?")),
]
_COMPILED = [(name, re.compile(rx, re.I)) for name, rx in CATEGORIES]

# Groups usable with --category (expand to several categories).
GROUPS = {"drums": ["kick", "snare", "clap", "hihat", "cymbal", "perc", "808"]}

# Not one-shots: loops, phrases, fills, runs, reversed sounds (onset at the end).
LOOP_NAME = re.compile(r"loop|\d{2,3} ?bpm|bpm|phrase|melody|top ?line|\[\d{2,3}|(?<![a-z])arps?(?![a-z])|fill|reverse|run_?(up|down)", re.I)
LOOP_FOLDER = re.compile(r"loop|phrase|top ?line|melod|(?<![a-z])arps?(?![a-z])|fills?|reverse|runs", re.I)

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}


def _match(text: str) -> Optional[str]:
    for name, rx in _COMPILED:
        if rx.search(text):
            return name
    return None


def classify(relative_path: str) -> str:
    """Category of a file: its own name first, then its folders, nearest first."""
    p = Path(relative_path)
    for part in [p.stem, *reversed(p.parts[:-1])]:
        found = _match(part.replace("_", " "))
        if found:
            return found
    return "other"


def is_loop(relative: Path) -> Optional[str]:
    """Reason string if this looks like a loop/phrase, else None."""
    if LOOP_NAME.search(relative.name):
        return "loop/fill name"
    if any(LOOP_FOLDER.search(part) for part in relative.parts[:-1]):
        return "loop/fill folder"
    return None
