# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Local, user-editable pronunciation corrections -- the LAST step
before Kokoro, applied after Markdown/emoji/path cleanup
(speech_format.to_speech). This affects SPEECH ONLY: the text that
gets logged, displayed, or handed back as a typed reply is never
touched -- only the separate copy that reaches the TTS engine (see
mouth.py's say()/say_chunk(), the one choke point every spoken word
passes through regardless of which brain or which hand-written
console-verb line produced it).

The dictionary lives in pronunciation.json, gitignored and personal
-- same pattern as brain_router.json. Captain edits it directly in a
text editor, no Python required, and it's reloaded automatically the
moment the file's mtime changes (see load()), so a mid-session edit
takes effect on the very next reply. pronunciation.json.example ships
the seed entries as the starting template; the real, active
pronunciation.json is seeded with the same entries so this works out
of the box.

Matching is whole-word/whole-phrase only, case-insensitive, longest
phrase first (so "RTX 3060" matches before anything that might
otherwise catch just "RTX"). This runs on already-clean spoken prose
(Markdown, links, and paths are already gone by this point), and it
only ever touches an EXACT configured phrase -- a password, an ID, a
code fragment, or a URL that somehow survived upstream cleanup is
never a dictionary entry, so this module has nothing to match against
it and leaves it untouched.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("PRONUNCIATION_CONFIG")
                   or (REPO / "pronunciation.json"))

DEFAULTS: dict[str, str] = {}

_lock = threading.Lock()
_cache: dict[str, str] | None = None
_cache_mtime: float | None = None


def load(force: bool = False) -> dict[str, str]:
    """The active dictionary, reloaded whenever the file's mtime
    changes. A missing or malformed file falls back to the last good
    dictionary (or empty) rather than ever breaking a turn -- a typo
    in a hand-edited JSON file must degrade speech quality, not take
    the voice line down."""
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = CONFIG_PATH.stat().st_mtime
        except OSError:
            if not force and _cache is not None:
                return _cache
            _cache, _cache_mtime = dict(DEFAULTS), None
            return _cache
        if not force and _cache is not None and mtime == _cache_mtime:
            return _cache
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("pronunciation.json must be a JSON object")
            _cache = {str(k): str(v) for k, v in data.items()}
        except (OSError, ValueError):
            _cache = _cache if _cache is not None else dict(DEFAULTS)
        _cache_mtime = mtime
        return _cache


def _build_pattern(entries: dict[str, str]) -> "re.Pattern[str] | None":
    if not entries:
        return None
    # Longest phrase first: "RTX 3060" must match as a whole before a
    # shorter entry could otherwise claim part of it.
    keys = sorted(entries.keys(), key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in keys)
    # (?<![\w-]) / (?![\w-]) rather than \b: an entry like "Pvt. Ltd."
    # ends in a non-word character (the period), and \b requires a
    # transition between a word char and a non-word char -- it can't
    # fire when BOTH the pattern's own last character and whatever
    # follows in real text (a space, another period) are non-word.
    # The lookaround checks each side independently instead, so
    # punctuation-ending entries match correctly too.
    #
    # Hyphen is folded into the excluded set on top of \w, on purpose:
    # \w alone treats "-" as a boundary, so "GPU" inside a compound
    # like "gpu-benchmarks" would count as a whole-word match (a real
    # bug, caught live -- "/gpu-benchmarks" in a URL got "GPU"
    # rewritten mid-token). Compound hyphenated words are common
    # enough that a bare acronym match inside one is never intended.
    return re.compile(
        rf"(?<![\w-])(?:{alternation})(?![\w-])", re.IGNORECASE)


def apply(text: str) -> str:
    """Substitute every configured whole-word/phrase match with its
    spoken form. Matching is case-insensitive (a model's own
    capitalization of a name or acronym varies); the replacement is
    always exactly the configured spoken form."""
    if not text:
        return text
    entries = load()
    pattern = _build_pattern(entries)
    if pattern is None:
        return text
    lower_map = {k.lower(): v for k, v in entries.items()}

    def _sub(m: "re.Match[str]") -> str:
        return lower_map.get(m.group(0).lower(), m.group(0))

    return pattern.sub(_sub, text)
