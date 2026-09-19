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
"""Deterministic, natural-language-tolerant detector for the handful of
brain-control questions a person actually says out loud, as opposed to
the terse EXACT phrases ("switch to qwen") that main.py's CONSOLE_VERBS
system requires spoken alone.

Caught live, in a real F8 field test: three natural phrasings, three
failures.
  - "Hello Jarvis, which brain are you using?" never matched
    CONSOLE_VERBS's "which brain are you using" because of the leading
    "Hello Jarvis," address -- console_match() requires the WHOLE
    normalized utterance to equal a phrase, verbatim.
  - "Jarvis, can you switch to DeepSeek R1 8B?" never matched
    "switch to deepseek" for the same reason, plus real speech names
    a model more freely than a fixed phrase list ever anticipates.
  - "Jarvis, can you switch to cloud brain?" doesn't match ANY existing
    phrase at all -- "cloud brain" was never a recognized synonym for
    anything, so Qwen answered it as an ordinary question and, having
    no idea a router even exists, invented an answer.

This module runs BEFORE the model sees a word (main.py tries it as a
fallback when the strict console_match() finds nothing), and is
deliberately narrow: it only ever recognizes brain-control intent, it
never touches ordinary conversation, and it never silently activates
Claude -- "cloud brain" gets an honest status report and a pointer at
the existing confirm-gated switch phrase, never a direct activation.
The asymmetry that governs every pattern below: a false positive here
costs one unwanted brain switch (annoying, but instantly reversible by
saying "switch to qwen"); a false negative means the exact bug this
module exists to close -- Qwen inventing an answer about capabilities
it doesn't have. Tuned toward catching the request, not toward
precision for its own sake.
"""
from __future__ import annotations

import re

_ADDRESS_PREFIX = re.compile(
    r"^\s*(hey|hi|hello|ok|okay)?\s*jarvis[,.]?\s*", re.IGNORECASE)
_FILLER_PREFIX = re.compile(
    r"^\s*(can you|could you|would you|please|i want to|i'd like to)\s+",
    re.IGNORECASE)


def _strip_wrapper(text: str) -> str:
    """Strip a leading name-address and/or a polite-request filler, so
    "Hello Jarvis, can you switch to DeepSeek?" reduces to "switch to
    deepseek" before matching. Only ever strips from the FRONT --
    never rewrites the middle of a sentence, so this can't misfire on
    ordinary conversation that happens to mention a brain by name
    partway through."""
    t = _ADDRESS_PREFIX.sub("", text)
    t = _FILLER_PREFIX.sub("", t)
    return t.strip(" ?.!")


_WHICH_BRAIN = re.compile(
    r"\bwhich brain\b|\bwhat brain\b|\bbrain (are you|is this|status)\b",
    re.IGNORECASE)

# "r1 8b" / "r 1 8 b" covers Whisper occasionally splitting the model
# name into separate tokens; "deep seek" covers it hearing two words.
_DEEPSEEK = re.compile(
    r"\bdeepseek\b|\bdeep\s*seek\b|\br\s*1\s*8\s*b\b|\bdeep reasoning\b",
    re.IGNORECASE)
_QWEN = re.compile(r"\bqwen\b|\blocal brain\b", re.IGNORECASE)
_CLAUDE = re.compile(r"\bclaude\b", re.IGNORECASE)
_CLOUD = re.compile(r"\bcloud brain\b|\bcloud model\b|\bthe cloud\b",
                    re.IGNORECASE)
_SWITCH_VERB = re.compile(
    r"\bswitch\b|\buse\b|\bchange to\b|\bgo to\b|\bcan you\b",
    re.IGNORECASE)


def detect(text: str) -> str | None:
    """Returns a CONSOLE_VERBS-compatible verb string, or None to mean
    "no opinion -- fall through to console_match() and then the
    model." Callers must treat a returned verb exactly like one
    console_match() found (main.py already does)."""
    stripped = _strip_wrapper(text)

    if _WHICH_BRAIN.search(stripped):
        return "whichbrain"

    # Checked BEFORE Claude on purpose: "cloud brain" must never
    # resolve to a direct Claude activation, even though Claude is
    # also, technically, a cloud brain -- Gemini is what "the cloud
    # brain" means here (Claude already has its own name and its own
    # exact phrase), and this path only ever informs, never switches.
    if _CLOUD.search(stripped) and not _CLAUDE.search(stripped):
        return "cloudbrain"

    wants_switch = bool(_SWITCH_VERB.search(stripped))
    if wants_switch and _DEEPSEEK.search(stripped):
        return "usedeepseek"
    if wants_switch and _QWEN.search(stripped):
        return "useqwen"
    if wants_switch and _CLAUDE.search(stripped):
        return "useclaude"

    return None
