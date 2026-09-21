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
Claude -- "cloud brain" (Claude Agent SDK is the only real cloud
brain) speaks the required cloud-specific line and enters the SAME
confirm gate "switch to Claude" uses; nothing switches until an
explicit "confirm".
The asymmetry that governs every pattern below: a false positive here
costs one unwanted brain switch (annoying, but instantly reversible by
saying "switch to qwen"); a false negative means the exact bug this
module exists to close -- Qwen inventing an answer about capabilities
it doesn't have. Tuned toward catching the request, not toward
precision for its own sake.
"""
from __future__ import annotations

import re

# Real live-test failure: Whisper transcribed "Jarvis" as "Java" and
# "Javis". Both are accepted as wake-word aliases here, anchored at
# the very start of the utterance same as "jarvis" always was -- this
# is what keeps "Explain the Java switch statement" untouched (it
# starts with "Explain", not "Java", so this regex never matches
# there at all), and it's also why the aliases are safe even for a
# hypothetical utterance that DID start with "Java ...": the rest of
# the text still has to match one of the narrow, specific patterns
# below to trigger anything, so stripping a leading "Java" that turns
# out to be the real word is harmless -- nothing downstream fires on
# arbitrary leftover text.
_ADDRESS_PREFIX = re.compile(
    r"^\s*(hey|hi|hello|ok|okay)?\s*(jarvis|javis|java)[,.]?\s*",
    re.IGNORECASE)
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


# Real live-test failure: "Which model you are running on Jarvis?"
# never matched -- it says "model" throughout, never "brain" once, so
# none of the original alternatives (all built on the word "brain")
# ever fired, and the question reached a local brain unfiltered, which
# hedged with something like "e.g., Qwen, DeepSeek, or similar"
# instead of naming its own real, current status.
_WHICH_BRAIN = re.compile(
    r"\bwhich brain\b|\bwhat brain\b|\bbrain (are you|is this|status)\b|"
    r"\bwhich model\b|\bwhat model\b|\bmodel (are you|is this|status)\b",
    re.IGNORECASE)
_BRAIN_COUNT = re.compile(
    r"how many brains\b|\bhow many brain modes\b|"
    r"\bwhat brains (do you have|are there|are available)\b",
    re.IGNORECASE)
# Real live-test miss: "How many memory modules do you have?" is a
# DIFFERENT question from "how many brains" -- it's about the vault,
# not the brain roster -- and never matched anything here, so a local
# brain answered it unfiltered and could invent a module count or
# offer to "inspect" vault sections that don't exist. There is exactly
# ONE shared vault; every brain reads the same context from it (see
# vault_context.py). Never checked against _VAULT_TARGET: this is an
# informational question about the vault, not a request to write to
# it, so tool_intent's vault-write refusal must never apply here.
_MEMORY_MODULES_COUNT = re.compile(
    r"\bhow many memory modules\b|\bhow many memory systems\b|"
    r"\bhow many (memory )?vaults\b",
    re.IGNORECASE)
# Broader inventory phrasing, caught live: "write about your models" and
# "what reasoning models are installed" never matched _BRAIN_COUNT (that
# one only recognized "how many"/"what brains ... do you have" shapes),
# so a local brain answered these in prose and invented model names. Any
# of these must produce the same deterministic, truthful inventory --
# EXCEPT when a vault-target word is also present ("write about your
# models IN MEMORY"), which is a vault-write request, not an inventory
# question, and must fall through to tool_intent's Claude-escalation
# refusal instead. See _VAULT_TARGET below.
_BRAIN_TOPIC = re.compile(
    r"\byour (configured )?brains\b|"
    r"\byour (installed |configured )?(reasoning )?models\b|"
    r"\breasoning models\b.{0,20}\b(do you have|installed|available)\b|"
    # Real live-test miss: "what ARE THE brains you have" never
    # matched -- the old pattern required "what" immediately followed
    # by "brains"/"models" with nothing in between. .{0,15} tolerates
    # "are the"/"are" insertions, and "you have" (no "do") is now an
    # accepted suffix on its own, not just "do you have".
    r"\bwhat\b.{0,15}\b(brains|models)\b.{0,20}\b(do you have|"
    r"are there|are available|are installed|you have)\b|"
    r"\bwhich (brains|models)\b.{0,20}\b(do you have|are available)\b|"
    # "list your brains/models" only -- bare "list brains" now means
    # the numbered menu instead (_LIST_BRAINS below), a later, more
    # specific requirement than this narrative-inventory phrasing.
    r"\blist your (brains|models)\b|"
    r"\btell me about (your )?(brains|models)\b|"
    r"\bwrite about (your )?(brains|models)\b",
    re.IGNORECASE)
_VAULT_TARGET = re.compile(
    r"\b(vault|memory|notes?|your world|daily note)\b", re.IGNORECASE)

# "r1 8b" / "r 1 8 b" covers Whisper occasionally splitting the model
# name into separate tokens; "deep seek" covers it hearing two words.
# Real live-test failure: "deep-seek" (hyphenated) never matched --
# \s* only allows WHITESPACE between "deep" and "seek", and a hyphen
# isn't whitespace, so the pattern silently failed on a transcript
# that used a dash instead of a space or nothing at all.
_DEEPSEEK = re.compile(
    r"\bdeepseek\b|\bdeep[\s-]*seek\b|\br\s*1\s*8\s*b\b|\bdeep reasoning\b",
    re.IGNORECASE)
# Real field-test miss: Whisper transcribed "Qwen" as "QN3", which
# matched nothing and fell through to Qwen inventing an answer via
# whichever brain was actually active. "qwen3?" catches "qwen"/"qwen3"
# /"qwen 3"; "q *n *3" and "q *n *three" catch Whisper spelling the
# name out letter by letter, spaced or not ("q n 3", "qn3", "q n three").
_QWEN = re.compile(
    r"\bqwen\s*3?\b|\bq\s*n\s*3\b|\bq\s*n\s*three\b|\blocal brain\b",
    re.IGNORECASE)
_CLAUDE = re.compile(r"\bclaude\b", re.IGNORECASE)
_GEMINI = re.compile(r"\bgemini\b", re.IGNORECASE)
# "switch to Gemini or Claude" must never silently pick one -- checked
# BEFORE the individual cloud/claude/gemini checks below so an
# ambiguous request can never resolve to either brain by accident of
# check order. Requires an actual switch verb: "Gemini or Claude,
# which is better?" is a real question, not a switch request, and must
# never trigger a clarification prompt.
_AMBIGUOUS_CLOUD_BRAIN = re.compile(
    r"\b(gemini|claude|cloud)\b[^.!?]{0,15}\bor\b[^.!?]{0,15}"
    r"\b(gemini|claude|cloud)\b",
    re.IGNORECASE)
# Real field-test miss: none of these six exact spoken aliases ever
# matched -- "switch to cloud" and "switch to cloud agent" have no
# "brain"/"model" suffix and no "the" before "cloud", and "switch to
# the third reasoning brain cloud" doesn't have "cloud" adjacent to
# "the" at all. Each fell through un-intercepted and reached whichever
# local brain was active, which invented an answer instead of routing
# to Claude (the only real cloud brain -- see the "cloudbrain" verb's
# handling in main.py, which now gates a real Claude switch here,
# never just an FYI).
_CLOUD = re.compile(
    r"\bswitch to (the )?cloud( brain| agent| model)?\b|"
    r"\buse (the )?cloud( brain| agent| model)?\b|"
    r"\bthe cloud\b|\bcloud brain\b|\bcloud model\b|\bcloud agent\b|"
    r"\b(third|3rd) reasoning brain\b.{0,20}\bcloud\b",
    re.IGNORECASE)
_SWITCH_VERB = re.compile(
    r"\bswitch\b|\buse\b|\bchange to\b|\bgo to\b|\bcan you\b",
    re.IGNORECASE)

# The deterministic NUMBERED brain interface: 1=Qwen, 2=DeepSeek,
# 3=Claude, 4=Gemini. Deliberately its own independent pattern, never
# routed through _CLOUD or _CLAUDE -- "numbers are canonical" means a
# number always means exactly that stable slot, with no dependency on
# word-based cloud/Claude phrasing at all. "tree" is a real Whisper
# mis-hearing of "three", included the same way "QN3"/"deep seek"
# cover other models' transcription drift elsewhere in this file.
_NUM_WORD = r"(?:1|one|2|two|3|three|tree|4|four)"
_NUM_TO_ID = {
    "1": "1", "one": "1",
    "2": "2", "two": "2",
    "3": "3", "three": "3", "tree": "3",
    "4": "4", "four": "4",
}
# Real live-test miss: "Java is switched to 4" uses a passive-voice
# form ("is switched to") that "switch to"/"use" never covered, and
# "switch brain to" (verb + object + "to") is a real spoken variant of
# "switch to brain" too. All three verb shapes accept the same
# optional "brain" before the number.
_SWITCH_VERB_PHRASE = (
    r"(?:switch(?:\s+brain)?\s+to|(?:is|are|was|were)\s+switched\s+to|"
    r"use)")
_SWITCH_BRAIN_NUMBER = re.compile(
    rf"\b{_SWITCH_VERB_PHRASE}\s+(?:brain\s+)?({_NUM_WORD})\b",
    re.IGNORECASE)
# A bare "switch" (or "switch brain(s)"/"switch the brain") with no
# target named at all -- asks which number, and main.py stores that as
# one-turn context so the VERY NEXT utterance, if it's just a bare
# number, resolves it. This must never fire on anything more specific:
# detect() only reaches this after every named-brain/cloud/number/
# memory-brain check above it has already failed to match, since
# Python returns on first hit. "switch the brain" is a real live-test
# transcript ("Java switch the brain") that "switch brains?" alone
# never covered.
_BARE_SWITCH = re.compile(
    r"^switch(\s+(the\s+)?brains?)?[.!]?$", re.IGNORECASE)
# Real live-test transcript: "Jarvis switch your memory brain" --
# Captain's own vault-context phrase for what the local brains read
# (see vault_context.py), not an actual fifth brain. Checked BEFORE
# _BARE_SWITCH so "switch your memory brain" gets the explanation
# instead of being swallowed by the plain "switch brain(s)" shape.
_MEMORY_BRAIN_SWITCH = re.compile(
    r"^switch (your |my )?memory brain[.!]?$", re.IGNORECASE)
# "list brains" / "brain options" / "which brains" -- the numbered
# menu, distinct from _BRAIN_COUNT's narrative sentence (which only
# names ENABLED brains): this always names all four stable slots.
_LIST_BRAINS = re.compile(
    r"\blist brains\b|\bbrain options\b|\bwhich brains\b", re.IGNORECASE)

# "Jarvis, close the day" and its three siblings -- caught with the
# same address-prefix/filler tolerance as every other verb here, since
# the exact bare phrase also lives in CONSOLE_VERBS for the fast path.
_CLOSE_DAY = re.compile(
    r"\bclose the day\b|\bsummarise today\b|\bsummarize today\b|"
    r"\bmake today'?s note\b|\bend[- ]of[- ]day summary\b",
    re.IGNORECASE)


def detect(text: str) -> str | None:
    """Returns a CONSOLE_VERBS-compatible verb string, or None to mean
    "no opinion -- fall through to console_match() and then the
    model." Callers must treat a returned verb exactly like one
    console_match() found (main.py already does)."""
    stripped = _strip_wrapper(text)
    wants_switch = bool(_SWITCH_VERB.search(stripped))

    if _CLOSE_DAY.search(stripped):
        return "closeday"

    # Checked before "which brain": a real field test showed Qwen
    # inventing an answer to "how many brains do you have?" instead of
    # this ever being intercepted -- distinct question from "which
    # brain are you ON right now", so it gets its own verb and its own
    # dynamically-built (never hardcoded) answer in main.py.
    if _BRAIN_COUNT.search(stripped):
        return "brainscount"

    # Distinct from brainscount: a question about the VAULT/memory
    # infrastructure, never about which brains exist.
    if _MEMORY_MODULES_COUNT.search(stripped):
        return "memorymodules"

    # A vault-target word means this is really a vault-write request
    # ("write about your models IN MEMORY") -- never intercept it here;
    # let it fall through to tool_intent's Claude-escalation refusal.
    if _BRAIN_TOPIC.search(stripped) and not _VAULT_TARGET.search(stripped):
        return "brainscount"

    # The numbered interface, checked early and independently of every
    # word-based brain check below it -- "numbers are canonical", so a
    # number never has to compete with or get reinterpreted by the
    # cloud/Claude word patterns.
    m = _SWITCH_BRAIN_NUMBER.search(stripped)
    if m:
        return f"switchnum:{_NUM_TO_ID[m.group(1).lower()]}"

    if _LIST_BRAINS.search(stripped):
        return "listbrains"

    if _MEMORY_BRAIN_SWITCH.search(stripped):
        return "switchmemorybrain"

    if _WHICH_BRAIN.search(stripped):
        return "whichbrain"

    # Checked BEFORE every cloud/claude/gemini check below: "switch to
    # Gemini or Claude" must never resolve to either one just because
    # of which check happens to run first.
    if wants_switch and _AMBIGUOUS_CLOUD_BRAIN.search(stripped):
        return "ambiguousbrain"

    # Checked BEFORE the plain "claude" check on purpose: this catches
    # every "cloud"-shaped phrasing that never says Claude's name
    # outright. Claude Agent SDK is the only real cloud brain -- it
    # gets its own verb here (never falling into useclaude directly)
    # so main.py can speak the required cloud-specific line before
    # entering the SAME confirm-gated switch flow "switch to claude"
    # uses. If someone actually says "claude", that's unambiguous and
    # skips straight to the plain Claude check below instead.
    if _CLOUD.search(stripped) and not _CLAUDE.search(stripped):
        return "cloudbrain"

    if wants_switch and _DEEPSEEK.search(stripped):
        return "usedeepseek"
    if wants_switch and _QWEN.search(stripped):
        return "useqwen"
    if wants_switch and _GEMINI.search(stripped):
        return "usegemini"
    if wants_switch and _CLAUDE.search(stripped):
        return "useclaude"

    # Lowest priority, checked last on purpose: a bare "switch" only
    # ever means "ask which number" once every more specific pattern
    # above -- a named brain, a number, cloud, list -- has already
    # failed to match.
    if _BARE_SWITCH.search(stripped):
        return "switchask"

    return None
