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
"""Converts raw model output (which may carry Markdown, LaTeX, emoji,
links, file paths, and other WRITTEN-text conventions) into clean,
natural SPOKEN text -- the one shared renderer every brain's reply
passes through before Kokoro (or ElevenLabs) ever sees it, so the
voice experience is identical regardless of which brain answered.

Caught live in a real F8 field test: Qwen/DeepSeek output routinely
carries **bold**, numbered/bulleted lists, and even LaTeX math blocks
(`$$x + 1 = 2$$`), none of which backtalk.brain's Claude path ever
produces -- Claude's own system prompt (DISCIPLINE, in config.py)
already instructs it to write for the ear. This module is the
backstop for every brain that doesn't reliably follow that on its
own, applied uniformly so Claude's already-clean output passes
through unchanged (this function is idempotent on plain prose).

Applied to a COMPLETE sentence/chunk at a time, never a raw streaming
fragment: a construct like **bold** can straddle two network chunks
mid-token, so sanitizing has to wait until each adapter's own
sentence-boundary detector has already assembled a whole unit of
text. router.py is where this actually gets called, once, for every
brain -- see its ask_stream().
"""
from __future__ import annotations

import re

# Fenced code blocks: drop entirely. Code is not speakable, and
# DISCIPLINE already tells Claude never to emit one in voice mode;
# this is the backstop for a brain that doesn't reliably follow that.
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")

# [text](url) -> text ; [[Note Name]] -> Note Name (Obsidian wikilink)
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
_BARE_URL = re.compile(r"https?://\S+")

# **bold**, __bold__ -> bold ; *italic* -> italic (asterisk only --
# single-underscore italic is deliberately NOT touched: it collides
# with ordinary identifiers like max_tokens far too often to be worth
# the false positives, and it's not the construct that was observed).
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)")

# Heading markers, bullet markers, horizontal rules -- all at the
# START of a line, stripped down to just the text.
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"(?m)^\s*[-*+]\s+")
_RULE = re.compile(r"(?m)^\s*([-*_])\1{2,}\s*$")

# Numbered-list markers get an ORDINAL WORD instead of being silently
# deleted: "1. Buy milk 2. Walk dog" used to collapse to "Buy milk
# Walk dog" with no transition at all -- correct written-text
# stripping, but unnatural spoken-aloud, since the ear needs the
# sequence marker prose already carries visually. "First, Buy milk.
# Second, Walk dog." reads the way a person actually narrates a list.
# Only lines 1 through 20 get a real ordinal; a longer list falls back
# to "Next," rather than growing an ever-longer word table for
# something nobody actually dictates that long aloud.
_NUMBERED = re.compile(r"(?m)^\s*(\d+)[.)]\s+")
_ORDINAL_WORDS = (
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh",
    "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth", "Thirteenth",
    "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth",
    "Eighteenth", "Nineteenth", "Twentieth")


def _ordinalize_numbered_marker(m: "re.Match") -> str:
    n = int(m.group(1))
    word = (_ORDINAL_WORDS[n - 1] if 1 <= n <= len(_ORDINAL_WORDS)
           else "Next")
    return f"{word}, "

# LaTeX math delimiters ($$...$$ / $...$): strip the delimiters so the
# TTS never reads literal dollar signs ("dollar dollar"). The math
# notation ITSELF is not translated to English -- that's a much bigger
# problem than this module solves -- but plain algebra (numbers,
# operators, parens) reads tolerably once the delimiters are gone.
_LATEX_BLOCK = re.compile(r"\${2}(.+?)\${2}", re.DOTALL)
_LATEX_INLINE = re.compile(r"\$(.+?)\$")

# A conservative emoji + pictograph range sweep. Deliberately broad:
# better to silently drop a rare harmless symbol than let "party
# popper" get read aloud character by character.
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U0001F900-\U0001F9FF\U00002B00-\U00002BFF\U0000FE0F]+")

# Windows/POSIX-shaped paths, collapsed to just the base file name --
# matching DISCIPLINE's existing rule for Claude ("say the file, not
# its address"). This is the backstop for brains that don't reliably
# follow that instruction on their own.
_WIN_PATH = re.compile(r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]+")
_POSIX_PATH = re.compile(r"(?:/[\w.\-]+){2,}")

# Generic chatbot closings, caught live from Qwen/DeepSeek: a habit of
# tacking a service-desk sign-off onto every reply regardless of
# whether one was warranted ("How may I assist you today?" after
# already answering the question in full). Claude's own DISCIPLINE
# prompt never produces these, and ollama_brain.py's system prompt now
# also tells Qwen/DeepSeek not to -- this is the backstop for when a
# local model does it anyway.
#
# A first version of this matched four exact fixed phrases -- and a
# SECOND live field test immediately produced a fifth ("What would you
# like to accomplish with your local AI assistant?") that was never
# any of them, wording-wise. A fixed phrase list can only ever chase
# the model's next paraphrase one field test behind, so this matches
# the SHAPE of a service-desk sign-off instead (anchored at the start
# of the sentence, since to_speech() runs on one already-complete
# sentence at a time -- a closing IS the whole sentence, not a
# fragment of a longer one). Still deliberately narrow: a real
# clarifying question ("Which daily note, today's or yesterday's?")
# matches none of these shapes, so it's never at risk of being swept
# up here.
_CLOSING_PATTERNS = tuple(re.compile(p) for p in (
    r"^how (can|may) i (assist|help) you\b",
    r"^what would you like to (explore|accomplish|do|achieve)\b",
    r"^let me know if you need anything else\b",
    r"^is there anything else i can (help|assist)( you)? with\b",
    # A THIRD live field test produced two more: "Let me know how I
    # can assist" / "...assist further" (no boundary on the end lets
    # "further" ride along), and "Let me know if you'd like an example
    # or clarification" -- normalization strips the apostrophe in
    # "you'd" down to "youd", not a space, so that's matched literally
    # alongside the un-contracted "you would" form.
    r"^let me know how i can assist\b",
    r"^let me know if you(d| would) like\b",
))
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def _is_generic_closing(sentence: str) -> bool:
    norm = _NON_ALNUM.sub("", sentence.lower())
    norm = " ".join(norm.split())
    return any(p.search(norm) for p in _CLOSING_PATTERNS)


def _basename_no_ext(path: str) -> str:
    name = re.split(r"[\\/]", path)[-1]
    return re.sub(r"\.[A-Za-z0-9]{1,6}$", "", name) or path


def _speak_path(m: "re.Match") -> str:
    return _basename_no_ext(m.group(0))


def to_speech(text: str) -> str:
    """Strip every written-text convention that has no business being
    spoken aloud, preserving the actual words. Idempotent: calling it
    on text that's already clean prose returns that text unchanged."""
    if not text:
        return text
    t = text
    t = _CODE_FENCE.sub(" ", t)
    t = _LATEX_BLOCK.sub(lambda m: m.group(1), t)
    t = _LATEX_INLINE.sub(lambda m: m.group(1), t)
    # Markdown links BEFORE the bare-URL fallback: a bare-URL match is
    # greedy on non-whitespace and would eat the link's closing paren,
    # leaving the markdown syntax broken and unmatchable afterward.
    t = _MD_LINK.sub(lambda m: m.group(1), t)
    t = _BARE_URL.sub("a link", t)
    t = _WIKILINK.sub(lambda m: m.group(1), t)
    t = _WIN_PATH.sub(_speak_path, t)
    t = _POSIX_PATH.sub(_speak_path, t)
    t = _INLINE_CODE.sub(lambda m: m.group(1), t)
    t = _HEADING.sub("", t)
    t = _RULE.sub("", t)
    t = _BULLET.sub("", t)
    t = _NUMBERED.sub(_ordinalize_numbered_marker, t)
    t = _BOLD.sub(lambda m: m.group(1) or m.group(2), t)
    t = _ITALIC.sub(lambda m: m.group(1), t)
    t = _EMOJI.sub("", t)
    # Leftover stray structural punctuation: backticks, table pipes,
    # lone asterisks/hashes that weren't part of a matched pair above
    # (e.g. a single trailing "*" the model never closed). Underscore
    # is deliberately NOT swept here -- unlike * and #, a lone
    # underscore is far more often a real identifier character
    # (max_tokens) than leftover markdown, so it's left alone.
    t = t.replace("`", "").replace("|", ", ")
    t = re.sub(r"[*#]+", "", t)
    t = " ".join(t.split())
    t = t.strip()
    if _is_generic_closing(t):
        return ""
    return t
