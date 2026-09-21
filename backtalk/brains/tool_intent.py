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
"""Heuristic detector for utterances that plainly ask for something
only Claude's Agent SDK tools can do (edit a file, run a command,
browse the web, write vault memory) -- so a brain with no tools can
refuse HONESTLY instead of answering in prose as if it had done the
thing, and instead of silently switching brains on the person's
behalf.

Deliberately conservative and keyword-based, not a real intent
classifier, and the asymmetry here is the point: a false positive
just costs one extra "switch to Claude?" line the person can ignore; a
false negative just means the local brain answers in plain
conversation, which is always safe since it has no tool access
regardless of what it says. This exists to catch the DANGEROUS
direction -- a brain implying it acted when it didn't -- not to be a
precise parser of every possible phrasing.
"""
from __future__ import annotations

import re

_PATTERNS = (
    r"\b(edit|write|create|delete|save)\b[^?!]{0,40}\bfile\b",
    r"\brun\b[^?!]{0,40}\b(command|script|powershell|bash|terminal)\b",
    r"\bexecute\b[^?!]{0,40}\b(command|script)\b",
    # "search the internet" is a real field-test miss: the original
    # pattern only recognized "browse/fetch/open" as the verb and
    # never matched "search", so it reached Qwen unfiltered and got a
    # confused, invented answer instead of the deterministic refusal.
    r"\b(browse|fetch|open|search)\b[^?!]{0,40}"
    r"\b(website|url|web page|link|http|internet|the web)\b",
    r"\b(remember|save|write|update|add)\b[^?!]{0,40}"
    r"\b(vault|memory|notes?|your world|daily note)\b",
)
# NOTE: the span excludes only "?" and "!" as hard stops, not ".", on
# purpose -- a bare period shows up constantly inside the exact things
# people ask to edit ("backtalk.json", "main.py"), and a heuristic that
# can't bridge that misses the request entirely (a real gap, caught by
# a live proof run: "edit my backtalk.json file" slipped past the
# period-excluding version and reached Ollama unfiltered). The 40-char
# cap alone is enough boundary control for this heuristic's purpose;
# see the module docstring for why a false positive here is cheap and
# a false negative is the direction that actually matters.
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

REFUSAL = (
    "This requires Claude Agent tools. Switch to Claude for this action? "
    "Say switch to Claude, then confirm, if you'd like to."
)


def looks_like_tool_request(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)
