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
"""Smart brain-recommendation layer: a light check of whether a
DIFFERENT brain would genuinely serve THIS request better --
reasoning complexity toward DeepSeek, a need for current public
information toward Gemini. Recommends, then asks; NEVER switches
anything itself, mirroring exactly how the router's own "cloudbrain"
response already behaves. Manual "switch to X" commands stay direct
and completely untouched by this module.

Deliberately does NOT also flag tool-shaped requests (file edit, run
a command, browse, vault write): tool_intent.py already refuses those
outright and names Claude as the escalation right there in its own
REFUSAL text. Suggesting a second time here would just be redundant
noise stacked on an already-firm refusal.

Every suggestion checks the CANDIDATE brain's own live status first
("available brain health" from the task spec) -- this never
recommends a brain that's disabled or currently unhealthy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backtalk.brains import tool_intent

_LIVE_INFO = re.compile(
    r"\b(current|latest|today'?s|right now|this week|recent news|"
    r"what'?s happening|up to date|as of today|breaking news|"
    r"stock price|weather (today|right now|now))\b",
    re.IGNORECASE)

_COMPLEX_REASONING = re.compile(
    r"\b(plan out|think through|step[- ]by[- ]step|"
    r"work through the logic|trade-?offs?|\bstrategy\b|"
    r"analy[sz]e (deeply|carefully)|prove that|optimi[sz]e|"
    r"multi-?step)\b",
    re.IGNORECASE)


@dataclass
class Recommendation:
    brain_id: str
    spoken_line: str


def recommend(utterance: str, router) -> "Recommendation | None":
    """Returns a suggestion, or None to mean "stay on the current
    brain, nothing to add." Short and natural, for the ear -- never a
    technical dump of what was detected or why."""
    if tool_intent.looks_like_tool_request(utterance):
        return None  # tool_intent's own refusal already covers this

    status = router.status()

    def _usable(bid: str) -> bool:
        s = status.brains.get(bid)
        return bool(s and s.enabled)

    if (status.active_id != "deepseek-r1-8b-local"
            and _COMPLEX_REASONING.search(utterance)
            and _usable("deepseek-r1-8b-local")):
        return Recommendation(
            "deepseek-r1-8b-local",
            "This sounds like it could use some careful reasoning -- "
            "want me to switch to deep reasoning for this one?")

    if (status.active_id != "gemini"
            and _LIVE_INFO.search(utterance)
            and _usable("gemini")):
        return Recommendation(
            "gemini",
            "That needs current information I don't have locally -- "
            "want me to check with Gemini? You'll approve exactly "
            "what gets sent first.")

    return None
