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
"""Bounded, read-only vault context for brains with no filesystem
tools (Qwen, DeepSeek). Mirrors local-jarvis/memory.py's exact policy
on purpose: read ONLY three files, only a concise excerpt of each,
never the whole vault -- this is a policy boundary, not a default to
widen casually. Duplicated here rather than imported across the
sibling local-jarvis folder so this fork stays self-contained.

Claude keeps its own, separate, full read/write vault access via
backtalk.json's extra_dirs -- unchanged, untouched by this module.
This is what a brain with NO tools gets instead: read-only, capped,
three files, loaded fresh on every turn so a same-day edit (a new
daily-note session, a /remember'd fact once that's wired up) shows up
without a restart.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

VAULT_ROOT = Path("D:/JARVIS-Vault")
VAULT_INDEX = VAULT_ROOT / "VAULT-INDEX.md"
CAPTAIN_PROFILE = VAULT_ROOT / "Captain Profile.md"

_EXCERPT_CHARS = 900


def _today_daily_note_path(today: _dt.date | None = None) -> Path:
    today = today or _dt.date.today()
    month_folder = f"{today.month:02d} - {today.strftime('%B')} {today.year}"
    filename = today.strftime("%Y-%m-%d.md")
    return VAULT_ROOT / "01 - Daily Notes" / month_folder / filename


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "\n...(truncated)"


def _daily_note_excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    """Prefer the '## Index' block over the full session bodies -- the
    one file that grows all day is where "concise, not the whole
    vault" actually has to be enforced."""
    marker = "## Index"
    idx = text.find(marker)
    if idx == -1:
        return _excerpt(text, limit)
    rest = text[idx:]
    session_idx = rest.find("## Session", len(marker))
    block = rest if session_idx == -1 else rest[:session_idx]
    return _excerpt(block, limit)


def load_context(today: _dt.date | None = None) -> str:
    """Concise context from the three approved vault files. A missing
    file (fresh vault, no daily note yet today) is skipped, never an
    error -- this must never block a turn."""
    parts = []

    index_text = _read_text(VAULT_INDEX)
    if index_text:
        parts.append("# Vault Index (excerpt)\n" + _excerpt(index_text))

    profile_text = _read_text(CAPTAIN_PROFILE)
    if profile_text:
        parts.append("# Captain Profile (excerpt)\n" + _excerpt(profile_text))

    daily_text = _read_text(_today_daily_note_path(today))
    if daily_text:
        parts.append("# Today's Daily Note (excerpt)\n"
                     + _daily_note_excerpt(daily_text))

    return "\n\n".join(parts)
