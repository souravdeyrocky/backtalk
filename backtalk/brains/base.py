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
"""Shared interface every brain adapter implements (Qwen, DeepSeek,
Gemini, Claude), plus the status contract the router reports from.

Phase 1: importing this module, or any adapter built on it, has NO
effect on the live F8 path -- nothing in backtalk.main constructs a
BrainAdapter yet. Every adapter also carries its own `enabled` flag,
read from brain_router.json, defaulting to False for all four brains.
There is no method anywhere in this file that flips one on by itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator


class BrainDisabledError(RuntimeError):
    """Raised when something tries to start or use an adapter that
    config has not enabled. Callers must surface this, never catch it
    and reroute to a different brain."""


class BrainUnavailableError(RuntimeError):
    """Raised when an adapter IS enabled but can't currently serve a
    turn (Ollama unreachable, a model not pulled, no API key). Distinct
    from BrainDisabledError so a status readout can tell "off on
    purpose" apart from "on but broken"."""


@dataclass
class BrainHealth:
    """One adapter's honest self-report from health(). A `healthy=False`
    reading must always carry a human-readable `reason` -- a blank
    reason on an unhealthy result is a bug in the adapter, not a valid
    state."""
    healthy: bool
    reason: str = ""
    checked_at: float = field(default_factory=time.time)


@dataclass
class BrainStatus:
    """What the router (and, later, a face or a test) reads to know
    where a brain stands: enabled by config, healthy right now, and WHY
    when it's neither. Whether this brain is the ACTIVE one is not
    stored here -- that lives in exactly one place, RouterStatus.active_id
    in router.py, so there is never a second copy of that fact to drift
    out of sync.

    capability_summary is the honest capability matrix, in one spoken-
    ready sentence: what this brain can ACTUALLY do right now, not what
    it might do once more is built. A brain must never claim a
    capability (file edits, commands, browsing, vault writes) it does
    not really have behind an explicit permission gate."""
    id: str
    label: str
    enabled: bool
    healthy: bool
    reason: str = ""
    requires_tools: bool = False
    requires_confirm_to_switch: bool = False
    capability_summary: str = ""


class BrainAdapter:
    """Base class every brain implements. See the module docstring for
    why constructing or subclassing this has no effect on F8 yet."""

    id: str = "base"
    label: str = "Base brain (do not use directly)"
    requires_tools: bool = False
    requires_confirm_to_switch: bool = False
    capability_summary: str = "no capabilities declared"

    def __init__(self, *, enabled: bool = False):
        self.enabled = enabled
        self.session = {"turns": 0, "out_tokens": 0, "in_tokens": 0,
                        "cost": 0.0}
        self._started = False
        self._last_health: BrainHealth | None = None

    async def health(self) -> BrainHealth:
        """Cheap, side-effect-free reachability check. Subclasses
        override _check_health(); this wrapper enforces the one rule
        every adapter shares: a disabled brain is always unhealthy, and
        never even asked to check itself."""
        if not self.enabled:
            h = BrainHealth(healthy=False, reason="disabled by configuration")
        else:
            h = await self._check_health()
        self._last_health = h
        return h

    async def _check_health(self) -> BrainHealth:
        """Subclasses override. Must never raise -- an adapter that
        can't tell whether it's healthy reports
        BrainHealth(healthy=False, reason=...), it doesn't let the
        exception propagate."""
        return BrainHealth(healthy=False,
                           reason="health check not implemented")

    def record_failed_health(self, reason: str) -> None:
        """Used by the router when an adapter's health() itself raises,
        so one broken adapter can't take down a whole status readout."""
        self._last_health = BrainHealth(healthy=False, reason=reason)

    def status(self) -> BrainStatus:
        """A disabled brain is always and immediately unhealthy -- that
        doesn't need a probe to know, so it's reported without one.
        An enabled brain reports its last health() reading, or "not
        checked yet" until one has run."""
        if not self.enabled:
            return BrainStatus(
                id=self.id, label=self.label, enabled=False, healthy=False,
                reason="disabled by configuration",
                requires_tools=self.requires_tools,
                requires_confirm_to_switch=self.requires_confirm_to_switch,
                capability_summary=self.capability_summary,
            )
        h = self._last_health
        return BrainStatus(
            id=self.id, label=self.label, enabled=True,
            healthy=bool(h and h.healthy),
            reason=(h.reason if h else "not checked yet"),
            requires_tools=self.requires_tools,
            requires_confirm_to_switch=self.requires_confirm_to_switch,
            capability_summary=self.capability_summary,
        )

    async def start(self) -> None:
        if not self.enabled:
            raise BrainDisabledError(
                f"{self.id} is disabled by configuration")
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover -- keeps this an async generator

    async def command(self, cmd: str) -> str:
        raise NotImplementedError

    async def interrupt(self) -> None:
        pass

    async def reset_turn(self, timeout: float = 8.0) -> None:
        pass

    async def set_permission_mode(self, mode: str) -> None:
        pass

    async def context_usage(self):
        return None
