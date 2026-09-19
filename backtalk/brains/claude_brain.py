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
"""Claude adapter -- wraps the existing, unmodified backtalk.brain.WarmBrain.
The only thing new here is the on/off gate: WarmBrain itself has no
concept of "disabled", so this class refuses to start() unless config
says this adapter is enabled. Phase 1 ships that flag off, same as every
other brain, and nothing in backtalk.main constructs a ClaudeBrain yet --
so today's live F8 path still talks to WarmBrain directly, unchanged.
"""
from __future__ import annotations

from typing import AsyncIterator

from backtalk.brain import WarmBrain
from backtalk.brains.base import BrainAdapter, BrainDisabledError, BrainHealth


class ClaudeBrain(BrainAdapter):
    id = "claude"
    label = "Claude (Agent SDK -- consumes your subscription usage)"
    requires_tools = True
    requires_confirm_to_switch = True
    capability_summary = (
        "the full Claude Agent SDK toolset -- file edits, running "
        "commands, web fetch, and vault writes -- every one of them "
        "gated by the spoken permission check before it acts.")

    def __init__(self, *, enabled: bool = False, model: str | None = None,
                can_use_tool=None, resume_id: str | None = None):
        super().__init__(enabled=enabled)
        self._model = model
        self._can_use_tool = can_use_tool
        # Consumed once, naturally: self._warm is only ever constructed
        # the first time start() runs, whether that's at boot (recovery
        # mode) or later via a live "switch to Claude" -- so a saved
        # resume_last_session id is honored exactly once per backtalk
        # launch, matching the original WarmBrain behavior this adapter
        # wraps.
        self._resume_id = resume_id
        self._warm: WarmBrain | None = None

    async def _check_health(self) -> BrainHealth:
        # A real reachability check means opening an SDK session, which
        # is too expensive to run on every status poll. Reported healthy
        # once enabled: actual connectivity is proven, honestly, at
        # start() -- exactly like backtalk.main's own guarded connect.
        if self._started and self._warm is not None:
            return BrainHealth(healthy=True)
        return BrainHealth(
            healthy=True,
            reason="not connected yet -- connects lazily on first use")

    async def start(self) -> None:
        if not self.enabled:
            raise BrainDisabledError("claude is disabled by configuration")
        if self._warm is None:
            self._warm = WarmBrain(model=self._model,
                                   can_use_tool=self._can_use_tool,
                                   resume_id=self._resume_id)
        await self._warm.start()
        self._started = True

    async def stop(self) -> None:
        if self._warm is not None:
            await self._warm.stop()
        self._started = False

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        if not self.enabled or self._warm is None:
            raise BrainDisabledError("claude is disabled or not started")
        async for sentence in self._warm.ask_stream(utterance):
            yield sentence
        self.session = self._warm.session

    async def command(self, cmd: str) -> str:
        if self._warm is None:
            raise BrainDisabledError("claude is disabled or not started")
        return await self._warm.command(cmd)

    async def interrupt(self) -> None:
        if self._warm is not None:
            await self._warm.interrupt()

    async def reset_turn(self, timeout: float = 8.0) -> None:
        if self._warm is not None:
            await self._warm.reset_turn(timeout)

    async def set_permission_mode(self, mode: str) -> None:
        if self._warm is not None:
            await self._warm.set_permission_mode(mode)

    async def context_usage(self):
        if self._warm is None:
            return None
        return await self._warm.context_usage()
