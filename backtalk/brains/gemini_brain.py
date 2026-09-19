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
"""Gemini adapter -- a TRUE STUB for Phase 1. No network call exists
here yet: writing one before an API key exists, and before Captain has
approved sending anything to Google, would be exactly the kind of
half-finished, unverifiable code the house rules exist to prevent.

What Phase 1 records instead is the contract: Gemini is external,
requires an explicit per-switch confirmation (requires_confirm_to_switch),
and stays unusable no matter what brain_router.json says until a real
request implementation lands in a later phase.
"""
from __future__ import annotations

import os
from typing import AsyncIterator

from backtalk.brains.base import BrainAdapter, BrainDisabledError, BrainHealth


class GeminiBrain(BrainAdapter):
    id = "gemini"
    label = "Gemini (external -- your own free tier, approval required)"
    requires_tools = False
    requires_confirm_to_switch = True

    def __init__(self, *, enabled: bool = False,
                api_key_env: str = "GEMINI_API_KEY"):
        super().__init__(enabled=enabled)
        self._api_key_env = api_key_env

    async def _check_health(self) -> BrainHealth:
        if not os.environ.get(self._api_key_env):
            return BrainHealth(
                healthy=False,
                reason=f"no {self._api_key_env} configured")
        return BrainHealth(
            healthy=False,
            reason="Gemini is a Phase 1 stub -- no request path exists yet")

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        raise BrainDisabledError(
            "Gemini is a Phase 1 stub: no live request path exists yet, "
            "and every real request will still need its own explicit "
            "approval once built.")
        yield ""  # pragma: no cover -- keeps this an async generator
