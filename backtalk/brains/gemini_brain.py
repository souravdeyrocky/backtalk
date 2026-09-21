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
"""Gemini adapter -- built out for real from the Phase 1 stub, but
still fully gated: enabled=false by default, requires an explicit
confirm to switch onto (like Claude), AND requires an external
consent LEASE (requires_external_lease) -- confirming the switch opens
a 30-minute idle window during which requests send without re-asking;
after 30 minutes of inactivity the next request shows its exact
outgoing text and asks for consent again. See main.py's
_EXTERNAL_LEASE for the mechanism, shared with Claude.

No vault context, ever -- not even the bounded excerpt Qwen/DeepSeek
get. That's not a policy promise enforced elsewhere, it's structural:
there is no vault_context import anywhere in this file, so there is
nothing here that COULD attach vault content even by mistake. The
only thing this adapter ever sends is the bare utterance itself (see
build_request_preview) -- never a file, never the profile, never
anything beyond what Captain actually said in that turn.

The API key is read fresh from the Windows user environment
(GEMINI_API_KEY by default) at the moment of each call -- never
cached on self, never logged, never written to brain_router.json,
.env, or any vault note. Sent as a request HEADER, not a query-string
parameter, specifically so it can never end up embedded in a URL that
an error message or a log line might later repeat.
"""
from __future__ import annotations

import os
import re
from typing import AsyncIterator

import httpx

from backtalk import phone_bridge
from backtalk.brains.base import (
    BrainAdapter,
    BrainDisabledError,
    BrainHealth,
    BrainUnavailableError,
)
from backtalk.identity import (GEMINI_CAPABILITY_RULES,
                               JARVIS_CORE_IDENTITY,
                               PHONE_REPLY_CAPABILITY_RULE,
                               PROACTIVE_CHECKIN_RULES)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Fixed, never vault-derived -- identical on every single call. This
# is what keeps adding it structurally safe alongside "exact-prompt-
# only transmission": `contents` (below) still carries nothing but
# Captain's own bare words; this is metadata about how Gemini should
# describe itself, not additional prompt content attributed to
# Captain, and it can never vary by turn or leak anything turn-
# specific because it's a module-level constant, not built per call.
_SYSTEM_INSTRUCTION = (f"{JARVIS_CORE_IDENTITY} {GEMINI_CAPABILITY_RULES} "
                       f"{PROACTIVE_CHECKIN_RULES}")


def _system_instruction() -> str:
    """_SYSTEM_INSTRUCTION plus the phone-reply note, appended fresh on
    every call rather than baked into the fixed module constant above
    -- whether a phone is actually paired and connected changes turn to
    turn, so it can't be a load-time-only fact like the rest of that
    constant. Still never vault- or utterance-derived: only a live
    True/False from phone_bridge.phone_reply_available()."""
    if phone_bridge.phone_reply_available():
        return f"{_SYSTEM_INSTRUCTION} {PHONE_REPLY_CAPABILITY_RULE}"
    return _SYSTEM_INSTRUCTION
# Short and explicit, not a single blanket number: a stuck connect
# (bad DNS, dead network) must fail fast rather than hang the turn for
# a full 30 seconds, and the read side gets a bit more room since a
# real generation can legitimately take a few seconds.
_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class GeminiBrain(BrainAdapter):
    id = "gemini"
    label = "Gemini (external -- your own free tier, approval required)"
    requires_tools = False
    requires_confirm_to_switch = True
    requires_external_lease = True
    capability_summary = (
        "quick current public research on your own Gemini free tier. "
        "No vault context, no files, ever -- only exactly what you "
        "say, shown to you and approved before your first request each "
        "session and again after any thirty-minute idle gap.")

    def __init__(self, *, enabled: bool = False,
                api_key_env: str = "GEMINI_API_KEY",
                model: str = "gemini-3.5-flash-lite"):
        super().__init__(enabled=enabled)
        self._api_key_env = api_key_env
        self._model = model

    def _api_key(self) -> str | None:
        # Read fresh every call -- never cached on self, never logged.
        return os.environ.get(self._api_key_env) or None

    async def _check_health(self) -> BrainHealth:
        # Key PRESENCE only, deliberately no network call: real
        # availability (rate limits, outages) can only be known at
        # request time, and a "just checking" call would itself be an
        # unapproved external request -- exactly what this adapter
        # exists to never do silently.
        if not self._api_key():
            return BrainHealth(
                healthy=False,
                reason=f"no {self._api_key_env} configured in your "
                       f"Windows user environment")
        return BrainHealth(healthy=True)

    # build_request_preview() is inherited unchanged from BrainAdapter
    # -- the bare utterance, nothing added -- since Gemini never had
    # anything of its own to attach in the first place.

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        if not self.enabled:
            raise BrainDisabledError("gemini is disabled by configuration")
        key = self._api_key()
        if not key:
            raise BrainUnavailableError(
                f"no {self._api_key_env} configured in your Windows "
                f"user environment")
        preview = self.build_request_preview(utterance)
        url = f"{GEMINI_API_BASE}/models/{self._model}:generateContent"
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": preview}]}],
            "systemInstruction": {"parts": [{"text": _system_instruction()}]},
        }
        self.session["turns"] += 1
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise BrainUnavailableError(
                "Gemini didn't respond in time") from exc
        except httpx.HTTPError as exc:
            raise BrainUnavailableError(
                f"Gemini is unreachable right now: {exc}") from exc
        if resp.status_code == 429:
            raise BrainUnavailableError(
                "Gemini is rate-limited on the free tier right now")
        if resp.status_code == 403:
            raise BrainUnavailableError(
                f"Gemini refused the request -- check that "
                f"{self._api_key_env} is a valid key")
        if resp.status_code != 200:
            raise BrainUnavailableError(
                f"Gemini returned an error (HTTP {resp.status_code})")
        try:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise BrainUnavailableError(
                f"Gemini returned an unexpected response shape: {exc}"
            ) from exc
        buf = text
        while True:
            m = _SENTENCE_END.search(buf)
            if not m:
                break
            sentence, buf = buf[:m.end()].strip(), buf[m.end():]
            if sentence:
                yield sentence
        tail = buf.strip()
        if tail:
            yield tail

    async def command(self, cmd: str) -> str:
        # Gemini has no slash-command console -- the router intercepts
        # console verbs before they reach an adapter at all.
        return ""
