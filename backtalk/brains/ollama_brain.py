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
"""Local-Ollama brain adapter. Qwen (the default id) and DeepSeek (deep
reasoning) are the SAME adapter class, parameterized only by model name
-- their behavior, health check, and streaming logic are identical.

Talks to Ollama the same way local-jarvis/ollama_client.py does:
hardcoded to 127.0.0.1, using httpx (already a backtalk dependency) for
async streaming instead of stdlib urllib. This adapter must never talk
to a remote or LAN-exposed Ollama host.

Streaming: Ollama's /api/chat accepts "stream": true and returns
newline-delimited JSON, each line carrying a message.content fragment
and a final {"done": true}. ask_stream re-chunks that into complete
sentences the same way backtalk.brain.WarmBrain.ask_stream does, so a
later phase's speak_reply() in main.py needs no changes to consume
either brain's output.
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator

import httpx

from backtalk.brains.base import BrainAdapter, BrainDisabledError, BrainHealth

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
_HEALTH_TIMEOUT_S = 5
_CHAT_TIMEOUT_S = 180
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class OllamaBrain(BrainAdapter):
    """Base for any local-Ollama-backed brain. Subclasses set
    id/label/model."""

    requires_tools = False
    requires_confirm_to_switch = False
    model: str = ""
    # The honest capability matrix (see base.BrainStatus): this brain
    # can talk and read a bounded vault excerpt. It must never claim
    # file edits, commands, browsing, or vault writes -- those aren't
    # implemented behind any permission gate here, so the router's
    # tool_intent check refuses those requests before they ever reach
    # this class, and this system prompt below says the same thing to
    # the model itself as a second line of defense.
    capability_summary = (
        "local conversation and a read-only excerpt of your vault (the "
        "index, your profile, and today's daily note). It cannot edit "
        "files, run commands, browse the web, or write to your vault -- "
        "those need Claude.")

    def __init__(self, *, enabled: bool = False, context_loader=None):
        super().__init__(enabled=enabled)
        # Called fresh on every turn, never cached at construction --
        # so a same-day vault edit shows up on the next turn without a
        # restart. None means "no vault context at all" (e.g. tests).
        self._context_loader = context_loader

    async def _check_health(self) -> BrainHealth:
        try:
            async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT_S) as client:
                resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            return BrainHealth(healthy=False,
                               reason=f"Ollama unreachable: {exc}")
        models = [m.get("name", "") for m in data.get("models", [])]
        if self.model not in models:
            return BrainHealth(
                healthy=False,
                reason=f"{self.model} is not installed in Ollama")
        return BrainHealth(healthy=True)

    async def ask_stream(self, utterance: str) -> AsyncIterator[str]:
        if not self.enabled:
            raise BrainDisabledError(
                f"{self.id} is disabled by configuration")
        vault_ctx = self._context_loader() if self._context_loader else ""
        preamble = (
            "You are Jarvis, a local AI assistant running entirely "
            "on-device through Ollama. No part of this conversation "
            "leaves this machine. You have NO tools: you cannot edit "
            "files, run commands, browse the web, or write to the "
            "vault. If asked to do one of those, say so plainly and "
            "suggest switching to Claude instead of pretending to do it.")
        system_prompt = (
            f"{preamble} Use the following vault context if relevant; "
            f"do not invent facts beyond it.\n\n{vault_ctx}"
            if vault_ctx else preamble)
        messages = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": utterance}]
        payload = {"model": self.model, "messages": messages, "stream": True}
        buf = ""
        self.session["turns"] += 1
        async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT_S) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat",
                                     json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        buf += piece
                        while True:
                            m = _SENTENCE_END.search(buf)
                            if not m:
                                break
                            sentence, buf = (buf[:m.end()].strip(),
                                             buf[m.end():])
                            if sentence:
                                yield sentence
                    if chunk.get("done"):
                        break
        tail = buf.strip()
        if tail:
            yield tail

    async def command(self, cmd: str) -> str:
        # Ollama has no slash-command console -- the router intercepts
        # console verbs before they reach an adapter at all, so this
        # only exists to satisfy the shared interface.
        return ""


class QwenBrain(OllamaBrain):
    id = "qwen3-8b-local"
    label = "Qwen3 8B Local"
    model = "qwen3:8b"


class DeepSeekBrain(OllamaBrain):
    id = "deepseek-r1-8b-local"
    label = "DeepSeek R1 8B Local (deep reasoning)"
    model = "deepseek-r1:8b"
