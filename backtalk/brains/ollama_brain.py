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

from backtalk import phone_bridge
from backtalk.brains.base import BrainAdapter, BrainDisabledError, BrainHealth
from backtalk.identity import (JARVIS_CORE_IDENTITY,
                               LOCAL_BRAIN_CAPABILITY_RULES,
                               PHONE_REPLY_CAPABILITY_RULE,
                               PROACTIVE_CHECKIN_RULES)

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

    # How many user/assistant TURN PAIRS to keep. Bounded so a long
    # session's history can't grow the request payload without limit;
    # 10 turns is generous for ordinary follow-up conversation while
    # staying well inside any local 8B model's context window even
    # alongside the vault excerpt.
    _MAX_HISTORY_TURNS = 10

    def __init__(self, *, enabled: bool = False, context_loader=None):
        super().__init__(enabled=enabled)
        # Called fresh on every turn, never cached at construction --
        # so a same-day vault edit shows up on the next turn without a
        # restart. None means "no vault context at all" (e.g. tests).
        self._context_loader = context_loader
        # Real field-test bug: this used to be a single [system, user]
        # call every turn, so a follow-up like "yes, Jarvis" had
        # nothing to resolve against and the model answered as if the
        # conversation had just started. Now a running transcript,
        # exactly like Claude's own conversation already had via the
        # SDK session -- see ask_stream() for how it's appended and
        # capped, and command() for how "/clear" empties it.
        self._history: list[dict] = []

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
        phone_note = (
            f" {PHONE_REPLY_CAPABILITY_RULE}"
            if phone_bridge.phone_reply_available() else "")
        preamble = (
            f"{JARVIS_CORE_IDENTITY} You're running entirely on-device "
            f"through Ollama right now -- no part of this conversation "
            f"leaves this machine. {LOCAL_BRAIN_CAPABILITY_RULES}"
            f"{phone_note} "
            f"{PROACTIVE_CHECKIN_RULES} "
            "Your reply is SPOKEN ALOUD, not displayed: write plain "
            "natural sentences only. Never use Markdown (no **bold**, "
            "no # headings, no - bullet points, no numbered lists, no "
            "code blocks, no [links](like this)), never use emoji, and "
            "never speak a raw file path or URL -- say the file or "
            "site by name instead. Answer the request and then stop -- "
            "do not tack on a generic closing line like 'How may I "
            "assist you today?', 'What would you like to explore "
            "next?', 'Let me know if you need anything else', 'How "
            "can I assist you?', or any rephrasing of those (for "
            "example 'What would you like to accomplish with your "
            "local AI assistant?' is just as banned as the exact "
            "wording above). Only ask a follow-up question when you "
            "genuinely need missing information to complete the "
            "request; for an explanation, end on a concise natural "
            "conclusion instead of a service-desk sign-off.")
        system_prompt = (
            f"{preamble} Use the following vault context if relevant; "
            f"do not invent facts beyond it.\n\n{vault_ctx}"
            if vault_ctx else preamble)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": utterance})
        payload = {"model": self.model, "messages": messages, "stream": True}
        buf = ""
        full_reply: list[str] = []
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
                                full_reply.append(sentence)
                                yield sentence
                    if chunk.get("done"):
                        break
        tail = buf.strip()
        if tail:
            full_reply.append(tail)
            yield tail
        # Remember this exchange -- WITHOUT this, "yes, Jarvis" after a
        # follow-up question has nothing to resolve against and the
        # model answers as if the conversation just started (the exact
        # bug this fixes, caught in a real field test). Capped to the
        # last _MAX_HISTORY_TURNS pairs so a long session's payload
        # can't grow without bound.
        if full_reply:
            self._history.append({"role": "user", "content": utterance})
            self._history.append(
                {"role": "assistant", "content": " ".join(full_reply)})
            max_messages = self._MAX_HISTORY_TURNS * 2
            if len(self._history) > max_messages:
                self._history = self._history[-max_messages:]

    async def command(self, cmd: str) -> str:
        if cmd.strip() == "/clear":
            self._history = []
            return "Cleared."
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
