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
"""BrainRouter -- Phase 1: the seam a later phase will wire backtalk.main
to instead of talking to backtalk.brain.WarmBrain directly. Nothing in
the live F8 path constructs or imports this module yet, so nothing here
can change how F8 behaves until that phase deliberately wires it in.

The one rule every method enforces: NO SILENT FALLBACK. Activating a
disabled, unconfirmed, or unhealthy brain always refuses explicitly --
it never substitutes a different brain, and it never activates anything
on its own initiative. A caller (a later phase's voice-console confirm
flow) decides WHEN to ask the person; this module only allows or
refuses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from backtalk import speech_format
from backtalk.brains import config as router_config
from backtalk.brains import tool_intent, vault_context
from backtalk.brains.base import (
    BrainAdapter,
    BrainDisabledError,
    BrainStatus,
    BrainUnavailableError,
)
from backtalk.brains.claude_brain import ClaudeBrain
from backtalk.brains.gemini_brain import GeminiBrain
from backtalk.brains.ollama_brain import DeepSeekBrain, QwenBrain


class NoBrainActiveError(RuntimeError):
    """Raised by ask_stream() when nothing has been activated yet."""


class ConfirmRequiredError(RuntimeError):
    """Raised when activating a gated brain (Gemini, Claude) without
    confirmed=True. The caller must have gotten an explicit yes from the
    person for THIS switch -- passing confirmed=True on a hunch defeats
    the whole point of this exception existing."""


@dataclass
class RouterStatus:
    """The accurate active-brain status contract -- one object, one
    truth source. `active_id` names what's actually live (or None);
    `brains` carries every registered adapter's own honest status,
    including WHY a disabled or unhealthy one isn't active. Anything
    that wants to know "what brain is Jarvis on right now" reads this,
    never an adapter directly."""
    active_id: str | None
    brains: dict[str, BrainStatus] = field(default_factory=dict)


def _build_registry(cfg: dict, *, can_use_tool=None,
                    resume_id=None) -> dict[str, BrainAdapter]:
    b = cfg["brains"]
    return {
        "qwen3-8b-local": QwenBrain(
            enabled=b.get("qwen3-8b-local", {}).get("enabled", False),
            context_loader=vault_context.load_context),
        "deepseek-r1-8b-local": DeepSeekBrain(
            enabled=b.get("deepseek-r1-8b-local", {}).get("enabled", False),
            context_loader=vault_context.load_context),
        "gemini": GeminiBrain(
            enabled=b.get("gemini", {}).get("enabled", False),
            api_key_env=b.get("gemini", {}).get("api_key_env",
                                                 "GEMINI_API_KEY"),
            **({"model": b["gemini"]["model"]}
               if b.get("gemini", {}).get("model") else {})),
        "claude": ClaudeBrain(
            enabled=b.get("claude", {}).get("enabled", False),
            model=b.get("claude", {}).get("model"),
            can_use_tool=can_use_tool,
            resume_id=resume_id),
    }


class BrainRouter:
    def __init__(self, *, config: dict | None = None, can_use_tool=None,
                resume_id=None):
        self._cfg = config if config is not None else router_config.load()
        self._brains = _build_registry(self._cfg, can_use_tool=can_use_tool,
                                       resume_id=resume_id)
        self._active_id: str | None = None

    @property
    def active_id(self) -> str | None:
        return self._active_id

    def registered_ids(self) -> list[str]:
        return list(self._brains.keys())

    def get(self, brain_id: str) -> BrainAdapter:
        try:
            return self._brains[brain_id]
        except KeyError:
            raise KeyError(
                f"no such brain: {brain_id!r}; registered: "
                f"{self.registered_ids()}"
            ) from None

    async def health_check_all(self) -> dict[str, BrainStatus]:
        """Await every adapter's health() and return each one's fresh
        status, keyed by id. Never raises: an adapter whose health check
        itself blows up is reported unhealthy, not allowed to take the
        whole readout down with it."""
        out = {}
        for bid, brain in self._brains.items():
            try:
                await brain.health()
            except Exception as exc:  # noqa: BLE001 -- must never propagate
                brain.record_failed_health(f"health check raised: {exc!r}")
            out[bid] = brain.status()
        return out

    def status(self) -> RouterStatus:
        """Cached status -- no network calls. Call health_check_all()
        first for a fresh reading."""
        return RouterStatus(
            active_id=self._active_id,
            brains={bid: b.status() for bid, b in self._brains.items()},
        )

    async def activate(self, brain_id: str, *,
                       confirmed: bool = False) -> BrainStatus:
        """Switch the active brain. Refuses outright, never substitutes,
        when: the id is unknown (KeyError), the brain is disabled
        (BrainDisabledError), it needs a confirm that wasn't given
        (ConfirmRequiredError), or it's enabled but not currently usable
        (BrainUnavailableError -- e.g. DeepSeek not pulled, Ollama down).
        This is the one chokepoint a later phase's confirm flow calls
        into, so the no-silent-fallback rule lives in exactly one
        place."""
        brain = self.get(brain_id)
        if not brain.enabled:
            raise BrainDisabledError(
                f"{brain.label} is disabled by configuration; enable it "
                f"in brain_router.json before switching to it")
        if brain.requires_confirm_to_switch and not confirmed:
            raise ConfirmRequiredError(
                f"{brain.label} requires an explicit confirmation before "
                f"switching to it")
        health = await brain.health()
        if not health.healthy:
            raise BrainUnavailableError(
                f"{brain.label} is enabled but not currently usable: "
                f"{health.reason}")
        if self._active_id and self._active_id != brain_id:
            await self._brains[self._active_id].stop()
        await brain.start()
        self._active_id = brain_id
        return brain.status()

    async def ask_stream(self, utterance: str):
        if not self._active_id:
            raise NoBrainActiveError("no brain has been activated yet")
        active = self._brains[self._active_id]
        # THE HONESTY GATE: a brain with no tools never gets to answer
        # a plainly tool-shaped request in prose as if it had done the
        # thing, and it never silently switches brains on its own. It
        # refuses, out loud, through the exact same reply channel as
        # any other sentence, and names the one phrase that actually
        # gets something done about it.
        if not active.requires_tools and tool_intent.looks_like_tool_request(
                utterance):
            yield tool_intent.REFUSAL
            return
        # THE SHARED SPEECH RENDERER: every brain's reply passes through
        # the same sanitizer here, in exactly one place, so the voice
        # experience never depends on which brain answered. Applied to
        # each already-complete sentence/chunk an adapter yields, never
        # to a raw streaming fragment (a markdown span can straddle two
        # network chunks mid-token). Claude's own output is already
        # clean prose (DISCIPLINE tells it to write for the ear), so
        # this is a no-op backstop for it and the actual fix for Qwen
        # and DeepSeek, which carry no such instruction of their own.
        async for sentence in active.ask_stream(utterance):
            cleaned = speech_format.to_speech(sentence)
            if cleaned:
                yield cleaned

    @property
    def session(self) -> dict:
        """Usage bookkeeping for the ACTIVE brain, or a zeroed dict when
        nothing is active -- main.py's "usage report" console verb reads
        this the same way it used to read WarmBrain.session directly."""
        if self._active_id:
            return self._brains[self._active_id].session
        return {"turns": 0, "out_tokens": 0, "in_tokens": 0, "cost": 0.0}

    async def interrupt(self) -> None:
        if self._active_id:
            await self._brains[self._active_id].interrupt()

    async def reset_turn(self, timeout: float = 8.0) -> None:
        if self._active_id:
            await self._brains[self._active_id].reset_turn(timeout)

    async def command(self, cmd: str) -> str:
        if not self._active_id:
            raise NoBrainActiveError("no brain has been activated yet")
        return await self._brains[self._active_id].command(cmd)

    async def set_permission_mode(self, mode: str) -> None:
        if self._active_id:
            await self._brains[self._active_id].set_permission_mode(mode)

    async def context_usage(self):
        if not self._active_id:
            return None
        return await self._brains[self._active_id].context_usage()

    async def shutdown(self) -> None:
        if self._active_id:
            await self._brains[self._active_id].stop()
            self._active_id = None
