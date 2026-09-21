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
"""backtalk — talk to your Claude Code agent out loud.

Flow: hold the key and speak -> local transcription -> your agent's
ACTIVE BRAIN streams the reply -> sentences go to the mouth the moment
they complete. Which brain answers is selected by a BrainRouter
(backtalk.router): Qwen3 8B Local by default (free, on-device, no
tools), Claude only as an explicit, confirm-gated emergency escalation
(the full Agent SDK toolset, consumes your subscription usage). The
identity, the vault, the permission gate, the voice, and the face are
all the same regardless of which brain is answering -- only the brain
changes. The greeting plays over a hidden warmup query so the first
real turn is already hot.

Typing in this terminal is a first-class turn too: same conversation,
spoken reply, and typing while it talks interrupts it.

THE VOICE CONSOLE: exact phrases, spoken (or typed) alone, control the
session itself so you never go back to the keyboard: "clear the
session" / "compact the session" / "switch to the deep model" / "back
to the fast model" / "set effort to low" (or medium, high, max) --
these five are Claude-only and say so honestly when a local brain is
active / "usage report" / "go hands free" and "push to talk mode" (the
MIC) / "stop asking for permission" and "start asking again"
(permissions, called auto-approve, a different axis than the
microphone on purpose) / "switch to Qwen" (immediate, no confirm) /
"switch to Claude" (always needs a spoken confirm, every time, no
exceptions) / "which brain are you using" (a truthful status readout,
brain identity and what it can actually do). And with permission_mode
"ask" (the default), gated tool calls Claude wants to make ASK OUT
LOUD and your spoken yes or no decides them; any other answer is
passed back to the agent as the reason. A brain with no tools (Qwen,
DeepSeek) never gets asked, because it never has anything gated to
ask about -- and if you ask it to do something that needs a tool, it
says so and points you at "switch to Claude" instead of pretending.

Flags:
  --open-mic   start in hands-free listening for this session (the
               config key mic_mode makes it the standing default, and
               the voice can switch live either way: "go hands free" /
               "push to talk mode"). Know the tradeoff: room audio (a
               video, music, another voice assistant) can trigger
               replies to speech never meant for the agent. The talk
               key keeps working: it interrupts, and holding it always
               gets you heard.
  --barge-in   with --open-mic: keep listening WHILE speaking.
               HEADPHONES REQUIRED — with open speakers the mic hears
               the reply and the agent interrupts itself.
  --model X    override the model for this session (full id).

Say "goodbye <name>" / "end voice mode" to hang up. Ctrl-C works.
"""
import asyncio
import datetime
import json
import queue
import re
import socket
import sys
import threading
import time
import zoneinfo

# Explicit, never assumed from the system clock's own configured
# timezone -- see _startup_greeting_line()'s docstring. Requires the
# tzdata package (pyproject.toml): Windows ships no IANA tz database
# of its own for stdlib zoneinfo to read.
IST = zoneinfo.ZoneInfo("Asia/Kolkata")

from backtalk import day_journal, phone_auth, phone_bridge, phone_tls, signals
from backtalk.brains import brain_intent
from backtalk.brains import config as brain_router_config
from backtalk.brains import recommend
from backtalk.brains.base import BrainDisabledError, BrainUnavailableError
from backtalk.config import CFG
from backtalk.ears import (Ears, explain_audio_failure, record_held,
                           warm as warm_ears)
from backtalk.mouth import Mouth
from backtalk.ptt import PTTListener
from backtalk.router import BrainRouter, ConfirmRequiredError
from backtalk.vlog import log

NAME = CFG["name"]
QUIT_PHRASES = CFG["quit_phrases"]

# ---- THE SPOKEN PERMISSION GATE (permission_mode "ask", the default).
# When the agent wants a gated tool, the SDK routes the decision here:
# the ask is spoken, the turn pauses (the SDK waits indefinitely; the
# timeout below is ours), and the NEXT utterance or typed line is the
# answer. "yes" approves; anything else denies, with the user's own
# words passed back as the reason. Silence means no.
PERM_TIMEOUT_S = 75
_PERM = {"fut": None, "asked_at": 0.0,   # pending ask + when it was posed
         "hinted": False}                # escape-hatch hint said yet?
_CONFIRM = {"verb": None, "at": 0.0}     # pending "say confirm" + when
# EXTERNAL CONSENT LEASE (Claude and Gemini): confirming the switch
# opens a window of this many seconds during which requests to that
# brain send without re-asking; each actual send refreshes it (an
# IDLE lease, not a fixed session length). Kept as its OWN dedicated
# state (never reusing _CONFIRM or _PERM) specifically so it can never
# interact with or risk the existing Claude tool-permission gate or
# the brain-SWITCH confirm flow.
_LEASE_DURATION_S = 30 * 60
_EXTERNAL_LEASE = {"brain_id": None, "expires_at": 0.0}
# Pending consent for the NEXT send to an external brain -- armed only
# when that brain has no currently-valid lease (never confirmed yet,
# or the lease expired). brain_id is checked on resolution so a stale
# pending consent from a brain Captain has since switched away from
# can never fire.
_EXTERNAL_PENDING = {"text": None, "brain_id": None, "at": 0.0}
# A bare "switch" asks which number and stores that as one-turn
# context: only the VERY NEXT utterance, and only if it's just a bare
# number ("3", "three"), resolves it -- anything else (including
# ordinary conversation that happens to be said next) falls through
# normally, exactly like every other pending-state gate here.
_BRAIN_NUM_PENDING = {"pending": False, "at": 0.0}
_INTERRUPT_ANSWER = "\x00interrupt"      # sentinel: turn is being killed
# Live AUTO-APPROVE is OUR flag, not an SDK mode flip: the CLI refuses
# a live switch INTO bypassPermissions unless it was launched with the
# danger flag, so instead the gate below auto-approves silently while
# this is on. Same behavior, no reconnect, conversation intact. A
# session that BOOTS in bypassPermissions never consults the gate at
# all; saying "start asking again" flips the SDK side live (that
# direction is allowed) and turns this off. ONLY the explicit
# bypassPermissions value arms this: any other mode (acceptEdits, plan)
# passes through to the SDK and keeps the spoken gate for whatever the
# SDK routes here. (Auto-approve is about PERMISSIONS; hands-free
# LISTENING is about the microphone: see _MIC below. Two different
# axes, deliberately never sharing a name.)
_AUTOAPPROVE = {"on": False}
# The microphone mode, switchable live by voice. "ptt" = mic closed
# except while the key is held. "open" = hands-free listening (VAD).
# The key keeps working in open mode: it interrupts, and holding it
# always gets you heard. gen bumps on every switch so an in-flight
# open-mic capture from before the switch gets discarded, never
# processed.
_MIC = {"mode": "ptt", "gen": 0, "btn": False}

# Active-platform reply routing (2026-09-21). None = automatic: each
# reply's audio goes to wherever ITS OWN utterance came from (rule 2/
# 3 -- a phone turn speaks only on the phone, a desktop turn speaks
# only on desktop). "phone"/"desktop"/"both" is a standing override
# set by an explicit command (rule 4: "reply on phone" etc, see the
# replyphone/replydesktop/replyboth/replyauto console verbs below),
# in force until changed again. Session-only by design, not written to
# backtalk.json -- a live routing choice, not a persistent preference.
_REPLY_TARGET_OVERRIDE = {"mode": None}


def _resolve_reply_targets(source_platform: str) -> frozenset:
    """Where a reply's AUDIO should actually play -- never gates text/
    transcript visibility, which phone_bridge always records
    regardless (see mouth.py's _on_drained and record_transcript_line
    call). An explicit override (rule 4) always wins; with none, the
    turn's own origin decides (rule 2/3)."""
    override = _REPLY_TARGET_OVERRIDE["mode"]
    if override == "phone":
        return frozenset({"phone"})
    if override == "desktop":
        return frozenset({"desktop"})
    if override == "both":
        return frozenset({"desktop", "phone"})
    return frozenset({source_platform})

# Approvals are EXACT matches after normalization, never prefixes:
# "yesterday", "yes or no", and "yes, but do not overwrite" must all
# fail. Anything that is not an exact yes DENIES, with the words passed
# back to the agent as the reason. Deny is always the default.
# Exact matches only, and the reason is in the comment on _norm_speech:
# prefix matching turns "yesterday" and "yes or no" into consent. So the
# set has to actually CONTAIN what people say -- and the phrase somebody
# reaches for is the one the prompt just put in their head. Asking for
# PERMISSION and then denying "permission granted" is the system tripping
# a user with its own vocabulary, and it quotes their words back as the
# reason for the refusal.
_YES = {"yes", "yeah", "yep", "yup", "sure", "approve", "approved",
        "go ahead", "do it", "yes please", "yes sir", "yes boss",
        "yes go ahead", "go for it", "green light", "okay", "ok", "y",
        "permission granted", "granted", "you have permission",
        "you may", "allowed", "allow it", "confirmed", "affirmative"}
_CHAIN_MARKS = ("&&", "||", ";", "|", "$(", "`", "\n")

# Resolves a PENDING brain-switch or external-lease confirm (_CONFIRM
# / _EXTERNAL_PENDING) -- deliberately separate from _YES above (that
# one answers a Claude tool permission ask, a different question
# entirely). Real live-test misses this closes: "conform"/"conformed"
# (Whisper mis-hearing "confirm", same shape as "QN3" for "Qwen"
# elsewhere in this file) and "Thank you, confirmed." (a polite prefix
# an exact-set-membership check could never match). Shared by every
# place that resolves a pending confirm so all of them accept exactly
# the same variants -- this must NEVER be consulted outside an actual
# pending confirmation (each caller only checks it inside its own "is
# something pending" branch).
_CONFIRM_WORD = r"(?:confirm|confirmed|conform|conformed)"
_CONFIRM_NEGATED = re.compile(
    r"\b(do not|don't|dont|never|not|won't|wont|will not|didn't|"
    r"didnt)\b[^.!?]{0,20}\b" + _CONFIRM_WORD + r"\b", re.IGNORECASE)
_CONFIRM_PHRASE = re.compile(
    r"^(thank you )?(yes )?" + _CONFIRM_WORD + r"$", re.IGNORECASE)


def _is_confirm_phrase(text: str) -> bool:
    """True iff `text` is an accepted confirmation -- case/punctuation
    ignored, a small set of real Whisper mis-hearings and a polite
    prefix tolerated -- but NEVER when negated ("do not confirm").
    The negation check runs on the raw lowercased text (contractions
    like "don't" intact); _norm_speech's own letters-only collapse
    would otherwise mangle "don't" into "don t" and lose the word."""
    if _CONFIRM_NEGATED.search(text.lower()):
        return False
    return bool(_CONFIRM_PHRASE.match(_norm_speech(text)))


def _norm_speech(text):
    """Lowercase, every non-letter to space, collapse. Whisper loves
    interior commas ("yes, confirm"); end-stripping alone misses them."""
    out = []
    for ch in text.lower():
        out.append(ch if "a" <= ch <= "z" else " ")
    return " ".join("".join(out).split())


# Deliberately NOT built on _norm_speech: that helper maps every
# non-letter (digits included) to a space, which would erase a bare
# "3" entirely before this could ever see it.
_BARE_NUM = re.compile(
    r"^(?:brain\s+)?(1|one|2|two|3|three|tree|4|four)[.!]?$",
    re.IGNORECASE)


def _bare_brain_number(text: str) -> str | None:
    """Only matches a BARE number (optionally "brain N"), nothing
    else -- this must never fire on ordinary conversation that happens
    to contain a number word, so it's only ever consulted while
    _BRAIN_NUM_PENDING is actually pending (see handle())."""
    m = _BARE_NUM.match(text.strip().strip("?.! "))
    if not m:
        return None
    return brain_intent._NUM_TO_ID[m.group(1).lower()]


def _deny_pending(reason=_INTERRUPT_ANSWER):
    """Resolve a pending spoken ask as a deny. Called whenever the turn
    that posed it is being interrupted, so the ask can never outlive its
    turn and hijack a later utterance (or stall the pipe drain)."""
    f = _PERM["fut"]
    if f is not None and not f.done():
        f.set_result(reason)


def _human_what(tool, tool_input, ctx):
    """The SHORT spoken form, built for a person who has never seen a
    terminal: plain words, no paths, no syntax. Built by code, never by
    the model, so it cannot understate; and every ask offers "details",
    which reads the full literal form below. (Field case: the gate read
    whole file paths and command syntax at a brand-new user.)"""
    d = tool_input or {}
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(d.get("file_path") or d.get("notebook_path")
                   or "a file").replace("\\", "/")
        name = path.rsplit("/", 1)[-1]
        import os as _os
        homes = [CFG.get("agent_dir", "")] + list(CFG.get("extra_dirs")
                                                  or [])
        in_vault = any(h and path.startswith(str(h).rstrip("/") + "/")
                       for h in (CFG.get("extra_dirs") or []))
        verb = "edit" if "Edit" in tool else "create or change"
        if in_vault and name.endswith(".md"):
            return f"{verb} a note in your vault called {name[:-3]}"
        return f"{verb} a file called {name}"
    if tool == "Bash":
        cmd = " ".join(str(d.get("command", "")).split())
        first = (cmd.split() or ["a"])[0].rsplit("/", 1)[-1]
        chained = any(m in cmd for m in _CHAIN_MARKS)
        return (f"run a {first} command in the terminal"
                + (", with several chained parts" if chained else ""))
    if tool == "WebFetch":
        url = str(d.get("url", ""))
        host = url.split("//", 1)[-1].split("/", 1)[0] or "a site"
        return f"read a web page at {host}"
    name = getattr(ctx, "display_name", None) or tool
    return f"use the {name} tool"


_DETAILS = {"details", "the details", "give me details",
            "give me the details", "what command", "what is it",
            "say more", "more", "what exactly", "the exact command"}


def _full_detail(tool, tool_input, ctx):
    """The full literal form, spoken only when the person asks for
    "details". Never lets a long command hide its tail: truncation is
    DISCLOSED and shell chaining is called out (the agent composes
    tool_input itself, so this line must not be steerable into
    understatement)."""
    d = tool_input or {}
    if tool == "Bash":
        cmd = " ".join(str(d.get("command", "")).split())
        chained = any(m in cmd for m in _CHAIN_MARKS)
        line = ("a chained command: " if chained else
                "run a command: ") + cmd[:90]
        if len(cmd) > 90:
            line += (f", and {len(cmd) - 90} more characters. "
                     "Check the log before approving")
        return line
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = str(d.get("file_path") or d.get("notebook_path")
                   or "a file").replace("\\", "/")
        bits = path.rsplit("/", 2)
        name = "/".join(bits[-2:]) if len(bits) >= 2 else path
        return f"{'edit' if 'Edit' in tool else 'write'} the file {name}"
    if tool == "WebFetch":
        return f"fetch a web page: {str(d.get('url', ''))[:70]}"
    desc = (getattr(ctx, "description", None) or "").strip()
    name = getattr(ctx, "display_name", None) or tool
    return f"use {name}" + (f", {desc[:70]}" if desc else "")


def make_permission_gate(mouth):
    from claude_agent_sdk import (PermissionResultAllow,
                                  PermissionResultDeny)

    async def gate(tool, tool_input, ctx):
        if _AUTOAPPROVE["on"]:
            return PermissionResultAllow(behavior="allow")
        what = _human_what(tool, tool_input, ctx)
        detail = _full_detail(tool, tool_input, ctx)
        loop = asyncio.get_running_loop()
        signals.static_stop()
        log(f"[perm]   asking: {what}")
        log(f"[perm]   detail: {detail}")
        if tool == "Bash":   # the FULL command always reaches the log
            log(f"[perm]   full command: {str((tool_input or {}).get('command', ''))[:2000]}")
        ask = f"Permission check. I want to {what}. Yes, no, or details?"
        if not _PERM["hinted"]:
            # the escape hatch announces itself exactly once, at the
            # moment it becomes relevant (a field case: a new user
            # couldn't find the phrase to turn the checks off)
            _PERM["hinted"] = True
            ask += (" And any time you're done with these checks, say "
                    "stop asking for permission.")
        mouth.say(ask)
        # Phone-visible mirror of this same ask (2026-09) -- the phone's
        # Yes/No buttons just POST "yes"/"no" into typed_q like any
        # other typed text; this line only lets the phone page know a
        # question is pending so it can show it. No second approval path.
        phone_bridge.set_pending_permission(ask)
        answer = None
        try:
            deadline = loop.time() + PERM_TIMEOUT_S
            while answer is None:
                fut = loop.create_future()
                _PERM["fut"] = fut
                _PERM["asked_at"] = time.monotonic()
                while True:
                    try:
                        got = await asyncio.wait_for(
                            asyncio.shield(fut), 1.0)
                        break
                    except asyncio.TimeoutError:
                        if loop.time() >= deadline:
                            fut.cancel()
                            mouth.say("No answer, so I didn't do it.")
                            log("[perm]   timed out, denied")
                            day_journal.record_event(
                                "denied_action", f"{what} (no answer)")
                            return PermissionResultDeny(
                                behavior="deny",
                                message="No spoken answer within the "
                                        "timeout; the action was not "
                                        "approved.",
                                interrupt=False)
                        # keep the ring honest while we wait
                        if not mouth.speaking:
                            signals.set_state("listening")
                if (got != _INTERRUPT_ANSWER
                        and _norm_speech(got) in _DETAILS):
                    # read the full literal form, then ask again with a
                    # fresh clock: asking for details is engagement,
                    # not silence
                    log("[perm]   details requested")
                    detail_ask = f"The details: I want to {detail}. Yes or no?"
                    mouth.say(detail_ask)
                    phone_bridge.set_pending_permission(detail_ask)
                    deadline = loop.time() + PERM_TIMEOUT_S
                    continue
                answer = got
        finally:
            _PERM["fut"] = None
            phone_bridge.set_pending_permission(None)
        if answer == _INTERRUPT_ANSWER:
            log("[perm]   turn interrupted, denied silently")
            return PermissionResultDeny(
                behavior="deny",
                message="Interrupted by the user; the turn is being "
                        "cancelled.",
                interrupt=False)
        approved = _norm_speech(answer) in _YES
        # the model keeps working either way: restore the working state
        signals.set_state("thinking")
        signals.static_start()
        if approved:
            log("[perm]   approved by voice")
            day_journal.record_event("approved_action", what)
            return PermissionResultAllow(behavior="allow")
        log(f"[perm]   denied: {answer!r}")
        day_journal.record_event("denied_action", what)
        return PermissionResultDeny(
            behavior="deny",
            message=f'Denied by voice. The user said: "{answer[:500]}"',
            interrupt=False)
    return gate


# ---- THE VOICE CONSOLE: session verbs, spoken. Exact phrases only,
# spoken alone, so ordinary sentences can never trigger them. (Grown
# from a community member's own build shared in the Discord.)
CONSOLE_VERBS = {
    "clear":     ("clear the session", "clear the context",
                  "clear context", "fresh slate", "slash clear"),
    "compact":   ("compact the session", "compact the context",
                  "compact context", "slash compact"),
    "deep":      ("switch to the deep model", "use the deep model",
                  "slash model deep"),
    "fast":      ("switch to the fast model", "use the fast model",
                  "back to the fast model", "slash model fast"),
    "usage":     ("usage report", "slash usage"),
    "micopen":   ("go hands free", "hands free mode",
                  "hands free listening", "open mic", "open the mic"),
    "micptt":    ("push to talk", "push to talk mode",
                  "back to push to talk", "back to the button"),
    "noask":     ("stop asking for permission",
                  "stop asking permission",
                  "stop asking me for permission",
                  "turn off the permission prompt",
                  "turn off the permission prompts",
                  "turn off the permissions prompt",
                  "turn off the permissions prompts",
                  "turn off permissions", "turn off permission checks",
                  "disable the permission checks",
                  "disable permission checks", "auto approve",
                  "auto approve mode"),
    "ask":       ("start asking again", "ask before acting",
                  "ask for permission again"),
    "useqwen":   ("switch to qwen", "use qwen", "back to qwen",
                  "switch to the local brain", "use the local model",
                  "switch to local", "local brain"),
    "useclaude": ("switch to claude", "use claude",
                  "switch to the claude brain", "emergency claude",
                  "escalate to claude"),
    "usedeepseek": ("switch to deep reasoning", "use deep reasoning",
                    "switch to deepseek", "use deepseek",
                    "deep reasoning mode", "think harder"),
    "usegemini": ("switch to gemini", "use gemini"),
    "whichbrain": ("which brain are you using", "what brain is this",
                   "which brain is active", "what brain are you on",
                   "brain status", "which brain"),
    "closeday": ("close the day", "summarise today", "summarize today",
                "make today's note", "make todays note",
                "end of day summary"),
    "listbrains": ("list brains", "brain options", "which brains"),
    "switchask": ("switch",),
    "memorymodules": ("how many memory modules do you have",
                      "how many memory systems do you have",
                      "how many vaults do you have"),
    "pairphone": ("pair a phone", "pair phone", "pair my phone",
                  "add a phone"),
    "listphones": ("list paired phones", "which phones are paired",
                   "list phones"),
    "showphonecert": ("show phone certificate", "show certificate fingerprint",
                      "show the certificate fingerprint"),
    "replyphone": ("reply on phone", "reply on my phone",
                  "answer on phone", "answer on my phone"),
    "replydesktop": ("reply on computer", "reply on the computer",
                     "reply on desktop", "answer on computer",
                     "answer on the computer", "answer on desktop"),
    "replyboth": ("reply on both", "answer on both", "reply everywhere"),
    "replyauto": ("reply automatically", "reply as normal",
                 "normal reply routing", "stop overriding replies"),
}
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
# "revoke phone N" / "remove phone N" / "unpair phone N" -- a one-shot
# regex rather than the numbered-pending two-step _BRAIN_NUM_PENDING
# machinery uses: revocation is a rare admin action, not a conversational
# flow, so requiring the number in the same utterance is the simpler,
# lower-risk choice here.
_REVOKE_PHONE_RE = re.compile(
    r"^(?:revoke|remove|unpair)\s+phone\s+(\d+)$")


def console_match(text):
    norm = " ".join(text.lower().replace("-", " ").split()).strip(" .,!?")
    for verb, phrases in CONSOLE_VERBS.items():
        if norm in phrases:
            return verb
    for lvl in _EFFORTS:
        if norm in (f"set effort to {lvl}", f"effort {lvl}",
                    f"slash effort {lvl}"):
            return f"effort:{lvl}"
    m = _REVOKE_PHONE_RE.match(norm)
    if m:
        return f"revokephone:{m.group(1)}"
    return None


def _write_config_key(key, value):
    """The agent rewrites the config; the person never hand-edits it.
    Returns True on a persisted write. A file that fails to PARSE is
    left untouched (rewriting from {} would wipe every other setting);
    the in-memory CFG updates either way so the session behaves."""
    from backtalk.config import CONFIG_PATH
    CFG[key] = value
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        log(f"[console] config not writable/parsable, session-only: {e}")
        return False
    data[key] = value
    try:
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except OSError as e:
        log(f"[console] config write failed, session-only: {e}")
        return False
    return True


def _fmt_tokens(n):
    if n >= 1_000_000:
        return f"about {round(n / 1_000_000, 1):g} million tokens"
    if n >= 1000:
        return f"about {round(n / 1000)} thousand tokens"
    return f"{n} tokens"


def _spoken_usage(sess, ctx_usage):
    """A short CFO brief of the session, written for the ear: plain
    numerals only (the TTS reads "40" fine; symbols come out garbled)."""
    turns = sess["turns"]
    parts = [f"{turns} turn{'s' if turns != 1 else ''} this session",
             _fmt_tokens(sess["out_tokens"]) + " spoken out"]
    cents = round(sess["cost"] * 100)
    if cents >= 1:
        parts.append(f"roughly {cents} cents" if cents < 100
                     else f"roughly {round(cents / 100)} dollars")
    try:
        cats = (getattr(ctx_usage, "categories", None)
                or (ctx_usage or {}).get("categories") or [])
        # the breakdown includes "Free space" and the autocompact
        # buffer; only OCCUPIED categories belong in the spoken number
        total = sum(int(c.get("tokens") or 0) for c in cats
                    if isinstance(c, dict)
                    and "free" not in str(c.get("name", "")).lower()
                    and "buffer" not in str(c.get("name", "")).lower())
        if total:
            parts.append(_fmt_tokens(total)
                         + " sitting in the context window")
    except Exception:
        pass
    return ". ".join(parts) + "."


def _greeting_period(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def _startup_greeting_line(now: "datetime.datetime | None" = None) -> str:
    """Exact required format, spoken ONCE per launch (see amain()'s one
    call site) after Kokoro has genuinely finished loading -- the
    queued text below is itself what triggers mouth.warm() the first
    time the playback thread dequeues it, so nothing can play before
    voice is ready. Uses Asia/Kolkata EXPLICITLY (zoneinfo, via the
    tzdata package -- stdlib zoneinfo has no IANA database of its own
    on Windows) rather than assuming the machine's own clock is set to
    IST: a wrong system timezone used to mean a wrong greeting with no
    way to tell from the text alone."""
    now = now or datetime.datetime.now(IST)
    period = _greeting_period(now.hour)
    time_str = now.strftime("%I:%M %p").lstrip("0")
    return f"Good {period}, Captain. Jarvis is online. It is {time_str}."


def _qwen_confirmation_line(router, *, already_active: bool) -> str:
    """Clean confirmation only -- one sentence, the brain's real name
    and nothing else. Never mentions any other interface: F8/Backtalk
    is the one Jarvis interface, and this line is the proof of it --
    it can never say "F9" or "Local Voice Bridge" because it never
    constructs text from anything but the brain's own label."""
    qwen_label = router.get("qwen3-8b-local").label
    if already_active:
        return f"{qwen_label} is already active, Captain."
    return f"{qwen_label} is active, Captain."


def _cloudbrain_line(*, already_active: bool) -> str:
    """Exact required text for the deterministic "cloud brain" fix --
    a fixed string, not built from router state, on purpose: never
    mentions Gemini (it isn't configured), and always names Claude by
    the one phrasing Captain approved live. Distinct from
    useclaude's own prompt ("Switching to Claude uses your
    subscription usage...") because this is the exact wording a real
    field test required for the "cloud brain" phrasing specifically."""
    if already_active:
        return "Already on Claude."
    return ("Claude Agent SDK is the available cloud brain. Switching "
            "gives it access to its Agent tools and may use my Claude "
            "subscription. Shall I switch to Claude? Say confirm to "
            "proceed.")


# The stable numbered interface: these four never move, regardless of
# which brains are currently enabled/active -- 1=Qwen, 2=DeepSeek,
# 3=Claude, 4=Gemini, always.
_NUM_LABELS = {
    "1": ("qwen3-8b-local", "Qwen3 8B Local"),
    "2": ("deepseek-r1-8b-local", "DeepSeek R1 8B Local"),
    "3": ("claude", "Claude Agent SDK"),
    "4": ("gemini", "Gemini"),
}


def _numbered_brain_menu(router) -> str:
    """Always names all four stable slots, Brain 4/Gemini included by
    name even when it's unconfigured -- an earlier version of this
    answer only narrated ENABLED brains and said "three configured
    brain modes" while silently omitting Gemini, a real live-test
    complaint. The numbers are canonical and never change shape based
    on config. Shared verbatim by both "brainscount" and "listbrains"
    in _run_console_inner, so the two questions always agree."""
    status = router.status()
    parts = []
    for num in ("1", "2", "3", "4"):
        bid, label = _NUM_LABELS[num]
        if bid == status.active_id:
            state = "is active"
        elif bid == "claude":
            state = "needs your confirmation to use"
        elif bid == "gemini":
            gemini = status.brains.get("gemini")
            state = ("is available, gated the same way as Claude"
                     if gemini and gemini.enabled
                     else "is not configured yet")
        else:
            state = "is available"
        parts.append(f"Brain {num}, {label}, {state}")
    return "Captain, here are the brain options: " + "; ".join(parts) + "."


_MEMORY_BRAIN_LINE = (
    "Your vault memory is shared by all brain modes; it is not a "
    "separate brain. Say switch to 1, 2, 3, or 4.")

_MEMORY_MODULES_LINE = (
    "Captain, I use one shared persistent memory system: your vault. "
    "All four brain modes use the same vault identity and selected "
    "context; they do not have separate memory vaults.")


def _switchnum3_line(*, already_active: bool) -> str:
    """Exact required text for brain 3 -- distinct wording from both
    useclaude's own prompt and _cloudbrain_line's, because a real
    field spec asked for this exact sentence when the NUMBER is what
    was said, not the word "claude" or "cloud"."""
    if already_active:
        return "Already on brain 3, Claude Agent SDK."
    return ("Brain 3 is Claude Agent SDK. It may use my Claude "
            "subscription and Agent tools. Say confirm to switch.")


def _switchnum4_line(*, already_active: bool) -> str:
    """Exact required text for brain 4 (Gemini) -- distinct wording
    from usegemini's own ask, because a real field spec asked for this
    exact sentence when the NUMBER is what was said. No longer a
    permanent refusal: Gemini activates through the SAME confirm gate
    as every other numbered brain once GEMINI_API_KEY is present and
    its health check passes -- that gating is enforced by
    router.activate() itself (BrainDisabledError/BrainUnavailableError),
    never duplicated or second-guessed here."""
    if already_active:
        return "Already on brain 4, Gemini."
    return ("Brain 4 is Gemini, an external free-tier service. I "
            "will send only your next approved prompt, not your "
            "vault. Say confirm to switch.")


def _gemini_activated_line() -> str:
    """Required announcement, verbatim, every time Gemini actually
    becomes active -- reached from BOTH "switch to Gemini" and "switch
    to 4" (they converge on the same usegemini:confirmed verb, see
    _run_console_inner). Never a silent swap into an external service.
    Names the external-consent LEASE explicitly: this confirm opens a
    thirty-minute idle window, not a promise to ask again on literally
    every request."""
    return ("Gemini free-tier external mode is active for the next "
            "thirty minutes of use. Requests leave this PC only "
            "during that window; I'll ask again after thirty minutes "
            "of inactivity. Say switch to Qwen any time to go back.")


def _claude_activated_line() -> str:
    """Companion to _gemini_activated_line() for Claude -- same
    external-consent lease, same thirty-minute idle window, distinct
    disclosure (subscription usage and Agent tools, not "leaves this
    PC" -- Claude's own separate per-TOOL permission gate is what
    actually governs file/command/fetch access, unchanged by this)."""
    return ("Claude is active for the next thirty minutes of use. "
            "I'll ask again after thirty minutes of inactivity. Say "
            "switch to Qwen any time to go back.")


def _gemini_preview_line(preview: str) -> str:
    """Exact required format for the external-consent prompt (first
    use each lease, or after a thirty-minute idle expiry) -- the
    EXACT outgoing text is embedded verbatim (never paraphrased),
    followed by the fixed disclosure and confirm instruction."""
    return f"Gemini preview: {preview} This leaves your PC. Say confirm to send."


def _claude_preview_line(preview: str) -> str:
    """Companion to _gemini_preview_line() for Claude."""
    return (f"Claude preview: {preview} This uses your Claude "
            f"subscription and Agent tools. Say confirm to send.")


def _external_preview_line(brain_id: str, preview: str) -> str:
    """Picks the right exact-wording preview for whichever external
    brain is asking -- single dispatch point so callers never have to
    know which brain uses which phrasing."""
    if brain_id == "gemini":
        return _gemini_preview_line(preview)
    return _claude_preview_line(preview)


def _start_external_lease(brain_id: str) -> None:
    """Opens (or refreshes) the external-consent lease for `brain_id`.
    Called once when a switch is confirmed, and again on every actual
    send while the lease is still valid -- an IDLE window, not a fixed
    session length."""
    _EXTERNAL_LEASE["brain_id"] = brain_id
    _EXTERNAL_LEASE["expires_at"] = time.monotonic() + _LEASE_DURATION_S


def _clear_external_lease() -> None:
    """Switching to Qwen or DeepSeek immediately clears any external
    lease -- called explicitly rather than left to an implicit
    brain_id mismatch, so the intent reads plainly at the call site."""
    _EXTERNAL_LEASE["brain_id"] = None
    _EXTERNAL_LEASE["expires_at"] = 0.0


def _external_lease_active(brain_id: str) -> bool:
    return (_EXTERNAL_LEASE["brain_id"] == brain_id
            and time.monotonic() < _EXTERNAL_LEASE["expires_at"])


def _log_external_send(brain_id: str, text: str) -> None:
    """Logged immediately before every real external send -- concise,
    and never carrying anything but the approved utterance itself: no
    secret, vault, or hidden context is ever available to log here in
    the first place (see build_request_preview() -- the bare
    utterance is the only thing either external brain ever receives)."""
    if brain_id == "gemini":
        log(f"[gemini] sending approved user request: {text}")
    else:
        log(f"[{brain_id}] sending approved user request: {text}")


_PASTE_ON = "\x1b[200~"    # bracketed-paste markers (we enable the mode below)
_PASTE_OFF = "\x1b[201~"


# <<anything>> is a stage direction: lifted out, never spoken, published on
# the bus when the audio carrying it starts. Bounded so a runaway model cannot
# swallow a paragraph into one "tag".
_DIRECTION_TAG = re.compile(r"<<([^<>]{1,80})>>")


def _clean_typed(line: str) -> str:
    """Scrub terminal-copy artifacts: blockquote gutter glyphs and stray
    whitespace (copying from a CLI chat render drags bars along)."""
    line = line.strip()
    while line[:1] in ("▎", "│", ">"):
        line = line[1:].lstrip()
    return line


def _join_paste(body: str) -> str:
    """Pasted blob -> one clean message (gutters scrubbed, lines joined)."""
    parts = [_clean_typed(l) for l in body.split("\n")]
    return " ".join(" ".join(p for p in parts if p).split())


def _typed_reader_pipe(q: "queue.Queue[str]", fd: int):
    """Non-tty stdin (pipes/tests): line assembly with paste markers."""
    import os
    pend = ""
    while True:
        try:
            b = os.read(fd, 65536)
        except OSError:
            return
        if not b:
            return
        pend += b.decode("utf-8", "replace")
        while True:
            if _PASTE_ON in pend:
                if _PASTE_OFF not in pend:
                    break
                head, rest = pend.split(_PASTE_ON, 1)
                body, pend = rest.split(_PASTE_OFF, 1)
                *hlines, hpart = head.split("\n")
                for l in hlines:
                    l = _clean_typed(l)
                    if l:
                        q.put(l)
                text = _join_paste(hpart + body)
                if text:
                    q.put(text)
                continue
            if "\n" in pend:
                line, pend = pend.split("\n", 1)
                line = _clean_typed(line)
                if line:
                    q.put(line)
                continue
            break


def _typed_reader_simple(q: "queue.Queue[str]"):
    """Windows (no termios): plain line input on a thread. Pastes work;
    they just echo normally instead of collapsing to a count."""
    while True:
        try:
            line = _clean_typed(input())
        except (EOFError, OSError):
            return
        if line:
            q.put(line)


def _typed_reader(q: "queue.Queue[str]"):
    """Terminal stdin -> typed messages (daemon thread). Typed lines are
    first-class turns: same pipeline as a spoken utterance, spoken reply.

    On a POSIX tty we OWN the input line (cbreak: no kernel echo, no
    canonical buffering — the little line editor below echoes keys,
    handles backspace, and assembles bracketed pastes invisibly). The
    kernel's canonical mode is unfixable for pastes: it echoes the
    markers as visible junk and holds unfinished marker lines hostage.
    Pastes show as `[pasted N chars]`; Enter sends everything as ONE
    message. Ctrl-C still works (ISIG stays on); termios restored at
    exit."""
    import atexit
    import os
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        _typed_reader_pipe(q, fd)
        return
    try:
        import termios
        import tty as _tty
    except ImportError:            # Windows: no termios — simple reader
        _typed_reader_simple(q)
        return
    old = termios.tcgetattr(fd)
    _tty.setcbreak(fd)                      # ECHO+ICANON off, ISIG kept
    sys.stdout.write("\x1b[?2004h")         # bracket pastes, please
    sys.stdout.flush()

    def _restore():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
    atexit.register(_restore)

    MARKS = (_PASTE_ON, _PASTE_OFF)

    def _partial_tail(s: str) -> int:
        """Length of a trailing partial paste-marker (hold it for the
        next read)."""
        for m in MARKS:
            for k in range(min(len(s), len(m) - 1), 0, -1):
                if m.startswith(s[-k:]):
                    return k
        return 0

    buf = ""          # the input line being composed
    paste = None      # accumulating paste body, or None
    pend = ""
    while True:
        try:
            b = os.read(fd, 4096)
        except OSError:
            _restore()
            return
        if not b:
            _restore()
            return
        pend += b.decode("utf-8", "replace")
        keep = _partial_tail(pend)
        proc = pend[:len(pend) - keep] if keep else pend
        pend = pend[len(pend) - keep:] if keep else ""
        i = 0
        while i < len(proc):
            if paste is not None:
                j = proc.find(_PASTE_OFF, i)
                if j < 0:
                    paste += proc[i:]
                    break
                paste += proc[i:j]
                i = j + len(_PASTE_OFF)
                text = _join_paste(paste)
                paste = None
                if text:
                    if buf and not buf.endswith(" "):
                        buf += " "
                    buf += text
                    sys.stdout.write(text if len(text) <= 60
                                     else f"[pasted {len(text)} chars]")
                    sys.stdout.flush()
                continue
            if proc.startswith(_PASTE_ON, i):
                paste = ""
                i += len(_PASTE_ON)
                continue
            ch = proc[i]
            i += 1
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                line = buf.strip()
                buf = ""
                if line:
                    q.put(line)
            elif ch in ("\x7f", "\x08"):     # backspace
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch >= " " or ch == "\t":    # printable: echo + collect
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()


_BOTH_TARGETS = frozenset({"desktop", "phone"})
_PHONE_ONLY_TARGET = frozenset({"phone"})


async def speak_reply(router: BrainRouter, mouth: Mouth, text: str,
                      targets: frozenset = _BOTH_TARGETS):
    """First sentence ships alone (fast start); the rest go in
    2-sentence breaths — fuller chunks get livelier prosody (single
    short sentences come out flat). `router` picks the sentences up
    from whichever brain is currently active -- this function has no
    idea which one that is, and doesn't need to.

    `targets` (2026-09-21, active-platform routing): which speaker(s)
    this reply's audio should actually reach -- forwarded to every
    mouth.say_chunk() call below. Defaults to both, the entire prior
    behavior of this function, for any caller that doesn't pass it.

    Rule 6 (a paired phone disconnecting mid-turn must fall back to
    desktop, never silently vanish): a phone-only turn whose phone
    isn't reachable gets an announced desktop fallback before the
    real answer starts; emit() re-checks on every sentence so a
    disconnect partway through the SAME reply is caught too, not just
    one that already existed before the turn began."""
    t0 = time.time()
    first = True
    batch: list[str] = []
    pending: list[str] = []          # directions waiting for their chunk
    fell_back = False
    if targets == _PHONE_ONLY_TARGET and not phone_bridge.phone_reply_available():
        targets = frozenset({"desktop"})
        fell_back = True
        log("[phone] reply was routed phone-only but the phone isn't "
            "reachable -- falling back to desktop")
        mouth.say_chunk(
            "Your phone isn't reachable right now, so I'm answering "
            "here instead.", targets=targets)

    def emit(raw: str):
        nonlocal first, batch, pending, targets, fell_back
        if (not fell_back and targets == _PHONE_ONLY_TARGET
                and not phone_bridge.phone_reply_available()):
            targets = frozenset({"desktop"})
            fell_back = True
            log("[phone] phone disconnected mid-reply -- falling back "
                "to desktop for the rest of this turn")
            mouth.say_chunk(
                "Your phone disconnected, so I'm finishing this reply "
                "here instead.", targets=targets)
        # STAGE DIRECTIONS: your agent may write <<anything>> inline. It is
        # lifted out here, never spoken, and published on the signal bus when
        # this chunk's audio starts (signals.direction). backtalk has no
        # opinion on what a direction means; something watching the bus does.
        #
        # This used to strip only the ANGLE BRACKETS, which left the tag body
        # in the sentence and the TTS read it aloud.
        found = _DIRECTION_TAG.findall(raw)
        if found:
            pending += [d.strip() for d in found if d.strip()]
        raw = _DIRECTION_TAG.sub(" ", raw)
        # TTS hygiene: backticks and markdown fences are never speakable.
        s = " ".join(raw.replace("`", "").split()).strip()
        if not s:
            return
        if first:
            log(f"[{NAME}] ({time.time()-t0:.1f}s to first) {s}"
                + (f"  <directions: {pending}>" if pending else ""))
            mouth.say_chunk(s, pending, targets=targets)
            pending = []
            first = False
        else:
            log(f"[{NAME}] {s}" + (f"  <directions: {pending}>" if pending else ""))
            batch.append(s)
            if len(batch) >= 2:
                mouth.say_chunk(" ".join(batch), pending, targets=targets)
                pending = []
                batch = []

    try:
        async for sentence in router.ask_stream(text):
            emit(sentence)
        if batch:
            mouth.say_chunk(" ".join(batch), pending, targets=targets)
            pending = []
        if first:
            # Zero sentences yielded (brain error / empty turn): nothing
            # will ever dequeue, so nothing resets the bus — park it here.
            signals.static_stop()
            signals.set_state("idle")
    except asyncio.CancelledError:
        try:
            await router.interrupt()
        except Exception:
            pass
        raise
    except (BrainDisabledError, BrainUnavailableError) as e:
        # A brain can fail MID-STREAM -- a Gemini timeout, an HTTP
        # error, a bad key, a rate-limit, a malformed response, etc.
        # Without this handler the exception propagated straight out
        # of this function unhandled (only CancelledError was ever
        # caught here), leaving the voice line stuck "thinking"
        # forever: nothing ever reset the state or spoke a word about
        # it. No retry, no fallback to a different brain -- the active
        # brain never changes just because one turn failed; Captain
        # hears exactly what went wrong and decides what to do next.
        # The exception message is already key-safe by construction
        # (see gemini_brain.py -- error text names the ENV VAR, never
        # the key's value), so this never risks printing it.
        log(f"[{NAME}] brain error mid-turn: {e}")
        day_journal.record_event("error", str(e))
        signals.static_stop()
        signals.set_state("idle")
        mouth.say_chunk(f"Sorry, I hit an error: {e}"[:300], targets=targets)


async def _answer_then_recommend(router: BrainRouter, mouth: Mouth,
                                 text: str, rec,
                                 targets: frozenset = _BOTH_TARGETS) -> None:
    """Answers normally on whatever brain is already active, THEN --
    only if the turn actually completed, never on a cancelled/
    interrupted one, since CancelledError propagates straight through
    the await below -- speaks a short suggestion if recommend.py found
    a genuinely better-suited brain for this request. Never switches
    anything itself. `targets` (active-platform routing) rides along
    to the suggestion too -- it's part of the same turn."""
    await speak_reply(router, mouth, text, targets=targets)
    if rec:
        mouth.say_chunk(rec.spoken_line, targets=targets)


async def amain():
    open_mic = "--open-mic" in sys.argv
    barge_in = "--barge-in" in sys.argv
    model = None
    if "--model" in sys.argv:
        try:
            model = sys.argv[sys.argv.index("--model") + 1]
        except IndexError:
            pass

    CFG_BOOT_MODE = CFG["permission_mode"]
    _AUTOAPPROVE["on"] = CFG_BOOT_MODE == "bypassPermissions"
    _MIC["mode"] = "open" if (open_mic
                              or CFG.get("mic_mode") == "open") else "ptt"
    # resume_last_session: reattach to the saved conversation, if any
    resume_id = None
    if CFG.get("resume_last_session"):
        try:
            from backtalk.brain import SESSION_FILE
            with open(SESSION_FILE) as f:
                resume_id = f.read().strip() or None
        except OSError:
            resume_id = None

    mouth = Mouth()
    ears = Ears()
    brain_cfg = brain_router_config.load()
    if model:
        # --model only ever meant "override Claude's model id for this
        # session" -- Qwen/DeepSeek don't take this kind of override,
        # so it's applied to the router config's claude entry only,
        # never silently reinterpreted for whichever brain happens to
        # be active.
        brain_cfg.setdefault("brains", {}).setdefault("claude", {})["model"] = model
    router = BrainRouter(config=brain_cfg,
                         can_use_tool=make_permission_gate(mouth),
                         resume_id=resume_id)
    default_id = brain_cfg.get("active_brain") or "qwen3-8b-local"
    default_brain = router.get(default_id)
    # RECOVERY MODE: the owner pointed brain_router.json's active_brain
    # at a gated brain (Claude) directly. That is a deliberate, offline
    # config edit -- not a live spoken "switch to Claude" -- so booting
    # trusts the config file itself as the approval and skips the
    # interactive confirm for THIS ONE boot-time activation only. It
    # must never be quiet about doing that: every recovery boot is both
    # logged and spoken, unmistakably, before anything else happens. A
    # LIVE "switch to Claude" later in this same session still always
    # goes through the real spoken confirm below -- recovery mode never
    # weakens that.
    recovery_mode = default_brain.requires_confirm_to_switch
    if recovery_mode:
        log(f"[backtalk] RECOVERY MODE: booting directly into "
            f"{default_brain.label} because brain_router.json's "
            f"active_brain is {default_id!r}. This bypasses the live "
            f"spoken confirm because the config file itself is the "
            f"approval -- a live 'switch to Claude' later in this "
            f"session still requires the spoken confirm as normal.")

    mode = ("hands-free listening (the talk key still works)"
            if _MIC["mode"] == "open"
            else f"push-to-talk ({CFG['ptt_key']})")
    log(f"[backtalk] up — agent={NAME} dir={CFG['agent_dir']} "
        f"brain={default_brain.label} mic={mode} "
        f"(say 'goodbye {NAME.lower()}' to hang up)")
    # The Jarvis-specific greeting REPLACES upstream backtalk's own
    # CFG["greeting"]/greeting_open_mic template here (that mechanism
    # stays untouched in config.py for upstream compatibility -- it's
    # just not what this call site uses anymore). Exactly once per
    # launch: this is the only call site, and amain() runs once per
    # process.
    mouth.say(_startup_greeting_line())
    if recovery_mode:
        mouth.say(f"Heads up — recovery mode. I'm booting straight "
                  f"into {default_brain.label} because that's what "
                  f"your config says, not because you approved it out "
                  f"loud this session.")

    loop = asyncio.get_event_loop()
    # Warm the engines while the greeting plays: the STT model load and
    # the brain's prompt-cache toll both hide behind the spoken line.
    loop.run_in_executor(None, warm_ears)
    # THE BRAIN CONNECT, guarded. For Claude this is the one startup
    # step that needs a signed-in Claude Code, internet, and available
    # usage; for a local brain it needs Ollama reachable and the model
    # pulled. Either way, when it fails or hangs, the mouth still
    # works, so SAY SO instead of dying silently with the face stuck
    # on idle (a real field case: the greeting played, then nothing,
    # and on Windows the window closed before anyone could read the
    # error).
    log("[backtalk] connecting the brain...")
    try:
        await asyncio.wait_for(
            router.activate(default_id, confirmed=True), 120)

        async def _warmup():
            async for _ in router.ask_stream(
                    "Warmup ping - reply with the single word: ready"):
                pass
        await asyncio.wait_for(_warmup(), 180)
    except (Exception, asyncio.TimeoutError) as e:
        kind = ("timed out" if isinstance(e, asyncio.TimeoutError)
                else f"failed: {e!r}"[:220])
        log(f"[backtalk] BRAIN CONNECT {kind}")
        mouth.say(f"Bad news. The voice and the face are fine, but I "
                  f"couldn't reach my brain, {default_brain.label}. "
                  f"Check this window for the error: {e}"[:400])
        mouth.wait_done(timeout=30)
        raise SystemExit(1)
    log("[backtalk] brain warm")
    # the hidden warmup ping is plumbing, not conversation
    router.session.update(turns=0, out_tokens=0, in_tokens=0, cost=0.0)
    # a configured effort level applies at launch (saved by the spoken
    # "set effort to X", or written by the person's agent on request)
    # -- Claude-only; a local brain has no equivalent, so this is
    # skipped rather than silently pretending it did something.
    boot_effort = str(CFG.get("effort") or "").strip().lower()
    if boot_effort in _EFFORTS and router.active_id == "claude":
        await router.command(f"/effort {boot_effort}")
        log(f"[backtalk] effort set to {boot_effort} (from config)")
    elif boot_effort and router.active_id != "claude":
        log(f"[backtalk] effort {boot_effort!r} in config skipped -- "
            f"not on Claude, no equivalent to set")
    elif boot_effort:
        log(f"[backtalk] ignoring unknown effort {boot_effort!r} in config")

    speak_task: asyncio.Task | None = None
    typed_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_typed_reader, args=(typed_q,), daemon=True).start()
    typed_fut: asyncio.Future | None = None

    # PHONE BRIDGE (2026-09, off by default): pushes phone-typed/spoken
    # text onto this SAME typed_q, so it reaches handle() exactly like
    # stdin -- no second brain route, no second approval path. See
    # phone_bridge.py's module docstring. Two independent ways to say
    # "off" both actually mean it: enabled=false always wins regardless
    # of mode, and mode="disabled" always wins regardless of enabled.
    async def _phone_interrupt() -> None:
        """Wired into phone_bridge as the immediate on-press barge-in:
        called the INSTANT Captain's thumb touches Hold-to-talk on the
        phone, before any recording or transcription happens -- unlike
        the interrupt block inside handle() below, which only fires
        once a full utterance has already arrived (seconds later, for
        voice). Same cancellation path as that block: cancel
        speak_task, mouth.shut_up() to silence whatever's mid-play on
        BOTH phone and desktop (Kokoro/ElevenLabs share the one
        OutputStream), then await the cancellation actually landing
        before returning -- so phone_bridge's /interrupt endpoint only
        acks once real silence has landed, matching "wait for
        acknowledgement, then capture/send the new request.\""""
        nonlocal speak_task
        _deny_pending()
        if speak_task and not speak_task.done():
            log("[turn] interrupted by phone hold-to-talk press")
            speak_task.cancel()
        mouth.shut_up()
        if speak_task:
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            speak_task = None

    _phone_cfg = CFG.get("phone") or {}
    if _phone_cfg.get("enabled") and (_phone_cfg.get("mode") or "disabled") != "disabled":
        phone_bridge.start(typed_q, router)
        phone_bridge.set_interrupt_handler(_phone_interrupt)

    async def run_console(verb):
        """One voice-console verb. The current reply was already
        cancelled and awaited by handle(); the pipe gets drained here
        before the command goes out. A verb that blows up must never
        take the whole voice session down with it."""
        try:
            await _run_console_inner(verb)
        except Exception as e:
            log(f"[console] {verb} failed: {e}")
            mouth.say("That command hit an error. Check the log.")
            signals.set_state("idle")

    async def _run_console_inner(verb):
        nonlocal speak_task
        _deny_pending()
        await router.reset_turn()
        say_after = None
        # Brain 1/2 switch directly and 3/4's confirmed step activates
        # exactly like "switch to Claude"/"switch to Gemini" already
        # do -- rather than duplicate that logic, the numbered verb is
        # translated onto the EXISTING verb here, so there's exactly
        # one place each actually runs. Brain 3/4's INITIAL ask keeps
        # its own distinct handling below (see _switchnum3_line/
        # _switchnum4_line) since those need their own exact wording.
        if verb == "switchnum:1":
            verb = "useqwen"
        elif verb == "switchnum:2":
            verb = "usedeepseek"
        elif verb == "switchnum:3:confirmed":
            verb = "useclaude:confirmed"
        elif verb == "switchnum:4:confirmed":
            verb = "usegemini:confirmed"

        def _claude_only_refusal(what):
            active = router.status().brains.get(router.active_id)
            active_label = active.label if active else "no brain"
            return (f"There's no {what} on {active_label} -- that's a "
                    f"Claude console command. Say switch to Claude "
                    f"first if you want it.")

        if verb == "clear":
            # Now meaningful for every brain: Claude's SDK session and
            # an Ollama brain's own running transcript (added alongside
            # the conversation-continuity fix) both understand /clear.
            resp = await router.command("/clear")
            say_after = "Cleared. Fresh slate."
        elif verb == "compact":
            if router.active_id != "claude":
                resp = ""
                say_after = _claude_only_refusal("session to compact")
            else:
                mouth.say("Compacting. One moment.")
                resp = await router.command("/compact")
                say_after = "Compacted. Same conversation, smaller footprint."
        elif verb == "deep":
            if router.active_id != "claude":
                resp = ""
                say_after = _claude_only_refusal("deep model to switch to")
            else:
                mouth.say("Switching to the deep model. Heads up, replies "
                          "get slower. Say back to the fast model when "
                          "you're done.")
                resp = await router.command(f"/model {CFG['deep_model']}")
                say_after = "Deep model online, for this session only."
        elif verb == "fast":
            if router.active_id != "claude":
                resp = ""
                say_after = _claude_only_refusal("fast/deep model switch")
            else:
                resp = await router.command(f"/model {CFG['model']}")
                say_after = "Back on the fast model."
        elif verb.startswith("effort:"):
            lvl = verb.split(":", 1)[1]
            if router.active_id != "claude":
                resp = ""
                say_after = _claude_only_refusal("effort level to set")
            else:
                resp = await router.command(f"/effort {lvl}")
                saved = _write_config_key("effort", lvl)
                say_after = (f"Effort set to {lvl}, and saved as your "
                             "default." if saved else
                             f"Effort set to {lvl} for this session. The "
                             "config file couldn't be written, so it "
                             "won't stick past a restart.")
        elif verb == "usage":
            resp = ""
            mouth.say(_spoken_usage(router.session,
                                    await router.context_usage()))
        elif verb == "micopen":
            resp = ""
            if _MIC["mode"] == "open":
                mouth.say("Already in hands-free listening.")
            else:
                _MIC["mode"] = "open"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "open")
                log("[console] mic_mode -> open (hands-free listening)")
                mouth.say("Hands-free listening on. I'm always "
                          "listening now, so anything said in the room "
                          "can reach me. The talk key still works, and "
                          "holding it always gets you heard. Say push "
                          "to talk mode to bring the button back.")
        elif verb == "micptt":
            resp = ""
            if _MIC["mode"] == "ptt":
                mouth.say("Already on push to talk.")
            else:
                _MIC["mode"] = "ptt"
                _MIC["gen"] += 1
                _write_config_key("mic_mode", "ptt")
                log("[console] mic_mode -> ptt")
                key = str(CFG.get("ptt_key", "home")).replace("_", " ")
                mouth.say(f"Push to talk. Hold the {key} key and "
                          "talk; the mic stays closed otherwise.")
        elif verb == "replyphone":
            resp = ""
            _REPLY_TARGET_OVERRIDE["mode"] = "phone"
            log("[console] reply routing -> phone only (override)")
            mouth.say("Replies go to your phone only, until you say "
                      "otherwise.")
        elif verb == "replydesktop":
            resp = ""
            _REPLY_TARGET_OVERRIDE["mode"] = "desktop"
            log("[console] reply routing -> desktop only (override)")
            mouth.say("Replies go to the desktop only, until you say "
                      "otherwise.")
        elif verb == "replyboth":
            resp = ""
            _REPLY_TARGET_OVERRIDE["mode"] = "both"
            log("[console] reply routing -> both (override)")
            mouth.say("Replies go to both the desktop and your phone, "
                      "until you say otherwise.")
        elif verb == "replyauto":
            resp = ""
            _REPLY_TARGET_OVERRIDE["mode"] = None
            log("[console] reply routing -> automatic (by origin)")
            mouth.say("Back to automatic reply routing -- each reply "
                      "goes to wherever you asked from.")
        elif verb == "noask":
            resp = ""
            _CONFIRM["verb"] = "noask"
            _CONFIRM["at"] = time.monotonic()
            mouth.say("Auto-approve means I act without asking "
                      "permission, and it becomes your saved default. "
                      "Say confirm to switch.")
        elif verb == "noask:confirmed":
            resp = ""
            saved = _write_config_key("permission_mode",
                                      "bypassPermissions")
            _AUTOAPPROVE["on"] = True
            log("[console] permission_mode -> bypassPermissions"
                + (" (saved)" if saved else " (session only)"))
            mouth.say(("Auto-approve on, and saved as your default. "
                       if saved else
                       "Auto-approve on for this session. The config "
                       "file couldn't be written, so it won't stick "
                       "past a restart. ")
                      + "Say start asking again any time to flip it "
                        "back.")
        elif verb == "ask":
            resp = ""
            saved = _write_config_key("permission_mode", "ask")
            _AUTOAPPROVE["on"] = False
            flipped = True
            if CFG_BOOT_MODE == "bypassPermissions":
                # a bypass-booted session never consults the gate, so
                # the SDK itself must flip (the safe direction is
                # allowed live). If that fails, saying "done" would be
                # a lie: the agent would keep acting silently.
                try:
                    await router.set_permission_mode("ask")
                except Exception as e:
                    flipped = False
                    log(f"[console] live flip to ask FAILED: {e}")
            log("[console] permission_mode -> ask"
                + (" (saved)" if saved else " (session only)"))
            if flipped:
                mouth.say("Done. I'll ask out loud before real "
                          "actions"
                          + (", and that's saved as your default."
                             if saved else
                             ". The config file couldn't be written, "
                             "so tell me again after a restart."))
            else:
                mouth.say("I saved asking as your default, but this "
                          "session couldn't switch over. Restart the "
                          "voice line to get asking back.")
        elif verb == "useqwen":
            resp = ""
            already = router.active_id == "qwen3-8b-local"
            _clear_external_lease()  # switching to a local brain clears it
            if already:
                line = _qwen_confirmation_line(router, already_active=True)
            else:
                try:
                    await router.activate("qwen3-8b-local")
                    line = _qwen_confirmation_line(router,
                                                   already_active=False)
                    day_journal.record_event("brain_switch",
                                             "Switched to Qwen3 8B Local")
                except (BrainDisabledError, BrainUnavailableError) as e:
                    line = f"Couldn't switch to Qwen: {e}"[:300]
            log(f"[console] useqwen -> {line}")
            mouth.say(line)
        elif verb == "usedeepseek":
            resp = ""
            _clear_external_lease()  # switching to a local brain clears it
            if router.active_id == "deepseek-r1-8b-local":
                line = "Already on DeepSeek local reasoning."
            else:
                try:
                    await router.activate("deepseek-r1-8b-local")
                    # Required announcement, verbatim, every time this
                    # brain becomes active: never a silent swap.
                    line = ("DeepSeek local reasoning, online. It's "
                            "local and free, same as Qwen, but slower "
                            "-- it thinks before it answers. Say "
                            "switch to Qwen when you want the fast "
                            "brain back.")
                    day_journal.record_event(
                        "brain_switch", "Switched to DeepSeek R1 8B Local")
                except (BrainDisabledError, BrainUnavailableError) as e:
                    line = f"Couldn't switch to deep reasoning: {e}"[:300]
            log(f"[console] usedeepseek -> {line}")
            mouth.say(line)
        elif verb == "useclaude":
            resp = ""
            if router.active_id == "claude":
                line = "Already on Claude."
            else:
                _CONFIRM["verb"] = "useclaude"
                _CONFIRM["at"] = time.monotonic()
                line = ("Switching to Claude uses your subscription "
                        "usage and Agent tools. Confirming opens a "
                        "thirty-minute session -- I'll ask again after "
                        "thirty minutes of inactivity. Say confirm to "
                        "switch.")
            log(f"[console] useclaude -> {line}")
            mouth.say(line)
        elif verb == "useclaude:confirmed":
            resp = ""
            try:
                await router.activate("claude", confirmed=True)
                _start_external_lease("claude")
                line = _claude_activated_line()
                day_journal.record_event("brain_switch",
                                         "Switched to Claude Agent SDK")
            except (BrainDisabledError, BrainUnavailableError,
                    ConfirmRequiredError) as e:
                line = f"Couldn't reach Claude: {e}"[:300]
            log(f"[console] useclaude:confirmed -> {line}")
            mouth.say(line)
        elif verb == "usegemini":
            resp = ""
            if router.active_id == "gemini":
                line = "Already on Gemini."
            else:
                _CONFIRM["verb"] = "usegemini"
                _CONFIRM["at"] = time.monotonic()
                line = ("Switching to Gemini sends your own words to "
                        "an external service on your free tier. "
                        "Confirming opens a thirty-minute session -- "
                        "I'll ask again after thirty minutes of "
                        "inactivity. Say confirm to switch.")
            log(f"[console] usegemini -> {line}")
            mouth.say(line)
        elif verb == "usegemini:confirmed":
            resp = ""
            try:
                await router.activate("gemini", confirmed=True)
                _start_external_lease("gemini")
                line = _gemini_activated_line()
                day_journal.record_event("brain_switch",
                                         "Switched to Gemini")
            except (BrainDisabledError, BrainUnavailableError,
                    ConfirmRequiredError) as e:
                line = f"Couldn't reach Gemini: {e}"[:300]
            log(f"[console] usegemini:confirmed -> {line}")
            mouth.say(line)
        elif verb == "whichbrain":
            resp = ""
            status = router.status()
            active = (status.brains.get(status.active_id)
                     if status.active_id else None)
            if active:
                line = (f"You're on {active.label}. It handles "
                       f"{active.capability_summary}")
            else:
                line = ("No brain is active right now, which "
                        "shouldn't happen. Check the log.")
            log(f"[console] whichbrain -> {line}")
            mouth.say(line)
        elif verb == "brainscount":
            resp = ""
            # Always names all four numbered positions now (Brain 4,
            # Gemini, explicitly as "not configured yet") -- the old
            # enabled-only phrasing said "three configured brain
            # modes" and silently dropped Gemini instead of naming it
            # as the fourth, unconfigured slot, a real live-test
            # complaint. Same function "list brains"/"brain options"
            # already use, so the two questions now agree exactly.
            line = _numbered_brain_menu(router)
            log(f"[console] brainscount -> {line}")
            mouth.say(line)
        elif verb == "switchmemorybrain":
            resp = ""
            log(f"[console] switchmemorybrain -> {_MEMORY_BRAIN_LINE}")
            mouth.say(_MEMORY_BRAIN_LINE)
        elif verb == "memorymodules":
            resp = ""
            # A FIXED string, deliberately not built from router state:
            # there is exactly one vault and that fact never varies by
            # which brains are enabled or active, so there is nothing
            # here that could ever go stale the way a brain-roster
            # answer could.
            log(f"[console] memorymodules -> {_MEMORY_MODULES_LINE}")
            mouth.say(_MEMORY_MODULES_LINE)
        elif verb == "cloudbrain":
            resp = ""
            # Claude Agent SDK is the only real cloud brain -- this
            # enters the SAME confirm gate "switch to Claude" uses
            # (reusing _CONFIRM, never a separate mechanism), so
            # "confirm" resolves it exactly like useclaude:confirmed
            # below. Never mentions Gemini: it isn't configured, and
            # naming it here would be exactly the kind of stale,
            # config-drifting claim this whole fix exists to prevent.
            already = router.active_id == "claude"
            if not already:
                _CONFIRM["verb"] = "useclaude"
                _CONFIRM["at"] = time.monotonic()
            line = _cloudbrain_line(already_active=already)
            log(f"[console] cloudbrain -> {line}")
            mouth.say(line)
        elif verb == "ambiguousbrain":
            resp = ""
            # "switch to Gemini or Claude" -- never guess. No state
            # change at all: the current brain stays exactly as it
            # was, and nothing is armed on _CONFIRM.
            line = "Do you mean Brain 3, Claude, or Brain 4, Gemini?"
            log(f"[console] ambiguousbrain -> {line}")
            mouth.say(line)
        elif verb == "listbrains":
            resp = ""
            line = _numbered_brain_menu(router)
            log(f"[console] listbrains -> {line}")
            mouth.say(line)
        elif verb == "switchask":
            resp = ""
            _BRAIN_NUM_PENDING["pending"] = True
            _BRAIN_NUM_PENDING["at"] = time.monotonic()
            mouth.say("Which brain number?")
        elif verb == "switchnum:3":
            resp = ""
            already = router.active_id == "claude"
            if not already:
                _CONFIRM["verb"] = "switchnum:3"
                _CONFIRM["at"] = time.monotonic()
            line = _switchnum3_line(already_active=already)
            log(f"[console] switchnum:3 -> {line}")
            mouth.say(line)
        elif verb == "switchnum:4":
            resp = ""
            # This ASK never constructs Gemini and never makes a
            # network call -- it only speaks the required line and
            # arms the same confirm gate "switch to Gemini" uses.
            # Whether the key is actually present and healthy is
            # decided at CONFIRM time by router.activate() itself
            # (translated to usegemini:confirmed above), never
            # pre-checked or duplicated here.
            already = router.active_id == "gemini"
            if not already:
                _CONFIRM["verb"] = "switchnum:4"
                _CONFIRM["at"] = time.monotonic()
            line = _switchnum4_line(already_active=already)
            log(f"[console] switchnum:4 -> {line}")
            mouth.say(line)
        elif verb == "closeday":
            resp = ""
            # The Day Journal service: a local, deterministic ledger
            # summary, no brain and no network involved in building
            # it. See day_journal.py -- this REPLACED the earlier
            # design that asked Claude to write the note.
            summary = day_journal.build_daily_summary()
            log(f"[console] closeday event_count={summary.event_count} "
                f"had_any_activity={summary.had_any_activity}")
            _CONFIRM["verb"] = "closeday"
            _CONFIRM["at"] = time.monotonic()
            mouth.say(summary.spoken() + " Shall I save this to "
                      "today's daily note? Say confirm to save.")
        elif verb == "closeday:confirmed":
            resp = ""
            # Recomputed fresh rather than carried over from the
            # initial ask: building the summary is cheap (one local
            # ledger read, no network, no brain), so recomputing
            # guarantees the note is never built from a stale
            # snapshot -- and it's the exact same deterministic
            # function, so nothing about the content can drift
            # between what was spoken and confirmed.
            summary = day_journal.build_daily_summary()
            try:
                day_journal.write_daily_note(summary)
                say_after = "Saved to today's daily note."
            except OSError as e:
                say_after = f"Couldn't save the daily note: {e}"[:300]
        elif verb == "pairphone":
            resp = ""
            phone_cfg = CFG.get("phone") or {}
            mode = phone_cfg.get("mode") or "disabled"
            if not (phone_cfg.get("enabled") and mode != "disabled"):
                mouth.say("Phone access is turned off. Set phone.enabled "
                          "to true and phone.mode to tailscale in "
                          "backtalk.json, and restart, to pair a phone.")
            elif not phone_bridge.bridge_url():
                # start() already logged the specific reason (Tailscale
                # not installed/running/signed in); the spoken line
                # stays short and points at the log rather than guessing
                mouth.say("The phone bridge isn't actually running -- "
                          "check the terminal log for why. Tailscale "
                          "may not be signed in.")
            else:
                # the trust step comes first, every time -- cheap to
                # show again if already verified, and correct the one
                # time it's genuinely a fresh phone or a rotated CA
                phone_tls.print_verification_screen(
                    phone_bridge._bind_ip, phone_bridge._port,
                    phone_bridge._tls_port)
                code, expires_at = phone_auth.create_pairing()
                url = phone_bridge.bridge_url()
                log("=" * 56)
                log(f"[phone] PAIRING CODE (expires in "
                    f"{phone_auth._CODE_TTL_S}s): {code}")
                log(f"[phone] On the phone (signed into this Tailscale "
                    f"network), open: {url}")
                log("=" * 56)
                # the code itself is deliberately never spoken aloud --
                # eight random letters/digits read out loud invites a
                # mis-hearing; the terminal is the trusted channel here
                mouth.say("Pairing code is ready. Check the terminal "
                          "screen for the code and the address to open "
                          "on the phone.")
        elif verb == "showphonecert":
            resp = ""
            phone_cfg = CFG.get("phone") or {}
            if (phone_cfg.get("mode") or "disabled") != "tailscale":
                mouth.say("There's no certificate to show -- phone.mode "
                          "isn't set to tailscale.")
            elif not phone_bridge.bridge_url():
                mouth.say("The phone bridge isn't actually running -- "
                          "check the terminal log for why.")
            else:
                phone_tls.print_verification_screen(
                    phone_bridge._bind_ip, phone_bridge._port,
                    phone_bridge._tls_port)
                mouth.say("Certificate fingerprint is on the terminal "
                          "screen.")
        elif verb == "listphones":
            resp = ""
            devices = phone_auth.list_devices()
            if not devices:
                mouth.say("No phones are paired yet.")
            else:
                lines = [f"{i}. {d['label']}" for i, d in enumerate(devices, 1)]
                log("[phone] paired devices: " + "; ".join(lines))
                mouth.say(f"{len(devices)} phone"
                          f"{'s' if len(devices) != 1 else ''} paired: "
                          + ", ".join(lines) + ". Say revoke phone and "
                          "the number to remove one.")
        elif verb.startswith("revokephone:"):
            resp = ""
            idx = int(verb.split(":", 1)[1])
            label = phone_auth.revoke_device(idx)
            if label:
                mouth.say(f"Revoked {label}. It will need to be paired "
                          "again to reconnect.")
            else:
                mouth.say("There's no phone at that number. Say list "
                          "paired phones to see the current list.")
        else:
            resp = ""
        if say_after:
            # the CLI answers slash commands with its own text
            # (confirmations, API errors); an error outranks our line
            low = (resp or "").lower()
            if resp and ("error" in low or "invalid" in low):
                mouth.say(resp[:160])
                log(f"[console] {verb} answered: {resp[:120]}")
            else:
                mouth.say(say_after)
        signals.set_state("idle")

    async def handle(text: str, spoke_from: float | None = None,
                     source_platform: str = "desktop") -> bool:
        """Process one utterance; returns False on quit. spoke_from is
        when the utterance STARTED (the PTT press), so an answer can be
        told apart from speech that began before the ask even existed.
        source_platform (2026-09-21, active-platform routing) is where
        THIS utterance came from -- "desktop" for every existing
        caller (typed stdin, PTT, open mic), "phone" only when
        typed_fut's result unwraps a phone_bridge.PhoneTurn. Used
        below (see _resolve_reply_targets) to decide which speaker(s)
        the reply's audio should reach."""
        nonlocal speak_task
        log(f"[you]    {text}")
        # A pending spoken permission ask owns the next utterance IF
        # that utterance started after the ask was posed. Speech that
        # began earlier is the user interrupting the turn, not
        # answering a question they never heard: the ask resolves as a
        # silent deny and the utterance falls through as a normal
        # interrupt. Quit wins either way, but only as an EXACT phrase
        # here ("No! Don't hang up, skip it" must stay a deny reason,
        # not kill the session).
        if _PERM["fut"] is not None and not _PERM["fut"].done():
            started_after = (spoke_from is None
                             or spoke_from >= _PERM["asked_at"])
            if _norm_speech(text) in {_norm_speech(q)
                                      for q in QUIT_PHRASES}:
                _PERM["fut"].set_result("no")
                # falls through to the quit body below
            elif started_after:
                _PERM["fut"].set_result(text)
                return True
            else:
                _deny_pending()
        # A pending auto-approve confirm owns it too, for two minutes;
        # after that it expires and speech flows normally again.
        verb = None
        if _CONFIRM["verb"]:
            pend, _CONFIRM["verb"] = _CONFIRM["verb"], None
            expired = time.monotonic() - _CONFIRM["at"] > 120
            if not expired and _is_confirm_phrase(text):
                verb = pend + ":confirmed"
            elif not expired and not any(q in text.lower()
                                         for q in QUIT_PHRASES):
                mouth.say("Staying as we are.")
                return True
        # A pending external-lease consent (Claude or Gemini, after
        # first switch-confirm or after a 30-minute idle expiry) owns
        # the next utterance too: only an accepted confirm phrase
        # sends the exact text that was previewed; anything else
        # declines it outright, never a silent retry, never a silent
        # skip. Separate from _CONFIRM above on purpose -- see
        # _EXTERNAL_PENDING and _EXTERNAL_LEASE.
        if _EXTERNAL_PENDING["text"] is not None:
            pending_text = _EXTERNAL_PENDING["text"]
            pending_brain = _EXTERNAL_PENDING["brain_id"]
            _EXTERNAL_PENDING["text"] = None
            _EXTERNAL_PENDING["brain_id"] = None
            expired = time.monotonic() - _EXTERNAL_PENDING["at"] > PERM_TIMEOUT_S
            if (not expired and _is_confirm_phrase(text)
                    and router.active_id == pending_brain):
                _start_external_lease(pending_brain)
                _log_external_send(pending_brain, pending_text)
                signals.set_state("thinking")
                signals.static_start()
                speak_task = asyncio.create_task(
                    speak_reply(router, mouth, pending_text,
                               targets=_resolve_reply_targets(source_platform)))
                return True
            elif not expired and not any(q in text.lower()
                                         for q in QUIT_PHRASES):
                mouth.say("Not sent.")
                signals.set_state("idle")
                return True
            # expired, the active brain changed meanwhile, or a quit
            # phrase: fall through to normal handling
        # A bare "switch" owns the next utterance too: only a bare
        # number resolves it (never plain conversation), and only
        # within the timeout window.
        if _BRAIN_NUM_PENDING["pending"]:
            _BRAIN_NUM_PENDING["pending"] = False
            expired = (time.monotonic() - _BRAIN_NUM_PENDING["at"]
                      > PERM_TIMEOUT_S)
            if not expired:
                num = _bare_brain_number(text)
                if num:
                    verb = f"switchnum:{num}"
                elif not any(q in text.lower() for q in QUIT_PHRASES):
                    mouth.say("That's not a brain number I recognize. "
                              "Say switch, then a number one through "
                              "four.")
                    return True
            # expired, unrecognized, or a quit phrase: fall through
        # "Confirm" said with nothing actually pending above must
        # never fall through to a local brain and get answered as an
        # ordinary question -- it explains itself and stops there.
        if verb is None and _is_confirm_phrase(text):
            mouth.say("There's no pending switch to confirm.")
            return True
        if any(q in text.lower() for q in QUIT_PHRASES):
            if speak_task and not speak_task.done():
                speak_task.cancel()
            mouth.shut_up()
            mouth.say(CFG["signoff"])
            mouth.wait_done(timeout=15)
            return False
        if speak_task and not speak_task.done():
            log("[turn] interrupted mid-reply by new input")
            _deny_pending()          # an ask never outlives its turn
            speak_task.cancel()
            mouth.shut_up()
        if speak_task:
            # Let the cancellation fully land (its router.interrupt()
            # included) BEFORE anything else touches the router —
            # otherwise the dead turn's stop signal can race in after
            # the new query and kill the new answer (half of the
            # off-by-one bug; see router.reset_turn for the other half).
            try:
                await speak_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            speak_task = None
        # The strict exact-phrase console verbs try first (cheapest,
        # zero false-positive risk); brain_intent is the tolerant
        # fallback for natural phrasing ("Hello Jarvis, which brain
        # are you using?") that console_match() was never built to
        # survive -- see brain_intent's module docstring for the real
        # field-test failures this closes.
        verb = verb or console_match(text) or brain_intent.detect(text)
        if verb:
            await run_console(verb)
            return True
        signals.set_state("thinking")
        signals.static_start()
        # Clean the pipe: drain the interrupted turn's leftovers so the
        # new question can't pair with a stale ResultMessage. A gate
        # that fired in the meantime resolves first, or the drain would
        # wait on a ResultMessage the CLI is withholding for an answer.
        _deny_pending()
        await router.reset_turn()
        active_brain = router.get(router.active_id) if router.active_id else None
        if active_brain and active_brain.requires_external_lease:
            if _external_lease_active(router.active_id):
                # Lease still valid: refresh the idle window (this IS
                # a use) and fall through to the normal answer path
                # below -- no repeated warning, no repeated confirm.
                _start_external_lease(router.active_id)
                _log_external_send(router.active_id, text)
            else:
                # No valid lease -- never confirmed this brain yet, or
                # the lease expired from thirty minutes of inactivity.
                # Show the EXACT text that would be sent and wait for
                # an explicit confirm before any real external call.
                # No answer is generated here -- see _EXTERNAL_PENDING's
                # resolution above for what happens once (or if) it's
                # approved.
                preview = active_brain.build_request_preview(text)
                _EXTERNAL_PENDING["text"] = text
                _EXTERNAL_PENDING["brain_id"] = router.active_id
                _EXTERNAL_PENDING["at"] = time.monotonic()
                line = _external_preview_line(router.active_id, preview)
                log(f"[console] {router.active_id} preview -> {line}")
                mouth.say(line)
                signals.set_state("idle")
                return True
        rec = recommend.recommend(text, router)
        speak_task = asyncio.create_task(
            _answer_then_recommend(router, mouth, text, rec,
                                   targets=_resolve_reply_targets(source_platform)))
        return True

    try:
        # ONE loop, two mic modes, switchable live (_MIC). The talk key
        # is constructed and honored in BOTH modes: in hands-free
        # listening it is the interrupt and the guaranteed way to be
        # heard over room noise. The open mic joins the wait-set only
        # in "open" mode; a mode switch bumps _MIC["gen"], the abort
        # callable closes the in-flight open mic promptly, and any
        # capture born under an old gen is discarded unprocessed.
        ptt = PTTListener(CFG["ptt_key"])
        press_fut: asyncio.Future | None = None
        mic_fut: asyncio.Future | None = None
        mic_gen_seen = _MIC["gen"]
        # The open mic yields while the BUTTON records (or the double
        # capture would turn one held utterance into two turns), and,
        # without barge-in, while the mouth speaks.
        mic_gate = (lambda: _MIC["btn"]
                    or (not barge_in and mouth.speaking))
        mic_fails = 0
        while True:
            if _MIC["gen"] != mic_gen_seen:
                mic_gen_seen = _MIC["gen"]
                # consume futures that completed under the old mode so
                # a stale press or capture can't fire after a switch
                if press_fut is not None and press_fut.done():
                    press_fut.result(); press_fut = None
                if mic_fut is not None and mic_fut.done():
                    mic_fut.result(); mic_fut = None
            if typed_fut is None:
                typed_fut = loop.run_in_executor(None, typed_q.get)
            if press_fut is None:
                press_fut = loop.run_in_executor(None, ptt.wait_press)
            waiters = {press_fut, typed_fut}
            if _MIC["mode"] == "open":
                if mic_fut is None:
                    g = _MIC["gen"]
                    mic_fut = loop.run_in_executor(
                        None, lambda g=g: (g, ears.listen_once(
                            gate=mic_gate,
                            abort=lambda: _MIC["gen"] != g)))
                waiters.add(mic_fut)
            done, _ = await asyncio.wait(
                waiters, return_when=asyncio.FIRST_COMPLETED)
            if typed_fut in done:
                item = typed_fut.result(); typed_fut = None
                # 2026-09-21, active-platform routing: phone_bridge
                # puts a PhoneTurn wrapper on this SAME queue for
                # phone-originated text/voice; every desktop reader
                # (stdin, both cbreak and pipe modes) still puts a bare
                # str, unchanged -- so this is the ONE place that tells
                # a phone turn apart from a desktop one.
                if isinstance(item, phone_bridge.PhoneTurn):
                    text, source_platform = item.text, "phone"
                else:
                    text, source_platform = item, "desktop"
                if text and not await handle(text, source_platform=source_platform):
                    return
                continue
            if mic_fut is not None and mic_fut in done:
                try:
                    g, text = mic_fut.result()
                except Exception as e:
                    mic_fut = None
                    mic_fails += 1
                    if not explain_audio_failure(e):
                        log(f"[ears] open mic failed ({mic_fails}): {e!r}")
                    if mic_fails >= 3:
                        _MIC["mode"] = "ptt"
                        _MIC["gen"] += 1
                        mic_fails = 0
                        mouth.say("The open microphone keeps failing, "
                                  "so I'm switching to push to talk. "
                                  "Hold the key to reach me, and "
                                  "check this window for the error.")
                    continue
                mic_fut = None
                if g != _MIC["gen"]:
                    continue             # captured before a switch
                if text and not await handle(text):
                    return
                continue
            if press_fut in done:
                press_fut.result(); press_fut = None
                press_t = time.monotonic()
                perm_wait = (_PERM["fut"] is not None
                             and not _PERM["fut"].done())
                if speak_task and not speak_task.done() and not perm_wait:
                    log("[turn] interrupted mid-reply — key pressed")
                    speak_task.cancel()          # the button = interrupt
                # During a permission ask the TURN stays alive; the
                # press only silences playback and records the answer.
                mouth.shut_up()
                signals.static_stop()            # button kills the static too
                signals.set_state("listening")
                mouth.ducker.speech_start()      # duck NOW, while you talk
                print("[ptt] recording (release to send)...", flush=True)
                _MIC["btn"] = True               # open mic yields to the button
                try:
                    text = await loop.run_in_executor(
                        None, lambda: record_held(ptt.is_held))
                except Exception as e:
                    # A device-level failure gets plain words instead of a
                    # raw exception. The pre-flight at startup cannot catch
                    # a microphone unplugged mid-session, and that is the
                    # case where the old message was worst: jargon, on
                    # every press, with the key hook still working so it
                    # looked like it was listening.
                    if explain_audio_failure(e):
                        mouth.say("I can't hear you. There's no working "
                                  "microphone I can use.")
                    else:
                        log(f"[ears] record/transcribe failed: {e!r}")
                        mouth.say("My ears hit an error. Check this "
                                  "window for the details.")
                    text = None
                finally:
                    _MIC["btn"] = False
                mouth.ducker.speech_end(0.2)     # snap back fast on release
                if not text:
                    log("[ptt] (tap or empty — ignored)")
                    signals.set_state("idle")
                    continue
                if not await handle(text, spoke_from=press_t):
                    return
    except KeyboardInterrupt:
        pass
    finally:
        _MIC["gen"] += 1     # abort any live open-mic capture promptly
        if speak_task and not speak_task.done():
            speak_task.cancel()
        mouth.shutdown()  # restores the music on Ctrl-C / crash paths too
        signals.static_stop()
        signals.set_state("idle")
        await router.shutdown()
        log("[backtalk] hung up")


# Loopback port used purely as a mutex. Nothing is ever served on it.
_INSTANCE_PORT = 8791
_instance_lock = None


def _claim_single_instance() -> bool:
    """Refuse to be the second voice line on this machine, out loud.

    Two instances both hold the keyboard hook and both open the
    microphone, and the result looks EXACTLY like a broken talk key:
    presses register, the audio goes to whichever process won the
    device, and the loser reports an ignored tap. Nothing warned about
    it, so a user who double-clicks the Talk icon twice concludes the
    product is broken. The tell, when it was finally caught, was the
    same sentence transcribed twice at an identical timestamp.

    A bound socket is the mutex rather than a pid file, because the
    operating system releases it when this process dies HOWEVER it dies.
    A pid file outlives a crash or a force-kill and then lies about a
    process that is long gone, which is the failure it would exist to
    prevent.
    """
    global _instance_lock
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # No SO_REUSEADDR here on purpose: reuse is exactly what would let a
    # second instance bind alongside the first and defeat the whole point.
    try:
        s.bind(("127.0.0.1", _INSTANCE_PORT))
        s.listen(1)
    except OSError:
        s.close()
        return False
    _instance_lock = s
    return True


def main():
    if not _claim_single_instance():
        print("[backtalk] ANOTHER VOICE LINE IS ALREADY RUNNING on this "
              "machine, so this one is stopping.", flush=True)
        print("[backtalk] Two of them fight over the microphone and the "
              "talk key, which looks exactly like the talk key being "
              "broken. Use the window that is already open, or close it "
              "and start again.", flush=True)
        sys.exit(1)
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\n[backtalk] interrupted — hanging up", flush=True)


if __name__ == "__main__":
    main()
