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
"""The phone bridge: a small FastAPI app, IN-PROCESS with backtalk's own
asyncio loop, so a phone-typed or phone-spoken message can be pushed
onto the SAME typed_q queue that stdin already feeds -- it reaches
main.py's handle() exactly like typed text, so brain switching, the
permission gate, and everything else stay the one deterministic path.
This module invents no second approval mechanism and no second brain
route: it is a text/audio DOOR into F8, nothing more.

REMOTE ONLY, VIA TAILSCALE (Captain's call, 2026-09-28 -- the earlier
LAN/home-Wi-Fi bridge is gone completely, not paused): there is no
LAN listener anywhere in this file, and none will be added back. The
only path in is mobile data -> an authenticated Tailscale phone ->
this bridge, bound ONLY to the desktop's Tailscale interface IP (see
phone_auth.detect_tailscale_ip -- a 100.64.0.0/10 address, never the
LAN, never 0.0.0.0). Two required, independent access layers: (1)
Tailscale network membership itself -- without it, packets to this
IP don't route, full stop, before any HTTP request is possible; (2)
Jarvis device pairing (phone_auth.py) -- a paired device token alone
is useless off the tailnet, and tailnet membership alone gets nothing
without a token.

Never imports backtalk.hermes or anything in local-jarvis: a phone
message can only ever reach F8's own typed_q, nothing downstream of it
directly.

TWO EXPLICIT MODES (CFG["phone"]["mode"], never overlapping booleans --
Captain's call, 2026-09-27), both still gated by CFG["phone"]["enabled"]
being true first:

  "disabled"   - no listeners. Also the safe fallback for any value
                 this module doesn't recognize (including the retired
                 "typed_http"/"tls" values from the abandoned LAN
                 design, if a stale config still has one).
  "tailscale"  - CFG["phone"]["port"] serves ONLY the CA certificate
                 bootstrap/verification page (build_bootstrap_app) --
                 never the app, never pairing, never an approval.
                 CFG["phone"]["tls_port"] serves the FULL app
                 (build_app) over HTTPS, mic included. See
                 phone_tls.py for the certificate design and its
                 SHA-256 fingerprint-verification protocol. There is
                 no HTTP fallback for the app anywhere in this file,
                 and none will be added -- fail closed, always. If
                 Tailscale isn't installed/running/signed in, start()
                 logs why and starts NOTHING, rather than guessing at
                 some other address to bind.
"""
import asyncio
import io
import json
import re
import socket
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from backtalk import ears, mouth, phone_auth, phone_tls
from backtalk.config import CFG
from backtalk.vlog import log

# NOTE: no `from __future__ import annotations` in this file on purpose.
# FastAPI resolves each endpoint's parameter types at RUNTIME (via
# typing.get_type_hints against the function's own __globals__) to
# decide what to inject -- Request, a query param, a body model. That
# future-import turns every annotation into a lazy string, which broke
# resolution the moment Request/HTTPException were imported LOCALLY
# inside build_app() instead of at module level: every "request:
# Request" silently became an unresolvable QUERY parameter named
# "request", and every route 422'd with "Field required". Fixed by
# importing fastapi's names here, at module level, and leaving
# annotations un-stringified.

_WEB_DIR = Path(__file__).resolve().parent / "phone_web"


@dataclass(frozen=True)
class PhoneTurn:
    """Wraps a phone-originated utterance on typed_q so main.py's
    amain() consumer loop can tell it apart from desktop-typed/stdin
    text (which stays a bare str, unchanged) -- see its typed_fut call
    site for handle()'s source_platform argument. Carries only the
    text; this module's own per-request auth/session-id/rate-limit
    checks already happened before this is ever put on the queue."""
    text: str

MAX_MESSAGE_CHARS = 4000
MAX_VOICE_BYTES = 8_000_000       # ~8MB: generous for a multi-minute 16kHz mono clip
VOICE_SAMPLE_RATE = 16000         # must match ears.transcribe()'s expectation

_msg_limiter = phone_auth.RateLimiter(max_calls=30, window_s=60)
_voice_limiter = phone_auth.RateLimiter(max_calls=10, window_s=60)
_pair_limiter = phone_auth.RateLimiter(max_calls=10, window_s=300)
# Generous on purpose: fires on EVERY Hold-to-talk press, not just
# every submitted message, so a fast back-to-back "no wait, actually--"
# press pattern must never trip this the way a message flood would.
_interrupt_limiter = phone_auth.RateLimiter(max_calls=120, window_s=60)

# 2026-09-21 field defect: near-silent audio (dead air, background
# noise) reaching Whisper produces hallucinated filler text like
# "You" or "Thank you" -- a well-known Whisper failure mode, not a
# transcription bug to patch around with a text blocklist (which
# would risk suppressing someone who genuinely said those exact
# words). Fixed upstream of transcription instead: an int16 RMS below
# this floor never reaches ears.transcribe() at all. Chosen low
# (~0.25% of full scale) so it only catches true silence/near-dead-air
# and never a genuinely quiet utterance recorded at normal mic gain.
VOICE_SILENCE_RMS = 80

# 2026-09-21 field defect, diagnosed: replies stayed quiet on a real
# phone (S23 Ultra) even after the original gain stage -- which only
# normalized PEAK amplitude (scale the single loudest sample up to a
# target). That does NOT equal perceived loudness: ordinary speech has
# brief loud transients (consonant bursts) sitting well above its
# sustained vowel/breath level, so a clip can hit its target peak on
# one syllable while the REST of it -- almost the entire perceived
# loudness -- stays quiet. Fixed by normalizing to a target RMS
# (average level, what a human ear actually judges as "loud") instead,
# with the old peak fraction now used only as the CEILING that gain is
# never allowed to cross, i.e. the limiter, not the loudness target.
# See _normalize_gain. A client-side WebAudio gain stage (index.html)
# sits on top of this as a real, user-controlled boost that can exceed
# what the file itself contains -- this is only the server-side floor.
PHONE_TARGET_RMS = 0.20
PHONE_TARGET_PEAK = 0.85
# The calibration clip /speak/test.wav renders -- runs through the
# EXACT same synthesis + gain pipeline a real reply uses, so it
# actually calibrates what Captain will hear, not an abstract tone.
PHONE_TEST_PHRASE = ("This is a test of your phone's audio. "
                     "One, two, three.")
# Default mobile-only playback speed (index.html's <audio> element);
# desktop's sounddevice output path in mouth.py is untouched by
# construction -- nothing in this file writes to that OutputStream.
PHONE_DEFAULT_PLAYBACK_RATE = 1.12

_transcript: deque = deque(maxlen=200)
_transcript_next_id = 0
_pending_permission: dict | None = None
_sse_open: dict[str, int] = {}    # device id -> open SSE connection count
_typed_q = None
_router = None
_mode = "disabled"
_bind_ip = ""
_port = 0
_tls_port = 0
_server = None              # the full app's HTTPS server, on _tls_port
_bootstrap_server = None    # the plain-HTTP cert bootstrap page, on _port

# 2026-09-29 field defect: main.py's speak_reply() splits ONE Jarvis
# reply into SEVERAL mouth.say_chunk() calls for TTS pacing (first
# sentence alone, then 2-sentence batches) -- each call used to become
# its own separate transcript entry, so the phone only ever saw
# fragments, never the complete reply, and had no way to auto-play a
# single complete rendition. Fragments are buffered here and published
# as ONE atomic transcript entry only when mouth.py's turn-complete
# hook fires (see finalize_jarvis_turn), which is the real "the whole
# reply has finished" signal (signals.reply_done()'s own trigger), not
# a per-sentence one.
_pending_jarvis_parts: list[str] = []
# 2026-09-21, active-platform routing: which platform(s) the reply
# CURRENTLY being buffered above should actually play audio on --
# every chunk of one turn carries the same targets by construction
# (speak_reply() computes it once per turn and passes it to every
# say_chunk() call for that turn), so just tracking the latest value
# is correct. None only if somehow nothing was ever buffered; defaults
# to "both" at finalize time in that case, matching the pre-routing
# behavior this feature was added on top of.
_pending_jarvis_targets: frozenset | None = None

# 2026-09-21 field defect: pressing Hold-to-talk again while Jarvis was
# still mid-reply didn't stop anything until the NEW recording had
# already been captured, uploaded, and transcribed -- seconds late, and
# with no guarantee the interrupted turn's half-buffered text (see
# _pending_jarvis_parts above) wouldn't bleed into the next reply. Set
# by main.py's amain() only when the phone bridge actually starts (see
# set_interrupt_handler) -- an async no-arg callable that cancels the
# in-flight brain turn and silences current speech; awaited by the
# /interrupt endpoint below so the phone only gets its "ok" once real
# silence has landed on both sides.
_interrupt_handler = None
_last_voice_session: dict[str, str] = {}  # device id -> last accepted capture-session id


def record_transcript_line(who: str, text: str, targets=None) -> None:
    """Called from mouth.py (via its transcript-sink hook, see
    mouth.set_transcript_sink) for EVERY spoken chunk, and from this
    module's own /message and /voice handlers for the CAPTAIN's side.
    Never raises -- a transcript is a convenience for the phone page,
    not something that may ever take the voice line down.

    Jarvis's chunks are buffered, not appended individually -- see the
    module-level note on _pending_jarvis_parts and finalize_jarvis_
    turn() below, which is what actually publishes them. `targets`
    (2026-09-21, active-platform routing) is mouth.py's per-chunk
    routing set, tracked so finalize_jarvis_turn can tag the published
    entry with it -- see /speak/{turn}/{chunk}.wav, which refuses to
    serve audio for a turn that was never routed to the phone."""
    global _pending_jarvis_targets
    try:
        if who == "jarvis":
            _pending_jarvis_parts.append(text)
            _pending_jarvis_targets = targets
            return
        _append_transcript_entry(who, text)
    except Exception as e:
        log(f"[phone] transcript append failed (non-fatal): {e}")


def _append_transcript_entry(who: str, text: str, targets=None) -> None:
    global _transcript_next_id
    _transcript_next_id += 1
    entry = {"id": _transcript_next_id, "ts": time.time(),
             "who": who, "text": text}
    if targets is not None:
        entry["targets"] = sorted(targets)
    _transcript.append(entry)


def finalize_jarvis_turn() -> None:
    """Called from mouth.py's turn-complete hook the moment a full
    reply has genuinely finished (the whole queue drained, not just a
    gap between two chunks of the same reply) -- joins every buffered
    fragment into ONE transcript entry, published atomically, so the
    phone's SSE stream only ever sees a complete line, never a partial
    one. A no-op if nothing was buffered (e.g. reply_done firing with
    no phone-relevant speech in between, or two fires in a row)."""
    global _pending_jarvis_targets
    if not _pending_jarvis_parts:
        _pending_jarvis_targets = None
        return
    full_text = " ".join(p for p in _pending_jarvis_parts if p).strip()
    _pending_jarvis_parts.clear()
    targets = _pending_jarvis_targets or frozenset({"desktop", "phone"})
    _pending_jarvis_targets = None
    if not full_text:
        return
    try:
        _append_transcript_entry("jarvis", full_text, targets=targets)
    except Exception as e:
        log(f"[phone] turn finalize failed (non-fatal): {e}")


def set_pending_permission(question: str | None) -> None:
    """Called from main.py's make_permission_gate right where it speaks
    the question, and again with None once it resolves. A no-op object
    write; safe to call whether or not the bridge is running."""
    global _pending_permission
    _pending_permission = ({"question": question, "asked_at": time.time()}
                           if question else None)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Splits a finalized reply into the same per-sentence units the
    phone plays back one at a time (see /speak/{turn_id}/{chunk}.wav).
    Falls back to the whole text as a single "sentence" if it has no
    terminal punctuation to split on, and to an empty list only for
    genuinely empty text (finalize_jarvis_turn never publishes one)."""
    stripped = text.strip()
    if not stripped:
        return []
    parts = [s.strip() for s in _SENTENCE_END.split(stripped) if s.strip()]
    return parts or [stripped]


def _normalize_gain(pcm: np.ndarray, target_rms: float = PHONE_TARGET_RMS,
                    target_peak: float = PHONE_TARGET_PEAK) -> np.ndarray:
    """Loudness-normalizes one int16 PCM chunk toward target_rms of
    full scale -- 2026-09-21 field defect: "Jarvis reply volume on the
    phone is too low," diagnosed as peak-only normalization not
    matching perceived loudness (see the module-level comment above
    PHONE_TARGET_RMS for the full diagnosis). The gain that would hit
    target_rms is computed first, then CAPPED by whichever gain would
    push the chunk's actual peak past target_peak -- that cap is the
    limiter, never the goal, so this can still never clip. Only ever
    LIFTS a chunk; one already loud enough (by either measure) is left
    exactly alone. A genuinely silent chunk (rms or peak 0) is
    returned unchanged rather than divided by zero."""
    if pcm.size == 0:
        return pcm
    peak = int(np.max(np.abs(pcm)))
    if peak == 0:
        return pcm
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms == 0:
        return pcm
    loudness_gain = (target_rms * 32767) / rms
    clip_safe_gain = (target_peak * 32767) / peak
    gain = min(loudness_gain, clip_safe_gain)
    if gain <= 1.0:
        return pcm
    return np.clip(pcm.astype(np.float64) * gain, -32767, 32767).astype(np.int16)


def set_interrupt_handler(fn) -> None:
    """Registered by main.py's amain() right after phone_bridge.start()
    -- the ONE way this module reaches back into F8's own turn/speech
    state without importing main.py (which already imports this
    module; importing it back would be circular). fn is an async
    no-arg callable; see the /interrupt endpoint for the only caller."""
    global _interrupt_handler
    _interrupt_handler = fn


def _voice_warm() -> bool:
    # Reaching into these modules' own warm-cache globals rather than
    # adding new public API just for a health check -- read-only, and
    # both are the exact objects warm() populates once and never
    # replaces (see mouth.py/ears.py's own module-level locks).
    return getattr(mouth, "_pipe", None) is not None or \
           getattr(ears, "_model", None) is not None


def _face_available() -> bool:
    from backtalk import signals
    return Path(signals._STATE_FILE).exists()


def _vault_available() -> bool:
    from backtalk.brains import vault_context
    return vault_context.VAULT_ROOT.exists() and vault_context.VAULT_INDEX.exists()


def _active_brain() -> str:
    try:
        return _router.active_id if _router else "unknown"
    except Exception:
        return "unknown"


def build_app():
    app = FastAPI(title="Jarvis Phone Bridge", docs_url=None, redoc_url=None)
    index_html = (_WEB_DIR / "index.html").read_text(encoding="utf-8")

    def _client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _authed_device(request: Request, token_qs: str | None = None) -> dict:
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else (token_qs or "")
        device = phone_auth.verify_token(token, _client_ip(request))
        if not device:
            raise HTTPException(status_code=401, detail="unpaired or revoked device")
        return device

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return index_html

    @app.post("/pair/redeem")
    async def pair_redeem(request: Request):
        ip = _client_ip(request)
        if not _pair_limiter.allow(ip):
            log(f"[phone] pairing REJECTED (bridge rate limit): ip={ip}")
            raise HTTPException(status_code=429, detail="too many attempts, wait a bit")
        body = await request.json()
        # "".join(...split()) strips ALL whitespace, not just leading/
        # trailing -- 2026-09-29 field failure: a mobile keyboard's
        # autocorrect can insert a space MID-code, which .strip() alone
        # never catches (the client now filters this live too, see
        # index.html's normalizeCodeField, but a code is never
        # legitimately anything but one contiguous token either way).
        code = "".join(str(body.get("code", "")).split()).upper()
        label = str(body.get("label", "") or "")[:60]
        ca_fp = phone_tls.current_ca_fingerprint() if _mode == "tailscale" else None
        token = phone_auth.redeem_pairing(code, ip, label, ca_fingerprint=ca_fp)
        if not token:
            raise HTTPException(status_code=401, detail="invalid or expired code")
        return {"token": token}

    @app.get("/health")
    async def health():
        return {"server_reachable": True, "phone_enabled": True}

    @app.get("/health/full")
    async def health_full(request: Request, token: str | None = None):
        device = _authed_device(request, token)
        return {
            "server_reachable": True,
            "phone_client_connected": _sse_open.get(device["id"], 0) > 0,
            # server-side Kokoro/Whisper readiness -- NOT the same thing
            # as whether a connecting phone's browser can use its OWN
            # microphone (see phone_mic_supported below). This stays
            # true even while that's false: text-in and Jarvis's spoken
            # replies both work fine over plain HTTP.
            "voice_available": _voice_warm(),
            # a phone browser's getUserMedia() needs a secure context --
            # the tailscale-mode app is always HTTPS, so this is True
            # for any real request that reaches this endpoint at all;
            # computed from the actual scheme (not hardcoded) so it
            # stays honest if this endpoint is ever reached any other way.
            "phone_mic_supported": request.url.scheme == "https",
            "face_available": _face_available(),
            "active_brain": _active_brain(),
            "vault_available": _vault_available(),
        }

    @app.post("/message")
    async def message(request: Request):
        device = _authed_device(request)
        if not _msg_limiter.allow(device["id"]):
            raise HTTPException(status_code=429, detail="slow down a little")
        body = await request.json()
        text = str(body.get("text", ""))[:MAX_MESSAGE_CHARS].strip()
        if not text:
            raise HTTPException(status_code=400, detail="empty message")
        record_transcript_line(f"captain (phone: {device['label']})", text)
        _typed_q.put(PhoneTurn(text))
        return {"ok": True}

    @app.post("/voice")
    async def voice(request: Request):
        device = _authed_device(request)
        if not _voice_limiter.allow(device["id"]):
            raise HTTPException(status_code=429, detail="slow down a little")
        # 2026-09-21: exactly-one-submission guarantee. The phone sends
        # a fresh UUID per Hold-to-talk press (X-Capture-Session); a
        # second /voice call carrying the SAME id for the SAME device
        # is a duplicate (a synthetic touch-then-mouse event, a retried
        # fetch after a flaky connection, ...), never a second real
        # recording -- ack it without transcribing or queuing it a
        # second time. No header at all (an older/other client) skips
        # this check entirely rather than failing closed on a missing
        # feature.
        session_id = request.headers.get("x-capture-session")
        if session_id and _last_voice_session.get(device["id"]) == session_id:
            return {"ok": True, "text": "", "duplicate": True}
        declared = request.headers.get("content-length")
        if declared and int(declared) > MAX_VOICE_BYTES:
            raise HTTPException(status_code=413, detail="clip too large")
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_VOICE_BYTES:
                raise HTTPException(status_code=413, detail="clip too large")
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            data, rate = sf.read(io.BytesIO(raw), dtype="int16")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"unreadable audio: {e}")
        if rate != VOICE_SAMPLE_RATE:
            raise HTTPException(
                status_code=400,
                detail=f"expected {VOICE_SAMPLE_RATE}Hz mono, got {rate}Hz")
        if data.ndim > 1:
            data = data[:, 0]
        if session_id:
            _last_voice_session[device["id"]] = session_id
        # 2026-09-21 field defect: near-silent audio reaching Whisper
        # hallucinates filler text ("You", "Thank you"). Gate on RMS
        # BEFORE transcription rather than filtering its output -- see
        # VOICE_SILENCE_RMS above for why, and why the floor is low.
        rms = (float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
              if data.size else 0.0)
        if rms < VOICE_SILENCE_RMS:
            return {"ok": True, "text": ""}
        text = await asyncio.to_thread(ears.transcribe, data)
        if not text:
            return {"ok": True, "text": ""}
        record_transcript_line(f"captain (phone voice: {device['label']})", text)
        _typed_q.put(PhoneTurn(text))
        return {"ok": True, "text": text}

    @app.post("/interrupt")
    async def interrupt(request: Request):
        """Fired by the phone the INSTANT Hold-to-talk is pressed,
        before any recording happens -- stops current Kokoro playback
        on the phone (client-side, on this same response) and on the
        desktop, and cancels the active F8 turn, via
        set_interrupt_handler's callback. Also drops any not-yet-
        finalized Jarvis reply fragments from the turn being
        interrupted (see _pending_jarvis_parts) so a late-arriving
        chunk from the OLD turn can never get stitched into or
        resurrect the NEW turn's transcript entry. Responds only after
        the handler's cancellation has actually landed, so the phone's
        "wait for acknowledgement, then record" sequencing is real,
        not just polite ordering."""
        global _pending_jarvis_targets
        device = _authed_device(request)
        if not _interrupt_limiter.allow(device["id"]):
            raise HTTPException(status_code=429, detail="slow down a little")
        # Handler FIRST, clear SECOND -- not the other way round. The
        # handler's mouth.shut_up() is what actually stops mouth.py's
        # worker thread from dequeuing any FURTHER sentence (its
        # transcript-sink call fires at dequeue time, before playback);
        # clearing before that lands would leave a window where one
        # more OLD-turn fragment could still land in
        # _pending_jarvis_parts AFTER the clear, undoing it.
        if _interrupt_handler is not None:
            try:
                await _interrupt_handler()
            except Exception as e:
                log(f"[phone] interrupt handler failed (non-fatal): {e}")
        _pending_jarvis_parts.clear()
        _pending_jarvis_targets = None
        return {"ok": True}

    @app.get("/speak/{turn_id}/{chunk}.wav")
    async def speak_chunk(turn_id: int, chunk: int, request: Request,
                          token: str | None = None):
        """One sentence of a finalized reply, as its own small,
        complete, ordinary WAV file -- 2026-09-21: "phone replies
        should start faster." The old /speak/{id}.wav blocked until
        Jarvis's WHOLE reply had synthesized before returning a single
        byte; index.html's playChunksSequentially now fetches and
        plays sentence 0 the moment IT is ready, then 1, then 2, ...
        instead of waiting for the whole thing. A chunk index past the
        reply's last sentence 404s -- the client's normal, expected
        stop signal, not an error."""
        _authed_device(request, token)
        entry = next((t for t in _transcript if t["id"] == turn_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="unknown turn")
        # 2026-09-21, active-platform routing: a turn explicitly routed
        # desktop-only (rule 3, "a desktop/F8 turn speaks only on
        # desktop") must be genuinely unfetchable from the phone, not
        # just hidden from autoplay -- enforced here, server-side,
        # rather than trusted to the client. 404, not 403: consistent
        # with "unknown turn"/"no more chunks" above, so a desktop-only
        # turn's existence isn't distinguishable from one that never
        # existed. Entries with no "targets" at all predate this
        # feature or came from a call site that never routes (say()) --
        # both default to "both", the whole prior behavior.
        if "phone" not in (entry.get("targets") or ["phone", "desktop"]):
            raise HTTPException(status_code=404, detail="unknown turn")
        sentences = _split_sentences(entry["text"])
        if chunk < 0 or chunk >= len(sentences):
            raise HTTPException(status_code=404, detail="no more chunks")

        def _render() -> bytes:
            # Off the event loop: Kokoro/ElevenLabs synthesis is
            # blocking, real work -- same reasoning as ears.transcribe
            # above. mouth.synth_stream() is read-only reuse of the
            # exact pipeline say_chunk() already drives; this never
            # touches the live desktop OutputStream (audio law #1).
            chunks, rate = [], VOICE_SAMPLE_RATE
            for sr, pcm in mouth.synth_stream(sentences[chunk]):
                rate = sr
                chunks.append(_normalize_gain(pcm))
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                for pcm in chunks:
                    w.writeframes(pcm.tobytes())
            return buf.getvalue()

        wav_bytes = await asyncio.to_thread(_render)
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.get("/speak/test.wav")
    async def speak_test(request: Request, token: str | None = None):
        """Calibration clip for the phone's "Test phone audio" control
        -- runs PHONE_TEST_PHRASE through the EXACT same synthesis +
        gain pipeline a real reply uses (mouth.synth_stream +
        _normalize_gain), so what Captain hears here is genuinely what
        a real reply will sound like, not an abstract tone. Not tied
        to any real transcript entry or turn id."""
        _authed_device(request, token)

        def _render() -> bytes:
            chunks, rate = [], VOICE_SAMPLE_RATE
            for sr, pcm in mouth.synth_stream(PHONE_TEST_PHRASE):
                rate = sr
                chunks.append(_normalize_gain(pcm))
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                for pcm in chunks:
                    w.writeframes(pcm.tobytes())
            return buf.getvalue()

        wav_bytes = await asyncio.to_thread(_render)
        return Response(content=wav_bytes, media_type="audio/wav")

    @app.get("/events")
    async def events(request: Request, token: str | None = None):
        device = _authed_device(request, token)
        did = device["id"]
        if _sse_open.get(did, 0) >= 3:
            raise HTTPException(status_code=429, detail="too many open connections")

        async def gen():
            _sse_open[did] = _sse_open.get(did, 0) + 1
            last_tid = _transcript[-1]["id"] if _transcript else 0
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    from backtalk import signals
                    try:
                        state = Path(signals._STATE_FILE).read_text().strip()
                    except OSError:
                        state = "unknown"
                    new_lines = [t for t in _transcript if t["id"] > last_tid]
                    if new_lines:
                        last_tid = new_lines[-1]["id"]
                    payload = {
                        "state": state,
                        "active_brain": _active_brain(),
                        "pending_permission": _pending_permission,
                        "new_lines": new_lines,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(1.0)
            finally:
                _sse_open[did] = max(0, _sse_open.get(did, 1) - 1)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def build_bootstrap_app():
    """The PLAIN-HTTP port, Tailscale-bound like everything else in
    this file: exactly two routes, both public (no device token --
    there's no app to protect here, only a certificate file and an
    instructions page). Never mounts /message, /voice, /pair, /events,
    or anything else the real app has -- see the module docstring's
    "no HTTP fallback for the app" commitment."""
    app = FastAPI(title="Jarvis Certificate Setup", docs_url=None, redoc_url=None)
    bootstrap_html = (_WEB_DIR / "bootstrap.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return bootstrap_html.replace("{{TLS_PORT}}", str(_tls_port))

    @app.get("/ca.crt")
    async def ca_cert():
        if not phone_tls.CA_CERT_FILE.exists():
            raise HTTPException(status_code=404, detail="no certificate yet")
        return Response(content=phone_tls.CA_CERT_FILE.read_bytes(),
                        media_type="application/x-x509-ca-cert")

    return app


def bridge_url() -> str:
    """The address of the FULL APP (HTTPS, on tls_port), once start()
    has succeeded. Empty string if the bridge was never started or
    failed to start (e.g. Tailscale unavailable)."""
    return f"https://{_bind_ip}:{_tls_port}/" if _bind_ip else ""


def bootstrap_url() -> str:
    """The certificate-setup page's address. Empty if the bridge was
    never started."""
    return f"http://{_bind_ip}:{_port}/" if _bind_ip else ""


def phone_reply_available() -> bool:
    """True only when Jarvis can genuinely speak a reply back through a
    paired phone right now: the HTTPS app actually started (bind_ip set
    -- start() fails closed if Tailscale wasn't reachable, so an empty
    bind_ip means no listener exists at all) AND at least one device is
    currently paired (a running-but-never-paired bridge is not a
    connected phone). Brains call this fresh each turn so a revoked or
    never-paired phone never gets an honest-sounding false claim, and a
    genuinely connected one never gets wrongly denied."""
    if not _bind_ip:
        return False
    try:
        return bool(phone_auth.list_devices())
    except Exception:
        return False


def start(typed_q, router) -> None:
    """Called once from main.py's amain(), right after typed_q exists,
    and only when CFG["phone"]["enabled"] is true and mode is not
    "disabled". Schedules each server as a task on the ALREADY-RUNNING
    loop -- no new thread, no new process, so the app shares typed_q/
    _PERM with zero IPC.

    FAILS CLOSED: if phone_auth.detect_tailscale_ip() can't get a real
    address (Tailscale not installed, not running, not signed in),
    NOTHING starts -- no listener on any interface, no fallback address
    ever guessed at. bridge_url()/bootstrap_url() stay empty strings so
    callers (main.py's "pair a phone") can tell honestly."""
    global _typed_q, _router, _mode, _bind_ip, _port, _tls_port
    global _server, _bootstrap_server

    phone_cfg = CFG.get("phone") or {}
    _typed_q = typed_q
    _router = router
    _mode = phone_cfg.get("mode") or "disabled"
    _port = int(phone_cfg.get("port") or 8765)
    _tls_port = int(phone_cfg.get("tls_port") or 8766)

    if _mode == "disabled":
        # Defensive only -- main.py's own gate (enabled AND mode !=
        # "disabled") should already keep this function from being
        # called at all in this case.
        log("[phone] start() called with mode=disabled -- not starting anything")
        return
    if _mode != "tailscale":
        log(f"[phone] unrecognized phone.mode {_mode!r} -- not starting "
            f"anything (treating like disabled)")
        return

    _bind_ip = phone_cfg.get("bind_ip") or phone_auth.detect_tailscale_ip()
    if not _bind_ip:
        log("[phone] Tailscale IP unavailable -- NOT starting the phone "
            "bridge (no fallback address, by design). Make sure "
            "Tailscale is installed, running, and signed in (`tailscale "
            "status` in a terminal), or set phone.bind_ip in "
            "backtalk.json if you know the address.")
        return

    mouth.set_transcript_sink(record_transcript_line)
    mouth.set_turn_complete_sink(finalize_jarvis_turn)
    phone_tls.ensure_certificates(_bind_ip, socket.gethostname())

    boot_app = build_bootstrap_app()
    boot_config = uvicorn.Config(boot_app, host=_bind_ip, port=_port,
                                 log_level="warning", access_log=False)
    _bootstrap_server = uvicorn.Server(boot_config)
    asyncio.create_task(_bootstrap_server.serve())

    app = build_app()
    tls_config = uvicorn.Config(
        app, host=_bind_ip, port=_tls_port,
        log_level="warning", access_log=False,
        ssl_keyfile=str(phone_tls.LEAF_KEY_FILE),
        ssl_certfile=str(phone_tls.LEAF_CERT_FILE))
    _server = uvicorn.Server(tls_config)
    asyncio.create_task(_server.serve())

    log(f"[phone] certificate bootstrap page (plain HTTP, Tailscale-bound) "
        f"at http://{_bind_ip}:{_port}/")
    log(f"[phone] app (HTTPS, Tailscale-bound) at https://{_bind_ip}:{_tls_port}/ "
        f"-- say \"pair a phone\" for the verification screen and a code")
