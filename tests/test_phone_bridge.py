"""Unit tests for backtalk.phone_bridge: the FastAPI app itself, run
through Starlette's TestClient (no real socket bind -- see
phone_bridge.start() for the only place that ever binds a real port,
which this file never calls). Confirms: unpaired requests are rejected,
a paired device's message reaches the SAME typed_q queue stdin uses,
voice upload validates sample rate and size, rate limits trip, and
revoking a device invalidates it immediately.
"""
import io
import queue
import unittest
import wave
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tests import isolation

isolation.ensure_isolated()  # explicit, invocation-agnostic -- see its docstring

from backtalk import mouth, phone_auth, phone_bridge


def _make_wav(seconds=0.2, rate=16000, amplitude=4000) -> bytes:
    """A synthetic clip for /voice tests. Defaults to a real (non-
    silent) 440Hz tone well above VOICE_SILENCE_RMS -- 2026-09-21: the
    RMS silence gate was added upstream of ears.transcribe(), so a
    literal all-zero clip (the old default) would now never even reach
    the mocked transcribe() these tests assert against. Pass
    amplitude=0 for tests that specifically want to exercise silence."""
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        if amplitude:
            t = np.arange(n)
            pcm = (amplitude * np.sin(2 * np.pi * 440 * t / rate)).astype(np.int16)
        else:
            pcm = np.zeros(n, dtype=np.int16)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class PhoneBridgeTestCase(unittest.TestCase):
    def setUp(self):
        phone_auth.DEVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
        phone_auth._save_devices([])
        phone_auth._pair_attempts.clear()
        phone_bridge._transcript.clear()
        phone_bridge._transcript_next_id = 0
        phone_bridge._pending_jarvis_parts.clear()
        phone_bridge._sse_open.clear()
        phone_bridge._last_voice_session.clear()
        phone_bridge._interrupt_handler = None
        phone_bridge._interrupt_limiter = phone_auth.RateLimiter(max_calls=120, window_s=60)
        phone_bridge.set_pending_permission(None)
        phone_bridge._msg_limiter = phone_auth.RateLimiter(max_calls=30, window_s=60)
        phone_bridge._voice_limiter = phone_auth.RateLimiter(max_calls=10, window_s=60)
        phone_bridge._pair_limiter = phone_auth.RateLimiter(max_calls=10, window_s=300)
        phone_bridge._typed_q = queue.Queue()
        phone_bridge._router = None
        phone_bridge._mode = "tailscale"
        self.app = phone_bridge.build_app()
        self.client = TestClient(self.app)

    def _pair(self, ip="10.9.9.1", label="Test Phone") -> str:
        code, _ = phone_auth.create_pairing()
        token = phone_auth.redeem_pairing(code, ip, label)
        self.assertIsNotNone(token)
        return token

    def auth(self, token):
        return {"Authorization": f"Bearer {token}"}


class HealthTests(PhoneBridgeTestCase):
    def test_health_unauthenticated_reports_server_reachable(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["server_reachable"])

    def test_health_full_requires_a_token(self):
        r = self.client.get("/health/full")
        self.assertEqual(r.status_code, 401)

    def test_health_full_reports_all_required_fields(self):
        token = self._pair()
        r = self.client.get("/health/full", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("server_reachable", "phone_client_connected",
                    "voice_available", "phone_mic_supported",
                    "face_available", "active_brain", "vault_available"):
            self.assertIn(key, body)

    def test_phone_mic_supported_reflects_the_actual_request_scheme(self):
        # Real production traffic only ever reaches this app over HTTPS
        # (the tailscale-mode app port -- see phone_bridge.start()).
        # This field is computed from request.url.scheme rather than
        # hardcoded or inferred from CFG so it can never silently claim
        # mic support that isn't real. TestClient never terminates real
        # TLS, so both branches are exercised via its base_url instead
        # of a real socket.
        token = self._pair()
        http_client = TestClient(self.app, base_url="http://testserver")
        r = http_client.get("/health/full", headers=self.auth(token))
        self.assertFalse(r.json()["phone_mic_supported"])
        https_client = TestClient(self.app, base_url="https://testserver")
        r2 = https_client.get("/health/full", headers=self.auth(token))
        self.assertTrue(r2.json()["phone_mic_supported"])


class PairingEndpointTests(PhoneBridgeTestCase):
    def test_redeem_with_bad_code_is_401(self):
        r = self.client.post("/pair/redeem", json={"code": "NOPE0000", "label": "x"})
        self.assertEqual(r.status_code, 401)

    def test_redeem_with_valid_code_returns_a_usable_token(self):
        code, _ = phone_auth.create_pairing()
        r = self.client.post("/pair/redeem", json={"code": code, "label": "My Phone"})
        self.assertEqual(r.status_code, 200)
        token = r.json()["token"]
        r2 = self.client.get("/health/full", headers=self.auth(token))
        self.assertEqual(r2.status_code, 200)

    def test_redeem_tolerates_lowercase_and_surrounding_whitespace(self):
        code, _ = phone_auth.create_pairing()
        r = self.client.post("/pair/redeem",
                             json={"code": f"  {code.lower()}  ", "label": "x"})
        self.assertEqual(r.status_code, 200)

    def test_redeem_tolerates_a_mid_code_space_from_keyboard_autocorrect(self):
        # 2026-09-29 field failure: a mobile keyboard's autocorrect/
        # predictive text inserted a space INSIDE a real pairing code
        # before submit. .strip() alone (the old behavior) never caught
        # that -- only whitespace ANYWHERE in the payload does.
        code, _ = phone_auth.create_pairing()
        mid = len(code) // 2
        spaced = code[:mid] + " " + code[mid:]
        r = self.client.post("/pair/redeem", json={"code": spaced, "label": "x"})
        self.assertEqual(r.status_code, 200)


class MessageEndpointTests(PhoneBridgeTestCase):
    def test_rejected_without_a_token(self):
        r = self.client.post("/message", json={"text": "hello"})
        self.assertEqual(r.status_code, 401)

    def test_rejected_with_a_garbage_token(self):
        r = self.client.post("/message", json={"text": "hello"},
                             headers=self.auth("garbage-token"))
        self.assertEqual(r.status_code, 401)

    def test_paired_device_message_reaches_typed_q(self):
        token = self._pair()
        r = self.client.post("/message", json={"text": "switch to qwen"},
                             headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        # 2026-09-21, active-platform routing: a PhoneTurn wrapper now,
        # not a bare str -- see phone_bridge.PhoneTurn / main.py's
        # typed_fut consumer for why (it's how the desktop-vs-phone
        # origin of a queued utterance is told apart).
        queued = phone_bridge._typed_q.get_nowait()
        self.assertIsInstance(queued, phone_bridge.PhoneTurn)
        self.assertEqual(queued.text, "switch to qwen")

    def test_empty_message_is_rejected(self):
        token = self._pair()
        r = self.client.post("/message", json={"text": "   "},
                             headers=self.auth(token))
        self.assertEqual(r.status_code, 400)

    def test_oversized_message_is_truncated_not_crashed(self):
        token = self._pair()
        huge = "x" * (phone_bridge.MAX_MESSAGE_CHARS + 5000)
        r = self.client.post("/message", json={"text": huge},
                             headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        sent = phone_bridge._typed_q.get_nowait()
        self.assertEqual(len(sent.text), phone_bridge.MAX_MESSAGE_CHARS)

    def test_revoked_device_is_rejected_on_its_very_next_request(self):
        token = self._pair()
        self.client.post("/message", json={"text": "one"}, headers=self.auth(token))
        phone_auth.revoke_device(1)
        r = self.client.post("/message", json={"text": "two"},
                             headers=self.auth(token))
        self.assertEqual(r.status_code, 401)

    def test_message_rate_limit_trips(self):
        token = self._pair()
        phone_bridge._msg_limiter = phone_auth.RateLimiter(max_calls=2, window_s=60)
        codes = []
        for _ in range(4):
            r = self.client.post("/message", json={"text": "hi"},
                                 headers=self.auth(token))
            codes.append(r.status_code)
        self.assertEqual(codes, [200, 200, 429, 429])


class VoiceEndpointTests(PhoneBridgeTestCase):
    def test_rejected_without_a_token(self):
        r = self.client.post("/voice", content=_make_wav(),
                             headers={"Content-Type": "audio/wav"})
        self.assertEqual(r.status_code, 401)

    def test_wrong_sample_rate_is_rejected(self):
        token = self._pair()
        r = self.client.post("/voice", content=_make_wav(rate=44100),
                             headers={**self.auth(token), "Content-Type": "audio/wav"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("16000", r.json()["detail"])

    def test_oversized_clip_is_rejected_by_content_length(self):
        token = self._pair()
        headers = {**self.auth(token), "Content-Type": "audio/wav",
                  "Content-Length": str(phone_bridge.MAX_VOICE_BYTES + 1)}
        r = self.client.post("/voice", content=b"x" * 10, headers=headers)
        self.assertEqual(r.status_code, 413)

    @patch("backtalk.phone_bridge.ears.transcribe", return_value="testing one two three")
    def test_valid_clip_transcribes_and_reaches_typed_q(self, mock_transcribe):
        token = self._pair()
        r = self.client.post("/voice", content=_make_wav(rate=16000),
                             headers={**self.auth(token), "Content-Type": "audio/wav"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "testing one two three")
        queued = phone_bridge._typed_q.get_nowait()
        self.assertIsInstance(queued, phone_bridge.PhoneTurn)
        self.assertEqual(queued.text, "testing one two three")
        mock_transcribe.assert_called_once()

    @patch("backtalk.phone_bridge.ears.transcribe", return_value="")
    def test_empty_transcribe_result_is_not_pushed_to_typed_q(self, mock_transcribe):
        # Distinct layer from the RMS gate below: a REAL (non-silent)
        # clip that Whisper itself decides has nothing worth saying.
        token = self._pair()
        r = self.client.post("/voice", content=_make_wav(rate=16000),
                             headers={**self.auth(token), "Content-Type": "audio/wav"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "")
        self.assertTrue(phone_bridge._typed_q.empty())
        mock_transcribe.assert_called_once()

    @patch("backtalk.phone_bridge.ears.transcribe", return_value="You")
    def test_near_silent_clip_never_reaches_transcribe(self, mock_transcribe):
        # 2026-09-21 field defect: near-silent audio reaching Whisper
        # hallucinates filler text like "You" -- fixed upstream of
        # transcription (see VOICE_SILENCE_RMS), not by filtering the
        # word "You" out of real output (which would risk suppressing
        # someone who genuinely said it). ears.transcribe is mocked to
        # return exactly that hallucination so this test would fail
        # loudly if the gate ever stopped short-circuiting before it.
        token = self._pair()
        r = self.client.post("/voice", content=_make_wav(amplitude=0),
                             headers={**self.auth(token), "Content-Type": "audio/wav"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "")
        self.assertTrue(phone_bridge._typed_q.empty())
        mock_transcribe.assert_not_called()

    @patch("backtalk.phone_bridge.ears.transcribe", return_value="hello there")
    def test_duplicate_capture_session_is_ignored_not_resubmitted(self, mock_transcribe):
        # 2026-09-21: "ensure exactly one capture session and exactly
        # one /voice submission per press/release" -- a second /voice
        # call carrying the SAME X-Capture-Session for the SAME device
        # (a synthetic duplicate touch/mouse event, a retried fetch)
        # must be acknowledged without transcribing or queuing it a
        # second time.
        token = self._pair()
        headers = {**self.auth(token), "Content-Type": "audio/wav",
                  "X-Capture-Session": "session-abc-123"}
        first = self.client.post("/voice", content=_make_wav(), headers=headers)
        second = self.client.post("/voice", content=_make_wav(), headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["text"], "hello there")
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.json().get("duplicate"))
        self.assertEqual(second.json()["text"], "")
        mock_transcribe.assert_called_once()
        self.assertEqual(phone_bridge._typed_q.get_nowait().text, "hello there")
        self.assertTrue(phone_bridge._typed_q.empty())

    @patch("backtalk.phone_bridge.ears.transcribe", return_value="hello there")
    def test_different_capture_sessions_both_submit(self, mock_transcribe):
        # The dedup key is per (device, session id) -- a genuinely NEW
        # press (a fresh UUID) from the same device must never be
        # mistaken for a duplicate of the previous one.
        token = self._pair()
        headers1 = {**self.auth(token), "Content-Type": "audio/wav",
                   "X-Capture-Session": "session-1"}
        headers2 = {**self.auth(token), "Content-Type": "audio/wav",
                   "X-Capture-Session": "session-2"}
        r1 = self.client.post("/voice", content=_make_wav(), headers=headers1)
        r2 = self.client.post("/voice", content=_make_wav(), headers=headers2)
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertFalse(r2.json().get("duplicate"))
        self.assertEqual(mock_transcribe.call_count, 2)

    @patch("backtalk.phone_bridge.ears.transcribe", return_value="no session header")
    def test_missing_session_header_never_fails_closed(self, mock_transcribe):
        # An older/other client that never sends X-Capture-Session at
        # all must keep working exactly as before -- the dedup check
        # only ever engages when the header is actually present.
        token = self._pair()
        headers = {**self.auth(token), "Content-Type": "audio/wav"}
        first = self.client.post("/voice", content=_make_wav(), headers=headers)
        second = self.client.post("/voice", content=_make_wav(), headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(second.json().get("duplicate"))
        self.assertEqual(mock_transcribe.call_count, 2)

    def test_voice_rate_limit_trips(self):
        token = self._pair()
        phone_bridge._voice_limiter = phone_auth.RateLimiter(max_calls=1, window_s=60)
        with patch("backtalk.phone_bridge.ears.transcribe", return_value="x"):
            first = self.client.post("/voice", content=_make_wav(),
                                     headers={**self.auth(token), "Content-Type": "audio/wav"})
            second = self.client.post("/voice", content=_make_wav(),
                                      headers={**self.auth(token), "Content-Type": "audio/wav"})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)


class InterruptEndpointTests(PhoneBridgeTestCase):
    """2026-09-21: "the instant I press Hold to talk, Jarvis must
    immediately stop any current speech... then listen to and answer
    only my newest query." /interrupt is the desktop half of that (the
    phone's own playback stop is pure client-side JS, tested
    structurally in VoiceCaptureHardeningTests-style checks below)."""

    def test_rejected_without_a_token(self):
        r = self.client.post("/interrupt")
        self.assertEqual(r.status_code, 401)

    def test_ok_with_no_handler_registered_is_a_safe_no_op(self):
        token = self._pair()
        phone_bridge._interrupt_handler = None
        r = self.client.post("/interrupt", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)

    def test_registered_handler_is_awaited(self):
        token = self._pair()
        calls = []

        async def handler():
            calls.append(1)

        phone_bridge.set_interrupt_handler(handler)
        r = self.client.post("/interrupt", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(calls, [1])

    def test_a_raising_handler_never_500s_the_phone(self):
        token = self._pair()

        async def bad_handler():
            raise RuntimeError("boom")

        phone_bridge.set_interrupt_handler(bad_handler)
        r = self.client.post("/interrupt", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)

    def test_clears_not_yet_finalized_jarvis_fragments(self):
        # 2026-09-21: "ensure late audio/text events from the
        # interrupted turn cannot resume or overwrite the new turn."
        # A chunk of the OLD reply that was mid-play when interrupted
        # already called record_transcript_line (mouth.py's transcript
        # sink fires at dequeue time, BEFORE playback finishes) -- if
        # this buffer weren't cleared, the NEXT turn's finalize would
        # silently prepend that stale fragment onto the new reply.
        token = self._pair()
        phone_bridge.record_transcript_line("jarvis", "the old, interrupted")
        self.assertTrue(phone_bridge._pending_jarvis_parts)
        self.client.post("/interrupt", headers=self.auth(token))
        self.assertEqual(phone_bridge._pending_jarvis_parts, [])
        # And the NEW turn's own text arrives clean, with no bleed-over.
        phone_bridge.record_transcript_line("jarvis", "the new reply")
        phone_bridge.finalize_jarvis_turn()
        self.assertEqual(phone_bridge._transcript[-1]["text"], "the new reply")


class SpeakChunkEndpointTests(PhoneBridgeTestCase):
    """2026-09-21: replies split into per-sentence WAV chunks (start
    faster) with gain normalization applied to each (too quiet)."""

    def _finalize_turn(self, text: str) -> int:
        phone_bridge.record_transcript_line("jarvis", text)
        phone_bridge.finalize_jarvis_turn()
        return phone_bridge._transcript[-1]["id"]

    def test_rejected_without_a_token(self):
        turn_id = self._finalize_turn("Hello there.")
        r = self.client.get(f"/speak/{turn_id}/0.wav")
        self.assertEqual(r.status_code, 401)

    def test_unknown_turn_is_404(self):
        token = self._pair()
        r = self.client.get("/speak/999999/0.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 404)

    def test_chunk_past_the_last_sentence_is_404_not_an_error(self):
        token = self._pair()
        turn_id = self._finalize_turn("Only one sentence here.")
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, np.zeros(100, dtype=np.int16))])):
            first = self.client.get(f"/speak/{turn_id}/0.wav",
                                    headers=self.auth(token))
            second = self.client.get(f"/speak/{turn_id}/1.wav",
                                     headers=self.auth(token))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 404)

    def test_multi_sentence_reply_exposes_one_chunk_per_sentence(self):
        token = self._pair()
        turn_id = self._finalize_turn("First sentence. Second sentence. Third one.")
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, np.zeros(100, dtype=np.int16))])):
            codes = [self.client.get(f"/speak/{turn_id}/{i}.wav",
                                     headers=self.auth(token)).status_code
                    for i in range(4)]
        self.assertEqual(codes, [200, 200, 200, 404])

    def test_quiet_audio_is_lifted_toward_the_target_loudness(self):
        # 2026-09-21 field defect, diagnosed: "Jarvis reply volume on
        # the phone is too low" traced back to peak-only normalization
        # not matching perceived loudness -- see NormalizeGainUnitTests
        # for the full diagnosis. A quiet, constant-value chunk (peak
        # == rms) must come back louder, lifted toward the RMS
        # (loudness) target; the response bytes are decoded and
        # checked for real, not just "the code path ran."
        token = self._pair()
        turn_id = self._finalize_turn("Quiet reply.")
        quiet = np.full(200, 1000, dtype=np.int16)   # peak 1000 of 32767
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, quiet)])):
            r = self.client.get(f"/speak/{turn_id}/0.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        data, rate = sf.read(io.BytesIO(r.content), dtype="int16")
        peak = int(np.max(np.abs(data)))
        expected_peak = round(phone_bridge.PHONE_TARGET_RMS * 32767)
        self.assertGreater(peak, 1000)
        self.assertAlmostEqual(peak, expected_peak, delta=2)

    def test_already_loud_audio_is_never_pushed_past_full_scale(self):
        # The other half of "clearly audible without clipping" -- a
        # chunk already near full scale must never be amplified
        # further (that would clip), only ever a quiet one lifted up.
        token = self._pair()
        turn_id = self._finalize_turn("Loud reply.")
        loud = np.full(200, 32000, dtype=np.int16)
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, loud)])):
            r = self.client.get(f"/speak/{turn_id}/0.wav", headers=self.auth(token))
        data, rate = sf.read(io.BytesIO(r.content), dtype="int16")
        peak = int(np.max(np.abs(data)))
        self.assertLessEqual(peak, 32767)
        self.assertEqual(peak, 32000)   # left alone, not turned down either

    def test_true_silence_is_returned_unchanged_no_divide_by_zero(self):
        token = self._pair()
        turn_id = self._finalize_turn("Silent reply.")
        silent = np.zeros(200, dtype=np.int16)
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, silent)])):
            r = self.client.get(f"/speak/{turn_id}/0.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        data, rate = sf.read(io.BytesIO(r.content), dtype="int16")
        self.assertTrue(np.all(data == 0))

    def _finalize_turn_targeted(self, text: str, targets) -> int:
        phone_bridge.record_transcript_line("jarvis", text, targets=targets)
        phone_bridge.finalize_jarvis_turn()
        return phone_bridge._transcript[-1]["id"]

    def test_desktop_only_turn_is_unfetchable_from_the_phone(self):
        # 2026-09-21, active-platform routing, rule 3: "a desktop/F8
        # turn speaks only on desktop." A turn routed desktop-only must
        # be genuinely unfetchable, not merely hidden from autoplay --
        # a 404, matching "unknown turn", so its existence can't be
        # told apart from one that never happened.
        token = self._pair()
        turn_id = self._finalize_turn_targeted(
            "Desktop only.", frozenset({"desktop"}))
        r = self.client.get(f"/speak/{turn_id}/0.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 404)

    def test_phone_only_turn_is_fetchable(self):
        token = self._pair()
        turn_id = self._finalize_turn_targeted(
            "Phone only.", frozenset({"phone"}))
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, np.zeros(100, dtype=np.int16))])):
            r = self.client.get(f"/speak/{turn_id}/0.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)

    def test_both_targeted_turn_is_fetchable(self):
        token = self._pair()
        turn_id = self._finalize_turn_targeted(
            "Both.", frozenset({"desktop", "phone"}))
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, np.zeros(100, dtype=np.int16))])):
            r = self.client.get(f"/speak/{turn_id}/0.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)

    def test_entry_with_no_targets_at_all_defaults_to_fetchable(self):
        # Predates the routing feature, or came from a call site that
        # never routes (say()) -- must default to the whole prior
        # behavior (both), never silently start refusing old entries.
        token = self._pair()
        turn_id = self._finalize_turn("No targets recorded.")
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, np.zeros(100, dtype=np.int16))])):
            r = self.client.get(f"/speak/{turn_id}/0.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)


class SpeakTestEndpointTests(PhoneBridgeTestCase):
    """2026-09-21: "add a short Test phone audio control and test
    tone/phrase for calibration." """

    def test_rejected_without_a_token(self):
        r = self.client.get("/speak/test.wav")
        self.assertEqual(r.status_code, 401)

    def test_returns_a_playable_wav_through_the_same_gain_pipeline(self):
        token = self._pair()
        quiet = np.full(200, 1000, dtype=np.int16)
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, quiet)])):
            r = self.client.get("/speak/test.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        data, rate = sf.read(io.BytesIO(r.content), dtype="int16")
        peak = int(np.max(np.abs(data)))
        # Same _normalize_gain pipeline as a real reply -- a quiet
        # input must come back measurably louder here too, not a raw,
        # un-gained tone that wouldn't actually calibrate anything.
        self.assertGreater(peak, 1000)

    def test_not_tied_to_any_real_transcript_entry(self):
        token = self._pair()
        self.assertEqual(len(phone_bridge._transcript), 0)
        with patch("backtalk.phone_bridge.mouth.synth_stream",
                   return_value=iter([(16000, np.zeros(100, dtype=np.int16))])):
            r = self.client.get("/speak/test.wav", headers=self.auth(token))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(phone_bridge._transcript), 0)


class NormalizeGainUnitTests(unittest.TestCase):
    """Pure-function coverage for _normalize_gain, independent of the
    HTTP layer above. 2026-09-21 diagnosis: the ORIGINAL implementation
    normalized to a target PEAK alone -- which is not the same thing
    as perceived loudness (this is why broadcast/streaming moved to
    LUFS/RMS-based loudness normalization years ago instead of peak
    normalization). A clip whose loudest single sample is a brief
    transient (a consonant burst) can hit its target peak while the
    REST of it -- almost the entire perceived loudness -- stays quiet.
    Fixed by normalizing to a target RMS (average level) instead, with
    the peak fraction now used only as a safety CEILING the resulting
    gain may never cross (the limiter), not the loudness goal itself:
    gain = min(loudness_gain, clip_safe_gain)."""

    def test_lifts_a_quiet_constant_chunk_toward_the_rms_target(self):
        # A perfectly constant-value signal has rms == peak, so
        # whichever target is the smaller FRACTION binds -- here
        # that's the RMS target (0.20 < 0.85).
        quiet = np.full(50, 500, dtype=np.int16)
        out = phone_bridge._normalize_gain(quiet, target_rms=0.20,
                                           target_peak=0.85)
        self.assertAlmostEqual(int(np.max(np.abs(out))),
                               round(0.20 * 32767), delta=2)

    def test_moderate_crest_factor_is_bound_by_the_loudness_target(self):
        # A sustained "speech body" at 3000 plus a moderate transient
        # at 9000 (crest factor ~2.5, well under 0.85/0.20 = 4.25) --
        # the gain that would hit the RMS target stays clip-safe, so
        # the RMS TARGET is what actually determines the gain, not the
        # peak ceiling. This is the case the old peak-only normalizer
        # got wrong: it would have gained toward the transient alone.
        pcm = np.full(1000, 3000, dtype=np.int16)
        pcm[:50] = 9000
        rms_in = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
        out = phone_bridge._normalize_gain(pcm, target_rms=0.20,
                                           target_peak=0.85)
        out_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        self.assertAlmostEqual(out_rms, 0.20 * 32767, delta=5)
        self.assertLess(int(np.max(np.abs(out))), round(0.85 * 32767))
        self.assertGreater(out_rms, rms_in)

    def test_high_crest_factor_is_bound_by_the_peak_limiter_not_clipped(self):
        # A quiet sustained body (300) with ONE sharp transient
        # (20000, crest factor ~28.6) -- lifting the body all the way
        # to the RMS target would push that transient past the peak
        # ceiling, so the LIMITER binds instead: the transient lands
        # at (not past) the peak target, and the body is lifted as far
        # as that same gain allows, never further.
        pcm = np.full(1000, 300, dtype=np.int16)
        pcm[500] = 20000
        out = phone_bridge._normalize_gain(pcm, target_rms=0.20,
                                           target_peak=0.85)
        self.assertLessEqual(int(np.max(np.abs(out))), round(0.85 * 32767) + 1)
        self.assertAlmostEqual(int(np.max(np.abs(out))),
                               round(0.85 * 32767), delta=2)

    def test_never_amplifies_an_already_loud_chunk(self):
        loud = np.full(50, 30000, dtype=np.int16)
        out = phone_bridge._normalize_gain(loud, target_rms=0.20,
                                           target_peak=0.85)
        self.assertTrue(np.array_equal(out, loud))

    def test_silence_is_a_safe_no_op(self):
        silent = np.zeros(50, dtype=np.int16)
        out = phone_bridge._normalize_gain(silent)
        self.assertTrue(np.all(out == 0))

    def test_result_never_exceeds_full_scale(self):
        edge = np.full(50, 32767, dtype=np.int16)
        out = phone_bridge._normalize_gain(edge, target_rms=0.99,
                                           target_peak=0.99)
        self.assertLessEqual(int(np.max(np.abs(out))), 32767)


class PhoneInterruptWiringTests(unittest.TestCase):
    """main.py's _phone_interrupt closure is exercised end-to-end by
    InterruptEndpointTests above via a fake handler -- this checks the
    REAL one actually wired in amain() uses the same cancel/shut_up/
    await sequence as the established interrupt-on-new-input path in
    handle(), and that it's registered with phone_bridge exactly once,
    the same source-scanning technique test_startup_greeting.py
    already uses for a similar single-call-site guarantee."""

    def test_phone_interrupt_defined_and_registered_exactly_once(self):
        import inspect

        from backtalk import main
        source = inspect.getsource(main)
        self.assertEqual(source.count("async def _phone_interrupt"), 1)
        self.assertEqual(
            source.count("phone_bridge.set_interrupt_handler(_phone_interrupt)"), 1)

    def test_phone_interrupt_cancels_and_silences_before_returning(self):
        import inspect

        from backtalk import main
        source = inspect.getsource(main)
        start = source.index("async def _phone_interrupt")
        end = source.index("\n    _phone_cfg = CFG.get", start)
        body = source[start:end]
        self.assertIn("speak_task.cancel()", body)
        self.assertIn("mouth.shut_up()", body)
        self.assertIn("await speak_task", body)
        self.assertIn("_deny_pending()", body)


class SplitSentencesUnitTests(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self):
        self.assertEqual(
            phone_bridge._split_sentences("First one. Second one! Third?"),
            ["First one.", "Second one!", "Third?"])

    def test_no_terminal_punctuation_is_one_whole_chunk(self):
        self.assertEqual(phone_bridge._split_sentences("just some words"),
                         ["just some words"])

    def test_empty_text_is_no_chunks(self):
        self.assertEqual(phone_bridge._split_sentences("   "), [])


class EventsEndpointTests(PhoneBridgeTestCase):
    def test_rejected_without_a_token(self):
        r = self.client.get("/events")
        self.assertEqual(r.status_code, 401)


class JarvisTurnBufferingTests(unittest.TestCase):
    """Regression coverage for the 2026-09-29 field defect: a phone
    only ever saw Jarvis's reply as several separate fragments (one per
    mouth.say_chunk() call), never one complete line, and had no way
    to know when a reply was actually finished. record_transcript_line
    buffers "jarvis" fragments; finalize_jarvis_turn (wired to mouth.
    py's turn-complete hook) publishes them as ONE atomic entry."""

    def setUp(self):
        phone_bridge._transcript.clear()
        phone_bridge._transcript_next_id = 0
        phone_bridge._pending_jarvis_parts.clear()

    def test_jarvis_fragments_do_not_appear_until_finalized(self):
        phone_bridge.record_transcript_line("jarvis", "Hello there.")
        phone_bridge.record_transcript_line("jarvis", "How can I help?")
        self.assertEqual(len(phone_bridge._transcript), 0,
                         "fragments must not be visible before the turn completes")

    def test_finalize_publishes_exactly_one_complete_entry(self):
        phone_bridge.record_transcript_line("jarvis", "First sentence.")
        phone_bridge.record_transcript_line("jarvis", "Second and third.")
        phone_bridge.record_transcript_line("jarvis", "Fourth, final.")
        phone_bridge.finalize_jarvis_turn()
        self.assertEqual(len(phone_bridge._transcript), 1)
        entry = phone_bridge._transcript[-1]
        self.assertEqual(entry["who"], "jarvis")
        self.assertEqual(entry["text"],
                         "First sentence. Second and third. Fourth, final.")

    def test_no_words_are_dropped_across_many_fragments(self):
        words = [f"word{i}" for i in range(20)]
        for w in words:
            phone_bridge.record_transcript_line("jarvis", w)
        phone_bridge.finalize_jarvis_turn()
        published = phone_bridge._transcript[-1]["text"]
        for w in words:
            self.assertIn(w, published)

    def test_finalize_with_nothing_buffered_is_a_safe_no_op(self):
        phone_bridge.finalize_jarvis_turn()
        self.assertEqual(len(phone_bridge._transcript), 0)

    def test_captain_lines_still_append_immediately_not_buffered(self):
        # buffering is a "jarvis"-only concept -- the phone's own typed/
        # spoken messages must keep appearing right away, unaffected.
        phone_bridge.record_transcript_line("captain (phone: Test)", "hi")
        self.assertEqual(len(phone_bridge._transcript), 1)

    def test_second_turn_after_finalize_starts_a_fresh_buffer(self):
        phone_bridge.record_transcript_line("jarvis", "First reply.")
        phone_bridge.finalize_jarvis_turn()
        phone_bridge.record_transcript_line("jarvis", "Second reply.")
        self.assertEqual(len(phone_bridge._transcript), 1,
                         "second turn's fragment must not be visible yet")
        phone_bridge.finalize_jarvis_turn()
        self.assertEqual(len(phone_bridge._transcript), 2)
        self.assertEqual(phone_bridge._transcript[-1]["text"], "Second reply.")

    def test_finalized_entry_carries_the_turns_targets(self):
        # 2026-09-21, active-platform routing: every chunk of one turn
        # carries the same targets by construction (speak_reply()
        # computes it once per turn) -- finalize publishes it on the
        # entry so /speak/{turn}/{chunk}.wav can gate on it.
        phone_bridge.record_transcript_line(
            "jarvis", "First.", targets=frozenset({"phone"}))
        phone_bridge.record_transcript_line(
            "jarvis", "Second.", targets=frozenset({"phone"}))
        phone_bridge.finalize_jarvis_turn()
        self.assertEqual(phone_bridge._transcript[-1]["targets"], ["phone"])

    def test_jarvis_entry_with_no_targets_ever_passed_defaults_to_both(self):
        # A jarvis chunk from a call site that never routes (e.g.
        # mouth.say(), which always uses mouth._BOTH_TARGETS) still
        # finalizes with an explicit "both" -- the same default this
        # whole feature's prior behavior was.
        phone_bridge.record_transcript_line("jarvis", "No targets given.")
        phone_bridge.finalize_jarvis_turn()
        self.assertEqual(phone_bridge._transcript[-1]["targets"],
                         ["desktop", "phone"])

    def test_non_jarvis_entry_never_carries_a_targets_field(self):
        # Captain's own phone messages go straight to
        # _append_transcript_entry with no targets -- routing is a
        # concept for JARVIS'S replies, not for what Captain said.
        phone_bridge.record_transcript_line("captain (phone: Test)", "hi")
        self.assertNotIn("targets", phone_bridge._transcript[-1])

    # NOT tested here: actually opening /events and reading a live SSE
    # tick through TestClient. That endpoint's generator is a genuine
    # `while True` loop -- Starlette's TestClient runs the ASGI app
    # through a synchronous portal, and driving a truly-infinite stream
    # through it deadlocked in practice (confirmed while writing this
    # file: the suite hung indefinitely on exactly that call). /events'
    # per-tick payload is built directly from _transcript (see
    # phone_bridge.py), which every test above already exercises
    # directly and safely -- that's the real coverage for "does the
    # phone only see the finalized line", without needing to drive the
    # live stream itself. EventsEndpointTests above sticks to the one
    # SSE case that's actually safe to test this way: the immediate
    # 401 on an unauthenticated request, which returns before any
    # streaming ever starts.


class MouthTurnCompleteHookTests(unittest.TestCase):
    """A minimal sanity check on the mouth.py side of this same fix --
    mouth.py has no dedicated real-audio test suite (it needs actual
    playback hardware), so this only proves the hook itself exists and
    is wired correctly, not the full worker thread."""

    def test_set_turn_complete_sink_registers_the_callback(self):
        calls = []
        mouth.set_turn_complete_sink(lambda: calls.append(True))
        try:
            self.assertIsNotNone(mouth._turn_complete_sink)
            mouth._turn_complete_sink()
            self.assertEqual(calls, [True])
        finally:
            mouth.set_turn_complete_sink(None)

    def test_default_is_none_a_safe_no_op(self):
        mouth.set_turn_complete_sink(None)
        self.assertIsNone(mouth._turn_complete_sink)


class PairingFormHardeningTests(PhoneBridgeTestCase):
    """The main app's index page (the pairing form itself) -- regression
    coverage for the 2026-09-29 field failure: a mobile keyboard's
    autocorrect/predictive text altered a real pairing code before
    submit. These assert the HTML actually carries the attributes that
    ask the keyboard not to do that, and the maxlength that matches the
    server's own code length."""

    def test_pair_code_field_disables_autocorrect_and_spellcheck(self):
        r = self.client.get("/")
        text = r.text
        self.assertIn('id="pairCode"', text)
        self.assertIn('autocorrect="off"', text)
        self.assertIn('spellcheck="false"', text)

    def test_pair_code_field_maxlength_matches_the_real_code_length(self):
        r = self.client.get("/")
        self.assertIn(f'maxlength="{phone_auth._CODE_LEN}"', r.text)

    def test_page_normalizes_and_clears_the_code_client_side(self):
        # regression guard: the actual fix functions must still be
        # present in the served page, not just documented in a comment
        r = self.client.get("/")
        text = r.text
        self.assertIn("function normalizeCodeField", text)
        self.assertIn('el.value.toUpperCase().replace(CODE_CHARS, "")', text)
        self.assertIn('document.getElementById("pairCode").value = "";', text)


class VoiceCaptureHardeningTests(PhoneBridgeTestCase):
    """Regression coverage for the 2026-09-29 clipped/mangled-
    transcription field defect. These assert the actual fix code is
    present in the served page -- a real device test is what proves it
    WORKS, but this proves the fix can't silently regress back to the
    forced-rate/no-drain/no-resample version."""

    def test_audiocontext_no_longer_forces_a_sample_rate(self):
        text = self.client.get("/").text
        # the OLD faulty constructor call, verbatim -- not a loose
        # substring check, since the fix's OWN explanatory comment
        # legitimately mentions "sampleRate: 16000" when describing the
        # bug it replaced
        self.assertNotIn(
            "new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000})",
            text)
        self.assertIn(
            "audioCtx = new (window.AudioContext || window.webkitAudioContext)();",
            text)

    def test_own_antialiased_resample_function_is_present(self):
        text = self.client.get("/").text
        self.assertIn("function resampleTo16k", text)
        self.assertIn("const cutoff = 7500", text)
        # a single pass measured ~9dB attenuation at 20kHz in testing --
        # not enough; the filter must be cascaded (see the numeric
        # verification this comment references in the source itself)
        self.assertIn("const passes = 3", text)
        self.assertIn("function _lowpassPass", text)

    def test_stop_recording_drains_before_tearing_down(self):
        text = self.client.get("/").text
        self.assertIn("await new Promise(r => setTimeout(r, 80));", text)

    def test_wav_is_always_encoded_at_the_resampled_16k_rate(self):
        text = self.client.get("/").text
        self.assertIn("encodeWav(floatTo16(resampled), 16000)", text)

    def test_debug_mode_is_off_by_default_and_gated_by_query_param(self):
        text = self.client.get("/").text
        self.assertIn(
            'new URLSearchParams(location.search).get("debug") === "1"', text)
        self.assertIn('id="voiceDebug" class="hint mono" style="display:none;',
                      text)

    def test_autoplay_queue_is_sequential_not_concurrent(self):
        text = self.client.get("/").text
        self.assertIn("function pumpAutoplayQueue", text)
        self.assertIn("if (autoplayBusy || autoplayQueue.length === 0) return;", text)
        # 2026-09-21: reply playback moved to per-SENTENCE chunks (see
        # playChunksSequentially, /speak/{turn_id}/{chunk}.wav) for
        # faster start -- pumpAutoplayQueue now delegates the actual
        # onended/onerror wiring to that function rather than setting
        # them directly, so the assertion follows the delegation.
        self.assertIn("playChunksSequentially(id, advance);", text)
        self.assertIn("player.onended = playNext;", text)


class PhoneExperienceMilestoneClientTests(PhoneBridgeTestCase):
    """2026-09-21 phone-experience milestone: structural regression
    coverage for the client-side half of each fix -- a real device
    test is what proves it WORKS, but these prove the fix can't
    silently regress out of the served page."""

    def test_pointer_events_replace_touch_and_mouse_listeners(self):
        # "sometimes duplicates the same phrase" -- caused by mobile
        # browsers firing a SYNTHETIC mousedown/mouseup after a real
        # touch sequence when both listener sets are present. The fix
        # is Pointer Events ONLY -- both old listener families must be
        # gone, not just supplemented.
        text = self.client.get("/").text
        self.assertIn('micBtn.addEventListener("pointerdown"', text)
        self.assertIn('micBtn.addEventListener("pointerup"', text)
        self.assertIn('micBtn.addEventListener("pointercancel"', text)
        self.assertNotIn('micBtn.addEventListener("touchstart"', text)
        self.assertNotIn('micBtn.addEventListener("touchend"', text)
        self.assertNotIn('micBtn.addEventListener("mousedown"', text)
        self.assertNotIn('micBtn.addEventListener("mouseup"', text)

    def test_recording_guard_rejects_a_second_start_while_already_active(self):
        text = self.client.get("/").text
        self.assertIn("if (recording) return;", text)

    def test_capture_session_id_is_generated_and_sent(self):
        text = self.client.get("/").text
        self.assertIn("crypto.randomUUID()", text)
        self.assertIn('"X-Capture-Session": session', text)

    def test_pre_roll_buffer_exists_and_is_spliced_into_the_capture(self):
        # "phone speech misses the beginning of sentences" -- when warm
        # mode is enabled (see PhoneMicPrivacyControlTests for the
        # opt-in gate itself) the mic stream stays open and a short
        # rolling buffer of already-captured audio is prepended to
        # every new recording.
        text = self.client.get("/").text
        self.assertIn("async function _openMicStream", text)
        self.assertIn("function pushPreRoll", text)
        self.assertIn("const PRE_ROLL_MS = 600;", text)
        self.assertIn("liveSamples = preRollChunks.slice();", text)

    def test_client_side_silence_gate_matches_the_server_threshold(self):
        text = self.client.get("/").text
        self.assertIn(f"const CLIENT_SILENCE_RMS = {phone_bridge.VOICE_SILENCE_RMS};",
                      text)
        self.assertIn("if (rms16 < CLIENT_SILENCE_RMS)", text)

    def test_press_sends_interrupt_and_stops_playback_before_capturing(self):
        # "the instant I press Hold to talk, Jarvis must immediately
        # stop any current speech on both phone and desktop" -- both
        # calls must happen in startRecording, before recording=true.
        text = self.client.get("/").text
        start = text.index("async function startRecording")
        end = text.index("async function stopRecording")
        body = text[start:end]
        self.assertIn("stopAllPlayback();", body)
        self.assertIn("interruptPromise = sendInterrupt();", body)
        self.assertIn("recording = true;", body)
        self.assertLess(body.index("stopAllPlayback();"),
                        body.index("recording = true;"))

    def test_release_waits_for_interrupt_ack_before_the_new_request(self):
        text = self.client.get("/").text
        self.assertIn("if (interruptPromise) await interruptPromise;", text)

    def test_interrupt_endpoint_exists_and_is_posted_to(self):
        text = self.client.get("/").text
        self.assertIn('api("/interrupt", {method: "POST"})', text)

    def test_volume_control_is_present_with_a_real_boost_default(self):
        # 2026-09-21 field defect: "add a visible phone volume control
        # and set a sensible default mobile playback gain... allow
        # controlled boost above normal browser audio volume." The
        # slider now defaults ABOVE 1.0 (a real boost via the WebAudio
        # gain graph, not just "full" native volume, which is capped
        # at 1.0 by spec and can't compensate for a quiet source).
        text = self.client.get("/").text
        self.assertIn('id="volCtl" type="range" min="0" max="2.5"', text)
        self.assertIn("const DEFAULT_VOLUME_GAIN = 1.5;", text)
        self.assertIn("gainNode.gain.value = ", text)

    def test_playback_speed_control_defaults_to_the_documented_rate(self):
        text = self.client.get("/").text
        self.assertIn('id="speedCtl" type="range"', text)
        self.assertIn(f"const DEFAULT_PLAYBACK_RATE = {phone_bridge.PHONE_DEFAULT_PLAYBACK_RATE};",
                      text)
        self.assertIn("player.playbackRate = ", text)

    def test_debug_mode_shows_the_final_raw_transcript(self):
        text = self.client.get("/").text
        self.assertIn('Whisper heard (raw transcript): "${j.text}"', text)

    def test_debug_text_is_never_written_to_localstorage(self):
        # "Keep debug transcript information test-only and
        # non-persistent." debugLog only ever touches the DOM
        # (overwritten, not accumulated); it must never call
        # localStorage.setItem with anything debug-related.
        text = self.client.get("/").text
        start = text.index("function debugLog")
        end = text.index("\n}", start)
        body = text[start:end]
        self.assertNotIn("localStorage", body)


class PhoneMicPrivacyControlTests(PhoneBridgeTestCase):
    """2026-09-21 privacy correction, same day as the phone-experience
    milestone above: the warm mic / pre-roll feature must never turn
    on by itself. These are structural regression checks on the served
    page -- a real device test is what proves the UX actually works,
    but these prove the opt-in gate can't silently regress away."""

    def test_warm_mic_state_defaults_to_off_and_is_never_read_from_storage(self):
        text = self.client.get("/").text
        self.assertIn("let warmMicEnabled = false;", text)
        # The one thing that would defeat "explicit tap required every
        # session": reading a previous choice back out of storage.
        self.assertNotIn('localStorage.getItem("jarvis_phone_mic', text)
        self.assertNotIn("localStorage.getItem(WARM_MIC_KEY", text)

    def test_enable_and_disable_buttons_exist_with_correct_default_visibility(self):
        text = self.client.get("/").text
        self.assertIn('id="micEnableBtn"', text)
        self.assertIn('id="micDisableBtn"', text)
        self.assertIn('id="micDisableBtn" style="display:none;"', text)
        self.assertIn('micEnableBtn").addEventListener("click", enableWarmMic)', text)
        self.assertIn('micDisableBtn").addEventListener("click", disableWarmMic)', text)

    def test_status_banner_has_the_exact_required_wording(self):
        # MIC_READY_TEXT wraps this as two concatenated JS string
        # literals in the served source -- assert the exact
        # concatenation, not just each half in isolation.
        text = self.client.get("/").text
        self.assertIn(
            '"MIC READY — last 600 ms held locally; "\n'
            '  + "nothing uploads until Hold to talk.";',
            text)

    def test_showapp_never_auto_enables_warm_mic(self):
        # "Do not keep the phone microphone warm automatically on page
        # load." showApp() runs on every load/pairing success; it must
        # call the STATUS-ONLY refresh, never the function that opens
        # a mic stream.
        text = self.client.get("/").text
        start = text.index("function showApp()")
        end = text.index("\n}", start)
        body = text[start:end]
        self.assertIn("updateMicStatusUI();", body)
        self.assertNotIn("enableWarmMic", body)
        self.assertNotIn("_openMicStream", body)

    def test_start_recording_never_auto_enables_warm_mode(self):
        # A press of Hold to talk must never itself flip warmMicEnabled
        # on -- only the explicit Enable button does that. Confirmed by
        # scanning startRecording's own body for the one call that
        # would set it, plus the direct assertion that the cold path
        # opens a stream WITHOUT setting warmMicEnabled = true.
        text = self.client.get("/").text
        start = text.index("async function startRecording")
        end = text.index("async function stopRecording")
        body = text[start:end]
        self.assertNotIn("warmMicEnabled = true", body)

    def test_disable_clears_preroll_and_stops_the_track(self):
        text = self.client.get("/").text
        start = text.index("async function _closeMicStream")
        end = text.index("\n}", start)
        body = text[start:end]
        self.assertIn("getTracks().forEach(t => t.stop())", body)
        self.assertIn("preRollChunks = [];", body)
        self.assertIn("preRollSampleCount = 0;", body)

    def test_disable_wired_to_visibilitychange_and_pagehide(self):
        text = self.client.get("/").text
        self.assertIn('document.addEventListener("visibilitychange"', text)
        self.assertIn('document.visibilityState === "hidden"', text)
        self.assertIn('window.addEventListener("pagehide"', text)

    def test_disable_wired_to_sse_disconnect_and_pairing_revocation(self):
        text = self.client.get("/").text
        # es.onerror (phone bridge / connection disconnect)
        start = text.index("es.onerror = () => {")
        end = text.index("};", start)
        self.assertIn("disableWarmMic();", text[start:end])
        # showPair() (pairing revocation and every other 401 bounce-back)
        start = text.index("function showPair()")
        end = text.index("\n}", start)
        self.assertIn("disableWarmMic();", text[start:end])
        # /voice's own 401 handling now routes through showPair() too
        self.assertIn("if (r.status === 401) {", text)

    def test_stop_recording_guards_against_stream_torn_down_mid_press(self):
        # A privacy trigger (page hidden/locked/disconnected) can close
        # the mic stream WHILE a press is still active -- stopRecording
        # must not blow up touching a null audioCtx/micStream when the
        # finger finally lifts.
        text = self.client.get("/").text
        start = text.index("async function stopRecording")
        end = text.index("const micBtn = document.getElementById", start)
        body = text[start:end]
        self.assertIn("if (!audioCtx || !micStream) {", body)

    def test_pre_roll_only_buffers_while_warm_mode_is_enabled(self):
        # onaudioprocess must gate pre-roll buffering on warmMicEnabled,
        # not merely on "a stream happens to be open" (the cold,
        # opt-out path also has an open stream mid-press).
        text = self.client.get("/").text
        start = text.index("processor.onaudioprocess = (e) => {")
        end = text.index("};", start)
        body = text[start:end]
        self.assertIn("} else if (warmMicEnabled) {", body)


class PhoneAudioBoostControlTests(PhoneBridgeTestCase):
    """2026-09-21 field defect, diagnosed: replies stayed quiet on a
    real S23 Ultra even after server-side WAV normalization, because
    HTMLMediaElement.volume is spec-capped at 1.0 -- never a boost past
    what the file itself contains. Structural regression coverage for
    the client-side fix: a real WebAudio GainNode (which CAN exceed
    1.0) routed through a DynamicsCompressorNode acting as a hard
    limiter, so a boosted gain can never clip. A real device test is
    what proves it's audibly louder; these prove the mechanism can't
    silently regress back to plain, capped player.volume."""

    def test_gain_graph_uses_a_real_gainnode_not_just_player_volume(self):
        text = self.client.get("/").text
        self.assertIn("function ensureAudioGraph", text)
        self.assertIn("createMediaElementSource(player)", text)
        self.assertIn("audioGraphCtx.createGain()", text)

    def test_limiter_is_configured_as_a_hard_limiter_not_a_musical_compressor(self):
        text = self.client.get("/").text
        start = text.index("function ensureAudioGraph")
        end = text.index("\n}", start)
        body = text[start:end]
        self.assertIn("createDynamicsCompressor()", body)
        # A real limiter: high ratio, fast attack -- not a gentle,
        # musical setting that would still let transients through.
        self.assertIn("limiterNode.ratio.value = 20;", body)
        self.assertIn("limiterNode.attack.value = 0.003;", body)

    def test_gain_can_exceed_native_full_volume(self):
        # HTMLMediaElement.volume tops out at 1.0 by spec -- the
        # slider's max must exceed that, and the actual player.volume
        # must be pinned to 1 once the graph exists (gainNode.gain is
        # the real control from that point on).
        text = self.client.get("/").text
        self.assertIn('id="volCtl" type="range" min="0" max="2.5"', text)
        self.assertIn("player.volume = 1;", text)

    def test_falls_back_to_plain_volume_if_webaudio_unavailable(self):
        # Audio must never go silent just because the boost path
        # failed to construct (an old/incompatible browser) -- confirm
        # the catch block sets a failure flag and applyPlaybackPrefs
        # has a real fallback branch.
        text = self.client.get("/").text
        self.assertIn("audioGraphFailed = true;", text)
        start = text.index("function applyPlaybackPrefs")
        end = text.index("\n}", start)
        body = text[start:end]
        self.assertIn("if (audioGraphReady && gainNode) {", body)
        self.assertIn("player.volume = Math.min(1, vol);", body)

    def test_graph_is_resumed_before_playback_not_just_built_once(self):
        # AudioContext can be suspended by the browser between
        # gestures -- every real playback path must resume it, not
        # assume a one-time construction is enough.
        text = self.client.get("/").text
        self.assertIn("async function _resumeAudioGraph", text)
        self.assertIn("await _resumeAudioGraph();", text)

    def test_test_phone_audio_control_exists_and_hits_the_calibration_endpoint(self):
        text = self.client.get("/").text
        self.assertIn('id="micTestBtn"', text)
        self.assertIn("async function testPhoneAudio", text)
        self.assertIn('"/speak/test.wav?token="', text)
        self.assertIn(
            'micTestBtn").addEventListener("click", testPhoneAudio)', text)


class ModeAwarePairingTests(PhoneBridgeTestCase):
    """/pair/redeem must record the active CA's fingerprint on the
    device in tailscale mode, and record nothing (None) if somehow
    called with no CA generated yet."""

    def test_pairing_with_no_ca_generated_yet_records_no_fingerprint(self):
        phone_bridge._mode = "disabled"   # _mode != "tailscale" -> no CA lookup
        code, _ = phone_auth.create_pairing()
        r = self.client.post("/pair/redeem", json={"code": code, "label": "x"})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(phone_auth.list_devices()[0]["ca_fingerprint_at_pairing"])

    def test_tailscale_mode_pairs_with_the_current_ca_fingerprint(self):
        from backtalk import phone_tls
        import shutil
        shutil.rmtree(phone_tls.TLS_DIR, ignore_errors=True)
        fp = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        phone_bridge._mode = "tailscale"
        code, _ = phone_auth.create_pairing()
        r = self.client.post("/pair/redeem", json={"code": code, "label": "x"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(phone_auth.list_devices()[0]["ca_fingerprint_at_pairing"],
                         fp["sha256"])


class PhoneReplyAvailableTests(PhoneBridgeTestCase):
    """phone_reply_available() is the live signal brains use to decide
    whether they may honestly claim the phone-reply path (2026-09-21
    field defect: Jarvis denied having phone audio output even while
    connected). It must require BOTH a started bridge (real bind_ip --
    empty means start() never got a Tailscale address, e.g. Tailscale
    down) AND at least one currently-paired device (a running-but-
    never-paired bridge is not "connected")."""

    def setUp(self):
        super().setUp()
        self._orig_bind_ip = phone_bridge._bind_ip

    def tearDown(self):
        phone_bridge._bind_ip = self._orig_bind_ip

    def test_false_when_bridge_never_started(self):
        phone_bridge._bind_ip = ""
        self._pair()
        self.assertFalse(phone_bridge.phone_reply_available())

    def test_false_when_started_but_no_device_paired(self):
        phone_bridge._bind_ip = "100.101.102.103"
        self.assertFalse(phone_bridge.phone_reply_available())

    def test_true_when_started_and_a_device_is_paired(self):
        phone_bridge._bind_ip = "100.101.102.103"
        self._pair()
        self.assertTrue(phone_bridge.phone_reply_available())

    def test_false_again_after_the_only_paired_device_is_revoked(self):
        phone_bridge._bind_ip = "100.101.102.103"
        self._pair()
        self.assertTrue(phone_bridge.phone_reply_available())
        phone_auth.revoke_device(1)
        self.assertFalse(phone_bridge.phone_reply_available())


class BootstrapAppTests(unittest.TestCase):
    """The plain-HTTP, Tailscale-bound bootstrap port: /ca.crt and the
    verification page only -- never the real app's routes."""

    def setUp(self):
        from backtalk import phone_tls
        import shutil
        shutil.rmtree(phone_tls.TLS_DIR, ignore_errors=True)
        self.fp = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        phone_bridge._tls_port = 8766
        self.client = TestClient(phone_bridge.build_bootstrap_app())

    def test_serves_the_actual_ca_certificate_bytes(self):
        from backtalk import phone_tls
        r = self.client.get("/ca.crt")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, phone_tls.CA_CERT_FILE.read_bytes())
        self.assertEqual(r.headers["content-type"], "application/x-x509-ca-cert")

    def test_index_page_mentions_the_https_app_port(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("8766", r.text)
        self.assertIn("More Details", r.text)   # the OS-native verification instruction

    def test_index_page_makes_sha256_the_only_accepted_value(self):
        # Captain's explicit rule, 2026-09-27: SHA-256 only; SHA-1 (if
        # shown at all) must never read as an acceptable substitute,
        # and the page must tell the reader not to install if SHA-256
        # can't be confirmed.
        r = self.client.get("/")
        text = r.text
        self.assertIn("SHA-256 is the ONLY value that counts", text)
        self.assertIn("never an acceptable substitute for SHA-256", text)
        self.assertIn("DO NOT INSTALL THIS CA ON THIS DEVICE", text)

    def test_no_app_routes_exist_on_the_bootstrap_server(self):
        for path in ("/message", "/voice", "/pair/redeem", "/events", "/health"):
            r = self.client.get(path)
            self.assertEqual(r.status_code, 404, f"{path} must not exist here")


if __name__ == "__main__":
    unittest.main()
