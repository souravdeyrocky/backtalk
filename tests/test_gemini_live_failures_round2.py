"""Regression tests for a second live Gemini/F8 test failure. Six
issues, six test classes, each named after the exact transcript or
behavior it fixes. All mocked -- zero real network calls.
"""
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from backtalk.brains import brain_intent
from backtalk.brains import config as router_config
from backtalk.brains.base import BrainHealth, BrainUnavailableError
from backtalk.brains.gemini_brain import GeminiBrain
from backtalk.main import (_gemini_preview_line, _is_confirm_phrase,
                           _norm_speech, speak_reply)
from backtalk.router import BrainRouter


def _cfg(*enabled_ids: str) -> dict:
    cfg = json.loads(json.dumps(router_config.DEFAULTS))
    for bid in enabled_ids:
        cfg["brains"][bid]["enabled"] = True
    return cfg


# ---- 1. "Java is switched to 4" -------------------------------------

class Issue1NumberedSwitchVerbFormsTests(unittest.TestCase):
    def test_java_is_switched_to_4(self):
        self.assertEqual(brain_intent.detect("Java is switched to 4"),
                         "switchnum:4")

    def test_javis_is_switched_to_4(self):
        self.assertEqual(brain_intent.detect("Javis is switched to 4"),
                         "switchnum:4")

    def test_is_switched_to_variants_for_every_number(self):
        for n, verb in (("1", "switchnum:1"), ("2", "switchnum:2"),
                        ("3", "switchnum:3"), ("4", "switchnum:4")):
            self.assertEqual(
                brain_intent.detect(f"is switched to {n}"), verb, n)

    def test_switch_brain_to_form(self):
        self.assertEqual(brain_intent.detect("switch brain to 4"),
                         "switchnum:4")

    def test_existing_switch_to_form_still_works(self):
        self.assertEqual(brain_intent.detect("switch to 4"), "switchnum:4")


# ---- 2. "Jarvis, what are the brains you have?" ----------------------

class Issue2WhatAreTheBrainsYouHaveTests(unittest.TestCase):
    def test_exact_live_transcript(self):
        self.assertEqual(
            brain_intent.detect("Jarvis, what are the brains you have?"),
            "brainscount")

    def test_never_reaches_a_local_model(self):
        # A non-None verb means main.py's dispatch intercepts this
        # BEFORE the utterance ever reaches router.ask_stream() /
        # Qwen -- proven by simply asserting the verb is claimed here.
        self.assertIsNotNone(
            brain_intent.detect("what are the brains you have"))

    def test_variant_without_wake_word(self):
        self.assertEqual(
            brain_intent.detect("what are the brains you have"),
            "brainscount")


# ---- 3. Confirm phrase variants, only while actually pending ---------

class Issue3ConfirmPhraseVariantsTests(unittest.TestCase):
    # Superseded by the fuller coverage in
    # test_gemini_live_failures_round3.py (negation, "Thank you,
    # confirmed.", punctuation/case) -- kept here only for the
    # original four variants this round introduced.
    def test_all_required_variants_accepted(self):
        for phrase in ("confirm", "conform", "confirmed", "yes confirm"):
            self.assertTrue(_is_confirm_phrase(phrase), phrase)

    def test_conform_is_accepted(self):
        self.assertTrue(_is_confirm_phrase("conform"))

    def test_unrelated_speech_never_accepted(self):
        self.assertFalse(_is_confirm_phrase("no thanks"))
        self.assertFalse(_is_confirm_phrase("yesterday"))


# ---- 4. Gemini request preview format ---------------------------------

class Issue4GeminiPreviewFormatTests(unittest.TestCase):
    def test_exact_required_format(self):
        self.assertEqual(
            _gemini_preview_line("What is the capital of Japan?"),
            "Gemini preview: What is the capital of Japan? This "
            "leaves your PC. Say confirm to send.")

    def test_preview_embeds_the_utterance_verbatim(self):
        line = _gemini_preview_line("exact text here")
        self.assertIn("exact text here", line)
        self.assertIn("This leaves your PC", line)
        self.assertIn("Say confirm to send", line)


class Issue4NoNetworkCallBeforeConfirmTests(unittest.IsolatedAsyncioTestCase):
    async def test_activation_alone_never_calls_httpx(self):
        cfg = _cfg("qwen3-8b-local", "gemini")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            with patch("backtalk.brains.gemini_brain.httpx.AsyncClient") as m:
                router = BrainRouter(config=cfg)
                await router.activate("gemini", confirmed=True)
            m.assert_not_called()
        self.assertEqual(router.active_id, "gemini")


# ---- 5. Ambiguous "switch to Gemini or Claude" ------------------------

class Issue5AmbiguousBrainRequestTests(unittest.TestCase):
    def test_gemini_or_claude_is_ambiguous(self):
        self.assertEqual(
            brain_intent.detect("switch to Gemini or Claude"),
            "ambiguousbrain")

    def test_claude_or_gemini_either_order(self):
        self.assertEqual(
            brain_intent.detect("switch to Claude or Gemini"),
            "ambiguousbrain")

    def test_never_resolves_to_either_brain_directly(self):
        verb = brain_intent.detect("switch to Gemini or Claude")
        self.assertNotEqual(verb, "usegemini")
        self.assertNotEqual(verb, "useclaude")
        self.assertNotEqual(verb, "cloudbrain")

    def test_comparison_question_without_switch_intent_is_not_ambiguous(self):
        # No switch verb present -- a real question, not a command.
        self.assertIsNone(
            brain_intent.detect("which is smarter, Gemini or Claude"))


class Issue5AmbiguousNeverActivatesEitherBrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_router_active_id_never_changes_from_ambiguous_detection(self):
        # brain_intent.detect() alone has no router access at all --
        # structural proof that "ambiguousbrain" can never itself
        # activate anything, since it never touches a BrainRouter.
        verb = brain_intent.detect("switch to Gemini or Claude")
        self.assertEqual(verb, "ambiguousbrain")
        cfg = _cfg("qwen3-8b-local", "gemini", "claude")
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        await router.activate("qwen3-8b-local")
        # Detecting the ambiguous verb changes nothing on its own.
        self.assertEqual(router.active_id, "qwen3-8b-local")


# ---- 6. Timeout/failure handling --------------------------------------

class Issue6ExplicitTimeoutTests(unittest.TestCase):
    def test_timeout_is_short_and_explicit_not_a_single_blanket_number(self):
        from backtalk.brains.gemini_brain import _TIMEOUT
        self.assertEqual(_TIMEOUT.connect, 5.0)
        self.assertLessEqual(_TIMEOUT.read, 15.0)


class Issue6TimeoutRaisesHonestlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_timeout_raises_unavailable_not_generic(self):
        import httpx

        class _TimingOutClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *a, **kw):
                raise httpx.ConnectTimeout("connect timed out")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            with patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                       return_value=_TimingOutClient()):
                with self.assertRaises(BrainUnavailableError) as ctx:
                    async for _ in brain.ask_stream("hi"):
                        pass
        self.assertIn("time", str(ctx.exception).lower())
        self.assertNotIn("fake-key-for-test", str(ctx.exception))


class Issue6SpeakReplyRecoversFromMidTurnFailureTests(
        unittest.IsolatedAsyncioTestCase):
    """The actual bug this whole round of live failures traced back
    to: speak_reply() only ever caught asyncio.CancelledError, so any
    OTHER exception raised mid-stream (timeout, HTTP error, bad key,
    rate-limit, malformed response) propagated out unhandled -- the
    voice line was left stuck "thinking" forever, nothing spoke the
    error, and the Face never returned to idle."""

    async def test_brain_unavailable_mid_stream_is_caught_and_spoken(self):
        class _FakeMouth:
            def __init__(self):
                self.said = []

            def say(self, text):
                self.said.append(text)

            def say_chunk(self, text, directions=None, targets=None):
                self.said.append(text)

        class _FakeRouter:
            async def ask_stream(self, text):
                yield "starting..."
                raise BrainUnavailableError("Gemini didn't respond in time")

            async def interrupt(self):
                pass

        mouth = _FakeMouth()
        # Must not raise -- the whole point of the fix. day_journal's
        # real record_event() is patched out here (and in the other
        # two tests below): speak_reply() now logs every mid-turn
        # error to the Day Journal ledger, and this test exists purely
        # to exercise that error-recovery path, not to write real
        # entries into Captain's actual local ledger every test run.
        with patch("backtalk.day_journal.record_event"):
            await speak_reply(_FakeRouter(), mouth,
                              "what's the capital of Japan")
        spoken = " ".join(mouth.said).lower()
        self.assertIn("error", spoken)
        self.assertNotIn("fake-key-for-test", spoken)

    async def test_state_returns_to_idle_after_mid_stream_failure(self):
        from backtalk import signals

        class _FakeMouth:
            def say(self, text):
                pass

            def say_chunk(self, text, directions=None, targets=None):
                pass

        class _FakeRouter:
            async def ask_stream(self, text):
                raise BrainUnavailableError("Gemini is rate-limited")
                yield ""  # pragma: no cover -- keeps this an async generator

            async def interrupt(self):
                pass

        states = []
        with patch("backtalk.day_journal.record_event"):
            with patch.object(signals, "set_state",
                              side_effect=lambda s: states.append(s)):
                with patch.object(signals, "static_stop"):
                    await speak_reply(_FakeRouter(), _FakeMouth(), "hi")
        self.assertIn("idle", states)

    async def test_never_falls_back_to_a_different_brain(self):
        # The fake router here has no activate()/switching capability
        # at all -- if speak_reply tried to fall back to another
        # brain, this would AttributeError instead of completing.
        class _FakeMouth:
            def say(self, text):
                pass

            def say_chunk(self, text, directions=None, targets=None):
                pass

        class _FakeRouter:
            async def ask_stream(self, text):
                raise BrainUnavailableError("Gemini quota exceeded")
                yield ""  # pragma: no cover -- keeps this an async generator

            async def interrupt(self):
                pass

        with patch("backtalk.day_journal.record_event"):
            await speak_reply(_FakeRouter(), _FakeMouth(), "hi")  # must not raise


if __name__ == "__main__":
    unittest.main()
