"""2026-09-21: active-platform output routing. A phone-originated turn
must speak only on the phone; a desktop-originated turn must speak
only on desktop; an explicit Captain command ("reply on phone" etc.)
overrides that; a phone barge-in interrupts and takes over; a
disconnected phone falls back to an announced desktop reply rather
than vanishing silently. See main.py's _resolve_reply_targets/
speak_reply, mouth.py's targets-aware say_chunk/_run, and
phone_bridge.py's PhoneTurn/targets-tagged transcript entries.
"""
import unittest
from unittest import mock

from backtalk import main, mouth as mouth_mod, phone_bridge


class ResolveReplyTargetsTests(unittest.TestCase):
    """Pure routing-decision logic, independent of the rest of the
    turn pipeline."""

    def setUp(self):
        main._REPLY_TARGET_OVERRIDE["mode"] = None

    def tearDown(self):
        main._REPLY_TARGET_OVERRIDE["mode"] = None

    def test_phone_turn_with_no_override_routes_phone_only(self):
        self.assertEqual(main._resolve_reply_targets("phone"),
                         frozenset({"phone"}))

    def test_desktop_turn_with_no_override_routes_desktop_only(self):
        self.assertEqual(main._resolve_reply_targets("desktop"),
                         frozenset({"desktop"}))

    def test_explicit_phone_override_wins_regardless_of_origin(self):
        main._REPLY_TARGET_OVERRIDE["mode"] = "phone"
        self.assertEqual(main._resolve_reply_targets("desktop"),
                         frozenset({"phone"}))
        self.assertEqual(main._resolve_reply_targets("phone"),
                         frozenset({"phone"}))

    def test_explicit_desktop_override_wins_regardless_of_origin(self):
        main._REPLY_TARGET_OVERRIDE["mode"] = "desktop"
        self.assertEqual(main._resolve_reply_targets("phone"),
                         frozenset({"desktop"}))

    def test_explicit_both_override(self):
        main._REPLY_TARGET_OVERRIDE["mode"] = "both"
        self.assertEqual(main._resolve_reply_targets("phone"),
                         frozenset({"desktop", "phone"}))
        self.assertEqual(main._resolve_reply_targets("desktop"),
                         frozenset({"desktop", "phone"}))


class ReplyRoutingConsoleVerbTests(unittest.TestCase):
    """Rule 4: "Explicit Captain commands override routing." Exact-
    phrase console verbs, same pattern/strictness as every other verb
    in CONSOLE_VERBS (ordinary sentences containing these words must
    never accidentally trigger them)."""

    def test_reply_on_phone_recognized(self):
        self.assertEqual(main.console_match("reply on phone"), "replyphone")
        self.assertEqual(main.console_match("Reply On My Phone"), "replyphone")

    def test_reply_on_computer_recognized(self):
        self.assertEqual(main.console_match("reply on computer"), "replydesktop")
        self.assertEqual(main.console_match("reply on the computer"), "replydesktop")
        self.assertEqual(main.console_match("reply on desktop"), "replydesktop")

    def test_reply_on_both_recognized(self):
        self.assertEqual(main.console_match("reply on both"), "replyboth")

    def test_reply_automatically_resets_the_override(self):
        self.assertEqual(main.console_match("reply automatically"), "replyauto")

    def test_ordinary_sentence_mentioning_phone_is_not_flagged(self):
        self.assertIsNone(
            main.console_match("I think my phone reply was slow today"))


class SpeakReplyTargetsTests(unittest.IsolatedAsyncioTestCase):
    """speak_reply() forwards `targets` to every mouth.say_chunk() call
    -- the actual mechanism that makes desktop stay silent for a
    phone-only turn and vice versa."""

    class _FakeMouth:
        def __init__(self):
            self.calls = []

        def say(self, text):
            self.calls.append((text, None))

        def say_chunk(self, text, directions=None, targets=None):
            self.calls.append((text, targets))

    class _FakeRouter:
        def __init__(self, sentences):
            self._sentences = sentences

        async def ask_stream(self, text):
            for s in self._sentences:
                yield s

        async def interrupt(self):
            pass

    async def test_desktop_only_turn_never_carries_phone_in_targets(self):
        fake_mouth = self._FakeMouth()
        with mock.patch("backtalk.main.phone_bridge.phone_reply_available",
                        return_value=True):
            await main.speak_reply(self._FakeRouter(["Hello there."]),
                                   fake_mouth, "hi",
                                   targets=frozenset({"desktop"}))
        self.assertTrue(fake_mouth.calls)
        for text, targets in fake_mouth.calls:
            self.assertNotIn("phone", targets)

    async def test_phone_only_turn_never_carries_desktop_in_targets(self):
        fake_mouth = self._FakeMouth()
        with mock.patch("backtalk.main.phone_bridge.phone_reply_available",
                        return_value=True):
            await main.speak_reply(self._FakeRouter(["Hello there."]),
                                   fake_mouth, "hi",
                                   targets=frozenset({"phone"}))
        self.assertTrue(fake_mouth.calls)
        for text, targets in fake_mouth.calls:
            self.assertNotIn("desktop", targets)

    async def test_both_targets_pass_through_unchanged(self):
        fake_mouth = self._FakeMouth()
        with mock.patch("backtalk.main.phone_bridge.phone_reply_available",
                        return_value=True):
            await main.speak_reply(self._FakeRouter(["Hi.", "There."]),
                                   fake_mouth, "hi",
                                   targets=frozenset({"desktop", "phone"}))
        for text, targets in fake_mouth.calls:
            self.assertEqual(targets, frozenset({"desktop", "phone"}))


class SpeakReplyDisconnectFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Rule 6: "If the paired phone disconnects during a phone-targeted
    turn, stop phone delivery and announce the fallback on desktop;
    never silently send a private phone reply elsewhere.\""""

    class _FakeMouth:
        def __init__(self):
            self.calls = []

        def say(self, text):
            self.calls.append((text, None))

        def say_chunk(self, text, directions=None, targets=None):
            self.calls.append((text, targets))

    class _FakeRouter:
        def __init__(self, sentences):
            self._sentences = sentences

        async def ask_stream(self, text):
            for s in self._sentences:
                yield s

        async def interrupt(self):
            pass

    async def test_phone_unreachable_before_the_turn_falls_back_with_announcement(self):
        fake_mouth = self._FakeMouth()
        with mock.patch("backtalk.main.phone_bridge.phone_reply_available",
                        return_value=False):
            await main.speak_reply(self._FakeRouter(["The actual answer."]),
                                   fake_mouth, "hi",
                                   targets=frozenset({"phone"}))
        # An announcement fired first, on desktop, followed by the real
        # answer -- also on desktop -- never lost, never sent to a
        # phone that was never reachable.
        self.assertGreaterEqual(len(fake_mouth.calls), 2)
        announcement, ann_targets = fake_mouth.calls[0]
        self.assertIn("phone", announcement.lower())
        self.assertEqual(ann_targets, frozenset({"desktop"}))
        for text, targets in fake_mouth.calls:
            self.assertNotIn("phone", targets)
        self.assertIn("The actual answer.",
                      " ".join(t for t, _ in fake_mouth.calls))

    async def test_both_targeted_turn_ignores_phone_reachability(self):
        # Desktop is ALREADY a target, so phone reachability is
        # irrelevant -- no fallback machinery should even engage.
        fake_mouth = self._FakeMouth()
        with mock.patch("backtalk.main.phone_bridge.phone_reply_available",
                        return_value=False):
            await main.speak_reply(self._FakeRouter(["Answer."]),
                                   fake_mouth, "hi",
                                   targets=frozenset({"desktop", "phone"}))
        texts = [t for t, _ in fake_mouth.calls]
        self.assertEqual(texts, ["Answer."])

    async def test_reachable_phone_only_turn_never_announces_a_fallback(self):
        fake_mouth = self._FakeMouth()
        with mock.patch("backtalk.main.phone_bridge.phone_reply_available",
                        return_value=True):
            await main.speak_reply(self._FakeRouter(["Answer."]),
                                   fake_mouth, "hi",
                                   targets=frozenset({"phone"}))
        texts = [t for t, _ in fake_mouth.calls]
        self.assertEqual(texts, ["Answer."])

    async def test_disconnect_mid_stream_falls_back_for_the_rest_of_the_turn(self):
        # Reachable for the FIRST sentence, gone by the second --
        # emit() re-checks every sentence, so this must still recover
        # rather than silently dropping the rest of the reply.
        fake_mouth = self._FakeMouth()
        availability = iter([True, False, False])
        with mock.patch("backtalk.main.phone_bridge.phone_reply_available",
                        side_effect=lambda: next(availability, False)):
            await main.speak_reply(
                self._FakeRouter(["First sentence.", "Second sentence."]),
                fake_mouth, "hi", targets=frozenset({"phone"}))
        texts = [t for t, _ in fake_mouth.calls]
        self.assertIn("First sentence.", texts)
        joined = " ".join(texts)
        self.assertIn("Second sentence.", joined)
        self.assertIn("disconnected", joined.lower())


class MouthSilentRoutingTests(unittest.TestCase):
    """Real Mouth instance, but _play_stream is mocked out -- proves
    a phone-only ("desktop" not in targets) chunk never reaches the
    actual desktop audio engine, while the transcript sink and turn-
    complete hook still fire exactly as they would for a normal reply.
    mouth.py has no dedicated real-audio test suite (needs playback
    hardware); this is the same structural-proof approach
    MouthTurnCompleteHookTests already uses."""

    def setUp(self):
        self.m = mouth_mod.Mouth()
        self.addCleanup(self.m.shutdown)
        self.play_calls = []
        self._play_patch = mock.patch.object(
            self.m, "_play_stream",
            side_effect=lambda *a, **k: self.play_calls.append(a))
        self._play_patch.start()
        self.addCleanup(self._play_patch.stop)
        self.transcript_calls = []
        mouth_mod.set_transcript_sink(
            lambda who, text, targets=None: self.transcript_calls.append(
                (who, text, targets)))
        self.addCleanup(mouth_mod.set_transcript_sink, None)
        self.turn_complete_calls = []
        mouth_mod.set_turn_complete_sink(
            lambda: self.turn_complete_calls.append(True))
        self.addCleanup(mouth_mod.set_turn_complete_sink, None)

    def test_phone_only_chunk_never_reaches_play_stream(self):
        self.m.say_chunk("Phone only reply.", targets=frozenset({"phone"}))
        self.m.wait_done(timeout=5)
        self.assertEqual(self.play_calls, [])

    def test_phone_only_chunk_still_reaches_the_transcript_sink(self):
        self.m.say_chunk("Phone only reply.", targets=frozenset({"phone"}))
        self.m.wait_done(timeout=5)
        self.assertEqual(len(self.transcript_calls), 1)
        who, text, targets = self.transcript_calls[0]
        self.assertEqual(who, "jarvis")
        self.assertEqual(text, "Phone only reply.")
        self.assertEqual(targets, frozenset({"phone"}))

    def test_phone_only_chunk_still_fires_turn_complete(self):
        self.m.say_chunk("Phone only reply.", targets=frozenset({"phone"}))
        self.m.wait_done(timeout=5)
        self.assertEqual(self.turn_complete_calls, [True])

    def test_phone_only_chunk_never_sets_the_speaking_flag(self):
        self.m.say_chunk("Phone only reply.", targets=frozenset({"phone"}))
        self.m.wait_done(timeout=5)
        self.assertFalse(self.m.speaking)

    def test_desktop_targeted_chunk_reaches_play_stream_as_before(self):
        self.m.say_chunk("Desktop reply.", targets=frozenset({"desktop"}))
        self.m.wait_done(timeout=5)
        self.assertEqual(len(self.play_calls), 1)

    def test_default_targets_still_play_on_desktop_unchanged(self):
        # No targets passed at all -- every pre-routing call site's
        # exact prior behavior.
        self.m.say_chunk("Untargeted reply.")
        self.m.wait_done(timeout=5)
        self.assertEqual(len(self.play_calls), 1)


class PhoneTurnDataclassTests(unittest.TestCase):
    def test_carries_the_text(self):
        t = phone_bridge.PhoneTurn("hello")
        self.assertEqual(t.text, "hello")

    def test_is_frozen_immutable(self):
        t = phone_bridge.PhoneTurn("hello")
        with self.assertRaises(Exception):
            t.text = "changed"


if __name__ == "__main__":
    unittest.main()
