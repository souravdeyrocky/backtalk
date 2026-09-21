"""Phase 3A: local DeepSeek reasoning mode.

These tests don't require deepseek-r1:8b to actually be pulled -- they
prove the wiring (phrase matching, config shape, no-silent-switch
behavior) using the same disabled-by-default / honest-refusal
machinery already proven in Phase 1/2. The live, unmocked proof against
a real pulled model is a separate proof script run after the pull, not
part of this file (there's nothing this suite can honestly assert about
a model that isn't on disk yet).
"""
import json
import unittest
from unittest.mock import AsyncMock

from backtalk.brains import config as router_config
from backtalk.brains.base import BrainHealth, BrainUnavailableError
from backtalk.main import CONSOLE_VERBS, console_match
from backtalk.router import BrainRouter


class DeepSeekConsoleVerbTests(unittest.TestCase):
    """Phase 2 wired voice-switching for Qwen and Claude only. This is
    the one piece of new user-facing surface Phase 3A adds."""

    def test_switch_phrases_map_to_usedeepseek(self):
        phrases = ("switch to deep reasoning", "use deep reasoning",
                  "switch to deepseek", "use deepseek",
                  "deep reasoning mode", "think harder")
        for p in phrases:
            self.assertEqual(console_match(p), "usedeepseek", f"{p!r}")

    def test_usedeepseek_phrases_dont_collide_with_claudes_deep_model(self):
        # "deep" (Claude's own /model deep) and "usedeepseek" (switching
        # the whole BRAIN) are different concepts and must stay
        # distinguishable -- neither phrase set should overlap.
        deep_model_phrases = set(CONSOLE_VERBS["deep"])
        deepseek_phrases = set(CONSOLE_VERBS["usedeepseek"])
        self.assertEqual(deep_model_phrases & deepseek_phrases, set())

    def test_ordinary_sentences_still_dont_match(self):
        self.assertIsNone(console_match(
            "I've been thinking really hard about this deep problem"))


class BrainRouterConfigDeepSeekTests(unittest.TestCase):
    """Guards the actual deployed brain_router.json shape -- the file
    itself, not a fixture -- so a future edit can't silently widen or
    narrow what Phase 3A actually shipped."""

    def test_real_config_has_deepseek_enabled_qwen_still_default(self):
        cfg = router_config.load()
        self.assertTrue(cfg["brains"]["deepseek-r1-8b-local"]["enabled"],
                        "DeepSeek should be enabled after Phase 3A")
        self.assertEqual(cfg["active_brain"], "qwen3-8b-local",
                         "Qwen must stay the default brain")
        self.assertTrue(cfg["brains"]["qwen3-8b-local"]["enabled"])
        # Gemini's "enabled" flag is deliberately true as of the
        # numbered-interface work: the REAL gate is GEMINI_API_KEY's
        # presence plus a passing health check (both enforced by
        # router.activate() itself), not this static config flag --
        # see test_numbered_brain_interface.py for that gate's tests.
        self.assertTrue(cfg["brains"]["gemini"]["enabled"])


class NoSilentSwitchTests(unittest.IsolatedAsyncioTestCase):
    """The core safety property this phase must not violate: DeepSeek
    is reachable ONLY through the explicit voice verb. A hard question
    asked while Qwen is active must never auto-route to DeepSeek."""

    def _cfg_with_deepseek_enabled(self):
        cfg = json.loads(json.dumps(router_config.DEFAULTS))
        cfg["brains"]["qwen3-8b-local"]["enabled"] = True
        cfg["brains"]["deepseek-r1-8b-local"]["enabled"] = True
        return cfg

    async def test_hard_reasoning_question_on_qwen_never_auto_switches(self):
        router = BrainRouter(config=self._cfg_with_deepseek_enabled())
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        await router.activate("qwen3-8b-local")

        async def _fake_ask_stream(utterance):
            yield "a Qwen answer, whatever the question actually was."

        qwen.ask_stream = _fake_ask_stream
        hard_question = (
            "Think step by step and reason carefully: a bat and a ball "
            "cost 1.10 together and the bat costs 1.00 more than the "
            "ball, how much does the ball cost")
        async for _ in router.ask_stream(hard_question):
            pass
        self.assertEqual(router.active_id, "qwen3-8b-local",
                         "a hard question must never silently move the "
                         "active brain off Qwen")

    async def test_deepseek_only_reachable_via_explicit_activate_call(self):
        router = BrainRouter(config=self._cfg_with_deepseek_enabled())
        deepseek = router.get("deepseek-r1-8b-local")
        deepseek.health = AsyncMock(return_value=BrainHealth(healthy=True))
        deepseek.start = AsyncMock()
        self.assertIsNone(router.active_id)
        await router.activate("deepseek-r1-8b-local")
        self.assertEqual(router.active_id, "deepseek-r1-8b-local")

    async def test_enabled_but_not_installed_refuses_honestly(self):
        # Structural proof, mocked so it stays true regardless of
        # whether this machine has actually pulled the model -- the
        # REAL, unmocked version of this same check (against whatever
        # Ollama actually reports right now) is run separately, ad hoc,
        # immediately before and after the real pull.
        router = BrainRouter(config=self._cfg_with_deepseek_enabled())
        deepseek = router.get("deepseek-r1-8b-local")
        deepseek.health = AsyncMock(return_value=BrainHealth(
            healthy=False,
            reason="deepseek-r1:8b is not installed in Ollama"))
        with self.assertRaises(BrainUnavailableError) as ctx:
            await router.activate("deepseek-r1-8b-local")
        self.assertIn("deepseek-r1:8b", str(ctx.exception))
        self.assertIsNone(router.active_id)


if __name__ == "__main__":
    unittest.main()
