"""Regression tests for Gemini as the numbered interface's Brain 4:
the "switch to 4" ask, its confirm gate, and the activation
announcement. All mocked -- zero real network calls, zero real key
usage. The underlying Gemini safety properties this wiring depends on
(absent/healthy key, exact-prompt-only transmission, no vault-context
call, rate-limit honesty, no silent fallback) are already covered in
depth by test_gemini_brain.py and test_gemini_recommendation_workflow.py;
this file covers what's NEW here -- the numbered verb reaching that
same gate, translated onto the existing usegemini:confirmed path.
"""
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from backtalk.brains import brain_intent
from backtalk.brains import config as router_config
from backtalk.brains.base import BrainHealth
from backtalk.main import (_gemini_activated_line, _switchnum4_line)
from backtalk.router import BrainRouter, ConfirmRequiredError


def _cfg(*enabled_ids: str) -> dict:
    cfg = json.loads(json.dumps(router_config.DEFAULTS))
    for bid in enabled_ids:
        cfg["brains"][bid]["enabled"] = True
    return cfg


class SwitchToFourDetectionTests(unittest.TestCase):
    def test_switch_to_4_and_variants(self):
        for phrase in ("switch to 4", "switch to four",
                       "switch to brain four", "use brain 4"):
            self.assertEqual(brain_intent.detect(phrase), "switchnum:4",
                             phrase)


class SwitchNum4AskLineTests(unittest.TestCase):
    def test_exact_required_ask_text(self):
        self.assertEqual(
            _switchnum4_line(already_active=False),
            "Brain 4 is Gemini, an external free-tier service. I "
            "will send only your next approved prompt, not your "
            "vault. Say confirm to switch.")


class GeminiActivatedAnnouncementTests(unittest.TestCase):
    # Updated for the external-consent LEASE model: confirming now
    # opens a thirty-minute idle window instead of promising to ask
    # again on literally every request.
    def test_exact_announcement_mentions_leaving_the_pc_and_the_lease(self):
        line = _gemini_activated_line()
        self.assertEqual(
            line,
            "Gemini free-tier external mode is active for the next "
            "thirty minutes of use. Requests leave this PC only "
            "during that window; I'll ask again after thirty minutes "
            "of inactivity. Say switch to Qwen any time to go back.")
        self.assertIn("leave this PC", line)
        self.assertIn("thirty minutes", line)


class AbsentKeyNeverActivatesTests(unittest.IsolatedAsyncioTestCase):
    """Brain 4 stays disabled -- structurally, via router.activate()'s
    own health check -- whenever GEMINI_API_KEY is absent, exactly
    like the numbered flow's confirm step (switchnum:4:confirmed ->
    usegemini:confirmed) would hit live."""

    async def test_confirm_without_a_key_raises_and_never_switches(self):
        cfg = _cfg("qwen3-8b-local", "gemini")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            router = BrainRouter(config=cfg)
            qwen = router.get("qwen3-8b-local")
            qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
            qwen.start = AsyncMock()
            await router.activate("qwen3-8b-local")

            with self.assertRaises(Exception):
                await router.activate("gemini", confirmed=True)
        self.assertEqual(router.active_id, "qwen3-8b-local")


class HealthyKeyActivatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirm_with_a_valid_key_activates_gemini(self):
        cfg = _cfg("qwen3-8b-local", "gemini")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            router = BrainRouter(config=cfg)
            status = await router.activate("gemini", confirmed=True)
        self.assertEqual(router.active_id, "gemini")
        self.assertTrue(status.healthy)


class ConfirmationRequiredTests(unittest.IsolatedAsyncioTestCase):
    """"switch to 4" alone must never activate Gemini -- only an
    explicit confirmed=True (the numbered flow's confirm step) does,
    same ConfirmRequiredError gate every other confirm-gated brain
    uses."""

    async def test_activate_without_confirm_raises(self):
        cfg = _cfg("qwen3-8b-local", "gemini")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            router = BrainRouter(config=cfg)
            with self.assertRaises(ConfirmRequiredError):
                await router.activate("gemini")
        self.assertIsNone(router.active_id)


class NoFallbackOnUnavailableTests(unittest.IsolatedAsyncioTestCase):
    """No silent fallback to Claude, Qwen, or DeepSeek when Gemini
    can't be reached -- the active brain never changes and Claude is
    never even constructed."""

    async def test_unavailable_gemini_never_touches_claude_or_active_id(self):
        cfg = _cfg("qwen3-8b-local", "gemini", "claude")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            with patch("backtalk.brains.claude_brain.WarmBrain") as MockWarm:
                router = BrainRouter(config=cfg)
                qwen = router.get("qwen3-8b-local")
                qwen.health = AsyncMock(
                    return_value=BrainHealth(healthy=True))
                qwen.start = AsyncMock()
                await router.activate("qwen3-8b-local")
                with self.assertRaises(Exception):
                    await router.activate("gemini", confirmed=True)
                MockWarm.assert_not_called()
        self.assertEqual(router.active_id, "qwen3-8b-local")


class RealConfigEnablesGeminiForTheHealthGateTests(unittest.TestCase):
    """The real, deployed brain_router.json now leaves Gemini's static
    "enabled" flag on -- the actual gate Captain asked for is the
    environment variable plus the health check, both enforced inside
    router.activate() regardless of this flag's own honesty about
    intent. This just proves the shipped file matches that design."""

    def test_real_config_has_gemini_enabled_for_the_health_gate(self):
        cfg = router_config.load()
        self.assertTrue(cfg["brains"]["gemini"]["enabled"])
        self.assertEqual(cfg["brains"]["gemini"]["api_key_env"],
                         "GEMINI_API_KEY")


if __name__ == "__main__":
    unittest.main()
