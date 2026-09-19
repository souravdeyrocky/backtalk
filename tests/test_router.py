import json
import unittest
from unittest.mock import AsyncMock

from backtalk.brains import config as router_config
from backtalk.brains.base import (
    BrainDisabledError,
    BrainHealth,
    BrainUnavailableError,
)
from backtalk.router import BrainRouter, ConfirmRequiredError, NoBrainActiveError


def _all_disabled_cfg() -> dict:
    return json.loads(json.dumps(router_config.DEFAULTS))


class RouterDefaultsTests(unittest.TestCase):
    def test_all_brains_disabled_by_default(self):
        router = BrainRouter(config=_all_disabled_cfg())
        status = router.status()
        self.assertIsNone(status.active_id)
        self.assertEqual(set(status.brains),
                         {"qwen3-8b-local", "deepseek-r1-8b-local",
                          "gemini", "claude"})
        for bid, s in status.brains.items():
            self.assertFalse(s.enabled, f"{bid} should default disabled")
            self.assertFalse(s.healthy, f"{bid} should default unhealthy")


class RouterActivationTests(unittest.IsolatedAsyncioTestCase):
    async def test_activating_disabled_brain_refuses(self):
        router = BrainRouter(config=_all_disabled_cfg())
        with self.assertRaises(BrainDisabledError):
            await router.activate("qwen3-8b-local")
        self.assertIsNone(router.active_id)

    async def test_activating_unknown_id_raises_keyerror(self):
        router = BrainRouter(config=_all_disabled_cfg())
        with self.assertRaises(KeyError):
            await router.activate("not-a-real-brain")

    async def test_ask_stream_before_activation_raises(self):
        router = BrainRouter(config=_all_disabled_cfg())
        with self.assertRaises(NoBrainActiveError):
            async for _ in router.ask_stream("hi"):
                pass

    async def test_claude_requires_confirm(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["claude"]["enabled"] = True
        router = BrainRouter(config=cfg)
        claude = router.get("claude")
        claude.start = AsyncMock()
        with self.assertRaises(ConfirmRequiredError):
            await router.activate("claude")
        self.assertIsNone(router.active_id)
        await router.activate("claude", confirmed=True)
        self.assertEqual(router.active_id, "claude")
        claude.start.assert_awaited_once()

    async def test_enabled_but_unhealthy_brain_refuses(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["deepseek-r1-8b-local"]["enabled"] = True
        router = BrainRouter(config=cfg)
        deepseek = router.get("deepseek-r1-8b-local")
        deepseek.health = AsyncMock(
            return_value=BrainHealth(
                healthy=False,
                reason="deepseek-r1:8b is not installed in Ollama"))
        with self.assertRaises(BrainUnavailableError):
            await router.activate("deepseek-r1-8b-local")
        self.assertIsNone(router.active_id)

    async def test_switching_stops_the_previous_brain(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["qwen3-8b-local"]["enabled"] = True
        cfg["brains"]["deepseek-r1-8b-local"]["enabled"] = True
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        deepseek = router.get("deepseek-r1-8b-local")
        for b in (qwen, deepseek):
            b.health = AsyncMock(return_value=BrainHealth(healthy=True))
            b.start = AsyncMock()
            b.stop = AsyncMock()
        await router.activate("qwen3-8b-local")
        await router.activate("deepseek-r1-8b-local")
        qwen.stop.assert_awaited_once()
        deepseek.start.assert_awaited_once()
        self.assertEqual(router.active_id, "deepseek-r1-8b-local")

    async def test_health_check_all_survives_a_broken_adapter(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["qwen3-8b-local"]["enabled"] = True
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")

        async def _boom():
            raise RuntimeError("simulated crash")

        qwen.health = _boom
        statuses = await router.health_check_all()
        self.assertFalse(statuses["qwen3-8b-local"].healthy)
        self.assertIn("simulated crash", statuses["qwen3-8b-local"].reason)


if __name__ == "__main__":
    unittest.main()
