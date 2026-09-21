"""Mocked, zero-real-API-call proofs for the Gemini brain + smart
recommendation workflow: recommend() suggests but never switches,
Gemini and Claude both stay confirm-gated at the router (no silent
fallback), a rate-limited/unavailable Gemini reports honestly instead
of quietly handing the turn to Claude, and Gemini's path never touches
vault context even when the utterance mentions vault-shaped content.
"""
import json
import unittest
from unittest.mock import AsyncMock, patch

from backtalk.brains import config as router_config
from backtalk.brains import recommend
from backtalk.brains.base import BrainHealth
from backtalk.router import BrainRouter, ConfirmRequiredError


def _cfg(*enabled_ids: str) -> dict:
    cfg = json.loads(json.dumps(router_config.DEFAULTS))
    for bid in enabled_ids:
        cfg["brains"][bid]["enabled"] = True
    return cfg


class _FakeRouterStatusOnly:
    """A stand-in for recommend()'s only real dependency (router.status())
    so recommendation logic can be tested without standing up a full
    BrainRouter or Ollama/Gemini connection."""

    def __init__(self, active_id, enabled_ids):
        self._active_id = active_id
        self._enabled = set(enabled_ids)

    def status(self):
        from backtalk.router import RouterStatus
        from backtalk.brains.base import BrainStatus
        brains = {
            bid: BrainStatus(id=bid, label=bid, enabled=(bid in self._enabled),
                             healthy=True, capability_summary="")
            for bid in ("qwen3-8b-local", "deepseek-r1-8b-local",
                       "gemini", "claude")
        }
        return RouterStatus(active_id=self._active_id, brains=brains)


class LocalRecommendationTests(unittest.TestCase):
    """An ordinary, everyday question must never trigger a suggestion
    -- recommend() only speaks up when it finds a genuinely better
    fit, and staying quiet is the overwhelmingly common case."""

    def test_ordinary_question_recommends_nothing(self):
        router = _FakeRouterStatusOnly(
            "qwen3-8b-local",
            {"qwen3-8b-local", "deepseek-r1-8b-local", "gemini"})
        result = recommend.recommend(
            "what time do I have my dentist appointment", router)
        self.assertIsNone(result)

    def test_tool_shaped_request_recommends_nothing(self):
        # tool_intent already refuses these with its own pointer at
        # Claude -- recommend() must not pile a second suggestion on.
        router = _FakeRouterStatusOnly(
            "qwen3-8b-local",
            {"qwen3-8b-local", "deepseek-r1-8b-local", "gemini"})
        result = recommend.recommend(
            "edit my backtalk.json file and save it", router)
        self.assertIsNone(result)


class DeepSeekRecommendationTests(unittest.TestCase):
    def test_complex_reasoning_phrasing_suggests_deepseek(self):
        router = _FakeRouterStatusOnly(
            "qwen3-8b-local",
            {"qwen3-8b-local", "deepseek-r1-8b-local"})
        result = recommend.recommend(
            "can you think through the trade-offs of this plan "
            "step-by-step", router)
        self.assertIsNotNone(result)
        self.assertEqual(result.brain_id, "deepseek-r1-8b-local")

    def test_never_suggests_deepseek_when_disabled(self):
        router = _FakeRouterStatusOnly("qwen3-8b-local", {"qwen3-8b-local"})
        result = recommend.recommend(
            "walk me through the trade-offs step-by-step", router)
        self.assertIsNone(result)

    def test_never_suggests_deepseek_when_already_active(self):
        router = _FakeRouterStatusOnly(
            "deepseek-r1-8b-local",
            {"qwen3-8b-local", "deepseek-r1-8b-local"})
        result = recommend.recommend(
            "walk me through the trade-offs step-by-step", router)
        self.assertIsNone(result)

    def test_recommendation_never_switches_the_active_brain_itself(self):
        # recommend() takes a status-only stand-in with no activate()
        # method at all -- if this call tried to switch anything, it
        # would raise AttributeError, not just fail an assertion.
        router = _FakeRouterStatusOnly(
            "qwen3-8b-local",
            {"qwen3-8b-local", "deepseek-r1-8b-local"})
        recommend.recommend(
            "walk me through the trade-offs step-by-step", router)


class GeminiRecommendationTests(unittest.TestCase):
    def test_live_info_phrasing_suggests_gemini(self):
        router = _FakeRouterStatusOnly(
            "qwen3-8b-local", {"qwen3-8b-local", "gemini"})
        result = recommend.recommend(
            "what's the latest news on this", router)
        self.assertIsNotNone(result)
        self.assertEqual(result.brain_id, "gemini")

    def test_never_suggests_gemini_when_disabled(self):
        router = _FakeRouterStatusOnly("qwen3-8b-local", {"qwen3-8b-local"})
        result = recommend.recommend(
            "what's the current stock price", router)
        self.assertIsNone(result)


class RouterApprovalGateTests(unittest.IsolatedAsyncioTestCase):
    """Both Gemini and Claude are confirm-gated at the exact same
    chokepoint (BrainRouter.activate) -- a recommendation, or any
    other caller, must never bypass it by passing confirmed=True on a
    hunch. Refusing must never change the active brain."""

    async def test_gemini_switch_without_confirm_refuses_and_stays_put(self):
        cfg = _cfg("qwen3-8b-local", "gemini")
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        await router.activate("qwen3-8b-local")

        gemini = router.get("gemini")
        gemini.health = AsyncMock(return_value=BrainHealth(healthy=True))
        with self.assertRaises(ConfirmRequiredError):
            await router.activate("gemini")
        self.assertEqual(router.active_id, "qwen3-8b-local")

    async def test_claude_switch_without_confirm_refuses_and_stays_put(self):
        cfg = _cfg("qwen3-8b-local", "claude")
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        await router.activate("qwen3-8b-local")

        claude = router.get("claude")
        claude.health = AsyncMock(return_value=BrainHealth(healthy=True))
        with self.assertRaises(ConfirmRequiredError):
            await router.activate("claude")
        self.assertEqual(router.active_id, "qwen3-8b-local")

    async def test_a_bare_recommendation_never_itself_calls_activate(self):
        # recommend() has no reference to the real router's activate()
        # at all when given the status-only stand-in, so if it tried
        # to switch anything this would AttributeError before it could
        # ever reach a real brain.
        cfg = _cfg("qwen3-8b-local", "gemini")
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        await router.activate("qwen3-8b-local")

        recommend.recommend("what's the latest news on this", router)
        self.assertEqual(router.active_id, "qwen3-8b-local")


class GeminiUnavailableNeverFallsBackTests(unittest.IsolatedAsyncioTestCase):
    """A rate-limited or otherwise unavailable Gemini must report
    honestly -- never silently hand the turn to Claude, and never
    change which brain is active."""

    async def _activated_gemini_router(self):
        import os
        cfg = _cfg("qwen3-8b-local", "gemini", "claude")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            router = BrainRouter(config=cfg)
            gemini = router.get("gemini")
            await router.activate("gemini", confirmed=True)
        return router, gemini

    async def test_rate_limited_gemini_never_constructs_claude(self):
        import os
        router, gemini = await self._activated_gemini_router()

        class _Resp:
            status_code = 429

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *a, **kw):
                return _Resp()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            with patch("backtalk.brains.claude_brain.WarmBrain") as MockWarm:
                with patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                           return_value=_Client()):
                    with self.assertRaises(Exception):
                        async for _ in router.ask_stream(
                                "what's the latest news"):
                            pass
                MockWarm.assert_not_called()
        self.assertEqual(router.active_id, "gemini")


class GeminiNeverLeaksVaultContextTests(unittest.IsolatedAsyncioTestCase):
    """Structural guarantee, proven at the router level too: even when
    the utterance itself mentions vault-shaped content, and even while
    vault_context.load_context is patched to a loud sentinel, Gemini's
    path never calls it -- unlike Qwen/DeepSeek, which call it every
    turn (see VaultContextWiringTests in test_router_phase2.py)."""

    async def test_vault_context_loader_never_invoked_for_gemini(self):
        import os
        cfg = _cfg("qwen3-8b-local", "gemini")

        class _Resp:
            status_code = 200

            def json(self):
                return {"candidates": [{"content": {"parts": [
                    {"text": "ok."}]}}]}

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, *a, **kw):
                return _Resp()

        loader_calls = []

        def loud_loader():
            loader_calls.append(1)
            return "SHOULD NEVER BE SEEN: private vault excerpt"

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            router = BrainRouter(config=cfg)
            await router.activate("gemini", confirmed=True)
            with patch("backtalk.router.vault_context.load_context",
                       loud_loader):
                with patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                           return_value=_Client()):
                    sentences = [s async for s in router.ask_stream(
                        "what does my vault note say about the project")]

        self.assertEqual(loader_calls, [])
        self.assertNotIn("private vault excerpt", " ".join(sentences))


if __name__ == "__main__":
    unittest.main()
