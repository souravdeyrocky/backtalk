"""Phase 2 router-level proofs that don't need main.py or a live
connection to anything:

1. A tool-shaped request on a no-tools brain is refused honestly and
   never reaches Ollama, and never changes the active brain (the
   "Qwen-side denied test" standing in for a live Claude permission
   check, per Captain's instruction not to spend real Claude usage
   just to prove the gate).
2. The permission-gate CLOSURE built in main.py is threaded, unchanged,
   all the way from BrainRouter through ClaudeBrain into WarmBrain's
   constructor -- proven by mocking WarmBrain itself, so this never
   opens a real Claude Agent SDK connection or spends any usage.
3. Vault context is loaded fresh per turn for a local brain, and never
   attempted for Claude (which has its own, separate, real filesystem
   access and needs no excerpt).
"""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backtalk.brains import config as router_config
from backtalk.brains.base import BrainHealth
from backtalk.router import BrainRouter


def _all_disabled_cfg() -> dict:
    return json.loads(json.dumps(router_config.DEFAULTS))


class ToolIntentRefusalTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_request_on_qwen_refuses_without_calling_ollama(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["qwen3-8b-local"]["enabled"] = True
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        await router.activate("qwen3-8b-local")

        with patch("backtalk.brains.ollama_brain.httpx.AsyncClient") as mock_client:
            sentences = [s async for s in router.ask_stream(
                "please edit the config file and save it")]

        self.assertEqual(len(sentences), 1)
        self.assertIn("Claude Agent tools", sentences[0])
        mock_client.assert_not_called()
        # Refusing a request must never itself change the active brain.
        self.assertEqual(router.active_id, "qwen3-8b-local")

    async def test_ordinary_question_on_qwen_still_reaches_ollama(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["qwen3-8b-local"]["enabled"] = True
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        await router.activate("qwen3-8b-local")

        async def _fake_ask_stream(utterance):
            yield "a harmless local reply."

        qwen.ask_stream = _fake_ask_stream
        sentences = [s async for s in router.ask_stream("what's 2 plus 2")]
        self.assertEqual(sentences, ["a harmless local reply."])


class PermissionGateWiringTests(unittest.IsolatedAsyncioTestCase):
    """Proves the can_use_tool closure reaches WarmBrain's constructor
    unchanged, WITHOUT ever constructing a real ClaudeSDKClient or
    spending any Claude usage -- WarmBrain itself is mocked out."""

    async def test_can_use_tool_reaches_warmbrain_construction(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["claude"]["enabled"] = True

        async def fake_gate(tool, tool_input, ctx):
            return "unused-in-this-test"

        with patch("backtalk.brains.claude_brain.WarmBrain") as MockWarm:
            instance = MockWarm.return_value
            instance.start = AsyncMock()
            router = BrainRouter(config=cfg, can_use_tool=fake_gate)
            claude = router.get("claude")
            claude.health = AsyncMock(return_value=BrainHealth(healthy=True))
            await router.activate("claude", confirmed=True)

        MockWarm.assert_called_once()
        _, kwargs = MockWarm.call_args
        self.assertIs(kwargs["can_use_tool"], fake_gate)
        instance.start.assert_awaited_once()

    async def test_resume_id_reaches_warmbrain_construction(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["claude"]["enabled"] = True
        with patch("backtalk.brains.claude_brain.WarmBrain") as MockWarm:
            instance = MockWarm.return_value
            instance.start = AsyncMock()
            router = BrainRouter(config=cfg, resume_id="saved-session-123")
            claude = router.get("claude")
            claude.health = AsyncMock(return_value=BrainHealth(healthy=True))
            await router.activate("claude", confirmed=True)
        _, kwargs = MockWarm.call_args
        self.assertEqual(kwargs["resume_id"], "saved-session-123")


class VaultContextWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_loads_vault_context_fresh_each_turn(self):
        cfg = _all_disabled_cfg()
        cfg["brains"]["qwen3-8b-local"]["enabled"] = True
        calls = []

        def fake_loader():
            calls.append(1)
            return f"fake vault excerpt #{len(calls)}"

        with patch("backtalk.router.vault_context.load_context", fake_loader):
            router = BrainRouter(config=cfg)
            qwen = router.get("qwen3-8b-local")
            qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
            qwen.start = AsyncMock()
            await router.activate("qwen3-8b-local")

            captured_payloads = []

            class _FakeStreamCtx:
                async def __aenter__(self):
                    class _Resp:
                        def raise_for_status(self):
                            pass

                        async def aiter_lines(self):
                            yield '{"message": {"content": "ok."}, "done": true}'

                    return _Resp()

                async def __aexit__(self, *exc):
                    return False

            class _FakeClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *exc):
                    return False

                def stream(self, method, url, json):
                    captured_payloads.append(json)
                    return _FakeStreamCtx()

            with patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=_FakeClient()):
                async for _ in router.ask_stream("first turn"):
                    pass
                async for _ in router.ask_stream("second turn"):
                    pass

        self.assertEqual(len(calls), 2, "context loader must run every turn")
        self.assertIn("fake vault excerpt #1",
                      captured_payloads[0]["messages"][0]["content"])
        self.assertIn("fake vault excerpt #2",
                      captured_payloads[1]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
