import unittest
from unittest import mock

import httpx

from backtalk.brains.ollama_brain import DeepSeekBrain, QwenBrain
from backtalk.identity import PHONE_REPLY_CAPABILITY_RULE


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient so tests never touch a real
    Ollama instance, running or not."""

    def __init__(self, tags_response=None, raise_exc=None):
        self._tags_response = tags_response
        self._raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        if self._raise_exc:
            raise self._raise_exc
        return _FakeResponse(self._tags_response)


class OllamaBrainIdentityTests(unittest.TestCase):
    def test_qwen_identity(self):
        self.assertEqual(QwenBrain.id, "qwen3-8b-local")
        self.assertEqual(QwenBrain.model, "qwen3:8b")
        self.assertFalse(QwenBrain.requires_tools)
        self.assertFalse(QwenBrain.requires_confirm_to_switch)

    def test_deepseek_identity(self):
        self.assertEqual(DeepSeekBrain.id, "deepseek-r1-8b-local")
        self.assertEqual(DeepSeekBrain.model, "deepseek-r1:8b")


class OllamaBrainHealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_never_calls_ollama(self):
        brain = QwenBrain(enabled=False)
        health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("disabled", health.reason)

    async def test_enabled_but_model_missing(self):
        brain = QwenBrain(enabled=True)
        fake = _FakeAsyncClient(
            tags_response={"models": [{"name": "llama3:8b"}]})
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("qwen3:8b", health.reason)

    async def test_enabled_and_model_installed(self):
        brain = QwenBrain(enabled=True)
        fake = _FakeAsyncClient(
            tags_response={"models": [{"name": "qwen3:8b"}]})
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            health = await brain.health()
        self.assertTrue(health.healthy)

    async def test_ollama_unreachable(self):
        brain = QwenBrain(enabled=True)
        fake = _FakeAsyncClient(raise_exc=httpx.HTTPError("connection refused"))
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("unreachable", health.reason)

    async def test_deepseek_not_installed_is_honest_not_a_fallback(self):
        brain = DeepSeekBrain(enabled=True)
        fake = _FakeAsyncClient(
            tags_response={"models": [{"name": "qwen3:8b"}]})
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("deepseek-r1:8b", health.reason)


class _FakeStreamResponse:
    """Stands in for the object httpx.AsyncClient.stream()'s async
    context manager yields -- just enough to let ask_stream() run its
    preamble-construction code and reach `done` on the first line."""

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        yield '{"message": {"content": "Okay."}, "done": true}'


class _StreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeStreamingClient:
    """Captures the JSON payload ask_stream() posts, so the test can
    inspect the exact system-prompt string sent to Ollama without a
    live server."""

    def __init__(self):
        self.last_payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json=None):
        self.last_payload = json
        return _StreamCtx(_FakeStreamResponse())


class OllamaBrainPhoneCapabilityTests(unittest.IsolatedAsyncioTestCase):
    """2026-09-21 field defect: Jarvis wrongly told Captain it could not
    speak through the paired phone app, even though the phone bridge
    was connected and the reply path genuinely works. Fix: the local
    brain's system prompt now says so, but ONLY when a phone is
    actually paired and reachable right now -- never unconditionally,
    since an idle/never-paired bridge makes the same claim false."""

    async def _run_and_capture(self, brain, phone_available: bool) -> str:
        fake = _FakeStreamingClient()
        with mock.patch(
                "backtalk.brains.ollama_brain.phone_bridge.phone_reply_available",
                return_value=phone_available), \
            mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            async for _ in brain.ask_stream("hello"):
                pass
        return fake.last_payload["messages"][0]["content"]

    async def test_phone_rule_present_when_phone_connected(self):
        brain = QwenBrain(enabled=True)
        system_prompt = await self._run_and_capture(brain, True)
        self.assertIn(PHONE_REPLY_CAPABILITY_RULE, system_prompt)

    async def test_phone_rule_absent_when_no_phone_connected(self):
        brain = QwenBrain(enabled=True)
        system_prompt = await self._run_and_capture(brain, False)
        self.assertNotIn(PHONE_REPLY_CAPABILITY_RULE, system_prompt)


if __name__ == "__main__":
    unittest.main()
