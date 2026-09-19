import unittest
from unittest import mock

import httpx

from backtalk.brains.ollama_brain import DeepSeekBrain, QwenBrain


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


if __name__ == "__main__":
    unittest.main()
