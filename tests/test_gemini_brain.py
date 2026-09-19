import os
import unittest
from unittest import mock

from backtalk.brains.base import BrainDisabledError
from backtalk.brains.gemini_brain import GeminiBrain


class GeminiStubTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_by_default(self):
        brain = GeminiBrain()
        self.assertFalse(brain.enabled)
        health = await brain.health()
        self.assertFalse(health.healthy)

    async def test_requires_confirm_to_switch(self):
        brain = GeminiBrain()
        self.assertTrue(brain.requires_confirm_to_switch)

    async def test_enabled_without_key_reports_honestly(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            brain = GeminiBrain(enabled=True)
            health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("GEMINI_API_KEY", health.reason)

    async def test_enabled_with_key_is_still_a_stub(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("stub", health.reason)

    async def test_ask_stream_always_refuses_in_phase_1(self):
        brain = GeminiBrain(enabled=True)
        with self.assertRaises(BrainDisabledError):
            async for _ in brain.ask_stream("hello"):
                pass


if __name__ == "__main__":
    unittest.main()
