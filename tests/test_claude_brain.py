import unittest

from backtalk.brains.base import BrainDisabledError
from backtalk.brains.claude_brain import ClaudeBrain


class ClaudeBrainGateTests(unittest.IsolatedAsyncioTestCase):
    """These tests must never construct a real claude_agent_sdk session --
    every case here stays on the disabled side of the gate, which raises
    before WarmBrain is ever touched."""

    async def test_disabled_refuses_to_start(self):
        brain = ClaudeBrain(enabled=False)
        with self.assertRaises(BrainDisabledError):
            await brain.start()

    async def test_disabled_refuses_ask_stream(self):
        brain = ClaudeBrain(enabled=False)
        with self.assertRaises(BrainDisabledError):
            async for _ in brain.ask_stream("hi"):
                pass

    def test_requires_confirm_and_tools(self):
        brain = ClaudeBrain()
        self.assertTrue(brain.requires_confirm_to_switch)
        self.assertTrue(brain.requires_tools)

    async def test_disabled_status_is_accurate_without_connecting(self):
        brain = ClaudeBrain(enabled=False)
        status = brain.status()
        self.assertFalse(status.enabled)
        self.assertFalse(status.healthy)


if __name__ == "__main__":
    unittest.main()
