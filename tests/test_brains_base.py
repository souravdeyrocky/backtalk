import unittest

from backtalk.brains.base import BrainAdapter, BrainDisabledError


class BrainAdapterDefaultsTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_by_default(self):
        brain = BrainAdapter()
        self.assertFalse(brain.enabled)

    async def test_disabled_health_is_never_healthy(self):
        brain = BrainAdapter(enabled=False)
        health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("disabled", health.reason)

    async def test_disabled_start_refuses(self):
        brain = BrainAdapter(enabled=False)
        with self.assertRaises(BrainDisabledError):
            await brain.start()

    def test_disabled_status_without_health_check(self):
        brain = BrainAdapter(enabled=False)
        status = brain.status()
        self.assertFalse(status.enabled)
        self.assertFalse(status.healthy)
        self.assertEqual(status.reason, "disabled by configuration")

    async def test_status_before_any_health_check(self):
        brain = BrainAdapter(enabled=True)
        status = brain.status()
        self.assertTrue(status.enabled)
        self.assertFalse(status.healthy)
        self.assertEqual(status.reason, "not checked yet")

    async def test_record_failed_health_updates_status(self):
        brain = BrainAdapter(enabled=True)
        brain.record_failed_health("simulated crash")
        status = brain.status()
        self.assertFalse(status.healthy)
        self.assertEqual(status.reason, "simulated crash")


if __name__ == "__main__":
    unittest.main()
