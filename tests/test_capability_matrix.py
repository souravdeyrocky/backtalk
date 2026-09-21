import unittest

from backtalk.brains.claude_brain import ClaudeBrain
from backtalk.brains.gemini_brain import GeminiBrain
from backtalk.brains.ollama_brain import DeepSeekBrain, QwenBrain

_FORBIDDEN_CLAIMS = ("can edit files", "can run commands",
                    "can browse", "can write to your vault")


class LocalBrainCapabilityHonestyTests(unittest.TestCase):
    """The house rule this file exists to enforce: a brain must never
    CLAIM a capability it doesn't actually have behind a permission
    gate. Qwen/DeepSeek get vault-read and conversation only."""

    def _assert_no_false_claims(self, summary: str):
        low = summary.lower()
        for claim in _FORBIDDEN_CLAIMS:
            self.assertNotIn(claim, low,
                             f"{summary!r} falsely claims {claim!r}")

    def test_qwen_capability_summary_is_honest(self):
        brain = QwenBrain(enabled=True)
        self._assert_no_false_claims(brain.capability_summary)
        self.assertIn("cannot edit files", brain.capability_summary)
        self.assertIn("vault", brain.capability_summary)

    def test_deepseek_capability_summary_is_honest(self):
        brain = DeepSeekBrain(enabled=True)
        self._assert_no_false_claims(brain.capability_summary)

    def test_gemini_capability_summary_names_approval_requirement(self):
        brain = GeminiBrain()
        self._assert_no_false_claims(brain.capability_summary)
        low = brain.capability_summary.lower()
        self.assertIn("approv", low)
        self.assertIn("no vault context", low)

    def test_claude_capability_summary_names_real_tools_and_gate(self):
        brain = ClaudeBrain()
        low = brain.capability_summary.lower()
        self.assertIn("file edit", low)
        self.assertIn("gated", low)

    def test_status_contract_carries_capability_summary(self):
        brain = QwenBrain(enabled=True)
        status = brain.status()
        self.assertEqual(status.capability_summary, brain.capability_summary)


if __name__ == "__main__":
    unittest.main()
