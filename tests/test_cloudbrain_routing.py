"""Regression tests for the deterministic "cloud brain" routing fix --
a real live-F8 failure where none of six spoken cloud-brain aliases
were intercepted, so each reached a local brain and it invented an
answer instead of routing to the confirm-gated Claude switch.
"""
import unittest

from backtalk.main import _cloudbrain_line


class CloudBrainLineTests(unittest.TestCase):
    def test_exact_required_text_when_not_already_on_claude(self):
        self.assertEqual(
            _cloudbrain_line(already_active=False),
            "Claude Agent SDK is the available cloud brain. Switching "
            "gives it access to its Agent tools and may use my Claude "
            "subscription. Shall I switch to Claude? Say confirm to "
            "proceed.")

    def test_already_active_is_a_short_clean_line(self):
        self.assertEqual(_cloudbrain_line(already_active=True),
                         "Already on Claude.")

    def test_never_mentions_gemini(self):
        # Gemini is not configured -- the required response must never
        # claim it as available, or even name it at all.
        for already in (True, False):
            self.assertNotIn(
                "gemini", _cloudbrain_line(already_active=already).lower())

    def test_names_claude_by_its_real_label(self):
        self.assertIn("Claude Agent SDK",
                      _cloudbrain_line(already_active=False))


if __name__ == "__main__":
    unittest.main()
