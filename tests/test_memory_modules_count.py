"""Regression test for a live F8 failure: "How many memory modules do
you have?" was never intercepted -- a different question from "how
many brains", about the vault/memory infrastructure, not the brain
roster -- so a local brain answered it unfiltered and could invent a
module count or offer to inspect vault sections that don't exist.
"""
import unittest

from backtalk.brains import brain_intent
from backtalk.main import _MEMORY_MODULES_LINE


class MemoryModulesDetectionTests(unittest.TestCase):
    def test_exact_live_transcript(self):
        self.assertEqual(
            brain_intent.detect("How many memory modules do you have?"),
            "memorymodules")

    def test_with_wake_word(self):
        self.assertEqual(
            brain_intent.detect(
                "Jarvis, how many memory modules do you have?"),
            "memorymodules")

    def test_variants(self):
        for text in ("how many memory systems do you have",
                     "how many vaults do you have",
                     "how many memory vaults do you have"):
            self.assertEqual(brain_intent.detect(text), "memorymodules",
                             text)

    def test_distinct_from_brainscount(self):
        self.assertEqual(brain_intent.detect("how many brains do you have"),
                         "brainscount")
        self.assertEqual(
            brain_intent.detect("how many memory modules do you have"),
            "memorymodules")

    def test_never_confused_with_vault_write_intent(self):
        # A real vault-WRITE request ("write about your models in
        # memory") must still refuse via tool_intent's Claude
        # escalation, never answer as if it were this question.
        self.assertNotEqual(
            brain_intent.detect("write about your models in memory"),
            "memorymodules")


class MemoryModulesExactAnswerTests(unittest.TestCase):
    def test_exact_required_text(self):
        self.assertEqual(
            _MEMORY_MODULES_LINE,
            "Captain, I use one shared persistent memory system: "
            "your vault. All four brain modes use the same vault "
            "identity and selected context; they do not have "
            "separate memory vaults.")

    def test_never_claims_unknown_modules_or_offers_to_inspect_sections(self):
        low = _MEMORY_MODULES_LINE.lower()
        for forbidden in ("unknown module", "inspect", "which section",
                          "let me check", "i'll look"):
            self.assertNotIn(forbidden, low)

    def test_never_invents_a_count_other_than_one(self):
        # The only numeral allowed to appear is the word "one" (one
        # shared system) and "four" (the four brain modes) -- no other
        # digit-shaped module count.
        import re
        self.assertIsNone(re.search(r"\b(two|three|five|six|\d+)\b"
                                    r"\s+(memory|module|vault)",
                                    _MEMORY_MODULES_LINE.lower()))


if __name__ == "__main__":
    unittest.main()
