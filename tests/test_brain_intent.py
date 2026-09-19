"""Regression tests for the three exact real-F8-transcript failures
this module exists to fix, plus the surrounding safety properties."""
import unittest

from backtalk.brains.brain_intent import detect


class RealFieldFailureRegressionTests(unittest.TestCase):
    """The exact three transcripts from the failed live test, verbatim."""

    def test_transcript_1_which_brain(self):
        self.assertEqual(
            detect("Hello Jarvis, which brain are you using?"),
            "whichbrain")

    def test_transcript_2_switch_to_deepseek(self):
        self.assertEqual(
            detect("Jarvis, can you switch to DeepSeek R1 8B?"),
            "usedeepseek")

    def test_transcript_3_switch_to_cloud_brain(self):
        self.assertEqual(
            detect("Jarvis, can you switch to cloud brain?"),
            "cloudbrain")


class DetectionVariantTests(unittest.TestCase):
    def test_which_brain_variants(self):
        for text in ("which brain are you using",
                     "what brain is this",
                     "Jarvis, what brain are you on right now?"):
            self.assertEqual(detect(text), "whichbrain", text)

    def test_deepseek_variants(self):
        for text in ("switch to deepseek",
                     "can you use deep seek instead",
                     "Jarvis can you switch to deep reasoning",
                     "hey jarvis switch to r1 8b"):
            self.assertEqual(detect(text), "usedeepseek", text)

    def test_qwen_variants(self):
        for text in ("switch to qwen",
                     "Jarvis, can you use qwen again",
                     "go to the local brain"):
            self.assertEqual(detect(text), "useqwen", text)

    def test_claude_variants(self):
        for text in ("switch to claude",
                     "Jarvis, can you use claude"):
            self.assertEqual(detect(text), "useclaude", text)

    def test_cloud_variants(self):
        for text in ("switch to cloud brain",
                     "can you use the cloud model",
                     "switch to the cloud"):
            self.assertEqual(detect(text), "cloudbrain", text)


class CloudBrainNeverActivatesClaudeDirectlyTests(unittest.TestCase):
    def test_cloud_alone_is_not_useclaude(self):
        self.assertNotEqual(detect("switch to cloud brain"), "useclaude")

    def test_explicit_claude_still_wins_when_both_named(self):
        # If someone actually says "claude", that's an unambiguous,
        # exact request -- it must still resolve to useclaude, not get
        # swallowed by the cloud-brain path.
        self.assertEqual(detect("switch to claude, the cloud one"),
                         "useclaude")


class NoFalsePositiveOnOrdinaryConversationTests(unittest.TestCase):
    def test_plain_questions_dont_trigger_anything(self):
        for text in ("what's a good name for a dog",
                     "explain how photosynthesis works",
                     "tell me a joke"):
            self.assertIsNone(detect(text), text)

    def test_asking_about_a_brain_without_a_switch_verb_is_not_a_command(self):
        # "what is deepseek" is a QUESTION, not a switch request -- no
        # switch verb present, so this must fall through to the model
        # rather than silently changing the active brain.
        self.assertIsNone(detect("what is deepseek anyway"))


if __name__ == "__main__":
    unittest.main()
