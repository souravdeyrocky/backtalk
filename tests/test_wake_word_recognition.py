"""Regression tests for the exact transcripts from a live F8 test
where Whisper transcribed "Jarvis" as "Java"/"Javis": exact wake-word
matching was unacceptable, so the address-prefix strip now tolerates
both, while a real Java programming-language question stays untouched.
"""
import unittest

from backtalk.brains import brain_intent
from backtalk.main import _MEMORY_BRAIN_LINE


class ExactLiveTranscriptTests(unittest.TestCase):
    """The six exact transcripts from the failed live test, verbatim."""

    def test_java_switch(self):
        self.assertEqual(brain_intent.detect("Java switch"), "switchask")

    def test_javis_switch(self):
        self.assertEqual(brain_intent.detect("Javis switch"), "switchask")

    def test_jarvis_switch_your_memory_brain(self):
        self.assertEqual(
            brain_intent.detect("Jarvis switch your memory brain"),
            "switchmemorybrain")

    def test_java_switch_the_brain(self):
        self.assertEqual(
            brain_intent.detect("Java switch the brain"), "switchask")

    def test_jarvis_can_you_switch_to_deep_seek(self):
        self.assertEqual(
            brain_intent.detect("Jarvis can you switch to deep-seek?"),
            "usedeepseek")

    def test_which_model_you_are_running_on_jarvis(self):
        self.assertEqual(
            brain_intent.detect("Which model you are running on Jarvis?"),
            "whichbrain")


class RealJavaQuestionsPreservedTests(unittest.TestCase):
    """The wake-word alias must never swallow an actual question about
    the Java programming language."""

    def test_explain_the_java_switch_statement(self):
        self.assertIsNone(
            brain_intent.detect("Explain the Java switch statement"))

    def test_java_switch_case_syntax_question(self):
        self.assertIsNone(brain_intent.detect(
            "Java switch case syntax, how does it work"))

    def test_java_mentioned_mid_sentence_is_never_touched(self):
        self.assertIsNone(brain_intent.detect(
            "I'm learning Java and switch statements confuse me"))


class WakeWordAliasVariantsTests(unittest.TestCase):
    def test_java_and_javis_both_work_for_other_verbs(self):
        self.assertEqual(brain_intent.detect("Java, which brain are you "
                                             "using"), "whichbrain")
        self.assertEqual(brain_intent.detect("Javis, switch to deepseek"),
                         "usedeepseek")

    def test_real_jarvis_still_works_unchanged(self):
        self.assertEqual(brain_intent.detect("Jarvis, switch to qwen"),
                         "useqwen")


class DeepSeekHyphenVariantTests(unittest.TestCase):
    """Real live-test miss: a hyphen between "deep" and "seek" never
    matched (\\s* only allows whitespace, not a dash)."""

    def test_hyphenated_form(self):
        self.assertEqual(brain_intent.detect("switch to deep-seek"),
                         "usedeepseek")

    def test_spaced_form_still_works(self):
        self.assertEqual(brain_intent.detect("switch to deep seek"),
                         "usedeepseek")

    def test_joined_form_still_works(self):
        self.assertEqual(brain_intent.detect("switch to deepseek"),
                         "usedeepseek")


class SwitchTheBrainVariantTests(unittest.TestCase):
    """"switch the brain" (with "the") must open the numbered menu,
    same as bare "switch"/"switch brain"."""

    def test_switch_the_brain(self):
        self.assertEqual(brain_intent.detect("switch the brain"),
                         "switchask")

    def test_bare_switch_still_works(self):
        self.assertEqual(brain_intent.detect("switch"), "switchask")

    def test_switch_brain_still_works(self):
        self.assertEqual(brain_intent.detect("switch brain"), "switchask")


class MemoryBrainExplanationTests(unittest.TestCase):
    def test_switch_your_memory_brain(self):
        self.assertEqual(
            brain_intent.detect("switch your memory brain"),
            "switchmemorybrain")

    def test_switch_my_memory_brain(self):
        self.assertEqual(
            brain_intent.detect("switch my memory brain"),
            "switchmemorybrain")

    def test_switch_memory_brain_bare(self):
        self.assertEqual(
            brain_intent.detect("switch memory brain"),
            "switchmemorybrain")

    def test_exact_required_explanation_text(self):
        self.assertEqual(
            _MEMORY_BRAIN_LINE,
            "Your vault memory is shared by all brain modes; it is "
            "not a separate brain. Say switch to 1, 2, 3, or 4.")

    def test_memory_brain_switch_never_confused_with_bare_switch(self):
        self.assertNotEqual(
            brain_intent.detect("switch your memory brain"), "switchask")


class WhichModelPhrasingTests(unittest.TestCase):
    """"which model"/"what model" must resolve exactly like "which
    brain"/"what brain" -- the router's real active-brain answer, never
    a hedge like "e.g., Qwen, DeepSeek, or similar."""

    def test_which_model_variants(self):
        for text in ("which model are you running",
                     "what model is this",
                     "which model you are running on Jarvis"):
            self.assertEqual(brain_intent.detect(text), "whichbrain", text)


if __name__ == "__main__":
    unittest.main()
