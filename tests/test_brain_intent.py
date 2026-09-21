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

    def test_transcript_4_qn3_alias_from_live_field_test(self):
        # Real live-test failure: Whisper transcribed "Qwen" as "QN3".
        # This fell through un-intercepted and DeepSeek (still active)
        # gave a confused, invented answer about not being able to
        # switch models itself.
        self.assertEqual(detect("Jarvis, can you switch to QN3?"),
                         "useqwen")


class QwenAliasNormalizationTests(unittest.TestCase):
    """The exact alias list from the fix requirement: qwen, qwen3,
    qwen 3, q n 3, qn3, q n three."""

    def test_all_required_aliases_switch_to_qwen(self):
        for phrase in ("switch to qwen",
                       "switch to qwen3",
                       "switch to qwen 3",
                       "switch to q n 3",
                       "switch to qn3",
                       "switch to q n three"):
            self.assertEqual(detect(phrase), "useqwen", phrase)


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


class BrainCountInterceptionTests(unittest.TestCase):
    """Regression for a real field-test failure: "how many brains do
    you have?" reached Qwen unintercepted and it invented an answer.
    This must resolve deterministically, distinct from "which brain
    are you ON" -- the answer text itself is built in main.py, live
    from router state, tested separately."""

    def test_exact_field_transcript(self):
        self.assertEqual(detect("How many brains do you have?"),
                         "brainscount")

    def test_variants(self):
        for text in ("how many brain modes do you have",
                     "what brains do you have",
                     "what brains are there",
                     "Jarvis, how many brains do you have?"):
            self.assertEqual(detect(text), "brainscount", text)

    def test_distinct_from_which_brain(self):
        self.assertEqual(detect("which brain are you using"), "whichbrain")
        self.assertEqual(detect("how many brains do you have"),
                         "brainscount")


class CloudBrainAliasRegressionTests(unittest.TestCase):
    """Regression for a real live-F8 failure: none of these six exact
    spoken aliases ever matched _CLOUD, so each reached a local brain
    unintercepted and it invented an answer instead of routing to the
    deterministic Claude-confirm flow."""

    def test_switch_to_cloud(self):
        self.assertEqual(detect("switch to cloud"), "cloudbrain")

    def test_switch_to_the_cloud(self):
        self.assertEqual(detect("switch to the cloud"), "cloudbrain")

    def test_switch_to_cloud_brain(self):
        self.assertEqual(detect("switch to cloud brain"), "cloudbrain")

    def test_switch_to_cloud_agent(self):
        self.assertEqual(detect("switch to cloud agent"), "cloudbrain")

    def test_switch_to_the_third_reasoning_brain_cloud(self):
        self.assertEqual(
            detect("switch to the third reasoning brain cloud"),
            "cloudbrain")

    def test_use_the_cloud_brain(self):
        self.assertEqual(detect("use the cloud brain"), "cloudbrain")

    def test_none_of_these_six_ever_reach_qwen_or_deepseek(self):
        # "Never reach Qwen or DeepSeek" means detect() must claim
        # ALL six itself (a non-None verb short-circuits main.py's
        # dispatch before the utterance ever reaches a brain).
        aliases = ("switch to cloud", "switch to the cloud",
                  "switch to cloud brain", "switch to cloud agent",
                  "switch to the third reasoning brain cloud",
                  "use the cloud brain")
        for a in aliases:
            verb = detect(a)
            self.assertIsNotNone(verb, a)
            self.assertNotIn(verb, ("useqwen", "usedeepseek"), a)


class BrainTopicInterceptionTests(unittest.TestCase):
    """Regression for a real field-test failure: questions/requests
    about Jarvis's configured brains or installed reasoning models,
    phrased more loosely than "how many brains", reached a local brain
    and it invented model names. Must resolve to the same deterministic
    "brainscount" answer as the narrower _BRAIN_COUNT phrasings."""

    def test_write_about_your_brains_is_an_inventory_question(self):
        self.assertEqual(detect("write about your brains"), "brainscount")

    def test_write_about_your_models_is_an_inventory_question(self):
        self.assertEqual(detect("write about your models"), "brainscount")

    def test_reasoning_models_variants(self):
        for text in ("what reasoning models do you have",
                     "what models do you have",
                     "which models do you have",
                     "tell me about your brains",
                     "tell me about your models",
                     "list your brains",
                     "list your models",
                     "your configured brains",
                     "your installed models"):
            self.assertEqual(detect(text), "brainscount", text)

    def test_vault_target_present_is_never_claimed_as_inventory(self):
        # "write about your models IN MEMORY" is a vault-write request,
        # not an inventory question -- this module must NOT intercept
        # it, so it falls through to tool_intent's Claude-escalation
        # refusal instead of Jarvis pretending to answer or save.
        self.assertIsNone(detect("write about your models in memory"))
        self.assertIsNone(detect("write about your brains in your world"))


if __name__ == "__main__":
    unittest.main()
