"""Regression tests for the shared speech sanitizer. Several cases use
real text observed in live Qwen/DeepSeek output during field testing
(bold names, numbered reasoning steps, LaTeX algebra), not synthetic
examples -- this is what actually came out of the model."""
import unittest

from backtalk.speech_format import to_speech


class MarkdownStrippingTests(unittest.TestCase):
    def test_bold_asterisks_removed(self):
        # Real observed output: "Your name is **Sourav Dey** --
        # always addressed as 'Captain.'"
        out = to_speech("Your name is **Sourav Dey**.")
        self.assertNotIn("*", out)
        self.assertIn("Sourav Dey", out)

    def test_bold_underscores_removed(self):
        out = to_speech("This is __very__ important.")
        self.assertNotIn("_", out)
        self.assertIn("very", out)

    def test_italic_asterisk_removed(self):
        out = to_speech("The word *strawberry* has three Rs.")
        self.assertNotIn("*", out)
        self.assertIn("strawberry", out)

    def test_underscored_identifier_not_mangled(self):
        # Deliberately NOT touched -- single-underscore italic is
        # skipped specifically because it collides with identifiers.
        out = to_speech("Set max_tokens to a higher value.")
        self.assertIn("max_tokens", out)

    def test_heading_marker_stripped(self):
        out = to_speech("## Section Title\nSome text follows.")
        self.assertNotIn("#", out)
        self.assertIn("Section Title", out)

    def test_bullet_points_become_plain_text(self):
        # Real observed shape: "- **Benzo** is a 10-year-old Pomeranian."
        out = to_speech("- Benzo is a dog.\n- Tyson is also a dog.")
        self.assertNotIn("-", out.split("Benzo")[0])
        self.assertIn("Benzo is a dog", out)
        self.assertIn("Tyson is also a dog", out)

    def test_numbered_list_markers_stripped(self):
        out = to_speech("1. First step.\n2. Second step.")
        self.assertIn("First step", out)
        self.assertIn("Second step", out)
        self.assertNotRegex(out, r"\b1\.\s")

    def test_numbered_list_becomes_ordinal_words_not_silent_deletion(self):
        # A real complaint: "1. Buy milk 2. Walk dog" used to collapse
        # to "Buy milk Walk dog" with no transition at all -- correct
        # written-text stripping, unnatural spoken aloud.
        out = to_speech("1. Buy milk.\n2. Walk dog.\n3. Call mom.")
        self.assertEqual(out, "First, Buy milk. Second, Walk dog. "
                             "Third, Call mom.")

    def test_ordinal_words_cover_up_to_twenty(self):
        out = to_speech("20. Twentieth item.")
        self.assertTrue(out.startswith("Twentieth,"))

    def test_beyond_twenty_falls_back_to_next(self):
        out = to_speech("21. Item beyond the table.")
        self.assertTrue(out.startswith("Next,"))

    def test_numbered_marker_with_parenthesis_also_ordinalized(self):
        out = to_speech("1) First option.\n2) Second option.")
        self.assertTrue(out.startswith("First,"))
        self.assertIn("Second,", out)

    def test_horizontal_rule_removed(self):
        out = to_speech("Above the line.\n---\nBelow the line.")
        self.assertNotIn("---", out)

    def test_code_fence_dropped(self):
        out = to_speech("Here is code:\n```python\nprint('hi')\n```\nDone.")
        self.assertNotIn("```", out)
        self.assertNotIn("print(", out)

    def test_inline_code_unwrapped(self):
        out = to_speech("Run `uv sync` to install.")
        self.assertNotIn("`", out)
        self.assertIn("uv sync", out)

    def test_markdown_link_speaks_just_the_text(self):
        out = to_speech("See [the docs](https://example.com/docs) for more.")
        self.assertNotIn("http", out)
        self.assertNotIn("[", out)
        self.assertIn("the docs", out)

    def test_wikilink_speaks_just_the_note_name(self):
        out = to_speech("Logged in [[Captain Profile]] already.")
        self.assertNotIn("[[", out)
        self.assertIn("Captain Profile", out)

    def test_bare_url_replaced(self):
        out = to_speech("Visit https://example.com/page?x=1 for info.")
        self.assertNotIn("http", out)


class LatexStrippingTests(unittest.TestCase):
    def test_block_math_delimiters_stripped(self):
        # Real observed output from the bat-and-ball benchmark.
        out = to_speech("$$\nx + (x + 1.00) = 1.10\n$$")
        self.assertNotIn("$", out)
        self.assertIn("1.10", out)

    def test_inline_math_delimiters_stripped(self):
        out = to_speech("The value $x = 0.05$ is the answer.")
        self.assertNotIn("$", out)
        self.assertIn("0.05", out)


class EmojiStrippingTests(unittest.TestCase):
    def test_emoji_removed(self):
        out = to_speech("Let me know if you need help! 🚀")
        self.assertNotIn("🚀", out)
        self.assertIn("Let me know if you need help", out)

    def test_paw_emoji_removed(self):
        out = to_speech("Woof! 🐾")
        self.assertNotIn("🐾", out)


class PathStrippingTests(unittest.TestCase):
    def test_windows_path_becomes_basename(self):
        out = to_speech(r"Edit D:\JARVISFullstack\backtalk\backtalk.json now.")
        self.assertNotIn("\\", out)
        self.assertIn("backtalk", out)

    def test_posix_path_becomes_basename(self):
        out = to_speech("Check /home/user/project/config.py first.")
        self.assertNotIn("/", out)
        self.assertIn("config", out)


class IdempotenceAndCleanProseTests(unittest.TestCase):
    def test_plain_prose_passes_through_unchanged(self):
        # Claude's own output, which already follows DISCIPLINE, should
        # never be altered by this sanitizer -- it's a no-op backstop.
        clean = ("Your name is Sourav Dey, always addressed as Captain. "
                 "The ball costs five cents.")
        self.assertEqual(to_speech(clean), clean)

    def test_sanitizing_twice_is_the_same_as_once(self):
        raw = "**Bold** and *italic* and `code` and 🚀 and [[Note]]."
        once = to_speech(raw)
        twice = to_speech(once)
        self.assertEqual(once, twice)

    def test_empty_string_stays_empty(self):
        self.assertEqual(to_speech(""), "")

    def test_whitespace_only_becomes_empty(self):
        self.assertEqual(to_speech("   \n\n  "), "")


class GenericClosingStrippedTests(unittest.TestCase):
    """Regression for a real field-test failure: Qwen/DeepSeek tacked a
    service-desk sign-off onto replies that had already fully answered
    the request. The shared renderer is the backstop -- even if a
    local model ignores its system prompt, none of these four exact
    phrases (or an obvious variant) ever reach Kokoro."""

    def test_how_may_i_assist_you_today_is_dropped(self):
        self.assertEqual(to_speech("How may I assist you today?"), "")

    def test_what_would_you_like_to_explore_next_is_dropped(self):
        self.assertEqual(
            to_speech("What would you like to explore next?"), "")

    def test_let_me_know_if_you_need_anything_else_is_dropped(self):
        self.assertEqual(
            to_speech("Let me know if you need anything else."), "")

    def test_how_can_i_assist_you_is_dropped(self):
        self.assertEqual(to_speech("How can I assist you?"), "")

    def test_variant_with_trailing_address_still_dropped(self):
        self.assertEqual(
            to_speech("How can I assist you, Captain?"), "")

    def test_closing_dropped_but_real_content_sentence_survives(self):
        # to_speech() is called once PER COMPLETE SENTENCE in real use
        # (router.ask_stream), never on a pre-joined multi-sentence
        # blob -- so this mirrors that: the real-content sentence and
        # the closing sentence are two separate calls, and only the
        # closing one should vanish.
        content = to_speech("The capital of France is Paris.")
        closing = to_speech("How may I assist you today?")
        self.assertEqual(content, "The capital of France is Paris.")
        self.assertEqual(closing, "")

    def test_real_clarifying_question_is_never_swept_up(self):
        # A genuine clarification is never one of the four fixed
        # phrases, so it must survive untouched.
        out = to_speech("Which daily note do you mean -- today's or "
                        "yesterday's?")
        self.assertIn("Which daily note", out)

    def test_second_live_failure_accomplish_with_local_ai_assistant(self):
        # Real second live-F8 failure: this exact wording was never
        # one of the original four fixed phrases, so the first version
        # of this filter (a literal phrase list) let it straight
        # through. The fix generalized to the SHAPE of a service-desk
        # sign-off instead of chasing each new paraphrase.
        self.assertEqual(
            to_speech("What would you like to accomplish with your "
                      "local AI assistant?"), "")

    def test_closing_shape_catches_unlisted_rephrasings(self):
        # Not one of the five exact phrases anywhere in this file, but
        # the same shape -- proves the fix generalizes rather than
        # only patching the specific wording reported live.
        for variant in (
                "How may I help you today?",
                "What would you like to explore with your local AI "
                "assistant?",
                "Is there anything else I can help with?",
                "Is there anything else I can assist you with, "
                "Captain?"):
            self.assertEqual(to_speech(variant), "", variant)

    def test_all_bans_survive_case_and_punctuation_variation(self):
        for variant in (
                "HOW MAY I ASSIST YOU TODAY???",
                "how may i assist you today",
                "How... may I assist you... today?!",
                "let me know if you need anything else!!",
                "How can I assist you, Captain??"):
            self.assertEqual(to_speech(variant), "", variant)

    def test_third_live_failure_let_me_know_how_i_can_assist(self):
        self.assertEqual(to_speech("Let me know how I can assist."), "")

    def test_third_live_failure_assist_further_variant(self):
        self.assertEqual(
            to_speech("Let me know how I can assist further."), "")

    def test_third_live_failure_example_or_clarification(self):
        self.assertEqual(
            to_speech("Let me know if you'd like an example or "
                      "clarification."), "")

    def test_uncontracted_you_would_like_variant_also_dropped(self):
        self.assertEqual(
            to_speech("Let me know if you would like an example."), "")


if __name__ == "__main__":
    unittest.main()
