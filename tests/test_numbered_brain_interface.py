"""Regression tests for the deterministic numbered brain-switch
interface: 1=Qwen, 2=DeepSeek, 3=Claude (confirm-gated), 4=Gemini
(confirm-gated, real activation only when GEMINI_API_KEY is present
and its health check passes -- see test_gemini_brain.py and
test_gemini_recommendation_workflow.py for that gate itself). These
numbers are canonical -- independent of the word-based cloud/Claude
detection entirely.
"""
import json
import unittest

from backtalk.brains import brain_intent
from backtalk.brains import config as router_config
from backtalk.main import (_bare_brain_number, _numbered_brain_menu,
                           _switchnum3_line, _switchnum4_line)
from backtalk.router import BrainRouter


def _cfg(*enabled_ids: str) -> dict:
    cfg = json.loads(json.dumps(router_config.DEFAULTS))
    for bid in enabled_ids:
        cfg["brains"][bid]["enabled"] = True
    return cfg


class NumberDetectionTests(unittest.TestCase):
    def test_each_number_switch_phrasing(self):
        cases = {
            "switch to 1": "switchnum:1", "switch to one": "switchnum:1",
            "switch to brain one": "switchnum:1",
            "use brain 1": "switchnum:1",
            "switch to 2": "switchnum:2", "switch to two": "switchnum:2",
            "use brain two": "switchnum:2",
            "switch to 3": "switchnum:3", "switch to three": "switchnum:3",
            "use brain 3": "switchnum:3",
            "switch to 4": "switchnum:4", "switch to four": "switchnum:4",
            "use brain four": "switchnum:4",
        }
        for phrase, expected in cases.items():
            self.assertEqual(brain_intent.detect(phrase), expected, phrase)

    def test_brain_tree_alias_for_three(self):
        # Real Whisper-shaped mis-hearing: "three" transcribed "tree".
        for phrase in ("switch to brain tree", "switch to tree"):
            self.assertEqual(brain_intent.detect(phrase), "switchnum:3",
                             phrase)

    def test_number_three_never_goes_through_cloud_or_claude_words(self):
        # Numbers are canonical -- proven by using a phrase that
        # contains NEITHER "cloud" nor "claude" at all, yet still
        # resolves to brain 3 exactly.
        self.assertEqual(brain_intent.detect("switch to brain three"),
                         "switchnum:3")
        self.assertNotEqual(brain_intent.detect("switch to brain three"),
                            "useclaude")
        self.assertNotEqual(brain_intent.detect("switch to brain three"),
                            "cloudbrain")

    def test_list_brains_variants(self):
        for phrase in ("list brains", "brain options", "which brains"):
            self.assertEqual(brain_intent.detect(phrase), "listbrains",
                             phrase)

    def test_bare_switch_asks_for_a_number(self):
        for phrase in ("switch", "switch brain", "switch brains"):
            self.assertEqual(brain_intent.detect(phrase), "switchask",
                             phrase)

    def test_named_brain_switches_still_win_over_bare_switch(self):
        # "switch to qwen" must never fall into the bare-switch
        # fallback -- it's checked last on purpose.
        self.assertEqual(brain_intent.detect("switch to qwen"), "useqwen")


class BareBrainNumberHelperTests(unittest.TestCase):
    """_bare_brain_number() is only ever consulted while a switchask
    is actually pending (see main.py's handle()) -- these tests just
    prove the parser itself is correct and conservative."""

    def test_digits_and_words_and_tree(self):
        cases = {"1": "1", "one": "1", "One": "1",
                 "2": "2", "two": "2",
                 "3": "3", "three": "3", "tree": "3",
                 "4": "4", "four": "4"}
        for text, expected in cases.items():
            self.assertEqual(_bare_brain_number(text), expected, text)

    def test_brain_prefix_still_matches(self):
        self.assertEqual(_bare_brain_number("brain three"), "3")
        self.assertEqual(_bare_brain_number("brain 3"), "3")

    def test_ordinary_sentence_containing_a_number_does_not_match(self):
        self.assertIsNone(_bare_brain_number("I need three eggs"))
        self.assertIsNone(_bare_brain_number("that's number one for sure"))

    def test_empty_and_unrelated_text(self):
        self.assertIsNone(_bare_brain_number(""))
        self.assertIsNone(_bare_brain_number("hello jarvis"))


class NumberedMenuTests(unittest.TestCase):
    def test_menu_names_all_four_slots_regardless_of_config(self):
        # Nothing enabled at all -- the menu still names all four.
        router = BrainRouter(config=_cfg())
        menu = _numbered_brain_menu(router)
        self.assertIn("Brain 1, Qwen3 8B Local", menu)
        self.assertIn("Brain 2, DeepSeek R1 8B Local", menu)
        self.assertIn("Brain 3, Claude Agent SDK", menu)
        self.assertIn("Brain 4, Gemini", menu)

    def test_menu_reflects_the_active_brain(self):
        router = BrainRouter(config=_cfg("qwen3-8b-local"))
        router._active_id = "qwen3-8b-local"
        menu = _numbered_brain_menu(router)
        self.assertIn("Brain 1, Qwen3 8B Local, is active", menu)

    def test_menu_says_four_positions_never_undercounts(self):
        # Real live-test complaint: the old brainscount answer said
        # "three configured brain modes" and silently dropped Gemini
        # instead of naming it as the fourth, unconfigured slot.
        router = BrainRouter(config=_cfg("qwen3-8b-local",
                                         "deepseek-r1-8b-local", "claude"))
        menu = _numbered_brain_menu(router)
        self.assertIn("Brain 1,", menu)
        self.assertIn("Brain 2,", menu)
        self.assertIn("Brain 3,", menu)
        self.assertIn("Brain 4, Gemini, is not configured yet", menu)

    def test_menu_never_mentions_non_brain_components(self):
        router = BrainRouter(config=_cfg("qwen3-8b-local"))
        menu = _numbered_brain_menu(router).lower()
        for non_brain in ("face", "hands", "vault", "jarvis eye",
                          "whisper", "kokoro"):
            self.assertNotIn(non_brain, menu)

    def test_menu_never_claims_gemini_is_configured_when_it_is_not(self):
        router = BrainRouter(config=_cfg("qwen3-8b-local"))
        menu = _numbered_brain_menu(router)
        self.assertIn("Brain 4, Gemini, is not configured yet", menu)


class SwitchNum3LineTests(unittest.TestCase):
    def test_exact_required_text_when_not_already_active(self):
        self.assertEqual(
            _switchnum3_line(already_active=False),
            "Brain 3 is Claude Agent SDK. It may use my Claude "
            "subscription and Agent tools. Say confirm to switch.")

    def test_already_active_is_short(self):
        self.assertEqual(_switchnum3_line(already_active=True),
                         "Already on brain 3, Claude Agent SDK.")


class SwitchNum4LineTests(unittest.TestCase):
    """Brain 4 is no longer a permanent refusal -- it asks the same
    way brain 3 does, and _switchnum4_line() itself never touches
    router state or the network (that's router.activate()'s job,
    exercised separately in test_gemini_recommendation_workflow.py)."""

    def test_exact_required_text_when_not_already_active(self):
        self.assertEqual(
            _switchnum4_line(already_active=False),
            "Brain 4 is Gemini, an external free-tier service. I "
            "will send only your next approved prompt, not your "
            "vault. Say confirm to switch.")

    def test_already_active_is_short(self):
        self.assertEqual(_switchnum4_line(already_active=True),
                         "Already on brain 4, Gemini.")

    def test_never_mentions_the_vault_or_files(self):
        line = _switchnum4_line(already_active=False)
        low = line.lower()
        for forbidden in ("vault context", "daily note", "profile",
                          "cctv", "password", "file contents"):
            self.assertNotIn(forbidden, low)


if __name__ == "__main__":
    unittest.main()
