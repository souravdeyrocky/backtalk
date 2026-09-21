"""Regression: F8/Backtalk is the one Jarvis interface. Nothing it
says about switching brains may ever reference F9, the Local Voice
Bridge, or any other separate voice process/UI/identity -- that
component is legacy/debug-only and never Captain-facing.
"""
import json
import unittest

from backtalk.brains import config as router_config
from backtalk.main import _qwen_confirmation_line
from backtalk.router import BrainRouter

_FORBIDDEN = ("f9", "local voice bridge", "voice-bridge", "voice bridge")


def _assert_single_interface(testcase, text: str):
    low = text.lower()
    for forbidden in _FORBIDDEN:
        testcase.assertNotIn(forbidden, low, f"{text!r} mentions {forbidden!r}")


class SwitchToQwenIsCleanTests(unittest.TestCase):
    def _cfg(self):
        cfg = json.loads(json.dumps(router_config.DEFAULTS))
        cfg["brains"]["qwen3-8b-local"]["enabled"] = True
        return cfg

    def test_switch_confirmation_matches_the_required_clean_format(self):
        router = BrainRouter(config=self._cfg())
        line = _qwen_confirmation_line(router, already_active=False)
        self.assertEqual(line, "Qwen3 8B Local is active, Captain.")

    def test_already_active_confirmation_is_also_clean(self):
        router = BrainRouter(config=self._cfg())
        line = _qwen_confirmation_line(router, already_active=True)
        self.assertEqual(line, "Qwen3 8B Local is already active, Captain.")

    def test_neither_confirmation_ever_mentions_f9_or_voice_bridge(self):
        router = BrainRouter(config=self._cfg())
        for already in (True, False):
            line = _qwen_confirmation_line(router, already_active=already)
            _assert_single_interface(self, line)


class NoUserFacingF9ReferencesAnywhereInBacktalkTests(unittest.TestCase):
    """Static sweep: every capability_summary and label in the brain
    registry -- the actual strings that reach spoken output -- must be
    clean, regardless of which brain or which phrasing path produced
    them."""

    def test_all_brain_labels_and_capability_summaries_are_clean(self):
        cfg = json.loads(json.dumps(router_config.DEFAULTS))
        for bid in cfg["brains"]:
            cfg["brains"][bid]["enabled"] = True
        router = BrainRouter(config=cfg)
        for bid in router.registered_ids():
            brain = router.get(bid)
            _assert_single_interface(self, brain.label)
            _assert_single_interface(self, brain.capability_summary)


if __name__ == "__main__":
    unittest.main()
