"""Regression tests for the local pronunciation-correction layer --
the exact seed entries, whole-word/phrase safety (never corrupt a
path, URL, password, ID, or code fragment), and live-reload behavior.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backtalk import pronunciation


def _with_dict(entries: dict):
    """Context manager-style helper: point pronunciation at a temp
    file with exactly these entries, for the duration of the block."""
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                    encoding="utf-8")
    json.dump(entries, f)
    f.close()
    return Path(f.name)


class SeedEntryTests(unittest.TestCase):
    """The exact entries and target pronunciations from the task."""

    def setUp(self):
        path = _with_dict({
            "Sourav Dey": "Soorav Day",
            "Kolkata": "Kol-ka-ta",
            "Jarvis": "Jarvis",
            "AWS": "A W S",
            "PVT LTD": "Private Limited",
            "Pvt. Ltd.": "Private Limited",
            "LLM": "L L M",
            "GPU": "G P U",
            "RTX 3060": "R T X thirty sixty",
            "Ollama": "Oh-lah-ma",
            "Qwen": "kwen",
            "DeepSeek": "deep seek",
        })
        self.addCleanup(path.unlink)
        patcher = mock.patch.object(pronunciation, "CONFIG_PATH", path)
        patcher.start()
        self.addCleanup(patcher.stop)
        pronunciation.load(force=True)

    def test_sourav_dey(self):
        self.assertEqual(pronunciation.apply("Sourav Dey"), "Soorav Day")

    def test_kolkata(self):
        self.assertEqual(pronunciation.apply("Kolkata"), "Kol-ka-ta")

    def test_jarvis(self):
        self.assertEqual(pronunciation.apply("Jarvis"), "Jarvis")

    def test_aws(self):
        self.assertEqual(pronunciation.apply("AWS"), "A W S")

    def test_pvt_ltd_all_caps_no_periods(self):
        self.assertEqual(pronunciation.apply("PVT LTD"), "Private Limited")

    def test_pvt_ltd_with_periods(self):
        self.assertEqual(pronunciation.apply("Pvt. Ltd."), "Private Limited")

    def test_llm(self):
        self.assertEqual(pronunciation.apply("LLM"), "L L M")

    def test_gpu(self):
        self.assertEqual(pronunciation.apply("GPU"), "G P U")

    def test_rtx_3060(self):
        self.assertEqual(pronunciation.apply("RTX 3060"),
                         "R T X thirty sixty")

    def test_ollama(self):
        self.assertEqual(pronunciation.apply("Ollama"), "Oh-lah-ma")

    def test_qwen(self):
        self.assertEqual(pronunciation.apply("Qwen"), "kwen")

    def test_deepseek(self):
        self.assertEqual(pronunciation.apply("DeepSeek"), "deep seek")

    def test_case_insensitive_matching(self):
        self.assertEqual(pronunciation.apply("qwen"), "kwen")
        self.assertEqual(pronunciation.apply("QWEN"), "kwen")
        self.assertEqual(pronunciation.apply("deepseek"), "deep seek")

    def test_in_a_natural_sentence(self):
        out = pronunciation.apply(
            "I'm running Qwen on an RTX 3060 through Ollama right now.")
        self.assertEqual(
            out,
            "I'm running kwen on an R T X thirty sixty through "
            "Oh-lah-ma right now.")

    def test_paragraph_flow_preserved(self):
        # The correction must not disturb surrounding punctuation or
        # sentence structure -- only the matched span changes.
        out = pronunciation.apply(
            "My name is Sourav Dey. I work with AWS and LLM tools daily.")
        self.assertEqual(
            out,
            "My name is Soorav Day. I work with A W S and L L M "
            "tools daily.")


class WholeWordSafetyTests(unittest.TestCase):
    """Never corrupt a path, URL, password, ID, or code fragment --
    only an EXACT configured word/phrase, at a real word boundary,
    ever gets touched."""

    def setUp(self):
        path = _with_dict({"AWS": "A W S", "Qwen": "kwen",
                           "GPU": "G P U"})
        self.addCleanup(path.unlink)
        patcher = mock.patch.object(pronunciation, "CONFIG_PATH", path)
        patcher.start()
        self.addCleanup(patcher.stop)
        pronunciation.load(force=True)

    def test_substring_inside_a_longer_word_is_not_touched(self):
        # "AWS" must not fire inside "AWStronaut" or similar.
        self.assertEqual(pronunciation.apply("AWStronaut"), "AWStronaut")

    def test_path_containing_the_word_is_untouched_by_shape(self):
        # A path or filename isn't a dictionary entry to begin with,
        # so nothing in it can match -- proven with a realistic one.
        text = r"open D:\JARVISFullstack\backtalk\backtalk.json"
        self.assertEqual(pronunciation.apply(text), text)

    def test_url_is_untouched(self):
        text = "see https://example.com/gpu-benchmarks for details"
        # "gpu" appears inside the URL but not as a whole word (it's
        # glued to "-benchmarks" with a hyphen, a word character to
        # regex, so no boundary exists there).
        self.assertEqual(pronunciation.apply(text), text)

    def test_password_like_token_is_untouched(self):
        text = "the password is Qwen4Life2026!"
        self.assertEqual(pronunciation.apply(text), text)

    def test_code_identifier_is_untouched(self):
        text = "call get_gpu_status() before reading GPU_COUNT"
        self.assertEqual(pronunciation.apply(text), text)


class EmptyDictionaryAndReloadTests(unittest.TestCase):
    def test_missing_file_returns_text_unchanged(self):
        missing = Path(tempfile.gettempdir()) / "no_such_pronunciation.json"
        with mock.patch.object(pronunciation, "CONFIG_PATH", missing):
            pronunciation.load(force=True)
            self.assertEqual(pronunciation.apply("Qwen is great"),
                             "Qwen is great")

    def test_malformed_json_falls_back_without_raising(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        f.write("{not valid json")
        f.close()
        path = Path(f.name)
        try:
            with mock.patch.object(pronunciation, "CONFIG_PATH", path), \
                mock.patch.object(pronunciation, "_cache", None), \
                mock.patch.object(pronunciation, "_cache_mtime", None):
                # Must not raise; with no prior good cache to fall
                # back to, an empty dictionary is used -- text passes
                # through unmodified.
                pronunciation.load(force=True)
                self.assertEqual(pronunciation.apply("Qwen"), "Qwen")
        finally:
            path.unlink()

    def test_edit_takes_effect_without_restart(self):
        path = _with_dict({"Qwen": "kwen"})
        try:
            with mock.patch.object(pronunciation, "CONFIG_PATH", path):
                pronunciation.load(force=True)
                self.assertEqual(pronunciation.apply("Qwen"), "kwen")
                # Simulate Captain hand-editing the file mid-session.
                path.write_text(json.dumps({"Qwen": "kwenn"}))
                self.assertEqual(pronunciation.apply("Qwen"), "kwenn")
        finally:
            path.unlink()

    def test_empty_text_returns_empty(self):
        self.assertEqual(pronunciation.apply(""), "")


if __name__ == "__main__":
    unittest.main()
