"""A/B regression: the same underlying answer, phrased the way Claude
already writes it (clean prose, per DISCIPLINE in config.py) versus
the way Qwen/DeepSeek were observed writing it live (Markdown-heavy),
must produce EQUIVALENT final spoken text once both pass through
BrainRouter.ask_stream()'s shared sanitizer. This is the proof that
the voice experience doesn't depend on which brain answered.
"""
import json
import unittest
from unittest.mock import AsyncMock

from backtalk.brains import config as router_config
from backtalk.brains.base import BrainHealth
from backtalk.router import BrainRouter


def _cfg_with(bid: str) -> dict:
    cfg = json.loads(json.dumps(router_config.DEFAULTS))
    cfg["brains"][bid]["enabled"] = True
    return cfg


class ClaudeStyleVsQwenStyleEquivalenceTests(unittest.IsolatedAsyncioTestCase):
    async def _spoken_words(self, bid: str, sentences: list[str]) -> str:
        router = BrainRouter(config=_cfg_with(bid))
        brain = router.get(bid)
        brain.health = AsyncMock(return_value=BrainHealth(healthy=True))
        brain.start = AsyncMock()
        await router.activate(bid, confirmed=True)

        async def _fake_ask_stream(utterance):
            for s in sentences:
                yield s

        brain.ask_stream = _fake_ask_stream
        out = []
        async for sentence in router.ask_stream("what's my name and dogs?"):
            out.append(sentence)
        return " ".join(out)

    async def test_bold_name_vs_plain_name_speak_identically(self):
        # Claude-style (DISCIPLINE-compliant): already clean prose.
        claude_text = await self._spoken_words(
            "claude", ["Your name is Sourav Dey, always addressed as "
                      "Captain."])
        # Qwen-style, as actually observed live: bold markdown.
        qwen_text = await self._spoken_words(
            "qwen3-8b-local", ["Your name is **Sourav Dey**, always "
                              "addressed as Captain."])
        self.assertEqual(claude_text, qwen_text)

    async def test_bulleted_list_vs_prose_speak_equivalent_content(self):
        claude_text = await self._spoken_words(
            "claude",
            ["Benzo is a ten-year-old Pomeranian.",
             "Tyson is an eight-year-old German Shepherd."])
        qwen_text = await self._spoken_words(
            "qwen3-8b-local",
            ["- Benzo is a ten-year-old Pomeranian.",
             "- Tyson is an eight-year-old German Shepherd."])
        self.assertEqual(claude_text, qwen_text)
        self.assertNotIn("-", qwen_text.replace("ten-year-old", "")
                         .replace("eight-year-old", ""))

    async def test_neither_path_ever_speaks_raw_markdown_punctuation(self):
        qwen_text = await self._spoken_words(
            "qwen3-8b-local",
            ["## Summary", "**Important**: check `config.py` at "
                          "[the repo](https://example.com)."])
        for forbidden in ("**", "##", "`", "[", "](", "http"):
            self.assertNotIn(forbidden, qwen_text)


class GenericClosingNeverReachesRouterOutputTests(unittest.IsolatedAsyncioTestCase):
    """Regression for a real field-test failure: Qwen and DeepSeek both
    habitually tacked a service-desk sign-off onto an already-complete
    answer. Proven here at the router level (not just to_speech() in
    isolation) for both local brains, since the fix requirement is the
    SAME shared policy applied to both."""

    async def _spoken_words(self, bid: str, sentences: list[str]) -> str:
        router = BrainRouter(config=_cfg_with(bid))
        brain = router.get(bid)
        brain.health = AsyncMock(return_value=BrainHealth(healthy=True))
        brain.start = AsyncMock()
        await router.activate(bid, confirmed=True)

        async def _fake_ask_stream(utterance):
            for s in sentences:
                yield s

        brain.ask_stream = _fake_ask_stream
        out = []
        async for sentence in router.ask_stream("what's the capital of France?"):
            out.append(sentence)
        return " ".join(out)

    async def test_qwen_closing_stripped(self):
        text = await self._spoken_words(
            "qwen3-8b-local",
            ["The capital of France is Paris.",
             "How may I assist you today?"])
        self.assertEqual(text, "The capital of France is Paris.")

    async def test_deepseek_closing_stripped(self):
        text = await self._spoken_words(
            "deepseek-r1-8b-local",
            ["The capital of France is Paris.",
             "Let me know if you need anything else."])
        self.assertEqual(text, "The capital of France is Paris.")


if __name__ == "__main__":
    unittest.main()
