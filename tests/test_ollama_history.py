"""Regression tests for conversation continuity on Ollama-backed
brains. Real field-test bug: every turn sent only [system, user] with
no memory of what was just said, so "Yes, Jarvis" after a follow-up
question had nothing to resolve against and the model answered as if
the conversation had just started. These tests prove the running
history is built, sent, capped, and clearable -- without touching a
real Ollama instance.
"""
import unittest
from unittest import mock

from backtalk.brains.ollama_brain import QwenBrain


class _FakeStreamResponse:
    def __init__(self, reply_text: str):
        self._reply_text = reply_text

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        yield f'{{"message": {{"content": {self._reply_text!r}}}, "done": false}}'.replace("'", '"')
        yield '{"done": true}'


class _FakeStreamCtx:
    def __init__(self, reply_text: str, captured_payloads: list):
        self._reply_text = reply_text
        self._captured = captured_payloads

    async def __aenter__(self):
        return _FakeStreamResponse(self._reply_text)

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncClient:
    """Records every payload it's asked to stream, and replies with a
    scripted, fixed sentence each call -- just enough to prove history
    accumulation and payload shape without any real model behind it."""

    def __init__(self, captured_payloads: list, replies: list):
        self._captured = captured_payloads
        self._replies = list(replies)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, json):
        self._captured.append(json)
        reply = self._replies.pop(0) if self._replies else "OK."
        return _FakeStreamCtx(reply, self._captured)


class ConversationHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_starts_empty(self):
        brain = QwenBrain(enabled=True)
        self.assertEqual(brain._history, [])

    async def test_first_turn_sends_no_history(self):
        payloads: list = []
        fake = _FakeAsyncClient(payloads, ["Hello there."])
        brain = QwenBrain(enabled=True)
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            async for _ in brain.ask_stream("Hi"):
                pass
        messages = payloads[0]["messages"]
        # Only system + this turn's user message -- no prior history yet.
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    async def test_second_turn_includes_first_turns_history(self):
        payloads: list = []
        fake = _FakeAsyncClient(
            payloads,
            ["Sure, would you like me to check the weather?",
             "Yes, please."])
        brain = QwenBrain(enabled=True)
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            async for _ in brain.ask_stream("Can you help me plan my day?"):
                pass
            async for _ in brain.ask_stream("Yes, Jarvis."):
                pass
        second_call_messages = payloads[1]["messages"]
        roles = [m["role"] for m in second_call_messages]
        # system, then the FIRST turn's user+assistant pair, then this
        # turn's fresh user message -- the model can now actually
        # resolve "yes" against what it just asked.
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertEqual(second_call_messages[1]["content"],
                         "Can you help me plan my day?")
        self.assertIn("weather", second_call_messages[2]["content"])
        self.assertEqual(second_call_messages[3]["content"], "Yes, Jarvis.")

    async def test_history_is_capped(self):
        payloads: list = []
        replies = [f"Reply number {i}." for i in range(30)]
        fake = _FakeAsyncClient(payloads, list(replies))
        brain = QwenBrain(enabled=True)
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            for i in range(15):
                async for _ in brain.ask_stream(f"Question {i}"):
                    pass
        # Capped at _MAX_HISTORY_TURNS pairs (2 messages per pair).
        self.assertLessEqual(len(brain._history),
                             brain._MAX_HISTORY_TURNS * 2)

    async def test_clear_command_empties_history(self):
        payloads: list = []
        fake = _FakeAsyncClient(payloads, ["Noted."])
        brain = QwenBrain(enabled=True)
        with mock.patch("backtalk.brains.ollama_brain.httpx.AsyncClient",
                       return_value=fake):
            async for _ in brain.ask_stream("Remember this for later."):
                pass
        self.assertNotEqual(brain._history, [])
        resp = await brain.command("/clear")
        self.assertEqual(brain._history, [])
        self.assertIn("Cleared", resp)

    async def test_disabled_brain_never_touches_history(self):
        brain = QwenBrain(enabled=False)
        from backtalk.brains.base import BrainDisabledError
        with self.assertRaises(BrainDisabledError):
            async for _ in brain.ask_stream("hi"):
                pass
        self.assertEqual(brain._history, [])


if __name__ == "__main__":
    unittest.main()
