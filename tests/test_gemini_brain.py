import os
import unittest
from unittest import mock

from backtalk.brains.base import BrainDisabledError, BrainUnavailableError
from backtalk.brains.gemini_brain import GeminiBrain


class GeminiGatingTests(unittest.IsolatedAsyncioTestCase):
    """The gates that must hold regardless of the real implementation
    underneath: disabled by default, confirm-gated to switch onto,
    approval-gated on every subsequent turn."""

    async def test_disabled_by_default(self):
        brain = GeminiBrain()
        self.assertFalse(brain.enabled)
        health = await brain.health()
        self.assertFalse(health.healthy)

    async def test_requires_confirm_to_switch(self):
        brain = GeminiBrain()
        self.assertTrue(brain.requires_confirm_to_switch)

    async def test_requires_external_lease(self):
        # Distinct from requires_confirm_to_switch: this one gates
        # the external-consent LEASE (main.py's _EXTERNAL_LEASE) --
        # confirming the switch opens a thirty-minute idle window,
        # replacing the old "approve every single turn" behavior.
        brain = GeminiBrain()
        self.assertTrue(brain.requires_external_lease)

    async def test_disabled_ask_stream_refuses_without_any_network_call(self):
        brain = GeminiBrain(enabled=False)
        with self.assertRaises(BrainDisabledError):
            async for _ in brain.ask_stream("hello"):
                pass


class GeminiHealthCheckTests(unittest.IsolatedAsyncioTestCase):
    """health() is key-PRESENCE only -- deliberately no network call,
    so even checking status never spends an unapproved request."""

    async def test_enabled_without_key_reports_honestly(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            brain = GeminiBrain(enabled=True)
            health = await brain.health()
        self.assertFalse(health.healthy)
        self.assertIn("GEMINI_API_KEY", health.reason)

    async def test_enabled_with_key_present_is_healthy(self):
        with mock.patch.dict(os.environ,
                             {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            health = await brain.health()
        self.assertTrue(health.healthy)

    async def test_health_check_never_makes_a_network_call(self):
        with mock.patch.dict(os.environ,
                             {"GEMINI_API_KEY": "fake-key-for-test"}):
            with mock.patch("backtalk.brains.gemini_brain.httpx.AsyncClient") as m:
                brain = GeminiBrain(enabled=True)
                await brain.health()
        m.assert_not_called()


class GeminiRequestPreviewTests(unittest.TestCase):
    """The exact text shown for approval must be exactly what gets
    sent -- no vault context, no wrapping, no extra words."""

    def test_preview_is_exactly_the_utterance(self):
        brain = GeminiBrain(enabled=True)
        self.assertEqual(
            brain.build_request_preview("what's the weather like"),
            "what's the weather like")

    def test_preview_strips_surrounding_whitespace_only(self):
        brain = GeminiBrain(enabled=True)
        self.assertEqual(
            brain.build_request_preview("  hello there  \n"),
            "hello there")

    def test_preview_never_imports_or_touches_vault_context(self):
        import backtalk.brains.gemini_brain as mod
        self.assertNotIn("vault_context", dir(mod))


class GeminiRealRequestTests(unittest.IsolatedAsyncioTestCase):
    """Real ask_stream() logic, fully mocked at the HTTP boundary --
    zero real network calls, zero real key usage."""

    def _client_returning(self, status_code, json_body=None):
        class _Resp:
            def __init__(self):
                self.status_code = status_code
            def json(self):
                return json_body

        class _Client:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False
            async def post(self, url, json, headers):
                self.captured = (url, json, headers)
                return _Resp()

        return _Client()

    async def test_disabled_never_reaches_the_network(self):
        brain = GeminiBrain(enabled=False)
        with mock.patch("backtalk.brains.gemini_brain.httpx.AsyncClient") as m:
            with self.assertRaises(BrainDisabledError):
                async for _ in brain.ask_stream("hi"):
                    pass
        m.assert_not_called()

    async def test_no_key_raises_unavailable_without_a_network_call(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            brain = GeminiBrain(enabled=True)
            with mock.patch(
                    "backtalk.brains.gemini_brain.httpx.AsyncClient") as m:
                with self.assertRaises(BrainUnavailableError):
                    async for _ in brain.ask_stream("hi"):
                        pass
            m.assert_not_called()

    async def test_key_is_sent_as_a_header_never_in_the_url(self):
        with mock.patch.dict(os.environ,
                             {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            client = self._client_returning(
                200, {"candidates": [{"content": {"parts": [
                    {"text": "Hello."}]}}]})
            with mock.patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                            return_value=client):
                async for _ in brain.ask_stream("hi"):
                    pass
        url, payload, headers = client.captured
        self.assertNotIn("fake-key-for-test", url)
        self.assertEqual(headers["x-goog-api-key"], "fake-key-for-test")

    async def test_payload_contents_contains_only_the_bare_utterance(self):
        # Note: the payload also carries a FIXED systemInstruction (see
        # test_identity.py) -- that's metadata about how Gemini
        # describes itself, never additional content attributed to
        # Captain. This test's own job is narrower and just as real:
        # `contents`, the field that actually represents what Captain
        # said, must never carry anything beyond the bare utterance.
        with mock.patch.dict(os.environ,
                             {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            client = self._client_returning(
                200, {"candidates": [{"content": {"parts": [
                    {"text": "ok"}]}}]})
            with mock.patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                            return_value=client):
                async for _ in brain.ask_stream("what's the weather"):
                    pass
        _, payload, _ = client.captured
        self.assertEqual(
            payload["contents"],
            [{"parts": [{"text": "what's the weather"}]}])

    def test_system_instruction_is_fixed_never_built_from_the_utterance(self):
        from backtalk.brains.gemini_brain import _SYSTEM_INSTRUCTION
        self.assertNotIn("what's the weather", _SYSTEM_INSTRUCTION)
        self.assertIn("Jarvis", _SYSTEM_INSTRUCTION)

    async def test_rate_limited_reports_honestly_never_falls_back(self):
        with mock.patch.dict(os.environ,
                             {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            client = self._client_returning(429)
            with mock.patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                            return_value=client):
                with self.assertRaises(BrainUnavailableError) as ctx:
                    async for _ in brain.ask_stream("hi"):
                        pass
        self.assertIn("rate-limited", str(ctx.exception).lower())

    async def test_reply_is_sentence_split(self):
        with mock.patch.dict(os.environ,
                             {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            client = self._client_returning(
                200, {"candidates": [{"content": {"parts": [
                    {"text": "First sentence. Second sentence."}]}}]})
            with mock.patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                            return_value=client):
                sentences = [s async for s in brain.ask_stream("hi")]
        self.assertEqual(sentences, ["First sentence.", "Second sentence."])


class GeminiPhoneCapabilityTests(unittest.IsolatedAsyncioTestCase):
    """2026-09-21 field defect, same fix as Qwen/DeepSeek: Gemini must
    say it can reply through the paired phone app when one is actually
    connected, and must NOT say so when none is -- computed fresh per
    call via phone_bridge.phone_reply_available(), never baked into
    the fixed module-level _SYSTEM_INSTRUCTION."""

    def _client_returning(self, status_code, json_body=None):
        class _Resp:
            def __init__(self):
                self.status_code = status_code
            def json(self):
                return json_body

        class _Client:
            def __init__(self):
                self.captured = None
            async def __aenter__(self):
                return self
            async def __aexit__(self, *exc):
                return False
            async def post(self, url, json, headers):
                self.captured = (url, json, headers)
                return _Resp()

        return _Client()

    async def _run_and_capture_instruction(self, phone_available: bool) -> str:
        with mock.patch.dict(os.environ,
                             {"GEMINI_API_KEY": "fake-key-for-test"}):
            brain = GeminiBrain(enabled=True)
            client = self._client_returning(
                200, {"candidates": [{"content": {"parts": [
                    {"text": "ok"}]}}]})
            with mock.patch(
                    "backtalk.brains.gemini_brain.phone_bridge.phone_reply_available",
                    return_value=phone_available), \
                mock.patch("backtalk.brains.gemini_brain.httpx.AsyncClient",
                           return_value=client):
                async for _ in brain.ask_stream("hi"):
                    pass
        _, payload, _ = client.captured
        return payload["systemInstruction"]["parts"][0]["text"]

    async def test_phone_rule_present_when_phone_connected(self):
        from backtalk.identity import PHONE_REPLY_CAPABILITY_RULE
        instruction = await self._run_and_capture_instruction(True)
        self.assertIn(PHONE_REPLY_CAPABILITY_RULE, instruction)

    async def test_phone_rule_absent_when_no_phone_connected(self):
        from backtalk.identity import PHONE_REPLY_CAPABILITY_RULE
        instruction = await self._run_and_capture_instruction(False)
        self.assertNotIn(PHONE_REPLY_CAPABILITY_RULE, instruction)


if __name__ == "__main__":
    unittest.main()
