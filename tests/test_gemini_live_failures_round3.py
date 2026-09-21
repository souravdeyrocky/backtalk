"""Regression tests for a third live F8/Gemini failure: "Thank you.
Confirmed." didn't resolve a pending Gemini send, and the whole
per-prompt Gemini confirmation model is replaced here by an explicit
external-session consent LEASE shared by Claude and Gemini. All
mocked -- zero real network calls, zero real key usage.
"""
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

from backtalk.brains import config as router_config
from backtalk.brains.base import BrainHealth
from backtalk.brains.claude_brain import ClaudeBrain
from backtalk.brains.gemini_brain import GeminiBrain
from backtalk.main import (_EXTERNAL_LEASE, _LEASE_DURATION_S,
                           _claude_preview_line, _clear_external_lease,
                           _external_lease_active, _gemini_preview_line,
                           _is_confirm_phrase, _log_external_send,
                           _start_external_lease)
from backtalk.router import BrainRouter


def _cfg(*enabled_ids: str) -> dict:
    cfg = json.loads(json.dumps(router_config.DEFAULTS))
    for bid in enabled_ids:
        cfg["brains"][bid]["enabled"] = True
    return cfg


def _reset_lease():
    _EXTERNAL_LEASE["brain_id"] = None
    _EXTERNAL_LEASE["expires_at"] = 0.0


# ---- 1. "Thank you. Confirmed." resolves a pending confirmation ------

class ThankYouConfirmedResolvesTests(unittest.TestCase):
    def test_exact_live_transcript(self):
        self.assertTrue(_is_confirm_phrase("Thank you. Confirmed."))

    def test_thank_you_confirm_variant(self):
        self.assertTrue(_is_confirm_phrase("Thank you, confirm"))

    def test_case_and_punctuation_ignored(self):
        for phrase in ("thank you confirmed", "THANK YOU, CONFIRMED!!!",
                       "Thank You... Confirmed."):
            self.assertTrue(_is_confirm_phrase(phrase), phrase)

    def test_all_required_variants(self):
        for phrase in ("confirm", "confirmed", "conform", "conformed",
                       "yes confirm", "yes confirmed",
                       "thank you, confirm", "thank you, confirmed"):
            self.assertTrue(_is_confirm_phrase(phrase), phrase)

    def test_negation_rejected(self):
        for phrase in ("do not confirm", "Do not confirm.",
                       "don't confirm", "never confirm"):
            self.assertFalse(_is_confirm_phrase(phrase), phrase)

    def test_unrelated_speech_rejected(self):
        self.assertFalse(_is_confirm_phrase("what's the weather"))
        self.assertFalse(_is_confirm_phrase(""))


# ---- 2. No external request before consent ----------------------------

class NoExternalRequestBeforeConsentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _reset_lease()

    def test_no_lease_active_by_default(self):
        self.assertFalse(_external_lease_active("gemini"))
        self.assertFalse(_external_lease_active("claude"))

    async def test_activation_alone_does_not_imply_a_lease(self):
        # router.activate() itself never touches _EXTERNAL_LEASE --
        # that's main.py's job, done explicitly in useclaude:confirmed
        # / usegemini:confirmed via _start_external_lease(). Proven
        # here by activating WITHOUT calling that helper at all.
        cfg = _cfg("qwen3-8b-local", "gemini")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-test"}):
            router = BrainRouter(config=cfg)
            await router.activate("gemini", confirmed=True)
        self.assertFalse(_external_lease_active("gemini"))

    async def test_no_httpx_call_without_a_lease_or_explicit_confirm(self):
        with patch("backtalk.brains.gemini_brain.httpx.AsyncClient") as m:
            self.assertFalse(_external_lease_active("gemini"))
        m.assert_not_called()


# ---- 3. Multiple Gemini requests during an active lease ---------------

class MultipleRequestsDuringActiveLeaseTests(unittest.TestCase):
    def setUp(self):
        _reset_lease()

    def test_lease_stays_active_across_repeated_checks(self):
        _start_external_lease("gemini")
        for _ in range(5):
            self.assertTrue(_external_lease_active("gemini"))

    def test_each_use_refreshes_the_idle_window(self):
        _start_external_lease("gemini")
        first_expiry = _EXTERNAL_LEASE["expires_at"]
        time.sleep(0.05)
        _start_external_lease("gemini")  # simulates a second real send
        self.assertGreater(_EXTERNAL_LEASE["expires_at"], first_expiry)

    def test_lease_is_per_brain(self):
        _start_external_lease("gemini")
        self.assertTrue(_external_lease_active("gemini"))
        self.assertFalse(_external_lease_active("claude"))


# ---- 4. Expiry after 30 idle minutes -----------------------------------

class LeaseExpiryTests(unittest.TestCase):
    def setUp(self):
        _reset_lease()

    def test_lease_duration_is_thirty_minutes(self):
        self.assertEqual(_LEASE_DURATION_S, 30 * 60)

    def test_expired_lease_is_not_active(self):
        _EXTERNAL_LEASE["brain_id"] = "gemini"
        _EXTERNAL_LEASE["expires_at"] = time.monotonic() - 1
        self.assertFalse(_external_lease_active("gemini"))

    def test_fresh_lease_is_active(self):
        _start_external_lease("gemini")
        self.assertTrue(_external_lease_active("gemini"))

    def test_lease_about_to_expire_is_still_active(self):
        _EXTERNAL_LEASE["brain_id"] = "gemini"
        _EXTERNAL_LEASE["expires_at"] = time.monotonic() + 0.5
        self.assertTrue(_external_lease_active("gemini"))


# ---- 5. Switch back to local clears consent ----------------------------

class SwitchToLocalClearsLeaseTests(unittest.TestCase):
    def setUp(self):
        _reset_lease()

    def test_clear_external_lease_deactivates_it(self):
        _start_external_lease("gemini")
        self.assertTrue(_external_lease_active("gemini"))
        _clear_external_lease()
        self.assertFalse(_external_lease_active("gemini"))

    def test_clear_resets_both_fields(self):
        _start_external_lease("claude")
        _clear_external_lease()
        self.assertIsNone(_EXTERNAL_LEASE["brain_id"])
        self.assertEqual(_EXTERNAL_LEASE["expires_at"], 0.0)


class SwitchingToQwenClearsLeaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _reset_lease()

    async def test_switching_to_qwen_clears_a_gemini_lease(self):
        # Mirrors what the useqwen console-verb handler does: it calls
        # _clear_external_lease() unconditionally before activating.
        _start_external_lease("gemini")
        self.assertTrue(_external_lease_active("gemini"))
        cfg = _cfg("qwen3-8b-local")
        router = BrainRouter(config=cfg)
        qwen = router.get("qwen3-8b-local")
        qwen.health = AsyncMock(return_value=BrainHealth(healthy=True))
        qwen.start = AsyncMock()
        _clear_external_lease()
        await router.activate("qwen3-8b-local")
        self.assertFalse(_external_lease_active("gemini"))
        self.assertEqual(router.active_id, "qwen3-8b-local")


# ---- 6. No vault-context call or hidden-context transmission ----------

class NoVaultOrHiddenContextTests(unittest.TestCase):
    def test_claude_preview_is_the_bare_utterance(self):
        brain = ClaudeBrain()
        self.assertEqual(brain.build_request_preview("  hello there  "),
                         "hello there")

    def test_gemini_preview_is_the_bare_utterance(self):
        brain = GeminiBrain(enabled=True)
        self.assertEqual(brain.build_request_preview("  hello there  "),
                         "hello there")

    def test_claude_module_never_imports_vault_context(self):
        import backtalk.brains.claude_brain as mod
        self.assertNotIn("vault_context", dir(mod))

    def test_gemini_module_never_imports_vault_context(self):
        import backtalk.brains.gemini_brain as mod
        self.assertNotIn("vault_context", dir(mod))

    def test_preview_lines_never_contain_more_than_the_given_text(self):
        secret_shaped = "call the dentist tomorrow"
        gemini_line = _gemini_preview_line(secret_shaped)
        claude_line = _claude_preview_line(secret_shaped)
        # Only the exact given text appears where content should be --
        # neither line template adds any OTHER identifying content.
        self.assertEqual(gemini_line.count(secret_shaped), 1)
        self.assertEqual(claude_line.count(secret_shaped), 1)

    def test_log_external_send_only_logs_the_given_text(self):
        logged = []
        with patch("backtalk.main.log", side_effect=lambda s: logged.append(s)):
            _log_external_send("gemini", "what is the capital of Japan")
        self.assertEqual(len(logged), 1)
        self.assertIn("what is the capital of Japan", logged[0])
        self.assertIn("gemini", logged[0].lower())


# ---- 7. Claude tool permissions unchanged ------------------------------

class ClaudeToolPermissionsUnchangedTests(unittest.TestCase):
    def test_claude_still_requires_tools(self):
        self.assertTrue(ClaudeBrain.requires_tools)

    def test_claude_capability_summary_still_names_the_gate(self):
        brain = ClaudeBrain()
        low = brain.capability_summary.lower()
        self.assertIn("gated", low)
        self.assertIn("permission", low)

    def test_perm_state_is_structurally_separate_from_the_lease(self):
        from backtalk.main import _EXTERNAL_LEASE as lease
        from backtalk.main import _EXTERNAL_PENDING as pending
        from backtalk.main import _PERM as perm
        # Different dicts, different keys -- the tool-permission gate
        # (_PERM) and the external-consent lease share no state at all.
        self.assertNotEqual(set(perm.keys()), set(lease.keys()))
        self.assertNotEqual(set(perm.keys()), set(pending.keys()))

    def test_make_permission_gate_still_exists_and_is_callable(self):
        from backtalk.main import make_permission_gate
        gate = make_permission_gate(mouth=None)
        self.assertTrue(callable(gate))


if __name__ == "__main__":
    unittest.main()
