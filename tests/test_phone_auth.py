"""Unit tests for backtalk.phone_auth: pairing codes, device tokens
(hash-only storage), revocation, and the rate limiter. Runs against the
FIXTURE device store tests/isolation.py redirected phone_auth.DEVICES_FILE
to -- never the real one (see test_zzz_real_state_unchanged.py and
test_isolation_regression.py).

Each test uses its OWN fake source IP so the module-level pairing-
attempt rate limiter (keyed by IP, and deliberately NOT reset between
tests -- it has no reason to know about test boundaries) can't leak
attempts from one test into another.
"""
import json
import subprocess
import time
import unittest
from unittest.mock import patch

from tests import isolation

isolation.ensure_isolated()  # explicit, invocation-agnostic -- see its docstring

from backtalk import phone_auth


def _reset_devices():
    phone_auth.DEVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    phone_auth._save_devices([])


class PairingTests(unittest.TestCase):
    def setUp(self):
        _reset_devices()

    def test_redeem_valid_code_returns_a_token_and_stores_only_its_hash(self):
        code, _ = phone_auth.create_pairing()
        token = phone_auth.redeem_pairing(code, "10.1.1.1", "Test Phone")
        self.assertIsNotNone(token)
        self.assertGreater(len(token), 20)
        stored = json.loads(phone_auth.DEVICES_FILE.read_text())
        self.assertEqual(len(stored), 1)
        self.assertNotEqual(stored[0]["token_hash"], token)
        self.assertEqual(stored[0]["label"], "Test Phone")

    def test_ca_fingerprint_is_none_when_not_supplied(self):
        code, _ = phone_auth.create_pairing()
        phone_auth.redeem_pairing(code, "10.1.1.9", "Test Phone")
        self.assertIsNone(phone_auth.list_devices()[0]["ca_fingerprint_at_pairing"])

    def test_ca_fingerprint_is_recorded_when_supplied(self):
        code, _ = phone_auth.create_pairing()
        fake_fp = "AA:BB:CC:DD"
        phone_auth.redeem_pairing(code, "10.1.1.10", "Test Phone",
                                  ca_fingerprint=fake_fp)
        self.assertEqual(phone_auth.list_devices()[0]["ca_fingerprint_at_pairing"],
                         fake_fp)

    def test_wrong_code_is_rejected(self):
        code, _ = phone_auth.create_pairing()
        wrong = "Z" * len(code) if code[0] != "Z" else "Y" * len(code)
        token = phone_auth.redeem_pairing(wrong, "10.1.1.2", "x")
        self.assertIsNone(token)

    def test_code_cannot_be_redeemed_twice(self):
        code, _ = phone_auth.create_pairing()
        first = phone_auth.redeem_pairing(code, "10.1.1.3", "a")
        second = phone_auth.redeem_pairing(code, "10.1.1.3", "b")
        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_expired_code_is_rejected(self):
        code, _ = phone_auth.create_pairing()
        with phone_auth._pending_lock:
            phone_auth._pending["expires_at"] = time.time() - 1
        token = phone_auth.redeem_pairing(code, "10.1.1.4", "x")
        self.assertIsNone(token)

    def test_repeated_bad_attempts_from_one_ip_get_rate_limited(self):
        code, _ = phone_auth.create_pairing()
        ip = "10.1.1.5"
        results = [phone_auth.redeem_pairing("WRONGCODE", ip, "x")
                  for _ in range(phone_auth._PAIR_ATTEMPT_LIMIT + 2)]
        self.assertTrue(all(r is None for r in results))
        # a CORRECT code from the same over-limit IP must also be
        # refused -- the limiter blocks the ip, not just bad codes
        still_blocked = phone_auth.redeem_pairing(code, ip, "x")
        self.assertIsNone(still_blocked)


class TokenVerificationTests(unittest.TestCase):
    def setUp(self):
        _reset_devices()
        code, _ = phone_auth.create_pairing()
        self.token = phone_auth.redeem_pairing(code, "10.2.1.1", "Verify Phone")

    def test_valid_token_verifies_and_updates_last_seen(self):
        before = phone_auth.list_devices()[0]["last_seen"]
        time.sleep(0.01)
        device = phone_auth.verify_token(self.token, "10.2.1.1")
        self.assertIsNotNone(device)
        self.assertEqual(device["label"], "Verify Phone")
        self.assertNotIn("token_hash", device)
        after = phone_auth.list_devices()[0]["last_seen"]
        self.assertGreater(after, before)

    def test_garbage_token_does_not_verify(self):
        self.assertIsNone(phone_auth.verify_token("not-a-real-token", "10.2.1.2"))

    def test_empty_token_does_not_verify(self):
        self.assertIsNone(phone_auth.verify_token("", "10.2.1.3"))

    def test_revoked_device_token_stops_verifying_immediately(self):
        self.assertIsNotNone(phone_auth.verify_token(self.token, "10.2.1.1"))
        label = phone_auth.revoke_device(1)
        self.assertEqual(label, "Verify Phone")
        self.assertIsNone(phone_auth.verify_token(self.token, "10.2.1.1"))


class DeviceListAndRevokeTests(unittest.TestCase):
    def setUp(self):
        _reset_devices()

    def _pair_one(self, ip, label):
        code, _ = phone_auth.create_pairing()
        return phone_auth.redeem_pairing(code, ip, label)

    def test_list_devices_never_exposes_token_hashes(self):
        self._pair_one("10.3.1.1", "Phone A")
        for d in phone_auth.list_devices():
            self.assertNotIn("token_hash", d)

    def test_revoke_by_out_of_range_index_returns_none(self):
        self._pair_one("10.3.1.2", "Phone A")
        self.assertIsNone(phone_auth.revoke_device(99))
        self.assertIsNone(phone_auth.revoke_device(0))

    def test_revoke_removes_only_the_targeted_device(self):
        self._pair_one("10.3.1.3", "Phone A")
        self._pair_one("10.3.1.4", "Phone B")
        label = phone_auth.revoke_device(1)
        self.assertEqual(label, "Phone A")
        remaining = phone_auth.list_devices()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["label"], "Phone B")


class RateLimiterTests(unittest.TestCase):
    def test_allows_up_to_max_calls_then_blocks(self):
        rl = phone_auth.RateLimiter(max_calls=3, window_s=60)
        results = [rl.allow("device-1") for _ in range(5)]
        self.assertEqual(results, [True, True, True, False, False])

    def test_different_keys_have_independent_budgets(self):
        rl = phone_auth.RateLimiter(max_calls=1, window_s=60)
        self.assertTrue(rl.allow("device-a"))
        self.assertTrue(rl.allow("device-b"))
        self.assertFalse(rl.allow("device-a"))

    def test_calls_outside_the_window_are_forgotten(self):
        # a comfortable margin (0.15s sleep vs. a 0.1s window) -- a
        # tighter one flaked under load once the suite grew large
        # enough for scheduling jitter to eat into it.
        rl = phone_auth.RateLimiter(max_calls=1, window_s=0.1)
        self.assertTrue(rl.allow("device-1"))
        self.assertFalse(rl.allow("device-1"))
        time.sleep(0.15)
        self.assertTrue(rl.allow("device-1"))


class TailscaleIpDetectionTests(unittest.TestCase):
    """detect_tailscale_ip() asks the Tailscale CLI directly rather
    than guessing via routing tricks -- these tests mock subprocess.run
    since Tailscale may not be installed/signed in wherever this suite
    runs (including this repo's own dev machine, before setup)."""

    def _fake_run(self, returncode=0, stdout="", stderr=""):
        return patch(
            "backtalk.phone_auth.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["tailscale", "ip", "-4"], returncode=returncode,
                stdout=stdout, stderr=stderr))

    def test_returns_the_cli_reported_address(self):
        with self._fake_run(stdout="100.101.102.103\n"):
            self.assertEqual(phone_auth.detect_tailscale_ip(), "100.101.102.103")

    def test_takes_only_the_first_line_if_several_are_returned(self):
        with self._fake_run(stdout="100.101.102.103\nfd7a:115c::1\n"):
            self.assertEqual(phone_auth.detect_tailscale_ip(), "100.101.102.103")

    def test_none_when_cli_exits_nonzero(self):
        with self._fake_run(returncode=1, stderr="not logged in"):
            self.assertIsNone(phone_auth.detect_tailscale_ip())

    def test_none_when_cli_returns_empty_output(self):
        with self._fake_run(stdout=""):
            self.assertIsNone(phone_auth.detect_tailscale_ip())

    def test_none_when_tailscale_binary_is_missing(self):
        with patch("backtalk.phone_auth.subprocess.run",
                  side_effect=FileNotFoundError("no such file")):
            self.assertIsNone(phone_auth.detect_tailscale_ip())

    def test_none_on_timeout(self):
        with patch("backtalk.phone_auth.subprocess.run",
                  side_effect=subprocess.TimeoutExpired(cmd="tailscale", timeout=10)):
            self.assertIsNone(phone_auth.detect_tailscale_ip())

    def test_never_returns_0_0_0_0_even_if_the_cli_somehow_reported_it(self):
        # Defense in depth: this documents the current contract (we
        # trust the CLI's own answer) rather than silently special-
        # casing it -- Tailscale itself would never legitimately hand
        # out 0.0.0.0, so this is here to catch a regression, not to
        # assert new filtering behavior.
        with self._fake_run(stdout="100.64.0.1\n"):
            ip = phone_auth.detect_tailscale_ip()
        self.assertNotEqual(ip, "0.0.0.0")


if __name__ == "__main__":
    unittest.main()
