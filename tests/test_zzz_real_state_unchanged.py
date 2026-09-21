"""Regression test proving the isolation fixture (tests/isolation.py)
actually held for the entire suite: the REAL Day Journal ledger, the
REAL vault's daily-notes directory, the REAL paired-phone device store,
the REAL local CA/leaf certificates, and the REAL backtalk.json must be
byte-for-byte unchanged after running every other test file.

Named to sort alphabetically LAST (unittest discover collects modules
in filename order) so this runs after every other test in the suite
has had its chance to write somewhere it shouldn't have. This is a
safety-net regression test for the fixture itself, not a unit test of
day_journal.py's, phone_auth.py's, or phone_tls.py's own logic -- see
test_day_journal.py, test_phone_auth.py, and test_phone_tls.py for
those.

This file being included in a discovery run is NOT what makes
isolation safe any more (that used to be true, and was the whole
problem -- see tests/isolation.py's docstring and
test_isolation_regression.py). It only proves, after the fact, that
isolation held for whichever run included it.
"""
import unittest

from tests import isolation

isolation.ensure_isolated()


class RealStateUntouchedByTheWholeSuiteTests(unittest.TestCase):
    def test_real_state_unchanged(self):
        current = isolation.real_state_hashes()
        baseline = isolation.baseline_hashes()
        for name in baseline:
            self.assertEqual(
                current[name], baseline[name],
                f"this machine's REAL {name!r} changed during this "
                f"test run -- some test wrote to it without going "
                f"through tests/isolation.py's redirect")

    def test_fixture_redirect_is_still_active(self):
        # A structural sanity check on the fixture itself: if this
        # ever failed, the test above would be meaningless -- it'd be
        # comparing the real path to itself.
        from pathlib import Path

        from backtalk import config, day_journal, phone_auth, phone_tls
        from backtalk.brains import vault_context

        self.assertTrue(isolation.is_isolated())
        self.assertNotEqual(day_journal.LEDGER_DIR, isolation.REAL_LEDGER_DIR)
        self.assertNotEqual(vault_context.VAULT_ROOT.resolve(),
                            Path("D:/JARVIS-Vault").resolve())
        self.assertNotEqual(phone_auth.DEVICES_FILE,
                            isolation.REAL_PHONE_DEVICES_FILE)
        self.assertNotEqual(phone_tls.TLS_DIR, isolation.REAL_PHONE_TLS_DIR)
        self.assertNotEqual(config.CONFIG_PATH, isolation.REAL_BACKTALK_CONFIG)


if __name__ == "__main__":
    unittest.main()
