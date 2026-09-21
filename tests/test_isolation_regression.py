"""Regression tests for the 2026-09-21 test-isolation incident: a
narrowed `python -m unittest discover -s tests -p "test_phone_*.py"`
run silently skipped tests/__init__.py's real-state redirect (see
tests/isolation.py's module docstring for the exact CPython mechanism)
and overwrote this machine's real .phone_tls/ with test-fixture
certificates issued for the fake IP 100.101.102.103.

These tests prove the fix holds by actually doing the dangerous thing:
spawning REAL subprocesses that run this suite exactly the way a
person or an agent would type it -- no BACKTALK_CONFIG override, no
special env, same cwd (the repo root) -- and hashing every real-state
path immediately before and immediately after each one. If isolation
ever regresses again, one of these three invocation shapes will catch
it before a human does.

Guarded against runaway recursion: the "full suite" case below spawns
a subprocess that itself discovers this very file. That child process
is launched with BACKTALK_ISOLATION_REGRESSION_CHILD=1 in its
environment, and every test in this file skips immediately when it
sees that variable set -- so the child runs (and is itself proof the
suite still passes end to end), but never spawns grandchildren.
"""
import os
import subprocess
import sys
import unittest

from tests import isolation

isolation.ensure_isolated()

_CHILD_ENV = "BACKTALK_ISOLATION_REGRESSION_CHILD"
_IS_CHILD = os.environ.get(_CHILD_ENV) == "1"

REPO_ROOT = isolation.REAL_PHONE_TLS_DIR.parent


class IsolationSurvivesRealInvocationsTests(unittest.TestCase):
    def setUp(self):
        if _IS_CHILD:
            self.skipTest(
                "running inside a nested regression-test subprocess -- "
                "skipped to avoid spawning grandchildren")

    def _run_and_check(self, argv, label):
        before = isolation.real_state_hashes()
        env = dict(os.environ)
        env[_CHILD_ENV] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "-m", "unittest", *argv],
                cwd=str(REPO_ROOT), env=env,
                capture_output=True, text=True, timeout=180)
        finally:
            after = isolation.real_state_hashes()
            self.assertEqual(
                before, after,
                f"{label}: real state changed during this run -- "
                f"isolation regressed")
        self.assertEqual(
            result.returncode, 0,
            f"{label} did not pass on its own terms (returncode="
            f"{result.returncode}); real state was unaffected but the "
            f"invocation itself is broken:\n{result.stdout[-3000:]}\n"
            f"{result.stderr[-3000:]}")

    def test_full_suite_leaves_real_state_untouched(self):
        self._run_and_check(
            ["discover", "-s", "tests", "-p", "test_*.py"],
            "full suite (discover -s tests -p test_*.py)")

    def test_narrowed_phone_pattern_leaves_real_state_untouched(self):
        # The exact command that caused the 2026-09-21 incident.
        self._run_and_check(
            ["discover", "-s", "tests", "-p", "test_phone_*.py"],
            "narrowed pattern (discover -s tests -p test_phone_*.py)")

    def test_single_test_leaves_real_state_untouched(self):
        self._run_and_check(
            ["tests.test_phone_tls.CertificateGenerationTests."
             "test_first_call_creates_all_four_files"],
            "single test (dotted name)")


if __name__ == "__main__":
    unittest.main()
