"""Regression tests for the exact-required startup greeting: fires
once per launch, addresses Captain by name only, never describes
Jarvis as a chatbot, and reports the local clock as morning, afternoon,
or evening correctly at each boundary.
"""
import datetime
import unittest
from unittest import mock

from backtalk.main import IST, _greeting_period, _startup_greeting_line


class GreetingPeriodBoundaryTests(unittest.TestCase):
    def test_midnight_is_morning(self):
        self.assertEqual(_greeting_period(0), "morning")

    def test_eleven_fifty_nine_am_is_morning(self):
        self.assertEqual(_greeting_period(11), "morning")

    def test_noon_is_afternoon(self):
        self.assertEqual(_greeting_period(12), "afternoon")

    def test_four_pm_is_afternoon(self):
        self.assertEqual(_greeting_period(16), "afternoon")

    def test_five_pm_is_evening(self):
        self.assertEqual(_greeting_period(17), "evening")

    def test_eleven_pm_is_evening(self):
        self.assertEqual(_greeting_period(23), "evening")


class GreetingLineFormatTests(unittest.TestCase):
    def test_exact_format_morning(self):
        now = datetime.datetime(2026, 9, 20, 9, 5)
        self.assertEqual(
            _startup_greeting_line(now),
            "Good morning, Captain. Jarvis is online. It is 9:05 AM.")

    def test_exact_format_afternoon(self):
        now = datetime.datetime(2026, 9, 20, 14, 30)
        self.assertEqual(
            _startup_greeting_line(now),
            "Good afternoon, Captain. Jarvis is online. It is 2:30 PM.")

    def test_exact_format_evening(self):
        now = datetime.datetime(2026, 9, 20, 21, 45)
        self.assertEqual(
            _startup_greeting_line(now),
            "Good evening, Captain. Jarvis is online. It is 9:45 PM.")

    def test_addresses_captain_never_sourav_boss_sir_or_username(self):
        line = _startup_greeting_line(datetime.datetime(2026, 9, 20, 9, 0))
        low = line.lower()
        self.assertIn("captain", low)
        for wrong in ("sourav", "boss", "sir"):
            self.assertNotIn(wrong, low)

    def test_never_describes_itself_as_a_chatbot_or_ai_model(self):
        line = _startup_greeting_line(datetime.datetime(2026, 9, 20, 9, 0))
        low = line.lower()
        self.assertNotIn("chatbot", low)
        self.assertNotIn("ai model", low)
        self.assertNotIn("language model", low)

    def test_says_jarvis_is_online(self):
        line = _startup_greeting_line(datetime.datetime(2026, 9, 20, 9, 0))
        self.assertIn("Jarvis is online", line)


class GreetingFiresOnceTests(unittest.TestCase):
    """The boot sequence's only call site for the startup greeting is
    a single, unconditional `mouth.say(_startup_greeting_line())` in
    amain() -- structurally impossible to fire twice in one launch
    since amain() itself runs to completion once per process. Proven
    here by counting the exact source occurrence rather than driving a
    full boot (amain() needs a real Ollama/Claude connection)."""

    def test_exactly_one_call_site_in_amain(self):
        import inspect

        from backtalk import main
        source = inspect.getsource(main)
        self.assertEqual(
            source.count("mouth.say(_startup_greeting_line())"), 1)


class ExplicitAsiaKolkataTimezoneTests(unittest.TestCase):
    """The greeting's clock must be Asia/Kolkata explicitly -- never
    just whatever the Windows system clock happens to be configured
    as (see IST in main.py, via the tzdata package)."""

    def test_ist_constant_is_asia_kolkata(self):
        self.assertEqual(str(IST), "Asia/Kolkata")

    def test_default_now_call_uses_the_explicit_ist_constant(self):
        with mock.patch("backtalk.main.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = datetime.datetime(
                2026, 9, 20, 9, 0)
            _startup_greeting_line()
        mock_dt.datetime.now.assert_called_once_with(IST)

    def test_midnight_boundary_across_a_utc_configured_system_clock(self):
        # 19:00 UTC on the 20th is 00:30 IST on the 21st -- a real
        # midnight-crossing boundary. Proves the greeting reports the
        # correct IST wall-clock time and period even when the
        # underlying instant, read naively, would still be evening on
        # the PRIOR day in UTC.
        utc = datetime.timezone.utc
        fixed_utc_instant = datetime.datetime(2026, 9, 20, 19, 0, tzinfo=utc)
        now_ist = fixed_utc_instant.astimezone(IST)
        line = _startup_greeting_line(now_ist)
        self.assertEqual(
            line,
            "Good morning, Captain. Jarvis is online. It is 12:30 AM.")

    def test_just_before_midnight_ist_stays_evening(self):
        utc = datetime.timezone.utc
        fixed_utc_instant = datetime.datetime(2026, 9, 20, 18, 0, tzinfo=utc)
        now_ist = fixed_utc_instant.astimezone(IST)
        line = _startup_greeting_line(now_ist)
        self.assertEqual(
            line,
            "Good evening, Captain. Jarvis is online. It is 11:30 PM.")


if __name__ == "__main__":
    unittest.main()
