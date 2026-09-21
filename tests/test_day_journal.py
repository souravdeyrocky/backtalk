"""Regression tests for the shared, brain-independent Day Journal
service: the structured local event ledger, the deterministic summary
built from it (no model call anywhere), the direct vault write (no
Claude, no Gemini, no network), and that manual and scheduled triggers
use the exact same underlying logic.
"""
import datetime as _dt
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backtalk import day_journal


class ExplicitAsiaKolkataTimezoneTests(unittest.TestCase):
    """The date boundary for the ledger and the vault daily note must
    be Asia/Kolkata explicitly -- never just whatever the Windows
    system clock happens to be configured as."""

    def test_ist_constant_is_asia_kolkata(self):
        self.assertEqual(str(day_journal.IST), "Asia/Kolkata")

    def test_today_in_ist_always_passes_an_explicit_tz(self):
        # A fake datetime.now() that raises unless called with an
        # explicit tz -- proves _today_in_ist() never falls back to a
        # naive, system-local reading.
        class _FakeDatetime(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    raise AssertionError("called datetime.now() without "
                                         "an explicit tz")
                return _dt.datetime(2026, 9, 20, 12, 0, tzinfo=tz)

        with mock.patch("backtalk.day_journal._dt.datetime", _FakeDatetime):
            day_journal._today_in_ist()  # must not raise

    def test_midnight_boundary_across_a_utc_configured_system_clock(self):
        # 19:00 UTC on the 20th is 00:30 IST on the 21st -- a real
        # midnight-crossing boundary case. If _today_in_ist() ever
        # silently used the system's own (here: UTC) interpretation of
        # "today" instead of converting through Asia/Kolkata
        # explicitly, this would report the 20th instead of the 21st.
        utc = _dt.timezone.utc
        fixed_utc_instant = _dt.datetime(2026, 9, 20, 19, 0, tzinfo=utc)

        class _FakeDatetime(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_utc_instant.astimezone(tz)

        with mock.patch("backtalk.day_journal._dt.datetime", _FakeDatetime):
            today = day_journal._today_in_ist()
        self.assertEqual(today, _dt.date(2026, 9, 21))

    def test_just_before_midnight_ist_is_still_the_earlier_day(self):
        # 18:00 UTC on the 20th is 23:30 IST on the 20th -- must NOT
        # roll over yet.
        utc = _dt.timezone.utc
        fixed_utc_instant = _dt.datetime(2026, 9, 20, 18, 0, tzinfo=utc)

        class _FakeDatetime(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_utc_instant.astimezone(tz)

        with mock.patch("backtalk.day_journal._dt.datetime", _FakeDatetime):
            today = day_journal._today_in_ist()
        self.assertEqual(today, _dt.date(2026, 9, 20))


class RecordEventTests(unittest.TestCase):
    def test_recorded_event_appears_in_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = _dt.date(2026, 9, 20)
            day_journal.record_event("brain_switch", "Switched to Claude",
                                     today=today, ledger_dir=ledger_dir)
            events = day_journal._read_events(today, ledger_dir=ledger_dir)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "brain_switch")
        self.assertEqual(events[0]["detail"], "Switched to Claude")

    def test_unrecognized_kind_is_filed_as_error_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = _dt.date(2026, 9, 20)
            day_journal.record_event("made_up_kind", "something",
                                     today=today, ledger_dir=ledger_dir)
            events = day_journal._read_events(today, ledger_dir=ledger_dir)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "error")

    def test_detail_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = _dt.date(2026, 9, 20)
            day_journal.record_event("error", "x" * 2000,
                                     today=today, ledger_dir=ledger_dir)
            events = day_journal._read_events(today, ledger_dir=ledger_dir)
        self.assertLessEqual(len(events[0]["detail"]), 500)

    def test_broken_ledger_write_never_raises(self):
        # An unwritable directory (a file where a directory should be)
        # must degrade silently, same rule vlog.log() follows.
        with tempfile.TemporaryDirectory() as tmp:
            blocked = Path(tmp) / "blocked"
            blocked.write_text("not a directory")
            day_journal.record_event(
                "error", "x", today=_dt.date(2026, 9, 20),
                ledger_dir=blocked)  # must not raise

    def test_corrupted_ledger_line_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = _dt.date(2026, 9, 20)
            path = day_journal._ledger_path(today, ledger_dir=ledger_dir)
            path.write_text("not valid json\n"
                            '{"at": "x", "kind": "error", "detail": "ok"}\n')
            events = day_journal._read_events(today, ledger_dir=ledger_dir)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["detail"], "ok")


class NeverInventsWorkTests(unittest.TestCase):
    """The hard requirement: a day with no ledger events must never
    produce a summary that implies something happened."""

    def test_no_events_with_tracking_started_earlier_gives_honest_no_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            # Tracking started YESTERDAY -- a genuinely quiet today is
            # honestly reportable as "no activity".
            (ledger_dir).mkdir(parents=True, exist_ok=True)
            (ledger_dir / "_tracking_started_at.txt").write_text(
                _dt.datetime(2026, 9, 19, 10, 0,
                            tzinfo=day_journal.IST).isoformat(),
                encoding="utf-8")
            summary = day_journal.build_daily_summary(
                _dt.date(2026, 9, 20), ledger_dir=ledger_dir)
        self.assertFalse(summary.had_any_activity)
        self.assertEqual(summary.event_count, 0)
        self.assertEqual(summary.body(), day_journal.NO_ACTIVITY_NOTE)
        self.assertEqual(summary.spoken(), day_journal.NO_ACTIVITY_NOTE)

    def test_no_events_categories_are_all_honest_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = day_journal.build_daily_summary(
                _dt.date(2026, 9, 20), ledger_dir=Path(tmp))
        self.assertEqual(summary.work_completed, day_journal._NO_WORK)
        self.assertEqual(summary.decisions_made, day_journal._NO_DECISIONS)
        self.assertEqual(summary.blockers_risks, day_journal._NO_RISKS)
        self.assertEqual(summary.next_actions, day_journal._NO_NEXT_ACTIONS)

    def test_summary_is_a_pure_function_of_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = _dt.date(2026, 9, 20)
            day_journal.record_event("approved_action", "use the Write tool",
                                     today=today, ledger_dir=ledger_dir)
            first = day_journal.build_daily_summary(today, ledger_dir=ledger_dir)
            second = day_journal.build_daily_summary(today, ledger_dir=ledger_dir)
        self.assertEqual(first.body(), second.body())

    def test_next_actions_are_never_derived_only_the_fixed_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = _dt.date(2026, 9, 20)
            day_journal.record_event("approved_action", "did something",
                                     today=today, ledger_dir=ledger_dir)
            day_journal.record_event("brain_switch", "switched brains",
                                     today=today, ledger_dir=ledger_dir)
            summary = day_journal.build_daily_summary(today, ledger_dir=ledger_dir)
        self.assertEqual(summary.next_actions, day_journal._NO_NEXT_ACTIONS)


class TrackingProvenanceTests(unittest.TestCase):
    """Requirement: an empty ledger must never claim "no activity"
    when tracking itself only began partway through today -- earlier
    activity that happened before tracking started is real activity
    this module structurally cannot know about, and must say so
    instead of implying a quiet day."""

    def test_first_ever_call_marks_tracking_as_starting_now(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = day_journal._today_in_ist()
            summary = day_journal.build_daily_summary(
                today, ledger_dir=ledger_dir)
        self.assertFalse(summary.had_any_activity)
        self.assertEqual(summary.tracking_started_at.date(), today)
        self.assertNotEqual(summary.no_activity_note,
                            day_journal.NO_ACTIVITY_NOTE)
        self.assertIn("tracking began", summary.no_activity_note)
        self.assertIn("earlier activity today is not included",
                      summary.no_activity_note)

    def test_tracking_started_today_never_claims_plain_no_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = day_journal._today_in_ist()
            # Marker written for TODAY, earlier this same day.
            marker = day_journal._marker_path(ledger_dir=ledger_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            started = day_journal._dt.datetime.now(day_journal.IST).replace(
                hour=9, minute=0, second=0, microsecond=0)
            marker.write_text(started.isoformat(), encoding="utf-8")
            summary = day_journal.build_daily_summary(
                today, ledger_dir=ledger_dir)
        self.assertFalse(summary.had_any_activity)
        self.assertNotEqual(summary.body(), day_journal.NO_ACTIVITY_NOTE)
        self.assertIn("9:00 AM", summary.body())

    def test_tracking_started_before_today_gives_honest_no_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            marker = day_journal._marker_path(ledger_dir=ledger_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                day_journal._dt.datetime(
                    2020, 1, 1, 8, 0, tzinfo=day_journal.IST).isoformat(),
                encoding="utf-8")
            summary = day_journal.build_daily_summary(
                day_journal._today_in_ist(), ledger_dir=ledger_dir)
        self.assertEqual(summary.body(), day_journal.NO_ACTIVITY_NOTE)
        self.assertEqual(summary.spoken(), day_journal.NO_ACTIVITY_NOTE)

    def test_marker_is_never_overwritten_once_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            first = day_journal._ensure_tracking_started_marker(
                ledger_dir=ledger_dir)
            second = day_journal._ensure_tracking_started_marker(
                ledger_dir=ledger_dir)
        self.assertEqual(first, second)

    def test_record_event_also_establishes_the_marker(self):
        # The marker must be set at the EARLIEST point day_journal
        # does anything at all, whether that's build_daily_summary()
        # or record_event() firing first.
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            marker = day_journal._marker_path(ledger_dir=ledger_dir)
            self.assertFalse(marker.exists())
            day_journal.record_event("brain_switch", "Switched to Qwen",
                                     ledger_dir=ledger_dir)
            self.assertTrue(marker.exists())

    def test_marker_has_activity_and_activity_summary_both_present(self):
        # When there WAS activity today, no_activity_note is computed
        # but simply unused by body()/spoken() -- proven here so a
        # future refactor can't accidentally start leaking it.
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = day_journal._today_in_ist()
            day_journal.record_event("brain_switch", "Switched to Claude",
                                     today=today, ledger_dir=ledger_dir)
            summary = day_journal.build_daily_summary(
                today, ledger_dir=ledger_dir)
        self.assertTrue(summary.had_any_activity)
        self.assertNotIn(summary.no_activity_note, summary.body())
        self.assertNotIn(summary.no_activity_note, summary.spoken())


class EvidenceGroundedCategoriesTests(unittest.TestCase):
    def _summary_with(self, events):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_dir = Path(tmp)
            today = _dt.date(2026, 9, 20)
            for kind, detail in events:
                day_journal.record_event(kind, detail, today=today,
                                         ledger_dir=ledger_dir)
            return day_journal.build_daily_summary(today, ledger_dir=ledger_dir)

    def test_approved_action_appears_in_work_completed(self):
        summary = self._summary_with(
            [("approved_action", "use the WebSearch tool")])
        self.assertIn("use the WebSearch tool", summary.work_completed)

    def test_denied_action_appears_in_blockers_never_work_completed(self):
        summary = self._summary_with([("denied_action", "run a rm command")])
        self.assertEqual(summary.work_completed, day_journal._NO_WORK)
        self.assertIn("run a rm command", summary.blockers_risks)

    def test_error_appears_in_blockers(self):
        summary = self._summary_with(
            [("error", "Gemini didn't respond in time")])
        self.assertIn("Gemini didn't respond in time", summary.blockers_risks)

    def test_brain_switch_appears_in_decisions_made(self):
        summary = self._summary_with(
            [("brain_switch", "Switched to Claude Agent SDK")])
        self.assertIn("Switched to Claude Agent SDK", summary.decisions_made)

    def test_had_any_activity_true_with_a_single_event(self):
        summary = self._summary_with([("brain_switch", "Switched to Qwen")])
        self.assertTrue(summary.had_any_activity)
        self.assertEqual(summary.event_count, 1)


class WriteDailyNoteScopeTests(unittest.TestCase):
    """The note-write is scoped to exactly one file, and never calls
    Claude, Gemini, or any other brain/network -- proven structurally,
    not just by mocking a network client that's never even imported
    here."""

    def test_module_never_imports_a_brain_or_network_library(self):
        source = inspect.getsource(day_journal)
        for forbidden in ("claude_brain", "gemini_brain", "ollama_brain",
                          "import httpx", "WarmBrain", "BrainRouter"):
            self.assertNotIn(forbidden, source)

    def _fake_summary(self, **overrides) -> "day_journal.DailySummary":
        """A directly-constructed DailySummary for tests that only
        care about write_daily_note()'s own behavior, not provenance
        logic (covered separately in TrackingProvenanceTests)."""
        defaults = dict(
            today=_dt.date(2026, 9, 20), work_completed="did a thing",
            decisions_made=day_journal._NO_DECISIONS,
            current_status="1 recorded event today.",
            blockers_risks=day_journal._NO_RISKS,
            next_actions=day_journal._NO_NEXT_ACTIONS,
            event_count=1, had_any_activity=True,
            tracking_started_at=_dt.datetime(2020, 1, 1, 0, 0,
                                             tzinfo=day_journal.IST),
            no_activity_note=day_journal.NO_ACTIVITY_NOTE)
        defaults.update(overrides)
        return day_journal.DailySummary(**defaults)

    def test_fresh_note_is_created_with_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            note_path = Path(tmp) / "2026-09-20.md"
            summary = self._fake_summary()
            returned_path = day_journal.write_daily_note(
                summary, note_path=note_path)
            self.assertEqual(returned_path, note_path)
            text = note_path.read_text(encoding="utf-8")
        self.assertIn("did a thing", text)
        self.assertIn("## Day Journal (automatic)", text)

    def test_no_activity_writes_the_short_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            note_path = Path(tmp) / "2026-09-20.md"
            ledger_dir = Path(tmp) / "ledger"
            # Tracking started well before today -- a genuinely quiet
            # day, not a "tracking just started" caveat (that behavior
            # is covered by TrackingProvenanceTests).
            marker = day_journal._marker_path(ledger_dir=ledger_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                _dt.datetime(2020, 1, 1, 0, 0,
                            tzinfo=day_journal.IST).isoformat(),
                encoding="utf-8")
            summary = day_journal.build_daily_summary(
                _dt.date(2026, 9, 20), ledger_dir=ledger_dir)
            day_journal.write_daily_note(summary, note_path=note_path)
            text = note_path.read_text(encoding="utf-8")
        self.assertIn(day_journal.NO_ACTIVITY_NOTE, text)

    def test_rerun_same_day_replaces_not_duplicates_the_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            note_path = Path(tmp) / "2026-09-20.md"
            summary1 = self._fake_summary(work_completed="first run")
            day_journal.write_daily_note(summary1, note_path=note_path)
            summary2 = self._fake_summary(work_completed="second run",
                                          current_status="y", event_count=2)
            day_journal.write_daily_note(summary2, note_path=note_path)
            text = note_path.read_text(encoding="utf-8")
        self.assertNotIn("first run", text)
        self.assertIn("second run", text)
        self.assertEqual(text.count("## Day Journal (automatic)"), 1)

    def test_existing_unrelated_note_content_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            note_path = Path(tmp) / "2026-09-20.md"
            note_path.write_text(
                "# 2026-09-20\n\n## Session 1\n\nSome real notes here.\n")
            summary = self._fake_summary()
            day_journal.write_daily_note(summary, note_path=note_path)
            text = note_path.read_text(encoding="utf-8")
        self.assertIn("Some real notes here.", text)
        self.assertIn("did a thing", text)

    def test_write_never_touches_any_path_other_than_note_path_and_the_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            note_path = tmp_path / "only_this_one.md"
            ledger_dir = tmp_path / "ledger"
            summary = day_journal.build_daily_summary(
                _dt.date(2026, 9, 20), ledger_dir=ledger_dir)
            day_journal.write_daily_note(summary, note_path=note_path)
            created = {p for p in tmp_path.rglob("*") if p.is_file()}
        # build_daily_summary() legitimately creates the tracking-start
        # marker under ledger_dir (never inside the vault note's own
        # directory) -- write_daily_note() itself still only ever
        # touches note_path.
        marker = day_journal._marker_path(ledger_dir=ledger_dir)
        self.assertEqual(created, {note_path, marker})


class ManualAndScheduledUseTheSameLogicTests(unittest.TestCase):
    """The manual "close the day now" console verb and the scheduled
    CLI entry point both call run_and_write(), which in turn calls the
    exact same build_daily_summary()/write_daily_note() pair -- one
    code path, two triggers."""

    def test_run_and_write_calls_the_same_build_and_write_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            note_path = tmp_path / "note.md"
            ledger_dir = tmp_path / "ledger"
            today = _dt.date(2026, 9, 20)
            day_journal.record_event("brain_switch", "Switched to Claude",
                                     today=today, ledger_dir=ledger_dir)
            returned = day_journal.run_and_write(
                today, ledger_dir=ledger_dir, note_path=note_path)
            direct_summary = day_journal.build_daily_summary(
                today, ledger_dir=ledger_dir)
            self.assertEqual(returned, note_path)
            text = note_path.read_text(encoding="utf-8")
        self.assertIn(direct_summary.work_completed
                       if direct_summary.had_any_activity
                       else day_journal.NO_ACTIVITY_NOTE, text)

    def test_cli_entry_point_calls_run_and_write(self):
        source = inspect.getsource(day_journal)
        self.assertIn("run_and_write()", source)
        self.assertIn('if __name__ == "__main__"', source)

    def test_main_py_console_verb_uses_the_same_module_functions(self):
        import inspect as _inspect

        from backtalk import main
        source = _inspect.getsource(main)
        self.assertIn("day_journal.build_daily_summary()", source)
        self.assertIn("day_journal.write_daily_note(summary)", source)


class NoCloudOrApiCallTests(unittest.TestCase):
    def test_no_httpx_import_anywhere_in_day_journal(self):
        source = inspect.getsource(day_journal)
        self.assertNotIn("httpx", source)

    def test_no_brain_router_import(self):
        source = inspect.getsource(day_journal)
        self.assertNotIn("from backtalk.router", source)
        self.assertNotIn("import router", source)


if __name__ == "__main__":
    unittest.main()
