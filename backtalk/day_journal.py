# backtalk: talk to your Claude Code agent out loud.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One shared, brain-independent Day Journal service.

Records meaningful Jarvis events locally through the day -- brain
switches, approved actions, denied actions, and mid-turn errors -- in
a per-day structured ledger (record_event(), called from main.py at
the natural points those things already happen). At day's end,
build_daily_summary() turns that ledger into a deterministic summary
with NO model call anywhere in it -- "never invent work that did not
occur" is structural here, not a prompted hope -- and write_daily_note()
writes it directly with plain Python file I/O, scoped to exactly one
file under D:\\JARVIS-Vault. Neither function ever imports or touches
Claude, Gemini, Qwen, DeepSeek, or the network.

This REPLACES the earlier eod_summary.py design, which asked Claude's
Agent SDK to perform the actual vault write -- Captain's explicit
instruction is that the daily summary must never call an external
service or another brain to do it, so eod_summary.py is retired.

Two triggers share this exact same path:
  - Manual: "Jarvis, close the day now" (and its siblings -- see
    brain_intent.py's _CLOSE_DAY) -- main.py's closeday/closeday:confirmed
    console verbs.
  - Scheduled: a Windows Scheduled Task invoking this module's own CLI
    entry point (`python -m backtalk.day_journal --write`) once daily
    at 9:30 PM local time -- see the bottom of this file for the exact
    registration command, shown to Captain for approval before it is
    ever run; nothing here registers that task itself.

Chosen behavior for a day with no recorded activity: WRITE a short
"No Jarvis-recorded activity today." note rather than skip writing
one -- a consistent, predictable daily-note history beats a silent
gap Captain would have to notice and explain to themselves later.
"""
from __future__ import annotations

import datetime as _dt
import json
import zoneinfo
from dataclasses import dataclass
from pathlib import Path

from backtalk.brains.vault_context import _today_daily_note_path
from backtalk.vlog import log

REPO = Path(__file__).resolve().parent.parent
LEDGER_DIR = REPO / "logs" / "day_journal"

# Explicit, never assumed from the system clock's own configured
# timezone: the ledger's day-boundary and the vault daily-note date it
# writes to must both be Asia/Kolkata regardless of how the machine
# itself is configured. Requires the tzdata package (pyproject.toml).
IST = zoneinfo.ZoneInfo("Asia/Kolkata")


def _today_in_ist() -> _dt.date:
    return _dt.datetime.now(IST).date()

# The only event kinds this module understands. Never silently drop
# something the caller thought was meaningful enough to record --
# an unrecognized kind is filed as "error" so it still shows up
# somewhere in the summary rather than vanishing.
_VALID_KINDS = ("brain_switch", "approved_action", "denied_action", "error")


def _ledger_path(today: _dt.date, *, ledger_dir: Path | None = None) -> Path:
    return (ledger_dir or LEDGER_DIR) / f"{today.isoformat()}.jsonl"


def _marker_path(*, ledger_dir: Path | None = None) -> Path:
    return (ledger_dir or LEDGER_DIR) / "_tracking_started_at.txt"


def _ensure_tracking_started_marker(*, ledger_dir: Path | None = None
                                    ) -> _dt.datetime:
    """Returns the timestamp Day Journal tracking first began -- reads
    the marker if one already exists, otherwise writes it NOW (the
    earliest point either record_event() or build_daily_summary() has
    ever run for this ledger) and returns that. Never overwritten once
    written, so it always answers "since when has anything at all been
    captured here" -- the fact requirement 1 depends on: an empty
    ledger for TODAY is only genuinely "no activity" if tracking was
    ALREADY running before today began. If tracking itself only
    started partway through today (or this is the very first call this
    ledger has ever seen), an empty ledger must say so instead of
    silently implying nothing happened all day."""
    path = _marker_path(ledger_dir=ledger_dir)
    try:
        text = path.read_text(encoding="utf-8").strip()
        return _dt.datetime.fromisoformat(text)
    except (OSError, ValueError):
        pass
    now = _dt.datetime.now(IST)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(now.isoformat(), encoding="utf-8")
    except OSError as e:
        log(f"[day_journal] couldn't record tracking-start marker: {e}")
    return now


def record_event(kind: str, detail: str, *, today: _dt.date | None = None,
                 ledger_dir: Path | None = None) -> None:
    """Appends one structured, timestamped event to today's ledger.
    Never raises -- a broken ledger write must not take the voice
    line down, the same rule vlog.log() already follows for the
    human-readable session log. `detail` is bounded to 500 characters:
    a ledger entry is a short FACT ("switched to Claude"), never a
    transcript of the conversation that led to it."""
    if kind not in _VALID_KINDS:
        kind = "error"
    today = today or _today_in_ist()
    _ensure_tracking_started_marker(ledger_dir=ledger_dir)
    entry = {
        "at": _dt.datetime.now(IST).isoformat(timespec="seconds"),
        "kind": kind,
        "detail": str(detail)[:500],
    }
    path = _ledger_path(today, ledger_dir=ledger_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        log(f"[day_journal] couldn't record event ({kind}): {e}")


def _read_events(today: _dt.date, *, ledger_dir: Path | None = None) -> list[dict]:
    path = _ledger_path(today, ledger_dir=ledger_dir)
    events: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue  # one corrupted line is skipped, never fatal
    return events


_NO_WORK = "No approved actions were recorded today."
_NO_DECISIONS = "No brain switches were recorded today."
_NO_RISKS = "No blockers or errors were recorded today."
_NO_NEXT_ACTIONS = "No next actions were noted today."
_NO_STATUS = "No Jarvis activity was recorded today."
# Only honest when tracking was ALREADY running before today began --
# see build_daily_summary()'s provenance check below. Never used
# directly; always read through a built DailySummary's own
# no_activity_note, which picks the right one of these two.
NO_ACTIVITY_NOTE = "No Jarvis-recorded activity today."


def _tracking_just_started_note(tracking_started_at: _dt.datetime) -> str:
    started_str = tracking_started_at.strftime("%I:%M %p").lstrip("0")
    return (f"No activity recorded since tracking began at "
            f"{started_str} today; earlier activity today is not "
            f"included.")


@dataclass
class DailySummary:
    today: _dt.date
    work_completed: str
    decisions_made: str
    current_status: str
    blockers_risks: str
    next_actions: str
    event_count: int
    had_any_activity: bool
    tracking_started_at: _dt.datetime
    # Computed once in build_daily_summary(): NO_ACTIVITY_NOTE when
    # tracking was already running before `today` began (a genuinely
    # quiet day), or _tracking_just_started_note(...) when tracking
    # itself only started sometime during `today` -- an empty ledger
    # in that case does NOT mean nothing happened, only that nothing
    # was recorded from that point on, and must never claim otherwise.
    no_activity_note: str

    def body(self) -> str:
        """The vault-note section body."""
        if not self.had_any_activity:
            return self.no_activity_note
        return (
            f"Work Completed: {self.work_completed}\n"
            f"Decisions Made: {self.decisions_made}\n"
            f"Current Status: {self.current_status}\n"
            f"Blockers or Errors: {self.blockers_risks}\n"
            f"Next Actions: {self.next_actions}\n")

    def spoken(self) -> str:
        if not self.had_any_activity:
            return self.no_activity_note
        return (
            f"Work completed: {self.work_completed} "
            f"Decisions made: {self.decisions_made} "
            f"Current status: {self.current_status} "
            f"Blockers or errors: {self.blockers_risks} "
            f"Next actions: {self.next_actions}")


def build_daily_summary(today: _dt.date | None = None, *,
                        ledger_dir: Path | None = None) -> DailySummary:
    """Pure function of the ledger (plus the tracking-start marker,
    itself just another file read, never a clock guess): the same
    events in always produce the same summary out. No model, no
    randomness, no network -- this is what makes "never invent work
    that did not occur" true by construction rather than by prompt
    instruction."""
    today = today or _today_in_ist()
    events = _read_events(today, ledger_dir=ledger_dir)
    tracking_started_at = _ensure_tracking_started_marker(ledger_dir=ledger_dir)

    approvals = [e.get("detail", "") for e in events
                if e.get("kind") == "approved_action"]
    denials = [e.get("detail", "") for e in events
              if e.get("kind") == "denied_action"]
    switches = [e.get("detail", "") for e in events
               if e.get("kind") == "brain_switch"]
    errors = [e.get("detail", "") for e in events
             if e.get("kind") == "error"]

    work_completed = "; ".join(approvals) if approvals else _NO_WORK
    decisions_made = "; ".join(switches) if switches else _NO_DECISIONS
    blockers = denials + errors
    blockers_risks = "; ".join(blockers) if blockers else _NO_RISKS
    current_status = (
        f"{len(events)} recorded event{'s' if len(events) != 1 else ''} "
        f"today." if events else _NO_STATUS)
    # Next actions are NEVER inferred from ledger content -- no
    # mechanism exists to record one, and guessing at conversation
    # content would be exactly the fabrication this module exists to
    # refuse.
    next_actions = _NO_NEXT_ACTIONS

    # Tracking that began BEFORE today started (strictly earlier
    # calendar date) means an empty ledger for today is a genuinely
    # quiet day. Tracking that began ON today (or this being the very
    # first call this ledger has ever seen) means an empty ledger
    # cannot honestly claim "no activity" -- only that nothing was
    # captured from the moment tracking started onward.
    if tracking_started_at.date() < today:
        no_activity_note = NO_ACTIVITY_NOTE
    else:
        no_activity_note = _tracking_just_started_note(tracking_started_at)

    return DailySummary(
        today=today, work_completed=work_completed,
        decisions_made=decisions_made, current_status=current_status,
        blockers_risks=blockers_risks, next_actions=next_actions,
        event_count=len(events), had_any_activity=bool(events),
        tracking_started_at=tracking_started_at,
        no_activity_note=no_activity_note)


_SECTION_HEADING = "## Day Journal (automatic)"


def write_daily_note(summary: DailySummary, *,
                     note_path: Path | None = None) -> Path:
    """Writes or updates EXACTLY one file: today's vault daily note.
    Direct Python file I/O -- no brain, no network, no external
    service of any kind. note_path is computed from `summary.today`
    (or overridden only for tests); nothing about this function's
    control flow can be steered toward any other file, since the path
    is never built from the summary's own content.

    Idempotent within a day: a second write (e.g. a manual "close the
    day now" after the 9:30 PM scheduled run already fired) replaces
    the PRIOR automatic section rather than duplicating it, on the
    assumption that this section is always the LAST thing this module
    ever appends to that day's note -- true as long as nothing else
    is added to the note after it runs. If Captain hand-edits the
    note later in the evening, a subsequent re-run the same day would
    still replace from the marker onward; this is a known, accepted
    trade-off for keeping the update logic simple and predictable
    rather than attempting a fragile structural merge."""
    note_path = note_path or _today_daily_note_path(summary.today)
    section = f"{_SECTION_HEADING}\n\n{summary.body()}"
    try:
        existing = note_path.read_text(encoding="utf-8")
    except OSError:
        existing = None

    if existing is None:
        note_path.parent.mkdir(parents=True, exist_ok=True)
        new_text = f"# {summary.today.isoformat()}\n\n{section}\n"
    elif _SECTION_HEADING in existing:
        before = existing.split(_SECTION_HEADING, 1)[0]
        new_text = before.rstrip() + "\n\n" + section + "\n"
    else:
        new_text = existing.rstrip() + "\n\n" + section + "\n"

    note_path.write_text(new_text, encoding="utf-8")
    return note_path


def run_and_write(today: _dt.date | None = None, *,
                  ledger_dir: Path | None = None,
                  note_path: Path | None = None) -> Path:
    """The one call both triggers (manual confirm and the scheduled
    CLI below) end up making: build the summary, write it, done."""
    summary = build_daily_summary(today, ledger_dir=ledger_dir)
    return write_daily_note(summary, note_path=note_path)


# ---- Scheduled entry point -------------------------------------------
#
# `python -m backtalk.day_journal --write` runs run_and_write() once
# and exits -- what a Windows Scheduled Task actually invokes at
# 9:30 PM daily. Registering that task is a SEPARATE, explicit step:
# this file only builds the mechanism. The exact command Captain must
# approve before it's ever run (from the backtalk/ directory, with
# your own absolute paths to python.exe and this repo substituted):
#
#   schtasks /Create /TN "Jarvis Day Journal" /SC DAILY /ST 21:30 ^
#     /TR "\"C:\path\to\python.exe\" -m backtalk.day_journal --write" ^
#     /F
#
# Task name: "Jarvis Day Journal"
# Target: this module's own CLI entry point below, run from the
#   backtalk/ repo root (so `python -m backtalk.day_journal` resolves)
# Rollback (removes the task, touches nothing else):
#
#   schtasks /Delete /TN "Jarvis Day Journal" /F
#
if __name__ == "__main__":
    import sys
    if "--write" in sys.argv:
        path = run_and_write()
        print(f"[day_journal] wrote {path}")
    else:
        print("usage: python -m backtalk.day_journal --write")
        sys.exit(1)
