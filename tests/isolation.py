"""Explicit, invocation-agnostic redirect of every real-state path this
test suite must never touch: the real Day Journal ledger, the real
D:\\JARVIS-Vault daily-notes directory, this machine's real paired-phone
device store, this machine's real local CA/leaf certificates
(.phone_tls/), and this machine's real backtalk.json.

2026-09-21 incident: the original version of this redirect lived only
in tests/__init__.py and relied on `unittest discover` importing the
tests PACKAGE (running __init__.py) before any test module's code
could run. That assumption is false. `python -m unittest discover -s
tests -p <pattern>` (no explicit -t) sets top_level_dir = start_dir =
"tests" (see CPython's unittest/loader.py, TestLoader.discover): since
start_dir == top_level_dir, the "does this directory have __init__.py"
importability check is skipped entirely, and every matched test file
is imported as a bare top-level module -- "tests" is never imported as
a package, and its __init__.py never runs. The old fixture only ever
activated because test_zzz_real_state_unchanged.py (sorted alphabetically
last, so imported last during collection, before any test method
executes) happened to contain its own `import tests` statement, an
independent absolute import that Python resolves normally regardless of
how unittest itself located that file. Any discovery pattern that
excludes that one file -- e.g. `-p "test_phone_*.py"`, or running a
single test by dotted name -- skipped the redirect completely and let
three test runs write real certificates (for the fake IP
100.101.102.103) into this machine's real .phone_tls/, destroying the
real CA. See tests/test_isolation_regression.py for the regression
tests that now prove this can't happen again.

The fix: isolation must never depend on which files a discovery run
happens to import. Every test file that touches phone_auth, phone_tls,
or any other real-state module without passing its own explicit
override path must import this module and call ensure_isolated() at
its own top level -- that is an independent `import` statement in that
file's own source, evaluated the moment THAT file is loaded, no matter
how it was selected (full suite, a narrowed -p pattern, or one test
picked by dotted name). tests/__init__.py still calls it too, purely
as a convenience for the common case -- it is no longer the only thing
standing between a test run and real state.
"""
from __future__ import annotations

import atexit
import hashlib
import shutil
import tempfile
import threading
from pathlib import Path

from backtalk import config, day_journal, phone_auth, phone_tls
from backtalk.brains import vault_context

_lock = threading.Lock()
_isolated = False
_fixture_dir: Path | None = None
_baseline_hashes: dict[str, str] = {}

# Populated by ensure_isolated(), once, from each module's own real
# path -- captured BEFORE that module is redirected. Never hardcoded
# here: if a module's real path computation ever changes, this stays
# correct because it reads the live attribute at that moment.
REAL_LEDGER_DIR: Path
REAL_VAULT_DAILY_NOTES_DIR: Path
REAL_PHONE_DEVICES_FILE: Path
REAL_PHONE_TLS_DIR: Path
REAL_BACKTALK_CONFIG: Path


def _hash_tree(path: Path) -> str:
    """A stable hash of every file's relative path + content under
    `path`, or a fixed sentinel if the path doesn't exist -- "doesn't
    exist" and "exists but empty" must hash differently so a test
    can't accidentally create-then-empty a directory and still pass."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()


def _hash_file(path: Path) -> str:
    """Same idea as _hash_tree, for a single real file."""
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def real_state_paths() -> dict[str, Path]:
    """This machine's actual, non-redirected paths. Only valid after
    ensure_isolated() has run at least once (that's when the REAL_*
    globals get captured, before anything is redirected)."""
    return {
        "ledger": REAL_LEDGER_DIR,
        "vault_daily_notes": REAL_VAULT_DAILY_NOTES_DIR,
        "phone_devices": REAL_PHONE_DEVICES_FILE,
        "phone_tls": REAL_PHONE_TLS_DIR,
        "backtalk_config": REAL_BACKTALK_CONFIG,
    }


def real_state_hashes() -> dict[str, str]:
    """Hashes of the REAL paths, computed fresh right now. Compare
    this against baseline_hashes() to prove a test run never touched
    real state."""
    out: dict[str, str] = {}
    for name, path in real_state_paths().items():
        if name in ("phone_devices", "backtalk_config"):
            out[name] = _hash_file(path)
        else:
            out[name] = _hash_tree(path)
    return out


def baseline_hashes() -> dict[str, str]:
    """The real-state hashes captured at the exact moment isolation
    first activated in this process -- before any redirect happened,
    so this is what "untouched" means for this process."""
    return dict(_baseline_hashes)


def is_isolated() -> bool:
    return _isolated


def fixture_dir() -> Path:
    """The one temp directory every redirected path lives under in
    this process. Raises if ensure_isolated() hasn't run yet."""
    if _fixture_dir is None:
        raise RuntimeError("isolation.ensure_isolated() has not run yet")
    return _fixture_dir


def ensure_isolated() -> None:
    """Redirect every real-state module to one isolated temp directory
    for the lifetime of this process. Idempotent and thread-safe: call
    this at the top level of every test file that touches phone_auth,
    phone_tls, day_journal, vault_context, or backtalk.config without
    passing its own explicit override -- only the FIRST call in this
    process does anything; every later call (from another test file,
    or from tests/__init__.py) is a fast no-op.
    """
    global _isolated, _fixture_dir
    global REAL_LEDGER_DIR, REAL_VAULT_DAILY_NOTES_DIR
    global REAL_PHONE_DEVICES_FILE, REAL_PHONE_TLS_DIR, REAL_BACKTALK_CONFIG

    with _lock:
        if _isolated:
            return

        REAL_LEDGER_DIR = day_journal.LEDGER_DIR
        REAL_VAULT_DAILY_NOTES_DIR = vault_context.VAULT_ROOT / "01 - Daily Notes"
        REAL_PHONE_DEVICES_FILE = phone_auth.DEVICES_FILE
        REAL_PHONE_TLS_DIR = phone_tls.TLS_DIR
        REAL_BACKTALK_CONFIG = config.CONFIG_PATH

        # Captured BEFORE anything below is redirected -- the real,
        # live state, exactly once, at the first activation in this
        # process.
        _baseline_hashes.update(real_state_hashes())

        fixture = Path(tempfile.mkdtemp(prefix="backtalk_test_fixture_"))
        _fixture_dir = fixture

        day_journal.LEDGER_DIR = fixture / "day_journal"

        vault_context.VAULT_ROOT = fixture / "vault"
        vault_context.VAULT_INDEX = vault_context.VAULT_ROOT / "VAULT-INDEX.md"
        vault_context.CAPTAIN_PROFILE = vault_context.VAULT_ROOT / "Captain Profile.md"

        phone_auth.DEVICES_FILE = fixture / ".phone_devices.json"

        phone_tls.TLS_DIR = fixture / ".phone_tls"
        phone_tls.CA_KEY_FILE = phone_tls.TLS_DIR / "ca.key"
        phone_tls.CA_CERT_FILE = phone_tls.TLS_DIR / "ca.crt"
        phone_tls.LEAF_KEY_FILE = phone_tls.TLS_DIR / "leaf.key"
        phone_tls.LEAF_CERT_FILE = phone_tls.TLS_DIR / "leaf.crt"

        # Points at a file that will never exist under the fixture dir
        # -- config.load() treats a missing file as "use defaults",
        # the same behavior a fresh real machine would see.
        config.CONFIG_PATH = fixture / "backtalk.json"

        # Canary: verify the redirect actually took, right now, rather
        # than trusting it silently. If this ever fails, every test
        # that would have used these modules refuses to run at all
        # instead of quietly writing to a real path again.
        for name, path in {
            "day_journal.LEDGER_DIR": day_journal.LEDGER_DIR,
            "vault_context.VAULT_ROOT": vault_context.VAULT_ROOT,
            "phone_auth.DEVICES_FILE": phone_auth.DEVICES_FILE,
            "phone_tls.TLS_DIR": phone_tls.TLS_DIR,
            "config.CONFIG_PATH": config.CONFIG_PATH,
        }.items():
            if fixture != path and fixture not in path.parents:
                raise RuntimeError(
                    f"test isolation FAILED for {name}: {path!r} is not "
                    f"under the fixture directory {fixture!r} -- refusing "
                    f"to let any test run against real state")

        _isolated = True
        atexit.register(_cleanup)


def _cleanup() -> None:
    if _fixture_dir is not None:
        shutil.rmtree(_fixture_dir, ignore_errors=True)
