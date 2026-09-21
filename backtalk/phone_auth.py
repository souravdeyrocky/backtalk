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
"""Phone pairing + device tokens + rate limiting for the phone bridge
(phone_bridge.py). No reusable PIN: a phone is paired ONCE, by typing or
opening a short-lived one-time code Jarvis prints on the desktop, and
gets back a random per-device token. Only a SHA-256 hash of that token
is ever written to disk (DEVICES_FILE) -- the raw token exists only in
the pairing response and whatever the phone's browser stores it in.

Nothing here grants a paired phone anything beyond what typed text
already could: every request this module authorizes still goes through
main.py's ordinary handle()/make_permission_gate() path. This module
answers exactly one question -- "is this request from a phone Captain
already paired, and hasn't since revoked?" -- and rate-limits/logs the
answer honestly.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import threading
import time
from pathlib import Path

from backtalk.vlog import log

# Overridable by tests (see tests/__init__.py's global fixture, which
# redirects this exactly like it does day_journal.LEDGER_DIR) -- every
# function below reads this at CALL time, never caches it, so a test
# reassignment takes effect immediately.
DEVICES_FILE = Path(__file__).resolve().parent.parent / ".phone_devices.json"

# Readable alphabet for spoken/typed pairing codes: excludes 0/O and 1/I,
# the two pairs people misread most often off a screen.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_CODE_LEN = 8
_CODE_TTL_S = 180          # 3 minutes -- long enough to walk to your phone
_PAIR_ATTEMPT_LIMIT = 5    # per source IP, per _CODE_TTL_S-ish window
_PAIR_ATTEMPT_WINDOW_S = 300

_devices_lock = threading.Lock()
_pending_lock = threading.Lock()
_pending: dict | None = None   # {"code": str, "expires_at": float, "consumed": bool}


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _load_devices() -> list[dict]:
    try:
        data = json.loads(DEVICES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        log(f"[phone-auth] device store unreadable, treating as empty: {e}")
        return []


def _save_devices(devices: list[dict]) -> None:
    try:
        DEVICES_FILE.write_text(json.dumps(devices, indent=2) + "\n",
                                encoding="utf-8")
    except OSError as e:
        log(f"[phone-auth] could not save device store: {e}")


def detect_tailscale_ip() -> str | None:
    """This machine's own Tailscale interface IP (a 100.64.0.0/10
    address), asked directly of the Tailscale CLI rather than guessed
    at via routing tricks -- Tailscale already knows its own address
    with certainty, so there's no reason to infer it.

    Returns None if Tailscale isn't installed, isn't running, or isn't
    logged into a tailnet -- deliberately NO fallback to any other
    address (never the LAN, never loopback, never 0.0.0.0). The phone
    bridge must fail to start rather than silently bind somewhere a
    phone on mobile data could never reach anyway; see phone_bridge.
    start()'s handling of a None return here."""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, OSError) as e:
        log(f"[phone-auth] Tailscale CLI not found ({e}) -- is it installed?")
        return None
    except subprocess.TimeoutExpired:
        log("[phone-auth] `tailscale ip -4` timed out")
        return None
    if result.returncode != 0:
        log(f"[phone-auth] `tailscale ip -4` failed (is Tailscale running "
            f"and signed in?): {result.stderr.strip()[:200]}")
        return None
    ip = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    if not ip:
        log("[phone-auth] `tailscale ip -4` returned no address")
        return None
    return ip


def create_pairing() -> tuple[str, float]:
    """Start a new pairing window, replacing any unconsumed one. Returns
    (code, expires_at). Only ONE pairing code is live at a time --
    simpler to reason about than a set, and pairing is a rare, deliberate
    action Captain takes one phone at a time."""
    global _pending
    code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))
    expires_at = time.time() + _CODE_TTL_S
    with _pending_lock:
        _pending = {"code": code, "expires_at": expires_at, "consumed": False}
    log(f"[phone-auth] pairing code issued, expires in {_CODE_TTL_S}s")
    return code, expires_at


_pair_attempts = {}   # source_ip -> [monotonic timestamps]
_pair_attempts_lock = threading.Lock()


def _pair_attempt_allowed(source_ip: str) -> bool:
    now = time.monotonic()
    with _pair_attempts_lock:
        hits = [t for t in _pair_attempts.get(source_ip, [])
                if now - t < _PAIR_ATTEMPT_WINDOW_S]
        if len(hits) >= _PAIR_ATTEMPT_LIMIT:
            _pair_attempts[source_ip] = hits
            return False
        hits.append(now)
        _pair_attempts[source_ip] = hits
        return True


def redeem_pairing(code: str, source_ip: str, label: str,
                   ca_fingerprint: str | None = None) -> str | None:
    """Exchange a valid, unexpired, unconsumed pairing code for a fresh
    device token. Returns the RAW token (shown/stored exactly once) or
    None if the code is wrong, expired, already used, or this source IP
    has made too many attempts recently. Every failure is logged with
    the source IP and reason only -- never the code or any token.

    `ca_fingerprint` (from phone_tls.current_ca_fingerprint()) is
    recorded on the device -- a public value, never a secret -- so
    "list paired phones" can show which CA a device was paired under,
    useful the day the CA is ever rotated."""
    global _pending
    if not _pair_attempt_allowed(source_ip):
        log(f"[phone-auth] pairing REJECTED (rate limited): ip={source_ip}")
        return None
    with _pending_lock:
        p = _pending
        if p is None or p["consumed"] or time.time() > p["expires_at"]:
            log(f"[phone-auth] pairing REJECTED (no live code): ip={source_ip}")
            return None
        if not secrets.compare_digest(code, p["code"]):
            log(f"[phone-auth] pairing REJECTED (wrong code): ip={source_ip}")
            return None
        p["consumed"] = True
    token = secrets.token_urlsafe(32)
    device = {
        "id": secrets.token_hex(4),
        "label": label or "Unlabeled phone",
        "token_hash": _hash_token(token),
        "paired_at": time.time(),
        "last_seen": time.time(),
        "source_ip_at_pairing": source_ip,
        "ca_fingerprint_at_pairing": ca_fingerprint,
    }
    with _devices_lock:
        devices = _load_devices()
        devices.append(device)
        _save_devices(devices)
    log(f"[phone-auth] paired new device id={device['id']} label={device['label']!r}")
    return token


def verify_token(token: str, source_ip: str) -> dict | None:
    """Returns the device record (without the hash) on a valid, non-
    revoked token, updating last_seen. None on anything else -- a
    missing/garbled Authorization header, an unknown token, or a token
    for a device that has since been revoked. Failures are logged by
    source IP only."""
    if not token:
        return None
    h = _hash_token(token)
    with _devices_lock:
        devices = _load_devices()
        for d in devices:
            if secrets.compare_digest(d["token_hash"], h):
                d["last_seen"] = time.time()
                _save_devices(devices)
                return {k: v for k, v in d.items() if k != "token_hash"}
    log(f"[phone-auth] request REJECTED (unknown/revoked token): ip={source_ip}")
    return None


def list_devices() -> list[dict]:
    with _devices_lock:
        devices = _load_devices()
    return [{k: v for k, v in d.items() if k != "token_hash"} for d in devices]


def revoke_device(index_1based: int) -> str | None:
    """Revokes by the 1-based position list_devices() would show
    (matches how listbrains/switchnum already number things for the
    ear) -- returns the revoked device's label, or None if the index is
    out of range. Immediate: the very next request bearing that token
    gets a 401, no grace period."""
    with _devices_lock:
        devices = _load_devices()
        if not (1 <= index_1based <= len(devices)):
            return None
        removed = devices.pop(index_1based - 1)
        _save_devices(devices)
    log(f"[phone-auth] revoked device id={removed['id']} label={removed['label']!r}")
    return removed["label"]


class RateLimiter:
    """A small sliding-window limiter, keyed by whatever the caller
    wants (device id, source IP, ...). No new dependency -- this is the
    whole thing."""

    def __init__(self, max_calls: int, window_s: float):
        self.max_calls = max_calls
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits.get(key, []) if now - t < self.window_s]
            if len(hits) >= self.max_calls:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True
