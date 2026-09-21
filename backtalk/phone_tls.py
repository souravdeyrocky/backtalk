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
"""Local certificate authority for phone.mode "tailscale" -- a self-
signed CA generated ONCE on this machine (pure Python, via
`cryptography`; no mkcert binary, no external service, EUR0/INR0),
which then signs a short-lived LEAF certificate covering whatever
Tailscale IP the phone bridge is currently bound to (100.64.0.0/10 --
see phone_auth.detect_tailscale_ip). There is no LAN mode any more:
the phone bridge only ever binds to the Tailscale interface.

THE TRUST MODEL, exactly as Captain specified: the CA's public
certificate (ca.crt) is NOT trusted just because it downloaded over
plain HTTP. The only thing that makes trust safe is comparing the CA's
SHA-256 fingerprint against a value Captain reads or scans from THIS
PROCESS'S OWN TERMINAL (print_verification_screen), a channel neither
a network attacker nor a compromised tailnet peer can touch. The phone
page's own displayed fingerprint (computed client-side from whatever
bytes it received) is documented there as a CONVENIENCE check only --
never the security-critical comparison. SHA-256 is the only accepted
value anywhere in this flow; SHA-1 is informational-only metadata.

FAIL CLOSED: rotating the CA (delete_all) produces a new CA and a new
leaf. A phone that only trusts the old CA gets a hard TLS handshake
failure against the new leaf -- there is no HTTP fallback anywhere in
this module, and none will be added.

The CA PRIVATE key never leaves this machine: never served by any
endpoint (only ca.crt, the public half, is ever handed to a phone),
never logged, never written under D:\\JARVIS-Vault, and file-permission
restricted to this Windows account via icacls the moment it's written.
"""
from __future__ import annotations

import datetime
import ipaddress
import subprocess
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from backtalk.vlog import log

TLS_DIR = Path(__file__).resolve().parent.parent / ".phone_tls"
CA_KEY_FILE = TLS_DIR / "ca.key"
CA_CERT_FILE = TLS_DIR / "ca.crt"
LEAF_KEY_FILE = TLS_DIR / "leaf.key"
LEAF_CERT_FILE = TLS_DIR / "leaf.crt"

CA_VALIDITY_DAYS = 1825          # ~5 years -- long-lived, rarely rotated
LEAF_VALIDITY_DAYS = 90          # short-lived, auto-renewed, zero phone action
LEAF_REISSUE_MARGIN_DAYS = 14    # reissue once within this many days of expiry

# Defense-in-depth: this CA can validly sign a cert for a TAILSCALE
# address only (100.64.0.0/10, the shared CGNAT range Tailscale
# allocates from) -- never a public IP, never a real internet domain,
# and no longer any home-LAN range either now that the LAN mode is
# gone. Even a fully compromised copy of ca.crt is cryptographically
# incapable of vouching for anything outside your own tailnet. (Honest
# caveat, stated once here rather than repeated at every call site:
# X.509 Name Constraints enforcement has historically been inconsistent
# on some older Android builds -- treat this as a second layer, not the
# sole guarantee. The primary guarantee is Captain's own eyes-on
# SHA-256 comparison in print_verification_screen/the phone page.)
_PERMITTED_NETS = ["100.64.0.0/10"]


def _lock_key_file_windows(path: Path) -> None:
    """Best-effort ACL lock: strip inherited permissions (which on some
    setups grant broader groups like Users/Everyone read access), grant
    ONLY this Windows account full control. Full control, not read-only
    -- the goal is excluding every OTHER account, never blocking this
    same process from reading OR later rewriting/rotating its own key;
    an earlier read-only version of this function locked itself out of
    its own reissue path, which is exactly the bug this comment is
    warning the next editor away from re-introducing.

    Never raises -- a locked-down filesystem permission is defense-in-
    depth on a single-user machine, not the only thing standing between
    the key and the world, and a failure here must never block the
    (much more important) key generation itself from completing."""
    if sys.platform != "win32":
        return
    import os
    user = os.environ.get("USERNAME", "")
    if not user:
        return
    try:
        subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
            capture_output=True, timeout=10, check=False)
    except Exception as e:
        log(f"[phone-tls] could not lock {path.name}'s file permissions "
            f"(non-fatal): {e}")


def fingerprints(cert: x509.Certificate) -> dict:
    """SHA-256 (primary) and SHA-1 (shown too -- some phones' native
    install screens only surface SHA-1) as colon-separated uppercase
    hex, the same convention `openssl x509 -fingerprint` uses, so it
    reads the same on the terminal and on a phone's own cert UI."""
    def _fmt(digest: bytes) -> str:
        return ":".join(f"{b:02X}" for b in digest)
    return {
        "sha256": _fmt(cert.fingerprint(hashes.SHA256())),
        "sha1": _fmt(cert.fingerprint(hashes.SHA1())),
    }


def _build_ca(hostname: str) -> tuple[bytes, bytes]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    subject = issuer = x509.Name([x509.NameAttribute(
        NameOID.COMMON_NAME,
        f"Jarvis Local CA - {hostname} - {now:%Y-%m-%d}")])
    builder = (x509.CertificateBuilder()
              .subject_name(subject).issuer_name(issuer)
              .public_key(key.public_key())
              .serial_number(x509.random_serial_number())
              .not_valid_before(now)
              .not_valid_after(now + datetime.timedelta(days=CA_VALIDITY_DAYS))
              .add_extension(x509.BasicConstraints(ca=True, path_length=0),
                             critical=True)
              .add_extension(x509.KeyUsage(
                  digital_signature=False, content_commitment=False,
                  key_encipherment=False, data_encipherment=False,
                  key_agreement=False, key_cert_sign=True, crl_sign=True,
                  encipher_only=False, decipher_only=False), critical=True)
              .add_extension(x509.SubjectKeyIdentifier.from_public_key(
                  key.public_key()), critical=False)
              .add_extension(x509.NameConstraints(
                  permitted_subtrees=[
                      x509.IPAddress(ipaddress.ip_network(net))
                      for net in _PERMITTED_NETS],
                  excluded_subtrees=None), critical=True))
    cert = builder.sign(key, hashes.SHA256())
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return key_pem, cert_pem


def _build_leaf(ca_key_pem: bytes, ca_cert_pem: bytes,
                bind_ip: str) -> tuple[bytes, bytes]:
    ca_key = serialization.load_pem_private_key(ca_key_pem, password=None)
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, bind_ip)])
    builder = (x509.CertificateBuilder()
              .subject_name(subject).issuer_name(ca_cert.subject)
              .public_key(key.public_key())
              .serial_number(x509.random_serial_number())
              .not_valid_before(now)
              .not_valid_after(now + datetime.timedelta(days=LEAF_VALIDITY_DAYS))
              .add_extension(x509.BasicConstraints(ca=False, path_length=None),
                             critical=True)
              .add_extension(x509.KeyUsage(
                  digital_signature=True, content_commitment=False,
                  key_encipherment=True, data_encipherment=False,
                  key_agreement=False, key_cert_sign=False, crl_sign=False,
                  encipher_only=False, decipher_only=False), critical=True)
              .add_extension(x509.ExtendedKeyUsage(
                  [x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
              .add_extension(x509.SubjectAlternativeName(
                  [x509.IPAddress(ipaddress.ip_address(bind_ip))]),
                  critical=False))
    cert = builder.sign(ca_key, hashes.SHA256())
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption())
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return key_pem, cert_pem


def _leaf_covers(cert: x509.Certificate, bind_ip: str) -> bool:
    try:
        san = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value
        ips = san.get_values_for_type(x509.IPAddress)
        return ipaddress.ip_address(bind_ip) in ips
    except x509.ExtensionNotFound:
        return False


def _needs_leaf_reissue(cert: x509.Certificate, bind_ip: str) -> bool:
    if not _leaf_covers(cert, bind_ip):
        return True
    remaining = cert.not_valid_after_utc - datetime.datetime.now(
        datetime.timezone.utc)
    return remaining < datetime.timedelta(days=LEAF_REISSUE_MARGIN_DAYS)


def ensure_certificates(bind_ip: str, hostname: str) -> dict:
    """Idempotent: generates the CA on first call, (re)issues the leaf
    whenever it's missing, covers the wrong IP, or is nearing expiry.
    Returns fingerprints() of the CA cert, for display. Called once per
    phone_bridge.start() in tailscale mode -- so a Tailscale IP that
    changed since the last launch (rare, but not impossible) is picked
    up automatically, with zero phone-side action (the phone already
    trusts the CA; any leaf that CA signs is trusted too)."""
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    if not (CA_KEY_FILE.exists() and CA_CERT_FILE.exists()):
        log("[phone-tls] generating a new local CA (first run in tailscale mode)")
        key_pem, cert_pem = _build_ca(hostname)
        # defensive: a prior partial/locked-down ca.key must not block
        # this write (see the identical reasoning on the leaf below)
        CA_KEY_FILE.unlink(missing_ok=True)
        CA_KEY_FILE.write_bytes(key_pem)
        CA_CERT_FILE.write_bytes(cert_pem)
        _lock_key_file_windows(CA_KEY_FILE)

    ca_cert_pem = CA_CERT_FILE.read_bytes()
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)

    reissue = True
    if LEAF_CERT_FILE.exists() and LEAF_KEY_FILE.exists():
        try:
            leaf_cert = x509.load_pem_x509_certificate(
                LEAF_CERT_FILE.read_bytes())
            reissue = _needs_leaf_reissue(leaf_cert, bind_ip)
        except ValueError:
            reissue = True
    if reissue:
        log(f"[phone-tls] issuing a new leaf certificate for {bind_ip} "
            f"(silent -- the phone already trusts the CA that signs it)")
        leaf_key_pem, leaf_cert_pem = _build_leaf(
            CA_KEY_FILE.read_bytes(), ca_cert_pem, bind_ip)
        # the previous leaf.key was locked read-only by _lock_key_file_
        # windows below on its own last write -- remove it first so this
        # write isn't blocked by its own prior lockdown.
        LEAF_KEY_FILE.unlink(missing_ok=True)
        LEAF_KEY_FILE.write_bytes(leaf_key_pem)
        LEAF_CERT_FILE.write_bytes(leaf_cert_pem)
        _lock_key_file_windows(LEAF_KEY_FILE)

    return fingerprints(ca_cert)


def current_ca_fingerprint() -> str | None:
    """The SHA-256 fingerprint of the CURRENTLY active CA, or None if
    none has been generated yet. Recorded on each newly paired device
    (phone_auth.redeem_pairing's ca_fingerprint param) -- never the
    private key, never a device token, just this one public value."""
    if not CA_CERT_FILE.exists():
        return None
    cert = x509.load_pem_x509_certificate(CA_CERT_FILE.read_bytes())
    return fingerprints(cert)["sha256"]


def print_verification_screen(bind_ip: str, port: int, tls_port: int) -> None:
    """The trusted reference: printed to THIS terminal (a channel no
    network attacker or tailnet peer controls), never spoken aloud (a
    misheard hex digit is worse than useless -- see the pairing code's
    own reasoning), shown every time "pair a phone" runs and on demand
    via "show certificate fingerprint".

    SHA-256 IS THE ONLY ACCEPTED VERIFICATION VALUE (Captain's explicit
    rule, 2026-09-27). SHA-1 is printed too, but strictly as
    informational/legacy metadata -- never framed as an acceptable
    substitute. A phone whose install screen can only show SHA-1 (or no
    fingerprint at all) must NOT be trusted; see bootstrap.html's
    matching instruction on the phone side."""
    import segno

    fp = fingerprints(x509.load_pem_x509_certificate(CA_CERT_FILE.read_bytes()))
    bar = "=" * 64
    log(bar)
    log("JARVIS PHONE - CERTIFICATE VERIFICATION (once per phone, or")
    log("after a CA rotation)")
    log(bar)
    log(f"On the phone's Wi-Fi browser, open: http://{bind_ip}:{port}/")
    log("")
    log("That page offers the certificate download. Before installing,")
    log("find the SHA-256 fingerprint on YOUR PHONE'S OWN certificate")
    log("install screen (not the web page) and compare it, character by")
    log("character, against the SHA-256 value below.")
    log("")
    log(f"  SHA-256 (the ONLY value that counts): {fp['sha256']}")
    log(f"  SHA-1 (informational only -- NEVER a valid substitute for")
    log(f"         SHA-256, never use this to approve installing):")
    log(f"         {fp['sha1']}")
    log("")
    log("If your phone's install screen shows SHA-256 and it matches")
    log("exactly: proceed. If it shows a DIFFERENT SHA-256 value, or")
    log("only shows SHA-1, or shows no fingerprint at all: DO NOT")
    log("INSTALL THIS CERTIFICATE ON THAT DEVICE.")
    log(bar)
    print(f"\nScan to compare the SHA-256 fingerprint independently "
          f"(your camera app, not this browser):\n")
    segno.make(f"SHA256:{fp['sha256']}", error="m").terminal(compact=True)
    print()
    log(f"Once trusted: https://{bind_ip}:{tls_port}/")


def delete_all() -> None:
    """CA rotation / full reset. FAIL CLOSED BY DESIGN: the next
    ensure_certificates() call generates a brand new CA and leaf: any
    phone that only trusts the OLD CA gets a hard TLS error against the
    new leaf, not a silent HTTP fallback (there is none anywhere in
    this module) and not silent re-trust. Re-pairing that phone means
    walking the verification screen again, on purpose."""
    for f in (CA_KEY_FILE, CA_CERT_FILE, LEAF_KEY_FILE, LEAF_CERT_FILE):
        try:
            f.unlink(missing_ok=True)
        except OSError as e:
            log(f"[phone-tls] could not remove {f.name}: {e}")
    log("[phone-tls] CA and leaf certificates deleted. Every previously "
        "paired phone will fail closed (a hard TLS error, never a "
        "silent fallback) until it re-verifies and re-trusts the next "
        "CA this machine generates.")
