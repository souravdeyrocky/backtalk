"""Unit tests for backtalk.phone_tls: local CA + leaf certificate
generation, idempotency, reissue-on-IP-change/near-expiry, fingerprint
format, and fail-closed rotation. Runs against the FIXTURE .phone_tls
directory tests/isolation.py redirected phone_tls.TLS_DIR to -- never
this machine's real one (see test_zzz_real_state_unchanged.py and
test_isolation_regression.py -- the latter exists because this file's
own real .phone_tls/ WAS overwritten once, on 2026-09-21, by a
narrowed `-p` test run that skipped the old, implicit-only redirect).

A real TLS handshake against a bound socket is deliberately NOT tested
here (unit tests shouldn't bind real ports) -- see phone_demo.py for
that end-to-end, live proof.
"""
import contextlib
import datetime
import io
import ipaddress
import unittest
from unittest.mock import patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes

from tests import isolation

isolation.ensure_isolated()  # explicit, invocation-agnostic -- see its docstring

from backtalk import phone_tls


def _reset():
    import shutil
    shutil.rmtree(phone_tls.TLS_DIR, ignore_errors=True)


class CertificateGenerationTests(unittest.TestCase):
    def setUp(self):
        _reset()

    def test_first_call_creates_all_four_files(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        self.assertTrue(phone_tls.CA_KEY_FILE.exists())
        self.assertTrue(phone_tls.CA_CERT_FILE.exists())
        self.assertTrue(phone_tls.LEAF_KEY_FILE.exists())
        self.assertTrue(phone_tls.LEAF_CERT_FILE.exists())

    def test_ca_is_a_valid_ca_with_path_length_zero(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        ca = x509.load_pem_x509_certificate(phone_tls.CA_CERT_FILE.read_bytes())
        bc = ca.extensions.get_extension_for_class(x509.BasicConstraints).value
        self.assertTrue(bc.ca)
        self.assertEqual(bc.path_length, 0)

    def test_ca_has_name_constraints_restricted_to_the_tailscale_range(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        ca = x509.load_pem_x509_certificate(phone_tls.CA_CERT_FILE.read_bytes())
        nc = ca.extensions.get_extension_for_class(x509.NameConstraints).value
        permitted = nc.permitted_subtrees
        self.assertIsNotNone(permitted)
        # a public address must never fall inside any permitted subtree
        public_ip = ipaddress.ip_address("8.8.8.8")
        self.assertFalse(any(public_ip in gn.value for gn in permitted))
        # nor should a home-LAN address -- that mode is gone entirely
        lan_ip = ipaddress.ip_address("192.168.1.5")
        self.assertFalse(any(lan_ip in gn.value for gn in permitted))
        # this machine's own Tailscale test IP must be permitted
        tailscale_ip = ipaddress.ip_address("100.101.102.103")
        self.assertTrue(any(tailscale_ip in gn.value for gn in permitted))

    def test_leaf_san_covers_the_bind_ip(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        leaf = x509.load_pem_x509_certificate(phone_tls.LEAF_CERT_FILE.read_bytes())
        san = leaf.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value
        ips = san.get_values_for_type(x509.IPAddress)
        self.assertIn(ipaddress.ip_address("100.101.102.103"), ips)

    def test_leaf_is_signed_by_the_ca(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        ca = x509.load_pem_x509_certificate(phone_tls.CA_CERT_FILE.read_bytes())
        leaf = x509.load_pem_x509_certificate(phone_tls.LEAF_CERT_FILE.read_bytes())
        self.assertEqual(leaf.issuer, ca.subject)
        # signature verifies against the CA's own public key
        ca.public_key().verify(
            leaf.signature, leaf.tbs_certificate_bytes,
            __import__("cryptography.hazmat.primitives.asymmetric.padding",
                       fromlist=["PKCS1v15"]).PKCS1v15(),
            leaf.signature_hash_algorithm)

    def test_returned_fingerprints_match_the_actual_ca_file(self):
        fp = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        ca = x509.load_pem_x509_certificate(phone_tls.CA_CERT_FILE.read_bytes())
        self.assertEqual(fp["sha256"],
                         ":".join(f"{b:02X}" for b in ca.fingerprint(hashes.SHA256())))

    def test_fingerprint_format_is_colon_separated_uppercase_hex(self):
        fp = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        parts = fp["sha256"].split(":")
        self.assertEqual(len(parts), 32)   # SHA-256 = 32 bytes
        self.assertTrue(all(len(p) == 2 and p == p.upper() for p in parts))


class IdempotencyAndReissueTests(unittest.TestCase):
    def setUp(self):
        _reset()

    def test_same_ip_does_not_regenerate_ca_or_reissue_leaf(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        ca_before = phone_tls.CA_CERT_FILE.read_bytes()
        leaf_before = phone_tls.LEAF_CERT_FILE.read_bytes()
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        self.assertEqual(phone_tls.CA_CERT_FILE.read_bytes(), ca_before)
        self.assertEqual(phone_tls.LEAF_CERT_FILE.read_bytes(), leaf_before)

    def test_ip_change_reissues_leaf_but_keeps_the_same_ca(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        ca_before = phone_tls.CA_CERT_FILE.read_bytes()
        leaf_before = phone_tls.LEAF_CERT_FILE.read_bytes()
        phone_tls.ensure_certificates("100.101.102.199", "TESTHOST")
        self.assertEqual(phone_tls.CA_CERT_FILE.read_bytes(), ca_before)
        self.assertNotEqual(phone_tls.LEAF_CERT_FILE.read_bytes(), leaf_before)
        leaf = x509.load_pem_x509_certificate(phone_tls.LEAF_CERT_FILE.read_bytes())
        san = leaf.extensions.get_extension_for_class(
            x509.SubjectAlternativeName).value
        self.assertIn(ipaddress.ip_address("100.101.102.199"),
                      san.get_values_for_type(x509.IPAddress))

    def test_near_expiry_leaf_is_reissued_even_on_the_same_ip(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        leaf = x509.load_pem_x509_certificate(phone_tls.LEAF_CERT_FILE.read_bytes())
        self.assertFalse(phone_tls._needs_leaf_reissue(leaf, "100.101.102.103"))
        # simulate a cert that expires in 1 day (well inside the
        # 14-day reissue margin)
        near_expiry = leaf.not_valid_after_utc - datetime.timedelta(
            days=phone_tls.LEAF_VALIDITY_DAYS - 1)

        class _Fake:
            not_valid_after_utc = datetime.datetime.now(
                datetime.timezone.utc) + datetime.timedelta(days=1)
            extensions = leaf.extensions
        self.assertTrue(phone_tls._needs_leaf_reissue(_Fake(), "100.101.102.103"))


class FailClosedRotationTests(unittest.TestCase):
    def setUp(self):
        _reset()

    def test_delete_all_removes_every_file(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        phone_tls.delete_all()
        for f in (phone_tls.CA_KEY_FILE, phone_tls.CA_CERT_FILE,
                 phone_tls.LEAF_KEY_FILE, phone_tls.LEAF_CERT_FILE):
            self.assertFalse(f.exists())

    def test_rotation_produces_a_genuinely_different_ca(self):
        fp1 = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        phone_tls.delete_all()
        fp2 = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        self.assertNotEqual(fp1["sha256"], fp2["sha256"])

    def test_current_ca_fingerprint_none_before_any_generation(self):
        self.assertIsNone(phone_tls.current_ca_fingerprint())

    def test_current_ca_fingerprint_matches_after_generation(self):
        fp = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        self.assertEqual(phone_tls.current_ca_fingerprint(), fp["sha256"])

    def test_current_ca_fingerprint_none_again_after_delete_all(self):
        phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")
        phone_tls.delete_all()
        self.assertIsNone(phone_tls.current_ca_fingerprint())


class VerificationScreenWordingTests(unittest.TestCase):
    """Captain's explicit rule, 2026-09-27: SHA-256 is the only
    accepted verification value; SHA-1 is informational-only metadata
    and must never be framed as an acceptable substitute or fallback."""

    def setUp(self):
        _reset()
        self.fp = phone_tls.ensure_certificates("100.101.102.103", "TESTHOST")

    def _captured_lines(self, port=8765) -> str:
        # the QR block is a bare print(), not log() -- swallow stdout
        # too so a test run doesn't spew a QR code to the console.
        with patch("backtalk.phone_tls.log") as mock_log, \
             contextlib.redirect_stdout(io.StringIO()):
            phone_tls.print_verification_screen("100.101.102.103", port, 8766)
        return " ".join(c.args[0] for c in mock_log.call_args_list)

    def test_sha256_is_labeled_as_the_only_accepted_value(self):
        lines = self._captured_lines()
        self.assertIn(self.fp["sha256"], lines)
        self.assertIn("ONLY value that counts", lines)

    def test_sha1_is_present_but_never_framed_as_a_substitute(self):
        lines = self._captured_lines()
        self.assertIn(self.fp["sha1"], lines)
        self.assertIn("NEVER a valid substitute for", lines)
        # the old wording this replaced treated SHA-1 as an acceptable
        # fallback for older phones -- must never reappear
        self.assertNotIn("older phones may only show this one", lines)

    def test_screen_instructs_do_not_install_when_sha256_unconfirmable(self):
        lines = self._captured_lines()
        self.assertIn("DO NOT", lines)
        self.assertIn("INSTALL", lines)

    def test_bootstrap_url_uses_the_actual_port_argument(self):
        lines = self._captured_lines(port=9999)
        self.assertIn("http://100.101.102.103:9999/", lines)


if __name__ == "__main__":
    unittest.main()
