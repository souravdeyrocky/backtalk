import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backtalk.brains import vault_context


class VaultContextTests(unittest.TestCase):
    def test_missing_files_never_error_and_return_empty(self):
        missing = Path(tempfile.gettempdir()) / "definitely-does-not-exist"
        with mock.patch.object(vault_context, "VAULT_ROOT", missing), \
            mock.patch.object(vault_context, "VAULT_INDEX",
                              missing / "VAULT-INDEX.md"), \
            mock.patch.object(vault_context, "CAPTAIN_PROFILE",
                              missing / "Captain Profile.md"):
            ctx = vault_context.load_context()
        self.assertEqual(ctx, "")

    def test_excerpt_is_capped_and_marks_truncation(self):
        long_text = "word " * 500  # well over the 900-char cap
        excerpt = vault_context._excerpt(long_text, limit=100)
        self.assertLessEqual(len(excerpt), 100 + len("\n...(truncated)"))
        self.assertIn("truncated", excerpt)

    def test_short_text_is_returned_whole(self):
        short = "short note"
        self.assertEqual(vault_context._excerpt(short, limit=900), short)

    def test_daily_note_excerpt_prefers_index_block(self):
        text = (
            "# Monday\n\n"
            "## Index\n- did a thing\n\n"
            "## Session 1\n" + ("filler " * 400)
        )
        excerpt = vault_context._daily_note_excerpt(text, limit=900)
        self.assertIn("did a thing", excerpt)
        self.assertNotIn("filler", excerpt)


if __name__ == "__main__":
    unittest.main()
