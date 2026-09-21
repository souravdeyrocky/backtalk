import unittest

from backtalk.brains.tool_intent import REFUSAL, looks_like_tool_request


class ToolIntentDetectionTests(unittest.TestCase):
    def test_detects_file_edit_request(self):
        self.assertTrue(looks_like_tool_request("edit the config file for me"))

    def test_detects_file_edit_request_with_a_filename_containing_a_dot(self):
        # Regression: caught live -- a period-excluding span couldn't
        # bridge "backtalk.json", so this exact phrase used to slip
        # past the gate and reach Ollama unfiltered.
        self.assertTrue(looks_like_tool_request(
            "please edit my backtalk.json file and change the ptt key"))

    def test_detects_run_command_request(self):
        self.assertTrue(looks_like_tool_request(
            "can you run a powershell command to list my files"))

    def test_detects_browse_request(self):
        self.assertTrue(looks_like_tool_request(
            "browse that website and tell me what it says"))

    def test_detects_vault_write_request(self):
        self.assertTrue(looks_like_tool_request(
            "remember this in my vault: call the dentist tomorrow"))

    def test_real_field_transcript_write_this_in_your_world(self):
        # Real live-test failure: "your world" is how Captain refers to
        # the vault/memory, and it never matched the vault-target
        # alternation (vault|memory|notes), so this reached Qwen
        # unfiltered instead of the deterministic Claude-escalation
        # refusal.
        self.assertTrue(looks_like_tool_request("write this in your world"))

    def test_real_field_transcript_write_this_in_memory(self):
        self.assertTrue(looks_like_tool_request("write this in memory"))

    def test_real_field_transcript_write_about_models_in_memory(self):
        self.assertTrue(looks_like_tool_request(
            "write about your models in memory"))

    def test_real_field_transcript_add_a_daily_note(self):
        # Real live-test failure: "add" was never in the verb
        # alternation (remember|save|write|update), so this slipped
        # past even though "daily note" is plainly a vault write.
        self.assertTrue(looks_like_tool_request("add a daily note"))

    def test_real_field_transcript_save_this_note(self):
        self.assertTrue(looks_like_tool_request("save this note"))

    def test_real_field_transcript_search_the_internet(self):
        # Real live-test failure: this exact phrase reached DeepSeek
        # unfiltered (the original pattern only recognized "browse/
        # fetch/open" as the verb, never "search"), and DeepSeek
        # answered in prose ("I'm sorry, but I can't search the
        # internet...") instead of the deterministic refusal -- and it
        # never offered the confirm-gated Claude escalation by name.
        self.assertTrue(looks_like_tool_request(
            "Jarvis, can you search the Internet and find about "
            "Kasturi Nursing Home?"))

    def test_search_the_web_variant(self):
        self.assertTrue(looks_like_tool_request(
            "can you search the web for that"))

    def test_ordinary_search_without_web_target_not_flagged(self):
        # "search" alone must not become a trigger word -- only
        # web-shaped search requests should refuse.
        self.assertFalse(looks_like_tool_request(
            "search your memory for what I said earlier"))

    def test_ordinary_conversation_is_not_flagged(self):
        self.assertFalse(looks_like_tool_request(
            "what's a good name for a dog"))
        self.assertFalse(looks_like_tool_request(
            "explain how photosynthesis works"))

    def test_refusal_text_names_the_next_step(self):
        self.assertIn("Claude Agent tools", REFUSAL)
        self.assertIn("switch to claude", REFUSAL.lower())


if __name__ == "__main__":
    unittest.main()
