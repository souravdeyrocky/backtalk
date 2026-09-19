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
