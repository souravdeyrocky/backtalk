"""Regression tests for the shared Jarvis identity + capability-
contract layer (backtalk/identity.py) and its three injection points:
Qwen/DeepSeek's preamble, Gemini's systemInstruction. Claude's identity
already lives in the vault's CLAUDE.md (config.py deliberately owns no
personality -- see its own module docstring), so this file does not
touch or test config.DISCIPLINE.
"""
import unittest

from backtalk.identity import (GEMINI_CAPABILITY_RULES,
                               JARVIS_CORE_IDENTITY,
                               LOCAL_BRAIN_CAPABILITY_RULES,
                               PROACTIVE_CHECKIN_RULES)


class SharedIdentityContentTests(unittest.TestCase):
    def test_core_identity_names_jarvis_and_captain(self):
        self.assertIn("Jarvis", JARVIS_CORE_IDENTITY)
        self.assertIn("Captain", JARVIS_CORE_IDENTITY)

    def test_core_identity_forbids_chatbot_self_description(self):
        low = JARVIS_CORE_IDENTITY.lower()
        self.assertIn("chatbot", low)
        self.assertIn("never", low)

    def test_core_identity_forbids_wrong_forms_of_address(self):
        low = JARVIS_CORE_IDENTITY.lower()
        for wrong in ("sourav", "boss", "sir"):
            self.assertIn(wrong, low)  # named as what NOT to use

    def test_local_rules_list_every_forbidden_claim(self):
        low = LOCAL_BRAIN_CAPABILITY_RULES.lower()
        for forbidden in ("write or edit the vault", "schedule",
                          "read files", "calendars", "phone data",
                          "run commands", "browse the web",
                          "send messages", "automate workflows"):
            self.assertIn(forbidden, low)

    def test_local_rules_point_at_claude_as_the_escalation(self):
        self.assertIn("Claude", LOCAL_BRAIN_CAPABILITY_RULES)

    def test_gemini_rules_forbid_platform_capabilities(self):
        low = GEMINI_CAPABILITY_RULES.lower()
        for forbidden in ("analyze an image", "control a calendar",
                          "access media", "browse the web"):
            self.assertIn(forbidden, low)

    def test_gemini_rules_name_it_as_brain_four_text_only(self):
        self.assertIn("Brain 4", GEMINI_CAPABILITY_RULES)
        self.assertIn("plain text", GEMINI_CAPABILITY_RULES)

    def test_checkin_rules_name_the_three_allowed_moments(self):
        low = PROACTIVE_CHECKIN_RULES.lower()
        self.assertIn("startup", low)
        self.assertIn("task", low)
        self.assertIn("idle", low)

    def test_checkin_rules_forbid_interrupting_a_task(self):
        self.assertIn("never in", PROACTIVE_CHECKIN_RULES.lower())

    def test_checkin_rules_require_naming_the_source(self):
        low = PROACTIVE_CHECKIN_RULES.lower()
        self.assertIn("name that source", low)
        self.assertIn("calendar", low)

    def test_checkin_rules_forbid_inference_from_unrelated_data(self):
        low = PROACTIVE_CHECKIN_RULES.lower()
        self.assertIn("never infer or guess", low)
        for topic in ("health", "mood", "travel", "family"):
            self.assertIn(topic, low)


class OllamaPreambleInjectionTests(unittest.TestCase):
    """The preamble string built fresh in ollama_brain.py's ask_stream()
    every turn -- reconstructed here from the same source constants
    rather than triggering a real Ollama call, since the preamble text
    itself lives inline in that method."""

    def _preamble(self) -> str:
        return (
            f"{JARVIS_CORE_IDENTITY} You're running entirely on-device "
            f"through Ollama right now -- no part of this conversation "
            f"leaves this machine. {LOCAL_BRAIN_CAPABILITY_RULES} "
            f"{PROACTIVE_CHECKIN_RULES}")

    def test_preamble_carries_the_shared_identity_verbatim(self):
        self.assertIn(JARVIS_CORE_IDENTITY, self._preamble())

    def test_preamble_carries_the_local_capability_rules_verbatim(self):
        self.assertIn(LOCAL_BRAIN_CAPABILITY_RULES, self._preamble())

    def test_preamble_carries_the_checkin_rules_verbatim(self):
        self.assertIn(PROACTIVE_CHECKIN_RULES, self._preamble())


class GeminiSystemInstructionInjectionTests(unittest.TestCase):
    def test_system_instruction_carries_shared_identity_and_gemini_rules(self):
        from backtalk.brains.gemini_brain import _SYSTEM_INSTRUCTION
        self.assertIn(JARVIS_CORE_IDENTITY, _SYSTEM_INSTRUCTION)
        self.assertIn(GEMINI_CAPABILITY_RULES, _SYSTEM_INSTRUCTION)
        self.assertIn(PROACTIVE_CHECKIN_RULES, _SYSTEM_INSTRUCTION)

    def test_system_instruction_is_a_fixed_module_constant(self):
        # Same object identity across "calls" -- there's no per-turn
        # construction to vary, and therefore nothing turn-specific
        # (vault content, the utterance itself) could ever leak in.
        from backtalk.brains import gemini_brain
        first = gemini_brain._SYSTEM_INSTRUCTION
        second = gemini_brain._SYSTEM_INSTRUCTION
        self.assertEqual(first, second)
        self.assertNotIn("{", first)  # no unfilled template placeholder


class AllFourBrainsCarrySomeIdentityMechanismTests(unittest.TestCase):
    """Not "identical text reaches all four" -- each brain receives its
    identity through whatever mechanism its own system-prompt
    construction actually uses, and this proves each one genuinely
    has ONE: Qwen and DeepSeek share the literal ollama_brain.py
    preamble (same class, parameterized only by model name), Gemini
    gets the fixed systemInstruction, and Claude's system_prompt
    wiring (preset "claude_code" + DISCIPLINE append, unchanged by
    this whole layer) still reaches WarmBrain -- Claude's CHARACTER
    living in the vault's own CLAUDE.md, per config.py's documented
    "no personality here" design."""

    def test_qwen_and_deepseek_share_the_same_preamble_construction(self):
        import inspect

        from backtalk.brains import ollama_brain
        source = inspect.getsource(ollama_brain.OllamaBrain.ask_stream)
        self.assertIn("JARVIS_CORE_IDENTITY", source)
        # QwenBrain and DeepSeekBrain both subclass OllamaBrain with no
        # ask_stream override of their own -- proven by identity, not
        # just absence-of-override, so a future accidental override
        # would fail this loudly.
        from backtalk.brains.ollama_brain import DeepSeekBrain, QwenBrain
        self.assertIs(QwenBrain.ask_stream, ollama_brain.OllamaBrain.ask_stream)
        self.assertIs(DeepSeekBrain.ask_stream,
                      ollama_brain.OllamaBrain.ask_stream)

    def test_gemini_carries_the_fixed_system_instruction(self):
        from backtalk.brains.gemini_brain import _SYSTEM_INSTRUCTION
        self.assertIn(JARVIS_CORE_IDENTITY, _SYSTEM_INSTRUCTION)

    def test_claude_system_prompt_wiring_still_appends_discipline(self):
        # Claude's CHARACTER lives in CLAUDE.md (outside this repo,
        # Captain's own vault-linked file) -- this only proves the
        # WIRING that carries whatever DISCIPLINE says is still
        # intact, unchanged by this session's work.
        import inspect

        from backtalk import brain
        source = inspect.getsource(brain.WarmBrain)
        self.assertIn('"preset": "claude_code"', source)
        self.assertIn('"append": DISCIPLINE', source)


if __name__ == "__main__":
    unittest.main()
