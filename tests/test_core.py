#!/usr/bin/env python3
"""Unit tests for the pure logic in the app's own code.

Run with the project venv:  .venv/bin/python -m unittest discover -s tests
No third-party deps — stdlib unittest only. numpy is used by a couple of
recorder helpers; those tests skip if numpy is missing.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dictation"))

import common  # noqa: E402


class TestReplacements(unittest.TestCase):
    def test_word_boundaries_and_case(self):
        cfg = {"replacements": {"gitlab": "GitLab"}}
        self.assertEqual(common.apply_replacements(cfg, "mein Gitlab und GITLAB"),
                         "mein GitLab und GitLab")
        # substring must NOT match
        self.assertEqual(common.apply_replacements(cfg, "Gitlabber"), "Gitlabber")

    def test_backslash_value_does_not_crash(self):
        cfg = {"replacements": {"pfad": r"C:\neu\1"}}
        self.assertEqual(common.apply_replacements(cfg, "der pfad hier"),
                         r"der C:\neu\1 hier")

    def test_dotted_term(self):
        cfg = {"replacements": {"z.B.": "zum Beispiel"}}
        self.assertEqual(common.apply_replacements(cfg, "wie z.B. dieses"),
                         "wie zum Beispiel dieses")

    def test_empty_and_bad_config(self):
        self.assertEqual(common.apply_replacements({}, "x"), "x")
        self.assertEqual(common.apply_replacements({"replacements": "nope"}, "x"), "x")


class TestPromptHotwords(unittest.TestCase):
    def test_merge_dictionary(self):
        cfg = {"initial_prompt": "Diktat.", "hotwords": "GNOME",
               "dictionary": ["Vicinae", "Völzke"]}
        prompt, hot = common.effective_prompt_and_hotwords(cfg)
        self.assertIn("Vicinae", prompt)
        self.assertIn("Begriffe:", prompt)
        self.assertEqual(hot, "GNOME, Vicinae, Völzke")

    def test_no_dictionary(self):
        cfg = {"initial_prompt": "P", "hotwords": ""}
        prompt, hot = common.effective_prompt_and_hotwords(cfg)
        self.assertEqual(prompt, "P")
        self.assertIsNone(hot)

    def test_comma_string_dictionary(self):
        cfg = {"dictionary": "A, B , ,C"}
        self.assertEqual(common.dictionary_terms(cfg), ["A", "B", "C"])


class TestSnippets(unittest.TestCase):
    def test_exact_trigger_only(self):
        cfg = {"snippets": {"Grußformel": "Viele Grüße"}}
        self.assertEqual(common.match_snippet(cfg, " grußformel. "), "Viele Grüße")
        self.assertIsNone(common.match_snippet(cfg, "die grußformel ist"))

    def test_normalize(self):
        self.assertEqual(common.normalize_utterance("  Hallo, Welt!  "), "hallo, welt")


class TestHotkeyParsing(unittest.TestCase):
    def test_key_label(self):
        self.assertEqual(common.key_label("ctrl_r"), "Right Ctrl")
        self.assertEqual(common.key_label("KEY_F8"), "F8")
        self.assertEqual(common.key_label("code:193:AI-Taste"), "AI-Taste")
        self.assertEqual(common.key_label(""), "")

    def test_evdev_code_for(self):
        class Ecodes:
            KEY_RIGHTCTRL = 97
            KEY_F8 = 66
        self.assertEqual(common.evdev_code_for("ctrl_r", Ecodes), 97)
        self.assertEqual(common.evdev_code_for("KEY_F8", Ecodes), 66)
        self.assertEqual(common.evdev_code_for("code:42:x", Ecodes), 42)
        self.assertIsNone(common.evdev_code_for("", Ecodes))
        self.assertIsNone(common.evdev_code_for("code:bad", Ecodes))


class TestDictationModes(unittest.TestCase):
    def test_modes_present(self):
        self.assertEqual(set(common.DICTATION_MODES), {"standard", "email", "chat", "raw"})


class TestParagraphs(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not available")
        import importlib
        self.recorder = importlib.import_module("recorder")

    def test_grouping_on_gap_and_span(self):
        segs = [(0.0, 3.0, " Hallo."), (3.2, 6.0, " Weiter."),
                (9.5, 12.0, " Nach Pause."), (12.5, 50.0, " Lang."),
                (50.2, 55.0, " Neuer.")]
        paras = self.recorder._segments_to_paragraphs(segs, offset=300.0)
        self.assertEqual(paras[0], (300.0, "Hallo. Weiter."))
        self.assertEqual(paras[1][0], 309.5)
        self.assertEqual(len(paras), 3)

    def test_fmt_and_strip(self):
        self.assertEqual(self.recorder._fmt_ts(65), "01:05")
        self.assertEqual(self.recorder._fmt_ts(3725), "1:02:05")
        self.assertEqual(self.recorder._strip_markers("[05:12] A [1:02:05] B"), "A B")


if __name__ == "__main__":
    unittest.main()
