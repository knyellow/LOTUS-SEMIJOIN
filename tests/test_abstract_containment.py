"""Offline regressions for empty strings and the exact 8,000-character boundary."""

import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "biodex" / "audit_abstract_containment.py"
spec = importlib.util.spec_from_file_location("abstract_audit", PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TestAbstractContainment(unittest.TestCase):
    def check(self, abstract, full):
        return audit.check_article(
            {"abstract": abstract, "fulltext_processed": full, "pmid": "test", "reactions": "A"},
            0,
            8000,
            {"test": ["A"]},
        )

    def test_empty_or_missing_text_never_proves_containment(self):
        for abstract in [None, "", " \n"]:
            with self.subTest(abstract=abstract):
                row = self.check(abstract, "text")
                self.assertFalse(row["exact_prefix"])
                self.assertEqual(row["status"], "empty_abstract")
        for full in [None, "", " \n"]:
            with self.subTest(full=full):
                self.assertEqual(self.check("abstract", full)["status"], "empty_fulltext")

    def test_last_character_at_limit_is_kept_but_one_past_is_not(self):
        inside = self.check("AB", "x" * 7998 + "AB")
        outside = self.check("AB", "x" * 7999 + "AB")
        self.assertTrue(inside["exact_prefix"])
        self.assertEqual(inside["abstract_end_exclusive"], 8000)
        self.assertFalse(outside["exact_prefix"])
        self.assertTrue(outside["exact_full"])
        self.assertEqual(outside["status"], "outside_prefix")
        summary = audit.summarize([inside, outside], 8000)
        self.assertEqual((summary["exact_prefix"], summary["n_not_contained"]), (1, 1))

    def test_comparison_preserves_whitespace_and_case(self):
        self.assertTrue(self.check(" Abstract\n", "title\n Abstract\nbody")["exact_prefix"])
        self.assertEqual(self.check(" Abstract\n", "title Abstract body")["status"], "not_found_in_fulltext")
        self.assertFalse(self.check("Abstract", "ABSTRACT")["exact_prefix"])


if __name__ == "__main__":
    unittest.main()
