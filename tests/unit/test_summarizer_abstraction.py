#!/usr/bin/env python3
"""
Concise tests for abstracted mCODE Summarizer syntactic structure.
All functionality is now within methods/functions - testing core rules.
"""

import unittest
from src.services.summarizer import McodeSummarizer


class TestSummarizerAbstraction(unittest.TestCase):
    """Test cases for abstracted mCODE summarizer with correct syntactic structure."""

    def setUp(self):
        """Set up test fixtures."""
        self.summarizer = McodeSummarizer(include_dates=True)

    def test_core_syntactic_rules(self):
        """Test all core syntactic rules in one comprehensive test."""
        # Test mCODE as subject structure
        elements = [
            {
                "element_name": "Age",
                "value": "45 years old",
                "codes": "SNOMED:424144002",
                "date_qualifier": "",
            },
            {
                "element_name": "CancerCondition",
                "value": "Breast Cancer",
                "codes": "SNOMED:254837009",
                "date_qualifier": " documented on 2020-01-01",
            },
        ]

        sentences = self.summarizer._generate_sentences_from_elements(
            elements, "Patient"
        )

        for sentence in sentences:
            if "mCODE:" in sentence:
                # mCODE should be subject (before verb)
                mcode_pos = sentence.find("(mCODE:")
                verb_pos = sentence.find(" is ")
                self.assertLess(
                    mcode_pos, verb_pos, f"mCODE not subject in: {sentence}"
                )

                # Should have codes in predicate
                self.assertIn("(", sentence.split("(mCODE:")[1].split(")")[1])

    def test_comprehensive_coding_coverage(self):
        """Test that all elements have comprehensive coding coverage."""
        test_cases = [
            ("Gender", "male", "SNOMED:407377005"),
            ("BirthDate", "1978-03-15", "SNOMED:184099003"),
            ("Age", "45 years old", "SNOMED:424144002"),
        ]

        for element_name, value, expected_code in test_cases:
            sentence = self.summarizer._create_abstracted_sentence(
                "Patient", element_name, value, "", ""
            )
            self.assertIn(
                f"({expected_code})",
                sentence,
                f"Missing code for {element_name}: {sentence}",
            )

    def test_priority_and_ordering(self):
        """Test priority ordering and element configuration."""
        # Verify configurations loaded
        self.assertEqual(len(self.summarizer.element_configs), 33)

        # Test priority ordering
        elements = [
            {
                "element_name": "CauseOfDeath",
                "value": "Cancer",
                "codes": "",
                "date_qualifier": "",
            },
            {
                "element_name": "Patient",
                "value": "John Doe",
                "codes": "",
                "date_qualifier": "",
            },
            {"element_name": "Age", "value": "45", "codes": "", "date_qualifier": ""},
        ]

        prioritized = self.summarizer._group_elements_by_priority(elements, "Patient")
        priorities = [elem["priority"] for elem in prioritized]
        self.assertEqual(priorities, sorted(priorities))

    def test_introduction_exceptions(self):
        """Test that introduction sentences follow exception pattern."""
        # Patient introduction
        patient_sentence = self.summarizer._create_abstracted_sentence(
            "Patient", "Patient", "John Doe (ID: 123)", "", ""
        )
        self.assertEqual(patient_sentence, "Patient is a Patient (mCODE: Patient).")

        # Trial introduction
        trial_sentence = self.summarizer._create_abstracted_sentence(
            "NCT123456", "Trial", "Clinical Trial", "", ""
        )
        self.assertEqual(
            trial_sentence, "NCT123456 is a Clinical Trial (mCODE: Trial)."
        )

    def test_complex_elements_and_validation(self):
        """Test complex elements with dates/codes and strict validation."""
        # Complex element with dates and codes
        complex_sentence = self.summarizer._create_abstracted_sentence(
            "Patient",
            "TNMStageGroup",
            "T2 N1 M0",
            "SNOMED:385633009",
            " documented on 2020-02-15",
        )
        expected = "Patient's tumor staging (mCODE: TNMStageGroup documented on 2020-02-15) is T2 N1 M0 (SNOMED:385633009)."
        self.assertEqual(complex_sentence, expected)

        # Strict validation - should fail for unconfigured elements
        with self.assertRaises(ValueError) as context:
            self.summarizer._create_abstracted_sentence(
                "Patient", "InvalidElement", "test", "", ""
            )
        self.assertIn("not configured", str(context.exception))


if __name__ == "__main__":
    unittest.main()
