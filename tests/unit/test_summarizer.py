#!/usr/bin/env python3
"""
Unit tests for the McodeSummarizer service.
"""

import unittest
import sys
import json
from pathlib import Path

# Add project root to path to allow absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.summarizer import McodeSummarizer

class TestMcodeSummarizer(unittest.TestCase):
    """Test suite for the McodeSummarizer."""

    def setUp(self):
        """Set up the test case."""
        self.summarizer = McodeSummarizer()
        self.complete_trial_path = project_root / "complete_trial.json"
        self.patient_data_path = project_root / "breast_cancer_patients_demo.json"

    def test_create_trial_summary_from_real_data(self):
        """Test trial summary generation with real, complete data."""
        print("\n--- Testing Trial Summary with Real Data ---")
        if not self.complete_trial_path.exists():
            self.skipTest(f"{self.complete_trial_path} not found.")

        with open(self.complete_trial_path, "r") as f:
            trial_data_list = json.load(f)
        
        trial_data = trial_data_list[0]
        summary = self.summarizer.create_trial_summary(trial_data)
        
        print("Generated Trial Summary:")
        print(summary)
        
        self.assertIn("NCT03633331", summary)
        self.assertIn("Palbociclib", summary)
        self.assertIn("Alliance for Clinical Trials in Oncology", summary)
        print("--- Test Passed ---")

    def test_create_patient_summary_from_real_data(self):
        """Test patient summary generation with real, complete data."""
        print("\n--- Testing Patient Summary with Real Data ---")
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)
        
        patient_data = patient_data_list[0]
        summary = self.summarizer.create_patient_summary(patient_data)
        
        print("Generated Patient Summary:")
        print(summary)
        
        self.assertIn("Patient", summary)
        self.assertIn("ID:", summary)
        self.assertIn("-year-old", summary)
        self.assertIn("is a deceased", summary)  # Patient is deceased
        self.assertIn("She has been diagnosed with Malignant neoplasm of breast disorder (mCODE: CancerCondition; SNOMED:254837009 documented on 2000-10-18)", summary)
        self.assertIn("Her tumor staging (mCODE: TNMStageGroup documented on 2000-10-18) is T4 (SNOMED:65565005) N1 (SNOMED:53623008), and M1 (SNOMED:55440008)", summary)
        self.assertIn("She has Stage 4", summary)
        self.assertIn("Her tumor markers (mCODE: TumorMarkerTest documented on 2000-10-20) show HER2 negative (SNOMED:260385009), ER positive (SNOMED:10828004), and PR negative (SNOMED:260385009)", summary)
        # Test new format with inline codes and "documented on"
        self.assertIn(" documented on ", summary)
        # Only clinically significant procedures should be included
        self.assertIn("She underwent", summary)  # Should include biopsy, chemotherapy, etc.
        self.assertNotIn("Medication Reconciliation", summary)  # Should exclude administrative procedures
        self.assertIn("She is prescribed", summary)  # Medications
        self.assertIn("Social determinant", summary)  # Social determinants
        self.assertIn("Her cause of death was Malignant neoplasm of breast disorder (mCODE: CauseOfDeath; SNOMED:254837009 documented on 2001-01-03)", summary)
        # Test that mCODE tags use "documented on" instead of "dates:"
        self.assertIn(" documented on ", summary)  # Should use "documented on" format
        # Test clinical note structure priority
        diagnosis_pos = summary.find("She has been diagnosed")
        staging_pos = summary.find("Her tumor staging")
        biomarkers_pos = summary.find("Her tumor markers")
        procedures_pos = summary.find("She underwent")
        medications_pos = summary.find("She is prescribed")
        # Ensure proper clinical note priority order
        self.assertTrue(diagnosis_pos < staging_pos < biomarkers_pos, "Clinical note sections should follow proper priority order")
        # Test that mCODE tags include system and code information
        self.assertIn("SNOMED:", summary)  # Should include SNOMED codes
        self.assertIn("She has other conditions including", summary)
        self.assertIn("Streptococcal sore throat", summary)
        self.assertIn("Viral sinusitis", summary)
        self.assertIn("Otitis media", summary)
        self.assertIn("Fracture of clavicle", summary)
        # Test de-duplication - primary diagnosis should not appear in other conditions
        other_conditions_part = summary.split("She has other conditions including")[1].split("Her cause of death")[0]
        self.assertNotIn("Malignant neoplasm of breast (disorder)", other_conditions_part)
        # Test date consolidation in mCODE tags
        self.assertIn("mCODE: CancerCondition", summary)  # Diagnosis with consolidated dates
        self.assertIn("mCODE: TNMStageGroup", summary)  # Staging with consolidated dates
        self.assertIn("mCODE: TumorMarkerTest", summary)  # Biomarkers with consolidated dates
        self.assertIn("mCODE: CauseOfDeath", summary)  # Cause of death with consolidated dates
        # Test comprehensivity - all major clinical domains covered
        self.assertIn("diagnosed with", summary)  # Diagnosis with dates
        self.assertIn("tumor staging", summary)  # Staging with dates
        self.assertIn("tumor markers", summary)  # Biomarkers with dates
        self.assertIn("underwent", summary)  # Procedures with dates
        self.assertIn("prescribed", summary)  # Medications
        self.assertIn("Social determinant", summary)  # Social determinants
        self.assertIn("cause of death", summary)  # Outcome with dates
        print("--- Test Passed ---")

    def test_create_trial_summary_no_data(self):
        """Test trial summary generation with no data."""
        trial_data = {}
        with self.assertRaises(ValueError):
            self.summarizer.create_trial_summary(trial_data)

    def test_create_patient_summary_no_data(self):
        """Test patient summary generation with no data."""
        patient_data = {}
        with self.assertRaises(ValueError):
            self.summarizer.create_patient_summary(patient_data)

    def test_format_mcode_display(self):
        """Test mCODE display formatting with various coding systems."""
        # Test with element name
        result = self.summarizer._format_mcode_display("CancerCondition", "http://snomed.info/sct", "254837009")
        self.assertEqual(result, "(mCODE: CancerCondition, SNOMED:254837009)")

        # Test without element name
        result = self.summarizer._format_mcode_display("", "http://loinc.org", "12345-6")
        self.assertEqual(result, "LOINC:12345-6")

        # Test with unknown system
        result = self.summarizer._format_mcode_display("", "http://example.com/system", "CODE123")
        self.assertEqual(result, "SYSTEM:CODE123")

    def test_format_date_simple(self):
        """Test date formatting to simple yyyy-mm-dd format."""
        # Test full ISO timestamp
        result = self.summarizer._format_date_simple("2000-10-18T07:22:35-04:00")
        self.assertEqual(result, "2000-10-18")

        # Test date only
        result = self.summarizer._format_date_simple("2000-10-18")
        self.assertEqual(result, "2000-10-18")

        # Test empty string
        result = self.summarizer._format_date_simple("")
        self.assertEqual(result, "")

        # Test None
        result = self.summarizer._format_date_simple(None)
        self.assertEqual(result, "")

    def test_format_list_with_separator(self):
        """Test list formatting with proper separators."""
        # This would be a helper method we could add for consistent list formatting
        # For now, test the expected behavior in the actual output
        pass

    def test_mcode_tag_consistency(self):
        """Test that all mCODE tags follow consistent formatting rules."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer.create_patient_summary(patient_data)

        # Test that diagnosis uses semicolon separator
        self.assertIn("(mCODE: CancerCondition; ", summary)

        # Test that staging uses inline format
        self.assertIn("(mCODE: TNMStageGroup documented on ", summary)
        self.assertIn("T4 (SNOMED:", summary)
        self.assertIn("N1 (SNOMED:", summary)
        self.assertIn("M1 (SNOMED:", summary)

        # Test that biomarkers use inline format
        self.assertIn("(mCODE: TumorMarkerTest documented on ", summary)
        self.assertIn("HER2 negative (SNOMED:", summary)
        self.assertIn("ER positive (SNOMED:", summary)
        self.assertIn("PR negative (SNOMED:", summary)

        # Test that cause of death uses semicolon separator
        self.assertIn("(mCODE: CauseOfDeath; ", summary)

    def test_date_formatting_consistency(self):
        """Test that all dates are formatted consistently as yyyy-mm-dd."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer.create_patient_summary(patient_data)

        # Should contain dates in yyyy-mm-dd format
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates_found = re.findall(date_pattern, summary)
        self.assertTrue(len(dates_found) > 0, "Should contain formatted dates")

        # Should not contain full timestamps
        self.assertNotIn("T", summary, "Should not contain full ISO timestamps")

    def test_natural_language_formatting(self):
        """Test natural language formatting throughout the summary."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer.create_patient_summary(patient_data)

        # Should use "documented on" instead of "documented:"
        self.assertIn(" documented on ", summary)
        self.assertNotIn(" documented: ", summary)

        # Should have inline codes in parentheses (this is expected for the new format)
        self.assertIn("(SNOMED:", summary)
        # Should not have leading parentheses before codes in mCODE tags (old format)
        self.assertNotIn("(mCODE: CancerCondition, SNOMED:", summary)

if __name__ == '__main__':
    unittest.main()