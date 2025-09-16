#!/usr/bin/env python3
"""
Unit tests for the McodeSummarizer service.
"""

import json
import re
import sys
import unittest
from pathlib import Path

# Add project root to path to allow absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.summarizer import McodeSummarizer


class TestMcodeSummarizer(unittest.TestCase):
    """Test suite for the McodeSummarizer."""

    def setUp(self):
        """Set up the test case."""
        self.summarizer_with_dates = McodeSummarizer(include_dates=True)
        self.summarizer_without_dates = McodeSummarizer(include_dates=False)
        self.complete_trial_path = (
            project_root / "examples" / "data" / "complete_trial.json"
        )
        self.patient_data_path = (
            project_root / "examples" / "data" / "breast_cancer_patients_demo.json"
        )

    def test_create_trial_summary_from_real_data(self):
        """Test trial summary generation with real, complete data."""
        print("\n--- Testing Trial Summary with Real Data ---")
        if not self.complete_trial_path.exists():
            self.skipTest(f"{self.complete_trial_path} not found.")

        with open(self.complete_trial_path, "r") as f:
            trial_data_list = json.load(f)

        trial_data = trial_data_list[0]
        summary = self.summarizer_with_dates.create_trial_summary(trial_data)

        print("Generated Trial Summary:")
        print(summary)

        self.assertIn("NCT03633331", summary)
        self.assertIn("Palbociclib", summary)
        self.assertIn("Alliance for Clinical Trials in Oncology", summary)

        # Test new active sentence structure with mCODE as subject
        self.assertIn(
            "Trial study type (mCODE: TrialStudyType) is interventional study.", summary
        )
        self.assertIn("Trial phase (mCODE: TrialPhase) is phase 2.", summary)
        self.assertIn(
            "Trial status (mCODE: TrialStatus) is active not recruiting.", summary
        )
        self.assertIn(
            "Trial lead sponsor (mCODE: TrialLeadSponsor) is Alliance for Clinical Trials in Oncology.",
            summary,
        )
        self.assertIn(
            "Trial enrollment (mCODE: TrialEnrollment) is 93 participants.", summary
        )
        self.assertIn(
            "Trial medication interventions (mCODE: TrialMedicationInterventions) include Palbociclib",
            summary,
        )
        print("--- Test Passed ---")

    def test_create_patient_summary_from_real_data(self):
        """Test patient summary generation with real, complete data."""
        print("\n--- Testing Patient Summary with Real Data ---")
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        print("Generated Patient Summary:")
        print(summary)

        self.assertIn("Patient", summary)
        self.assertIn("ID:", summary)
        self.assertIn(
            "Her age (mCODE: Age) is", summary
        )  # Age is now in separate sentence
        self.assertIn("is a deceased", summary)  # Patient is deceased
        self.assertIn(
            "Her diagnosis (mCODE: CancerCondition documented on 2000-10-18) is Malignant neoplasm of breast (SNOMED:254837009)",
            summary,
        )
        self.assertIn(
            "Her tumor staging (mCODE: TNMStageGroup documented on 2000-10-18) is T4 (SNOMED:65565005) N1 (SNOMED:53623008), and M1 (SNOMED:55440008)",
            summary,
        )
        self.assertIn(
            "Her disease (mCODE: TNMStageGroup documented on 2000-10-20) is Stage 4 disease (SNOMED:258228008)",
            summary,
        )
        self.assertIn(
            "Her tumor markers (mCODE: TumorMarkerTest documented on 2000-10-20) show HER2 negative (SNOMED:260385009), ER positive (SNOMED:10828004), and PR negative (SNOMED:260385009)",
            summary,
        )
        # Test new format with inline codes and "documented on"
        self.assertIn(" documented on ", summary)
        # Only clinically significant procedures should be included
        self.assertIn(
            "Her procedures (mCODE: Procedure documented on", summary
        )  # Should include biopsy, chemotherapy, etc.
        self.assertNotIn(
            "Medication Reconciliation", summary
        )  # Should exclude administrative procedures
        self.assertIn(
            "Her medications (mCODE: MedicationRequest) are", summary
        )  # Medications
        self.assertIn(
            "Her social determinants (mCODE: SocialDeterminant) are", summary
        )  # Social determinants
        self.assertIn(
            "Her cause of death (mCODE: CauseOfDeath documented on 2001-01-03) is Malignant neoplasm of breast (SNOMED:254837009)",
            summary,
        )
        # Test that mCODE tags use "documented on" instead of "dates:"
        self.assertIn(" documented on ", summary)  # Should use "documented on" format
        # Test clinical note structure priority
        diagnosis_pos = summary.find("Her diagnosis is")
        staging_pos = summary.find("Her tumor staging")
        biomarkers_pos = summary.find("Her tumor markers")
        procedures_pos = summary.find("She underwent")
        medications_pos = summary.find("She is prescribed")
        # Ensure proper clinical note priority order
        self.assertTrue(
            diagnosis_pos < staging_pos < biomarkers_pos,
            "Clinical note sections should follow proper priority order",
        )
        # Test that mCODE tags include system and code information
        self.assertIn("SNOMED:", summary)  # Should include SNOMED codes
        self.assertIn("Her conditions (mCODE: Condition documented on", summary)
        self.assertIn("Streptococcal sore throat", summary)
        self.assertIn("Viral sinusitis", summary)
        self.assertIn("Otitis media", summary)
        self.assertIn("Fracture of clavicle", summary)
        # Test de-duplication - primary diagnosis should not appear in other conditions
        # Check that no condition sentence contains the primary diagnosis
        condition_lines = [
            line
            for line in summary.split(".")
            if "Her conditions (mCODE: Condition" in line
        ]
        for line in condition_lines:
            self.assertNotIn("Malignant neoplasm of breast", line)
        # Test date consolidation in mCODE tags
        self.assertIn(
            "mCODE: CancerCondition", summary
        )  # Diagnosis with consolidated dates
        self.assertIn(
            "mCODE: TNMStageGroup", summary
        )  # Staging with consolidated dates
        self.assertIn(
            "mCODE: TumorMarkerTest", summary
        )  # Biomarkers with consolidated dates
        self.assertIn(
            "mCODE: CauseOfDeath", summary
        )  # Cause of death with consolidated dates
        # Test comprehensivity - all major clinical domains covered
        self.assertIn(
            "diagnosis (mCODE: CancerCondition documented on", summary
        )  # Diagnosis with dates
        self.assertIn("tumor staging", summary)  # Staging with dates
        self.assertIn("tumor markers", summary)  # Biomarkers with dates
        self.assertIn(
            "Her procedures (mCODE: Procedure documented on", summary
        )  # Procedures with dates
        self.assertIn(
            "Her medications (mCODE: MedicationRequest) are", summary
        )  # Medications
        self.assertIn(
            "Her social determinants (mCODE: SocialDeterminant) are", summary
        )  # Social determinants
        self.assertIn("cause of death", summary)  # Outcome with dates
        print("--- Test Passed ---")

    def test_create_trial_summary_no_data(self):
        """Test trial summary generation with no data."""
        trial_data = {}
        with self.assertRaises(ValueError):
            self.summarizer_with_dates.create_trial_summary(trial_data)

    def test_create_patient_summary_no_data(self):
        """Test patient summary generation with no data."""
        patient_data = {}
        with self.assertRaises(ValueError):
            self.summarizer_with_dates.create_patient_summary(patient_data)

    def test_format_mcode_display(self):
        """Test mCODE display formatting with various coding systems."""
        # Test with element name
        result = self.summarizer_with_dates._format_mcode_display(
            "CancerCondition", "http://snomed.info/sct", "254837009"
        )
        self.assertEqual(result, "(mCODE: CancerCondition, SNOMED:254837009)")

        # Test without element name
        result = self.summarizer_with_dates._format_mcode_display(
            "", "http://loinc.org", "12345-6"
        )
        self.assertEqual(result, "LOINC:12345-6")

        # Test with unknown system
        result = self.summarizer_with_dates._format_mcode_display(
            "", "http://example.com/system", "CODE123"
        )
        self.assertEqual(result, "SYSTEM:CODE123")

    def test_format_date_simple(self):
        """Test date formatting to simple yyyy-mm-dd format."""
        # Test full ISO timestamp
        result = self.summarizer_with_dates._format_date_simple(
            "2000-10-18T07:22:35-04:00"
        )
        self.assertEqual(result, "2000-10-18")

        # Test date only
        result = self.summarizer_with_dates._format_date_simple("2000-10-18")
        self.assertEqual(result, "2000-10-18")

        # Test empty string
        result = self.summarizer_with_dates._format_date_simple("")
        self.assertEqual(result, "")

        # Test None
        result = self.summarizer_with_dates._format_date_simple(None)
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
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        # Test that diagnosis uses mCODE immediately after clinical feature
        self.assertIn("Her diagnosis (mCODE: CancerCondition documented on", summary)

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

        # Test that cause of death uses mCODE immediately after clinical feature
        self.assertIn("Her cause of death (mCODE: CauseOfDeath documented on", summary)

    def test_date_formatting_consistency(self):
        """Test that all dates are formatted consistently as yyyy-mm-dd."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        # Should contain dates in yyyy-mm-dd format
        import re

        date_pattern = r"\d{4}-\d{2}-\d{2}"
        dates_found = re.findall(date_pattern, summary)
        self.assertTrue(len(dates_found) > 0, "Should contain formatted dates")

        # Should not contain full timestamps
        import re

        timestamp_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"
        self.assertNotRegex(
            summary, timestamp_pattern, "Should not contain full ISO timestamps"
        )

    def test_natural_language_formatting(self):
        """Test natural language formatting throughout the summary."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        # Should use "documented on" instead of "documented:"
        self.assertIn(" documented on ", summary)
        self.assertNotIn(" documented: ", summary)

        # Should have inline codes in parentheses (this is expected for the new format)
        self.assertIn("(SNOMED:", summary)
        # Should not have leading parentheses before codes in mCODE tags (old format)
        self.assertNotIn("(mCODE: CancerCondition, SNOMED:", summary)

    def test_active_sentence_structure_for_nlp_clarity(self):
        """Test that clinical features are subjects in active sentence structure for NLP clarity."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        # Test that diagnosis uses active sentence structure with clinical feature as subject
        self.assertIn("Her diagnosis (mCODE: CancerCondition documented on", summary)
        self.assertNotIn("She has been diagnosed with", summary)

        # Test that cause of death uses active sentence structure with clinical feature as subject
        self.assertIn("Her cause of death (mCODE: CauseOfDeath documented on", summary)
        self.assertNotIn("Her cause of death was", summary)

        # Test that mCODE precedes detailed codes inline
        self.assertIn("(mCODE: CancerCondition documented on", summary)
        self.assertIn("(mCODE: CauseOfDeath documented on", summary)

        # Test that staging maintains active structure (already correct)
        self.assertIn("Her tumor staging (mCODE: TNMStageGroup", summary)

        # Test that biomarkers maintain active structure (already correct)
        self.assertIn("Her tumor markers (mCODE: TumorMarkerTest", summary)

    def test_include_dates_flag(self):
        """Test that include_dates flag controls date inclusion in summary."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]

        # Test with dates included
        summary_with_dates = self.summarizer_with_dates.create_patient_summary(
            patient_data
        )
        self.assertIn("documented on", summary_with_dates)
        self.assertIn("documented on 2000-10-18", summary_with_dates)

        # Test with dates excluded
        summary_without_dates = self.summarizer_without_dates.create_patient_summary(
            patient_data
        )
        self.assertNotIn("documented on", summary_without_dates)
        self.assertNotIn("documented on 2000-10-18", summary_without_dates)

        # Both should still have the basic structure
        self.assertIn("Her diagnosis (mCODE: CancerCondition)", summary_without_dates)
        self.assertIn("Her cause of death (mCODE: CauseOfDeath)", summary_without_dates)

    def test_comprehensive_aggregation_rules(self):
        """Test that all sections follow consistent aggregation rules for maximal knowledge graph information."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        # Test that procedures are aggregated by date with correct plural grammar
        procedure_lines = [
            line
            for line in summary.split(".")
            if "procedures" in line.lower() and "mCODE" in line
        ]
        for line in procedure_lines:
            self.assertIn("are", line, "Plural procedures should use 'are' not 'is'")
            self.assertIn(
                "documented on", line, "Procedures should include date qualification"
            )
            self.assertIn("SNOMED:", line, "Procedures should include detailed codes")

        # Test that medications are aggregated with correct plural grammar
        medication_lines = [
            line
            for line in summary.split(".")
            if "medications" in line.lower() and "mCODE" in line
        ]
        for line in medication_lines:
            self.assertIn("are", line, "Plural medications should use 'are' not 'is'")
            self.assertIn("RxNorm:", line, "Medications should include detailed codes")

        # Test that conditions are aggregated by date with correct plural grammar
        condition_lines = [
            line
            for line in summary.split(".")
            if "conditions" in line.lower() and "mCODE" in line
        ]
        for line in condition_lines:
            self.assertIn("are", line, "Plural conditions should use 'are' not 'is'")
            self.assertIn(
                "documented on", line, "Conditions should include date qualification"
            )
            self.assertIn("SNOMED:", line, "Conditions should include detailed codes")

        # Test that social determinants are aggregated with correct plural grammar
        social_lines = [
            line
            for line in summary.split(".")
            if "social determinants" in line.lower() and "mCODE" in line
        ]
        for line in social_lines:
            self.assertIn(
                "are", line, "Plural social determinants should use 'are' not 'is'"
            )

    def test_natural_language_grammar_consistency(self):
        """Test that all sentences follow natural language grammar rules."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        # Test that sentences are properly separated and end with periods
        # Split by period and check that we have multiple sentences
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        self.assertGreater(
            len(sentences), 1, "Should have multiple sentences separated by periods"
        )

        # Test that the original summary contains periods (indicating proper sentence separation)
        self.assertIn(
            ".", summary, "Summary should contain periods for sentence separation"
        )

        # Test that mCODE tags are properly formatted
        for line in sentences:
            if "(mCODE:" in line:
                self.assertIn(")", line, f"mCODE tag should be closed: {line}")

            # Test subject-verb agreement for plural subjects
            if any(
                word in line.lower()
                for word in ["procedures", "medications", "conditions", "determinants"]
            ):
                self.assertIn("are", line, f"Plural subject should use 'are': {line}")
                self.assertNotIn(
                    " is ", line, f"Plural subject should not use 'is': {line}"
                )

    def test_knowledge_graph_optimization(self):
        """Test that summaries are optimized for knowledge graph matching between patients and trials."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        # Test that all clinical elements have mCODE tags
        clinical_elements = [
            "diagnosis",
            "tumor staging",
            "tumor markers",
            "procedures",
            "medications",
            "conditions",
            "cause of death",
        ]
        for element in clinical_elements:
            element_lines = [
                line for line in summary.split(".") if element in line.lower()
            ]
            for line in element_lines:
                self.assertIn(
                    "(mCODE:",
                    line,
                    f"Clinical element '{element}' should have mCODE tag: {line}",
                )

        # Test that detailed codes are present and properly formatted
        self.assertIn("SNOMED:", summary, "Should contain SNOMED codes")
        self.assertIn("RxNorm:", summary, "Should contain RxNorm codes")
        self.assertIn("CDC-RACE:", summary, "Should contain CDC-RACE codes")

        # Test that dates are included where clinically relevant
        self.assertIn("documented on", summary, "Should include date qualifications")

        # Test that the summary flows naturally as clinical documentation
        # Check that dates are properly integrated into sentence structure
        # Allow dates that are part of proper sentence structure (like birth dates)
        isolated_dates = re.findall(
            r"(?<!documented on )(?<!on )(?<!BirthDate)(?<!birth date)(?<!is )\d{4}-\d{2}-\d{2}(?=\.)",
            summary,
        )
        self.assertEqual(
            len(isolated_dates),
            0,
            f"Should not have isolated dates at end of sentences: {isolated_dates}",
        )
        # Check that codes are properly integrated
        isolated_codes = re.findall(r"(?<!\w)SNOMED:\d+(?=\.)", summary)
        self.assertEqual(
            len(isolated_codes),
            0,
            f"Should not have isolated SNOMED codes at end of sentences: {isolated_codes}",
        )

    def test_subject_predicate_consistency(self):
        """Test that all mCODE sentences follow consistent subject-predicate format."""
        if not self.patient_data_path.exists():
            self.skipTest(f"{self.patient_data_path} not found.")

        with open(self.patient_data_path, "r") as f:
            patient_data_list = json.load(f)

        patient_data = patient_data_list[0]
        summary = self.summarizer_with_dates.create_patient_summary(patient_data)

        lines = [
            line.strip()
            for line in summary.split(".")
            if line.strip() and "(mCODE:" in line
        ]

        for line in lines:
            # Test that mCODE is the subject (comes after the clinical subject)
            mcode_pos = line.find("(mCODE:")
            if mcode_pos > 0:
                # Ensure there's a clinical subject before mCODE
                subject_part = line[:mcode_pos].strip()
                self.assertTrue(
                    len(subject_part) > 0,
                    f"Should have clinical subject before mCODE: {line}",
                )

            # Test that detailed codes are properly formatted
            if "SNOMED:" in line or "RxNorm:" in line or "CDC-RACE:" in line:
                # Codes should be in parentheses
                self.assertIn("(", line, "Codes should be in parentheses")
                self.assertIn(")", line, "Codes should be in parentheses")


if __name__ == "__main__":
    unittest.main()
