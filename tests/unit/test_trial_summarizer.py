"""
Unit tests for TrialSummarizer class.
"""

from unittest.mock import patch

import pytest

from src.workflows.trial_summarizer import TrialSummarizer


class TestTrialSummarizer:
    """Test cases for TrialSummarizer class."""

    @pytest.fixture
    def trial_summarizer(self):
        """Create TrialSummarizer instance."""
        return TrialSummarizer()

    @pytest.fixture
    def trial_summarizer_no_dates(self):
        """Create TrialSummarizer instance without dates."""
        return TrialSummarizer(include_dates=False)

    @pytest.fixture
    def sample_mcode_elements(self):
        """Sample mCODE elements for testing."""
        return {
            "TrialIdentifier": {
                "system": "https://clinicaltrials.gov",
                "code": "NCT12345678",
                "display": "Clinical Trial NCT12345678",
            },
            "TrialCancerConditions": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "C50",
                    "display": "Breast Cancer",
                    "interpretation": "Confirmed",
                },
                {
                    "system": "http://snomed.info/sct",
                    "code": "C34",
                    "display": "Lung Cancer",
                    "interpretation": "Confirmed",
                },
            ],
            "TrialMedicationInterventions": [
                {
                    "system": "http://snomed.info/sct",
                    "code": "Unknown",
                    "display": "Test Drug",
                    "description": "Test drug description",
                    "interventionType": "drug",
                }
            ],
            "TrialStudyType": {
                "display": "Interventional",
                "code": "interventional",
            },
            "TrialStatus": {
                "display": "Completed",
                "code": "completed",
            },
        }

    @pytest.fixture
    def sample_trial_data(self):
        """Sample trial data for testing."""
        return {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678", "briefTitle": "Test Trial"}
            }
        }

    def test_init_with_dates(self, trial_summarizer):
        """Test initialization with dates enabled."""
        assert trial_summarizer.summarizer is not None
        assert hasattr(trial_summarizer.summarizer, "include_dates")
        assert trial_summarizer.summarizer.include_dates is True

    def test_init_without_dates(self, trial_summarizer_no_dates):
        """Test initialization with dates disabled."""
        assert trial_summarizer_no_dates.summarizer is not None
        assert hasattr(trial_summarizer_no_dates.summarizer, "include_dates")
        assert trial_summarizer_no_dates.summarizer.include_dates is False

    @patch("src.services.summarizer.McodeSummarizer.create_trial_summary")
    def test_generate_trial_natural_language_summary_success(
        self, mock_create_summary, trial_summarizer, sample_trial_data
    ):
        """Test successful generation of trial natural language summary."""
        mock_create_summary.return_value = "This is a comprehensive trial summary."

        result = trial_summarizer.generate_trial_natural_language_summary(
            "NCT12345678", {}, sample_trial_data
        )

        assert result == "This is a comprehensive trial summary."
        mock_create_summary.assert_called_once_with(sample_trial_data)

    @patch("src.services.summarizer.McodeSummarizer.create_trial_summary")
    def test_generate_trial_natural_language_summary_exception(
        self, mock_create_summary, trial_summarizer, sample_trial_data
    ):
        """Test exception handling in trial summary generation."""
        mock_create_summary.side_effect = Exception("Test error")

        result = trial_summarizer.generate_trial_natural_language_summary(
            "NCT12345678", {}, sample_trial_data
        )

        assert "Error generating comprehensive summary" in result
        assert "Test error" in result
        assert "NCT12345678" in result

    def test_convert_trial_mcode_to_mappings_format_complex_elements(
        self, trial_summarizer, sample_mcode_elements
    ):
        """Test conversion of complex mCODE elements to mappings format."""
        result = trial_summarizer.convert_trial_mcode_to_mappings_format(sample_mcode_elements)

        assert isinstance(result, list)
        assert (
            len(result) == 6
        )  # 1 identifier + 2 conditions + 1 intervention + 1 study type + 1 status + 1 comorbid condition

        # Check TrialIdentifier mapping
        identifier_mapping = next(m for m in result if m["mcode_element"] == "TrialIdentifier")
        assert identifier_mapping["value"] == "Clinical Trial NCT12345678"
        assert identifier_mapping["system"] == "https://clinicaltrials.gov"
        assert identifier_mapping["code"] == "NCT12345678"
        assert identifier_mapping["interpretation"] is None

        # Check TrialCancerConditions mappings
        condition_mappings = [m for m in result if m["mcode_element"] == "TrialCancerConditions"]
        assert len(condition_mappings) == 2
        assert condition_mappings[0]["value"] == "Breast Cancer"
        assert condition_mappings[0]["interpretation"] == "Confirmed"
        assert condition_mappings[1]["value"] == "Lung Cancer"

        # Check TrialMedicationInterventions mapping
        intervention_mapping = next(
            m for m in result if m["mcode_element"] == "TrialMedicationInterventions"
        )
        assert intervention_mapping["value"] == "Test Drug"
        assert intervention_mapping["system"] == "http://snomed.info/sct"
        assert intervention_mapping["code"] == "Unknown"

    def test_convert_trial_mcode_to_mappings_format_simple_elements(self, trial_summarizer):
        """Test conversion of simple mCODE elements to mappings format."""
        simple_elements = {
            "TrialStudyType": "Interventional",
            "TrialPhase": "Phase 1",
            "TrialStatus": {"display": "Completed", "code": "completed"},
        }

        result = trial_summarizer.convert_trial_mcode_to_mappings_format(simple_elements)

        assert len(result) == 3

        # Check string element
        study_type_mapping = next(m for m in result if m["mcode_element"] == "TrialStudyType")
        assert study_type_mapping["value"] == "Interventional"
        assert study_type_mapping["system"] is None
        assert study_type_mapping["code"] is None

        # Check dict element
        status_mapping = next(m for m in result if m["mcode_element"] == "TrialStatus")
        assert status_mapping["value"] == "Completed"
        assert status_mapping["code"] == "completed"

    def test_convert_trial_mcode_to_mappings_format_list_with_strings(self, trial_summarizer):
        """Test conversion of list elements containing strings."""
        elements_with_strings = {"TrialPhases": ["Phase 1", "Phase 2", "Phase 3"]}

        result = trial_summarizer.convert_trial_mcode_to_mappings_format(elements_with_strings)

        assert len(result) == 3
        for i, mapping in enumerate(result):
            assert mapping["mcode_element"] == "TrialPhases"
            assert mapping["value"] == f"Phase {i+1}"
            assert mapping["system"] is None
            assert mapping["code"] is None

    def test_convert_trial_mcode_to_mappings_format_empty(self, trial_summarizer):
        """Test conversion of empty mCODE elements."""
        result = trial_summarizer.convert_trial_mcode_to_mappings_format({})
        assert result == []

    def test_convert_trial_mcode_to_mappings_format_exception(self, trial_summarizer):
        """Test exception handling in mCODE conversion."""
        # Mock the print function to avoid output during test
        with patch("builtins.print"):
            # Create an element that will cause an exception during processing
            problematic_elements = {"ProblemElement": {"nested": {"deeply": {"nested": "value"}}}}

            # The method should handle exceptions gracefully and continue processing
            result = trial_summarizer.convert_trial_mcode_to_mappings_format(problematic_elements)
            # Should return a mapping for the element that can be processed
            assert len(result) == 1
            assert result[0]["mcode_element"] == "ProblemElement"

    def test_format_trial_mcode_element_snomed(self, trial_summarizer):
        """Test formatting of SNOMED mCODE elements."""
        result = trial_summarizer.format_trial_mcode_element(
            "CancerCondition", "http://snomed.info/sct", "C50"
        )
        assert result == "(mCODE: CancerCondition; SNOMED:C50)"

    def test_format_trial_mcode_element_loinc(self, trial_summarizer):
        """Test formatting of LOINC mCODE elements."""
        result = trial_summarizer.format_trial_mcode_element(
            "LabTest", "http://loinc.org", "12345-6"
        )
        assert result == "(mCODE: LabTest; LOINC:12345-6)"

    def test_format_trial_mcode_element_cvx(self, trial_summarizer):
        """Test formatting of CVX mCODE elements."""
        result = trial_summarizer.format_trial_mcode_element(
            "Vaccine", "http://hl7.org/fhir/sid/cvx", "207"
        )
        assert result == "(mCODE: Vaccine; CVX:207)"

    def test_format_trial_mcode_element_rxnorm(self, trial_summarizer):
        """Test formatting of RxNorm mCODE elements."""
        result = trial_summarizer.format_trial_mcode_element(
            "Medication", "http://www.nlm.nih.gov/research/umls/rxnorm", "123456"
        )
        assert result == "(mCODE: Medication; RxNorm:123456)"

    def test_format_trial_mcode_element_icd(self, trial_summarizer):
        """Test formatting of ICD mCODE elements."""
        result = trial_summarizer.format_trial_mcode_element(
            "Diagnosis", "http://hl7.org/fhir/sid/icd-10", "C50.1"
        )
        assert result == "(mCODE: Diagnosis; ICD:C50.1)"

    def test_format_trial_mcode_element_clinicaltrials_gov(self, trial_summarizer):
        """Test formatting of ClinicalTrials.gov mCODE elements."""
        result = trial_summarizer.format_trial_mcode_element(
            "TrialIdentifier", "https://clinicaltrials.gov", "NCT12345678"
        )
        assert result == "(mCODE: TrialIdentifier; ClinicalTrials.gov:NCT12345678)"

    def test_format_trial_mcode_element_generic_url(self, trial_summarizer):
        """Test formatting of generic URL system."""
        result = trial_summarizer.format_trial_mcode_element(
            "GenericElement", "http://example.com/system/custom", "CODE123"
        )
        assert result == "(mCODE: GenericElement; CUSTOM:CODE123)"

    def test_format_trial_mcode_element_unknown_system(self, trial_summarizer):
        """Test formatting with unknown system."""
        result = trial_summarizer.format_trial_mcode_element(
            "UnknownElement", "unknown-system", "UNKNOWN"
        )
        assert result == "(mCODE: UnknownElement; UNKNOWN-SYSTEM:UNKNOWN)"

    def test_format_trial_mcode_element_case_insensitive(self, trial_summarizer):
        """Test that system formatting is case insensitive."""
        result = trial_summarizer.format_trial_mcode_element(
            "TestElement", "HTTP://SNOMED.INFO/SCT", "C50"
        )
        assert result == "(mCODE: TestElement; SNOMED:C50)"

    def test_summarizer_initialization(self, trial_summarizer):
        """Test that McodeSummarizer is initialized correctly."""
        # Verify that the summarizer was created and has the expected attribute
        assert hasattr(trial_summarizer, "summarizer")
        assert trial_summarizer.summarizer is not None

    def test_summarizer_initialization_no_dates(self, trial_summarizer_no_dates):
        """Test that McodeSummarizer is initialized correctly without dates."""
        # Verify that the summarizer was created and has the expected attribute
        assert hasattr(trial_summarizer_no_dates, "summarizer")
        assert trial_summarizer_no_dates.summarizer is not None
