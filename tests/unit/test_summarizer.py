"""
Unit tests for McodeSummarizer with mocked dependencies.
"""
import pytest
from unittest.mock import Mock, patch
from src.services.summarizer import McodeSummarizer


@pytest.mark.mock
class TestMcodeSummarizer:
    """Test McodeSummarizer functionality with mocks."""

    def test_init(self):
        """Test McodeSummarizer initialization."""
        summarizer = McodeSummarizer(include_dates=True)
        assert summarizer.include_dates is True

        summarizer_no_dates = McodeSummarizer(include_dates=False)
        assert summarizer_no_dates.include_dates is False

    def test_format_mcode_display(self):
        """Test formatting mCODE display."""
        summarizer = McodeSummarizer()

        result = summarizer._format_mcode_display("PrimaryCancerCondition", "http://snomed.info/sct", "12345")
        expected = "(mCODE: PrimaryCancerCondition, SNOMED:12345)"

        assert result == expected

    def test_format_date_simple(self):
        """Test simple date formatting."""
        summarizer = McodeSummarizer()

        result = summarizer._format_date_simple("2023-01-15")
        assert result == "2023-01-15"

        result_invalid = summarizer._format_date_simple("invalid")
        assert result_invalid == "invalid"

    def test_create_mcode_sentence(self):
        """Test creating mCODE sentence."""
        summarizer = McodeSummarizer()

        result = summarizer._create_mcode_sentence(
            "PrimaryCancerCondition",
            "http://snomed.info/sct",
            "254837009",
            "Malignant neoplasm of breast"
        )

        assert "PrimaryCancerCondition" in result
        assert "254837009" in result

    def test_create_patient_demographics_sentence(self):
        """Test creating patient demographics sentence."""
        summarizer = McodeSummarizer()

        patient_data = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "12345",
                        "gender": "female",
                        "birthDate": "1980-01-01",
                        "name": [{"family": "Doe", "given": ["Jane"]}]
                    }
                }
            ]
        }

        result = summarizer._create_patient_demographics_sentence(patient_data)

        assert isinstance(result, dict)
        assert "demographics_sentence" in result
        assert "Jane Doe" in result["demographics_sentence"]
        assert "Patient" in result["demographics_sentence"]
        assert "mCODE" in result["demographics_sentence"]

    def test_create_trial_subject_predicate_sentence(self):
        """Test creating trial subject predicate sentence."""
        summarizer = McodeSummarizer()

        result = summarizer._create_trial_subject_predicate_sentence(
            "Trial eligibility", "TrialEligibility", "Age >= 18 years"
        )

        assert "Trial eligibility" in result
        assert "Age >= 18 years" in result
        assert "(mCODE: TrialEligibility)" in result

    def test_create_patient_summary(self):
        """Test creating patient summary."""
        summarizer = McodeSummarizer()

        patient_data = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "12345",
                        "gender": "female",
                        "birthDate": "1980-01-01",
                        "name": [{"family": "Doe", "given": ["Jane"]}]
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "254837009",
                                    "display": "Malignant neoplasm of breast"
                                }
                            ]
                        }
                    }
                }
            ]
        }

        result = summarizer.create_patient_summary(patient_data)

        assert isinstance(result, str)
        assert "Jane Doe" in result
        assert "female" in result
        assert "1980" in result
        assert "breast" in result.lower()

    def test_create_trial_summary(self):
        """Test creating trial summary."""
        summarizer = McodeSummarizer()

        trial_data = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Trial"
                },
                "conditionsModule": {
                    "conditions": [{"name": "Breast Cancer"}]
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Age >= 18"
                }
            }
        }

        result = summarizer.create_trial_summary(trial_data)

        assert isinstance(result, str)
        assert "NCT12345678" in result
        assert "Test Trial" in result
        assert "Breast Cancer" in result

    def test_check_trial_data_completeness_complete(self):
        """Test checking complete trial data."""
        summarizer = McodeSummarizer()

        trial_data = {
            "nct_id": "NCT12345678",
            "title": "Complete Trial",
            "eligibility": {"criteria": "Age >= 18"},
            "conditions": ["Cancer"]
        }

        result = summarizer._check_trial_data_completeness(trial_data)

        assert "complete" in result.lower()

    def test_check_trial_data_completeness_incomplete(self):
        """Test checking incomplete trial data."""
        summarizer = McodeSummarizer()

        trial_data = {
            "nct_id": "NCT12345678"
            # Missing title, eligibility, conditions
        }

        result = summarizer._check_trial_data_completeness(trial_data)

        assert "incomplete" in result.lower() or "missing" in result.lower()

    def test_create_patient_summary_empty_data(self):
        """Test creating patient summary with empty data."""
        summarizer = McodeSummarizer()

        with pytest.raises(ValueError, match="Patient data is missing or not in the expected format"):
            summarizer.create_patient_summary({})

    def test_create_trial_summary_empty_data(self):
        """Test creating trial summary with empty data."""
        summarizer = McodeSummarizer()

        with pytest.raises(ValueError, match="Trial data is missing or not in the expected format"):
            summarizer.create_trial_summary({})

    def test_format_mcode_display_edge_cases(self):
        """Test mCODE display formatting edge cases."""
        summarizer = McodeSummarizer()

        # Empty values
        result = summarizer._format_mcode_display("", "", "")
        assert result == ""

        # None values
        result = summarizer._format_mcode_display(None, None, None)
        assert result == "None"

    def test_format_date_simple_edge_cases(self):
        """Test date formatting edge cases."""
        summarizer = McodeSummarizer()

        # None input
        result = summarizer._format_date_simple(None)
        assert result == ""

        # Empty string
        result = summarizer._format_date_simple("")
        assert result == ""

        # Invalid format
        result = summarizer._format_date_simple("not-a-date")
        assert result == "not-a-date"
