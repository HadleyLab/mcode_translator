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

    def test_create_abstracted_sentence(self):
        """Test creating abstracted mCODE sentence using new interface."""
        summarizer = McodeSummarizer()

        result = summarizer._create_abstracted_sentence(
            subject="Patient",
            element_name="CancerCondition",
            value="Malignant neoplasm of breast",
            codes="SNOMED:254837009",
            date_qualifier=" documented on 2023-01-01"
        )

        assert "Patient's diagnosis" in result
        assert "Malignant neoplasm of breast" in result
        assert "SNOMED:254837009" in result
        assert "(mCODE: CancerCondition documented on 2023-01-01)" in result

    def test_extract_patient_elements(self):
        """Test extracting patient elements using new abstracted interface."""
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

        elements = summarizer._extract_patient_elements(patient_data, include_dates=True)

        assert isinstance(elements, list)
        assert len(elements) >= 1

        # Check for Patient element
        patient_element = next((e for e in elements if e.get('element_name') == 'Patient'), None)
        assert patient_element is not None
        assert "Jane Doe" in patient_element['value']
        assert "ID: 12345" in patient_element['value']

    def test_extract_trial_elements(self):
        """Test extracting trial elements using new abstracted interface."""
        summarizer = McodeSummarizer()

        trial_data = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Trial"
                },
                "statusModule": {
                    "overallStatus": "Recruiting"
                }
            }
        }

        elements = summarizer._extract_trial_elements(trial_data)

        assert isinstance(elements, list)
        assert len(elements) >= 1

        # Check for Trial element
        trial_element = next((e for e in elements if e.get('element_name') == 'Trial'), None)
        assert trial_element is not None
        assert "Clinical Trial" in trial_element['value']

        # Check for TrialTitle element
        title_element = next((e for e in elements if e.get('element_name') == 'TrialTitle'), None)
        assert title_element is not None
        assert "Test Trial" in title_element['value']

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
        # The new implementation focuses on core trial elements, not conditions
        assert "Clinical Trial" in result

    def test_group_elements_by_priority(self):
        """Test grouping elements by clinical priority."""
        summarizer = McodeSummarizer()

        elements = [
            {'element_name': 'TrialTitle', 'value': 'Test Trial', 'priority': 16},
            {'element_name': 'Trial', 'value': 'Clinical Trial', 'priority': 15},
            {'element_name': 'TrialStatus', 'value': 'recruiting', 'priority': 19}
        ]

        grouped = summarizer._group_elements_by_priority(elements, "Trial")

        assert isinstance(grouped, list)
        assert len(grouped) == 3
        # Should be sorted by priority (lower number = higher priority)
        assert grouped[0]['element_name'] == 'Trial'
        assert grouped[1]['element_name'] == 'TrialTitle'
        assert grouped[2]['element_name'] == 'TrialStatus'

    def test_create_patient_summary_empty_data(self):
        """Test creating patient summary with empty data."""
        summarizer = McodeSummarizer()

        with pytest.raises(ValueError, match="Patient data missing required format"):
            summarizer.create_patient_summary({})

    def test_create_trial_summary_empty_data(self):
        """Test creating trial summary with empty data."""
        summarizer = McodeSummarizer()

        with pytest.raises(ValueError, match="Trial data missing required format"):
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
