"""
Unit tests for shared extractors module.
"""

from src.shared.extractors import DataExtractor


class TestDataExtractor:
    """Test cases for DataExtractor class."""

    def test_extract_trial_id_success(self):
        """Test successful trial ID extraction."""
        trial = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678"
                }
            }
        }
        result = DataExtractor.extract_trial_id(trial)
        assert result == "NCT12345678"

    def test_extract_trial_id_missing_protocol_section(self):
        """Test trial ID extraction when protocolSection is missing."""
        trial = {"someOtherField": "value"}
        result = DataExtractor.extract_trial_id(trial)
        assert result == ""

    def test_extract_trial_id_missing_identification_module(self):
        """Test trial ID extraction when identificationModule is missing."""
        trial = {"protocolSection": {"someOtherField": "value"}}
        result = DataExtractor.extract_trial_id(trial)
        assert result == ""

    def test_extract_trial_id_missing_nct_id(self):
        """Test trial ID extraction when nctId is missing."""
        trial = {
            "protocolSection": {
                "identificationModule": {
                    "someOtherField": "value"
                }
            }
        }
        result = DataExtractor.extract_trial_id(trial)
        assert result == ""

    def test_extract_trial_id_empty_dict(self):
        """Test trial ID extraction with empty dict."""
        trial = {}
        result = DataExtractor.extract_trial_id(trial)
        assert result == ""

    def test_extract_trial_id_none_input(self):
        """Test trial ID extraction with None input."""
        result = DataExtractor.extract_trial_id(None)
        assert result == ""

    def test_extract_patient_id_direct_id_field(self):
        """Test patient ID extraction using direct 'id' field."""
        patient = {"id": "patient123"}
        result = DataExtractor.extract_patient_id(patient)
        assert result == "patient123"

    def test_extract_patient_id_resource_id_field(self):
        """Test patient ID extraction using resource.id field."""
        patient = {"resource": {"id": "resource456"}}
        result = DataExtractor.extract_patient_id(patient)
        assert result == "resource456"

    def test_extract_patient_id_identifier_value(self):
        """Test patient ID extraction using identifier.value field."""
        patient = {
            "identifier": [
                {"value": "identifier789"}
            ]
        }
        result = DataExtractor.extract_patient_id(patient)
        assert result == "identifier789"

    def test_extract_patient_id_multiple_identifiers(self):
        """Test patient ID extraction with multiple identifiers."""
        patient = {
            "identifier": [
                {"system": "system1", "value": "value1"},
                {"system": "system2", "value": "value2"}
            ]
        }
        result = DataExtractor.extract_patient_id(patient)
        assert result == "value1"

    def test_extract_patient_id_empty_identifier_list(self):
        """Test patient ID extraction with empty identifier list."""
        patient = {"identifier": []}
        result = DataExtractor.extract_patient_id(patient)
        assert result == ""

    def test_extract_patient_id_missing_fields(self):
        """Test patient ID extraction when no ID fields are present."""
        patient = {"name": "John Doe", "age": 30}
        result = DataExtractor.extract_patient_id(patient)
        assert result == ""

    def test_extract_patient_id_empty_dict(self):
        """Test patient ID extraction with empty dict."""
        patient = {}
        result = DataExtractor.extract_patient_id(patient)
        assert result == ""

    def test_extract_patient_id_none_input(self):
        """Test patient ID extraction with None input."""
        result = DataExtractor.extract_patient_id(None)
        assert result == ""

    def test_extract_patient_id_non_string_id(self):
        """Test patient ID extraction with non-string ID (should convert to string)."""
        patient = {"id": 12345}
        result = DataExtractor.extract_patient_id(patient)
        assert result == "12345"

    def test_extract_provider_from_model_deepseek(self):
        """Test provider extraction for DeepSeek models."""
        result = DataExtractor.extract_provider_from_model("deepseek-coder")
        assert result == "DeepSeek"

    def test_extract_provider_from_model_gpt(self):
        """Test provider extraction for GPT models."""
        result = DataExtractor.extract_provider_from_model("gpt-4")
        assert result == "OpenAI"

    def test_extract_provider_from_model_gpt_3_5(self):
        """Test provider extraction for GPT-3.5 models."""
        result = DataExtractor.extract_provider_from_model("gpt-3.5-turbo")
        assert result == "OpenAI"

    def test_extract_provider_from_model_other(self):
        """Test provider extraction for other/unknown models."""
        result = DataExtractor.extract_provider_from_model("claude-3")
        assert result == "Other"

    def test_extract_provider_from_model_empty_string(self):
        """Test provider extraction with empty string."""
        result = DataExtractor.extract_provider_from_model("")
        assert result == "Other"

    def test_extract_provider_from_model_none(self):
        """Test provider extraction with None input."""
        result = DataExtractor.extract_provider_from_model(None)
        assert result == "Other"