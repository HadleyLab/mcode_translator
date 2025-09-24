"""
Unit tests for PatientsProcessorWorkflow.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.workflows.patients_processor_workflow import PatientsProcessorWorkflow
from src.utils.config import Config


class TestPatientsProcessorWorkflow:
    """Test cases for PatientsProcessorWorkflow."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config()

    @pytest.fixture
    def workflow(self, config):
        """Create a test workflow instance."""
        return PatientsProcessorWorkflow(config)

    @pytest.fixture
    def mock_patient_data(self):
        """Create mock patient data for testing."""
        return {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient_123",
                        "name": [{"given": ["John"], "family": "Doe"}],
                        "birthDate": "1980-01-01",
                        "gender": "male"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": "254837009",
                                "display": "Breast Cancer"
                            }]
                        }
                    }
                }
            ]
        }

    def test_workflow_initialization(self, workflow):
        """Test workflow initializes correctly."""
        assert workflow is not None
        assert hasattr(workflow, 'execute')
        assert hasattr(workflow, 'summarizer')

    @patch('src.workflows.patients_processor_workflow.TaskQueue')
    def test_execute_success(self, mock_task_queue, workflow, mock_patient_data):
        """Test successful patient processing execution."""
        # Mock task queue
        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance

        # Mock task results
        mock_result = Mock()
        mock_result.success = True
        mock_result.result = mock_patient_data.copy()
        mock_queue_instance.execute_tasks.return_value = [mock_result]

        result = workflow.execute(patients_data=[mock_patient_data])

        assert result.success is True
        assert len(result.data) == 1
        assert result.metadata["total_patients"] == 1
        assert result.metadata["successful"] == 1
        assert result.metadata["failed"] == 0

    def test_execute_no_patients_data(self, workflow):
        """Test execute fails with no patient data."""
        result = workflow.execute()

        assert result.success is False
        assert "No patient data provided" in result.error_message

    @patch('src.workflows.patients_processor_workflow.TaskQueue')
    def test_execute_with_task_failures(self, mock_task_queue, workflow, mock_patient_data):
        """Test execute handles task failures gracefully."""
        # Mock task queue
        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance

        # Mock failed task result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Processing failed"
        mock_result.task_id = "patient_0"
        mock_queue_instance.execute_tasks.return_value = [mock_result]

        result = workflow.execute(patients_data=[mock_patient_data])

        assert result.success is False  # Fails when all tasks fail
        assert len(result.data) == 1
        assert "McodeProcessingError" in result.data[0]

    @patch('src.workflows.patients_processor_workflow.TaskQueue')
    def test_execute_all_tasks_fail(self, mock_task_queue, workflow, mock_patient_data):
        """Test execute when all tasks fail."""
        # Mock task queue
        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance

        # Mock failed task result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Processing failed"
        mock_result.task_id = "patient_0"
        mock_queue_instance.execute_tasks.return_value = [mock_result]

        result = workflow.execute(patients_data=[mock_patient_data])

        assert result.success is False  # Fails when all tasks fail
        assert len(result.data) == 1

    def test_execute_unexpected_error(self, workflow):
        """Test execute handles unexpected errors."""
        with patch.object(workflow, '_create_result') as mock_create_result:
            mock_create_result.side_effect = Exception("Unexpected error")

            with pytest.raises(Exception, match="Unexpected error"):
                workflow.execute(patients_data=[{"test": "data"}])

    def test_process_single_patient_success(self, workflow, mock_patient_data):
        """Test successful single patient processing."""
        result = workflow._process_single_patient(
            patient=mock_patient_data,
            patient_index=0,
            trials_criteria=None,
            store_in_memory=False
        )

        assert isinstance(result, dict)
        assert "filtered_mcode_elements" in result
        assert "mcode_processing_metadata" in result

    def test_process_single_patient_with_memory_storage(self, workflow, mock_patient_data):
        """Test single patient processing with memory storage."""
        # Mock memory storage
        workflow.memory_storage = Mock()
        workflow.memory_storage.store_patient_mcode_summary.return_value = True

        result = workflow._process_single_patient(
            patient=mock_patient_data,
            patient_index=0,
            trials_criteria=None,
            store_in_memory=True
        )

        assert isinstance(result, dict)
        workflow.memory_storage.store_patient_mcode_summary.assert_called_once()

    def test_process_single_patient_memory_storage_failure(self, workflow, mock_patient_data):
        """Test single patient processing when memory storage fails."""
        # Mock memory storage
        workflow.memory_storage = Mock()
        workflow.memory_storage.store_patient_mcode_summary.return_value = False

        result = workflow._process_single_patient(
            patient=mock_patient_data,
            patient_index=0,
            trials_criteria=None,
            store_in_memory=True
        )

        assert isinstance(result, dict)
        workflow.memory_storage.store_patient_mcode_summary.assert_called_once()

    def test_process_single_patient_with_trials_criteria(self, workflow, mock_patient_data):
        """Test single patient processing with trial criteria filtering."""
        trials_criteria = {"CancerCondition": True}

        result = workflow._process_single_patient(
            patient=mock_patient_data,
            patient_index=0,
            trials_criteria=trials_criteria,
            store_in_memory=False
        )

        assert isinstance(result, dict)
        assert "filtered_mcode_elements" in result
        assert result["mcode_processing_metadata"]["trial_criteria_applied"] is True

    def test_process_single_patient_error_handling(self, workflow):
        """Test single patient processing handles errors gracefully."""
        # Pass invalid patient data to trigger error
        result = workflow._process_single_patient(
            patient="invalid_data",
            patient_index=0,
            trials_criteria=None,
            store_in_memory=False
        )

        assert isinstance(result, dict)  # Error returns dict with error info
        assert "McodeProcessingError" in result

    def test_extract_patient_mcode_elements_patient_resource(self, workflow):
        """Test extracting mCODE elements from Patient resource."""
        patient_resource = {
            "resourceType": "Patient",
            "id": "patient_123",
            "name": [{"given": ["John"], "family": "Doe"}],
            "birthDate": "1980-01-01",
            "gender": "male"
        }

        elements = workflow._extract_patient_mcode_elements({
            "entry": [{"resource": patient_resource}]
        })

        assert isinstance(elements, dict)
        assert "name" in elements  # Demographics should be extracted

    def test_extract_patient_mcode_elements_condition_resource(self, workflow):
        """Test extracting mCODE elements from Condition resource."""
        condition_resource = {
            "resourceType": "Condition",
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "254837009",
                    "display": "Breast Cancer"
                }]
            }
        }

        elements = workflow._extract_patient_mcode_elements({
            "entry": [{"resource": condition_resource}]
        })

        assert isinstance(elements, dict)
        assert "CancerCondition" in elements

    def test_extract_patient_mcode_elements_invalid_data(self, workflow):
        """Test extracting mCODE elements handles invalid data."""
        elements = workflow._extract_patient_mcode_elements("invalid_data")

        assert isinstance(elements, dict)
        assert len(elements) == 0

    def test_extract_demographics_complete(self, workflow):
        """Test extracting complete demographics."""
        patient_resource = {
            "name": [{"given": ["John"], "family": "Doe"}],
            "birthDate": "1980-01-01",
            "gender": "male",
            "maritalStatus": {"coding": [{"display": "Married"}]},
            "communication": [{"language": {"coding": [{"display": "English"}]}}],
            "address": [{"city": "New York", "state": "NY", "country": "USA"}]
        }

        demographics = workflow._extract_demographics(patient_resource)

        assert demographics["name"] == "John Doe"
        assert demographics["birthDate"] == "1980-01-01"
        assert demographics["gender"] == "Male"
        assert demographics["maritalStatus"] == "Married"
        assert demographics["language"] == "English"
        assert "New York" in demographics["address"]

    def test_extract_demographics_minimal(self, workflow):
        """Test extracting minimal demographics."""
        patient_resource = {
            "name": [{"family": "Doe"}],
            "gender": "female"
        }

        demographics = workflow._extract_demographics(patient_resource)

        assert demographics["name"] == "Doe"
        assert demographics["gender"] == "Female"
        assert demographics["birthDate"] == "Unknown"
        assert demographics["age"] == "Unknown"

    def test_extract_condition_mcode_cancer(self, workflow):
        """Test extracting cancer condition mCODE."""
        condition = {
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "254837009",
                    "display": "Breast Cancer"
                }]
            }
        }

        result = workflow._extract_condition_mcode(condition)

        assert result is not None
        assert result["display"] == "Breast Cancer"
        assert result["code"] == "254837009"

    def test_extract_condition_mcode_non_cancer(self, workflow):
        """Test extracting non-cancer condition mCODE."""
        condition = {
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "123456",
                    "display": "Hypertension"
                }]
            }
        }

        result = workflow._extract_condition_mcode(condition)

        assert result is None

    def test_extract_observation_mcode_receptor_status(self, workflow):
        """Test extracting receptor status from observation."""
        observation = {
            "code": {"coding": [{"display": "Estrogen receptor status"}]},
            "valueCodeableConcept": {
                "coding": [{"code": "positive", "display": "Positive"}]
            }
        }

        elements = workflow._extract_observation_mcode(observation)

        assert "ERReceptorStatus" in elements
        assert elements["ERReceptorStatus"]["interpretation"] == "Positive"

    def test_extract_observation_mcode_stage(self, workflow):
        """Test extracting stage information from observation."""
        observation = {
            "code": {"coding": [{"display": "TNM Stage"}]},
            "valueCodeableConcept": {
                "coding": [{"code": "IIA", "display": "Stage IIA"}]
            }
        }

        elements = workflow._extract_observation_mcode(observation)

        assert "TNMStage" in elements
        assert elements["TNMStage"]["display"] == "Stage IIA"

    def test_extract_receptor_status(self, workflow):
        """Test extracting receptor status details."""
        observation = {
            "valueCodeableConcept": {
                "coding": [{"system": "SNOMED", "code": "positive", "display": "Positive"}]
            }
        }

        result = workflow._extract_receptor_status(observation)

        assert result["system"] == "SNOMED"
        assert result["code"] == "positive"
        assert result["display"] == "Positive"

    def test_extract_stage_info(self, workflow):
        """Test extracting stage information details."""
        observation = {
            "valueCodeableConcept": {
                "coding": [{"system": "SNOMED", "code": "IIA", "display": "Stage IIA"}]
            }
        }

        result = workflow._extract_stage_info(observation)

        assert result["system"] == "SNOMED"
        assert result["code"] == "IIA"
        assert result["display"] == "Stage IIA"

    def test_extract_procedure_mcode_surgery(self, workflow):
        """Test extracting surgical procedure mCODE."""
        procedure = {
            "code": {
                "coding": [{
                    "system": "SNOMED",
                    "code": "12345",
                    "display": "Mastectomy"
                }]
            },
            "performedDateTime": "2023-01-01"
        }

        result = workflow._extract_procedure_mcode(procedure)

        assert result is not None
        assert result["display"] == "Mastectomy"
        assert result["date"] == "2023-01-01"

    def test_extract_procedure_mcode_non_surgery(self, workflow):
        """Test extracting non-surgical procedure mCODE."""
        procedure = {
            "code": {
                "coding": [{
                    "system": "SNOMED",
                    "code": "12345",
                    "display": "Blood test"
                }]
            }
        }

        result = workflow._extract_procedure_mcode(procedure)

        assert result is None

    def test_extract_allergy_mcode(self, workflow):
        """Test extracting allergy mCODE."""
        allergy = {
            "code": {
                "coding": [{
                    "system": "SNOMED",
                    "code": "12345",
                    "display": "Penicillin allergy"
                }]
            },
            "criticality": "high",
            "recordedDate": "2023-01-01"
        }

        result = workflow._extract_allergy_mcode(allergy)

        assert result is not None
        assert result["display"] == "Penicillin allergy"
        assert result["criticality"] == "high"
        assert result["recordedDate"] == "2023-01-01"

    def test_extract_immunization_mcode(self, workflow):
        """Test extracting immunization mCODE."""
        immunization = {
            "vaccineCode": {
                "coding": [{
                    "system": "CVX",
                    "code": "123",
                    "display": "COVID-19 vaccine"
                }]
            },
            "occurrenceDateTime": "2023-01-01",
            "status": "completed"
        }

        result = workflow._extract_immunization_mcode(immunization)

        assert result is not None
        assert result["display"] == "COVID-19 vaccine"
        assert result["occurrenceDateTime"] == "2023-01-01"
        assert result["status"] == "completed"

    def test_extract_family_history_mcode(self, workflow):
        """Test extracting family history mCODE."""
        family_history = {
            "relationship": {
                "coding": [{
                    "system": "SNOMED",
                    "code": "12345",
                    "display": "Mother"
                }]
            },
            "condition": [{
                "code": {
                    "coding": [{
                        "system": "SNOMED",
                        "code": "67890",
                        "display": "Breast Cancer"
                    }]
                }
            }],
            "born": "1950-01-01"
        }

        result = workflow._extract_family_history_mcode(family_history)

        assert result is not None
        assert result["relationship"]["display"] == "Mother"
        assert len(result["conditions"]) == 1
        assert result["born"] == "1950-01-01"

    def test_extract_observation_mcode_comprehensive_vitals(self, workflow):
        """Test extracting comprehensive vital signs."""
        observation = {
            "code": {"coding": [{"display": "Body weight"}]},
            "valueQuantity": {"value": 70, "unit": "kg"}
        }

        elements = workflow._extract_observation_mcode_comprehensive(observation)

        assert "BodyWeight" in elements
        assert elements["BodyWeight"]["value"] == 70
        assert elements["BodyWeight"]["unit"] == "kg"

    def test_extract_observation_mcode_comprehensive_labs(self, workflow):
        """Test extracting comprehensive lab results."""
        observation = {
            "code": {"coding": [{"display": "Hemoglobin"}]},
            "valueQuantity": {"value": 12.5, "unit": "g/dL"}
        }

        elements = workflow._extract_observation_mcode_comprehensive(observation)

        assert "Hemoglobin" in elements
        assert elements["Hemoglobin"]["value"] == 12.5
        assert elements["Hemoglobin"]["unit"] == "g/dL"

    def test_filter_by_trial_criteria(self, workflow):
        """Test filtering mCODE elements by trial criteria."""
        patient_mcode = {
            "CancerCondition": {"display": "Breast Cancer"},
            "ERReceptorStatus": {"interpretation": "Positive"},
            "ComorbidCondition": {"display": "Hypertension"}
        }

        trial_criteria = {"CancerCondition": True, "ERReceptorStatus": True}

        filtered = workflow._filter_by_trial_criteria(patient_mcode, trial_criteria)

        assert "CancerCondition" in filtered
        assert "ERReceptorStatus" in filtered
        assert "ComorbidCondition" not in filtered

    def test_convert_to_mappings_format_single(self, workflow):
        """Test converting single mCODE element to mappings format."""
        mcode_elements = {
            "CancerCondition": {
                "system": "SNOMED",
                "code": "12345",
                "display": "Breast Cancer"
            }
        }

        mappings = workflow._convert_to_mappings_format(mcode_elements)

        assert len(mappings) == 1
        assert mappings[0]["mcode_element"] == "CancerCondition"
        assert mappings[0]["system"] == "SNOMED"
        assert mappings[0]["code"] == "12345"

    def test_convert_to_mappings_format_list(self, workflow):
        """Test converting list of mCODE elements to mappings format."""
        mcode_elements = {
            "ComorbidCondition": [
                {"system": "SNOMED", "code": "12345", "display": "Hypertension"},
                {"system": "SNOMED", "code": "67890", "display": "Diabetes"}
            ]
        }

        mappings = workflow._convert_to_mappings_format(mcode_elements)

        assert len(mappings) == 2
        assert all(m["mcode_element"] == "ComorbidCondition" for m in mappings)

    def test_generate_natural_language_summary(self, workflow):
        """Test generating clinical note summary."""
        patient_id = "patient_123"
        mcode_elements = {
            "CancerCondition": {"display": "Breast Cancer", "code": "12345", "system": "SNOMED"},
            "ERReceptorStatus": {"display": "Positive", "code": "45678", "system": "SNOMED", "interpretation": "Positive"}
        }
        demographics = {
            "name": "John Doe",
            "age": "43",
            "gender": "Male",
            "birthDate": "1980-01-01"
        }

        summary = workflow._generate_natural_language_summary(
            patient_id, mcode_elements, demographics
        )

        assert isinstance(summary, str)
        assert "John Doe" in summary
        assert "Breast Cancer" in summary
        assert "ER receptor status is Positive" in summary

    def test_decode_birth_sex(self, workflow):
        """Test decoding birth sex codes."""
        assert workflow._decode_birth_sex("F") == "Female"
        assert workflow._decode_birth_sex("M") == "Male"
        assert workflow._decode_birth_sex("UNK") == "Unknown"
        assert workflow._decode_birth_sex("OTH") == "Other"

    def test_decode_marital_status(self, workflow):
        """Test decoding marital status codes."""
        assert workflow._decode_marital_status("M") == "Married"
        assert workflow._decode_marital_status("S") == "Single"
        assert workflow._decode_marital_status("D") == "Divorced"
        assert workflow._decode_marital_status("UNK") == "Unknown"

    def test_format_mcode_element(self, workflow):
        """Test formatting mCODE elements consistently."""
        result = workflow._format_mcode_element("CancerCondition", "http://snomed.info/sct", "12345")

        assert "mCODE: CancerCondition" in result
        assert "SNOMED:12345" in result

    def test_extract_patient_id_from_identifier(self, workflow):
        """Test extracting patient ID from identifier."""
        patient = {
            "entry": [{
                "resource": {
                    "resourceType": "Patient",
                    "identifier": [{"value": "PAT123"}]
                }
            }]
        }

        patient_id = workflow._extract_patient_id(patient)

        assert patient_id == "PAT123"

    def test_extract_patient_id_from_resource_id(self, workflow):
        """Test extracting patient ID from resource ID."""
        patient = {
            "entry": [{
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient_456"
                }
            }]
        }

        patient_id = workflow._extract_patient_id(patient)

        assert patient_id == "patient_456"

    def test_extract_patient_id_fallback_hash(self, workflow):
        """Test extracting patient ID falls back to hash."""
        patient = {
            "entry": [{
                "resource": {
                    "resourceType": "Patient"
                    # No identifier or id
                }
            }]
        }

        patient_id = workflow._extract_patient_id(patient)

        assert patient_id.startswith("patient_")
        assert len(patient_id) > 8  # Should include hash