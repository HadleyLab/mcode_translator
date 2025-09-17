"""
Unit tests for Pydantic data models.
"""

from datetime import datetime

import pytest

from src.shared.models import (BenchmarkResult, ClinicalTrialData, FHIRBundle,
                               FHIRPatient, McodeElement, PatientData,
                               PipelineResult, ProcessingMetadata,
                               SourceReference, TokenUsage, ValidationResult,
                               WorkflowResult)


class TestClinicalTrialData:
    """Test ClinicalTrialData model."""

    def test_valid_clinical_trial_data(self):
        """Test creating valid clinical trial data."""
        data = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Trial",
                    "officialTitle": "Official Test Trial Title",
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion: Age 18+",
                    "healthyVolunteers": False,
                    "sex": "ALL",
                    "minimumAge": "18 Years",
                    "maximumAge": "65 Years",
                },
            },
            "hasResults": False,
            "studyType": "Interventional",
            "overallStatus": "Recruiting",
            "phase": "Phase 2",
        }

        trial = ClinicalTrialData(**data)
        assert trial.nct_id == "NCT12345678"
        assert trial.brief_title == "Test Trial"
        assert (
            trial.protocolSection.eligibilityModule.eligibilityCriteria
            == "Inclusion: Age 18+"
        )

    def test_invalid_clinical_trial_data(self):
        """Test validation of invalid clinical trial data."""
        invalid_data = {
            "protocolSection": {
                "identificationModule": {
                    "briefTitle": "Test Trial"
                    # Missing required nctId
                }
            }
        }

        with pytest.raises(Exception) as exc_info:
            ClinicalTrialData(**invalid_data)
        assert "nctId" in str(exc_info.value)


class TestMcodeElement:
    """Test McodeElement model."""

    def test_valid_mcode_element(self):
        """Test creating valid mCODE element."""
        element_data = {
            "element_type": "CancerCondition",
            "code": "C4872",
            "display": "Breast Carcinoma",
            "system": "NCIT",
            "confidence_score": 0.95,
            "evidence_text": "Patient diagnosed with breast cancer",
        }

        element = McodeElement(**element_data)
        assert element.element_type == "CancerCondition"
        assert element.confidence_score == 0.95

    def test_invalid_confidence_score(self):
        """Test validation of invalid confidence score."""
        with pytest.raises(ValueError):
            McodeElement(
                element_type="CancerCondition", confidence_score=1.5  # Invalid: > 1.0
            )


class TestPipelineResult:
    """Test PipelineResult model."""

    def test_valid_pipeline_result(self):
        """Test creating valid pipeline result."""
        result_data = {
            "extracted_entities": [],
            "mcode_mappings": [
                {
                    "element_type": "CancerCondition",
                    "code": "C4872",
                    "display": "Breast Carcinoma",
                }
            ],
            "source_references": [
                {
                    "document_type": "protocol",
                    "section_name": "eligibility",
                    "text_snippet": "breast cancer patients",
                }
            ],
            "validation_results": {
                "compliance_score": 0.85,
                "validation_errors": [],
                "required_elements_present": ["CancerCondition"],
            },
            "metadata": {"engine_type": "LLM", "entities_count": 0, "mapped_count": 1},
            "original_data": {"test": "data"},
        }

        result = PipelineResult(**result_data)
        assert len(result.mcode_mappings) == 1
        assert result.validation_results.compliance_score == 0.85
        assert result.metadata.engine_type == "LLM"


class TestWorkflowResult:
    """Test WorkflowResult model."""

    def test_valid_workflow_result(self):
        """Test creating valid workflow result."""
        result_data = {
            "success": True,
            "data": {"trials_processed": 5},
            "metadata": {"duration": 120.5},
        }

        result = WorkflowResult(**result_data)
        assert result.success is True
        assert result.data["trials_processed"] == 5


class TestBenchmarkResult:
    """Test BenchmarkResult model."""

    def test_valid_benchmark_result(self):
        """Test creating valid benchmark result."""
        result_data = {
            "task_id": "test-task-123",
            "trial_id": "NCT12345678",
            "pipeline_result": {
                "extracted_entities": [],
                "mcode_mappings": [],
                "source_references": [],
                "validation_results": {"compliance_score": 0.0},
                "metadata": {"engine_type": "test"},
                "original_data": {},
            },
            "execution_time_seconds": 45.2,
            "status": "success",
            "precision": 0.85,
            "recall": 0.92,
            "f1_score": 0.88,
        }

        result = BenchmarkResult(**result_data)
        assert result.task_id == "test-task-123"
        assert result.precision == 0.85
        assert result.execution_time_seconds == 45.2


class TestPatientData:
    """Test PatientData model."""

    def test_valid_patient_data(self):
        """Test creating valid patient data."""
        patient_data = {
            "bundle": {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "patient-123",
                            "identifier": [{"use": "usual", "value": "PATIENT123"}],
                            "name": [{"family": "Doe", "given": ["John"]}],
                            "gender": "male",
                            "birthDate": "1980-01-01",
                        }
                    }
                ],
            },
            "source_file": "test_patient.json",
            "archive_name": "breast_cancer_10_years.zip",
        }

        patient = PatientData(**patient_data)
        assert patient.patient_id == "patient-123"
        assert patient.patient.gender == "male"
        assert patient.source_file == "test_patient.json"

    def test_invalid_patient_data(self):
        """Test validation of invalid patient data."""
        invalid_data = {
            "bundle": {
                "resourceType": "Invalid",  # Wrong resource type
                "type": "collection",
            }
        }

        with pytest.raises(Exception) as exc_info:
            PatientData(**invalid_data)
        assert "resourceType" in str(exc_info.value)


class TestTokenUsage:
    """Test TokenUsage model."""

    def test_token_usage_with_total(self):
        """Test token usage with explicit total."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150

    def test_token_usage_calculate_total(self):
        """Test automatic total calculation."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150


class TestValidationResult:
    """Test ValidationResult model."""

    def test_validation_result(self):
        """Test creating validation result."""
        result = ValidationResult(
            compliance_score=0.92,
            validation_errors=["Missing required element"],
            required_elements_present=["CancerCondition", "CancerTreatment"],
            missing_elements=["TumorMarker"],
        )
        assert result.compliance_score == 0.92
        assert len(result.validation_errors) == 1
        assert "CancerCondition" in result.required_elements_present


if __name__ == "__main__":
    pytest.main([__file__])
