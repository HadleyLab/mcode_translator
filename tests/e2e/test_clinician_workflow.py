#!/usr/bin/env python3
"""
End-to-End Tests for Clinician Workflow

Tests the complete clinician workflow from patient assessment to treatment recommendations:
1. Patient Assessment (patients_fetcher) → 2. Trial Matching (patients_processor) → 3. Treatment Recommendations (patients_summarizer)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.patients_fetcher import main as patients_fetcher_main
from src.cli.patients_processor import main as patients_processor_main
from src.cli.patients_summarizer import main as patients_summarizer_main
from src.shared.models import WorkflowResult


class TestClinicianWorkflowE2E:
    """End-to-end tests for the complete clinician workflow."""

    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data in FHIR Bundle format."""
        return {
            "resourceType": "Bundle",
            "id": "bundle-12345",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "12345",
                        "name": [
                            {
                                "family": "Smith",
                                "given": ["Jane"]
                            }
                        ],
                        "gender": "female",
                        "birthDate": "1975-03-15"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "condition-1",
                        "subject": {
                            "reference": "Patient/12345"
                        },
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "254837009",
                                    "display": "Malignant neoplasm of breast"
                                }
                            ]
                        },
                        "clinicalStatus": {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                    "code": "active"
                                }
                            ]
                        }
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-1",
                        "subject": {
                            "reference": "Patient/12345"
                        },
                        "code": {
                            "coding": [
                                {
                                    "system": "http://loinc.org",
                                    "code": "16112-5",
                                    "display": "Estrogen receptor Ag [Presence] in Breast cancer specimen by Immune stain"
                                }
                            ]
                        },
                        "valueCodeableConcept": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "10828004",
                                    "display": "Positive"
                                }
                            ]
                        }
                    }
                }
            ]
        }

    @pytest.fixture
    def sample_trial_criteria(self):
        """Sample trial data with mCODE mappings for eligibility filtering."""
        return {
            "trial_id": "NCT12345678",
            "McodeResults": {
                "mcode_mappings": [
                    {
                        "mcode_element": "TrialCancerConditions",
                        "value": "Breast Cancer",
                        "system": "http://snomed.info/sct",
                        "code": "254837009"
                    },
                    {
                        "mcode_element": "TrialSexCriteria",
                        "value": "Female",
                        "system": "http://hl7.org/fhir/administrative-gender",
                        "code": "female"
                    }
                ]
            }
        }

    @pytest.fixture
    def mock_memory_storage(self):
        """Mock CORE Memory storage."""
        mock_storage = MagicMock()
        mock_storage.store_patient_data.return_value = True
        mock_storage.store_processed_patient.return_value = True
        mock_storage.store_patient_summary.return_value = True
        return mock_storage

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for patient processing."""
        from src.shared.models import McodeElement

        mock_service = MagicMock()
        # Mock the async map_to_mcode method to return McodeElement instances
        async def mock_map_to_mcode(text):
            return [
                McodeElement(
                    element_type="PrimaryCancerCondition",
                    code="254837009",
                    system="http://snomed.info/sct",
                    display="Breast Cancer",
                    confidence_score=0.95,
                    evidence_text="Patient has malignant neoplasm of breast"
                ),
                McodeElement(
                    element_type="CancerBiomarker",
                    code="16112-5",
                    system="http://loinc.org",
                    display="ER Positive",
                    confidence_score=0.90,
                    evidence_text="Estrogen receptor positive in breast cancer specimen"
                )
            ]
        mock_service.map_to_mcode = mock_map_to_mcode
        return mock_service

    @pytest.fixture
    def mock_summarizer_service(self):
        """Mock summarizer service for patient summaries."""
        mock_summarizer = MagicMock()
        mock_summarizer.create_patient_summary.return_value = "Patient Jane Smith is a 49-year-old female with active breast cancer. She has estrogen receptor positive disease. Based on her clinical profile, she may be eligible for clinical trials targeting hormone receptor positive breast cancer."
        return mock_summarizer

    @patch('src.cli.patients_fetcher.PatientsFetcherWorkflow')
    def test_clinician_workflow_patient_assessment(self, mock_workflow_class, sample_patient_data, tmp_path):
        """Test the patient assessment phase of clinician workflow."""
        # Setup mock workflow
        mock_workflow = MagicMock()
        mock_workflow.execute.return_value = WorkflowResult(
            success=True,
            data=[sample_patient_data],
            metadata={"total_fetched": 1, "fetch_type": "archive", "archive_path": "breast_cancer_10_years"}
        )
        mock_workflow_class.return_value = mock_workflow

        # Create output file
        output_file = tmp_path / "patients.ndjson"

        # Test CLI execution
        import argparse
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            patient_id=None,
            limit=10,
            list_archives=False,
            output_file=str(output_file),
            verbose=False,
            log_level="INFO",
            config=None
        )

        import io
        import sys
        from contextlib import redirect_stdout
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            patients_fetcher_main(args)

        # Verify workflow was called correctly
        mock_workflow_class.assert_called_once()
        mock_workflow.execute.assert_called_once_with(
            archive_path="breast_cancer_10_years",
            limit=10,
            output_path=str(output_file)
        )

        # Verify success message in output
        output = stdout_capture.getvalue()
        assert "✅ Patients fetch completed successfully!" in output
        assert "📊 Total patients fetched: 1" in output

    @patch('src.pipeline.llm_service.LLMService')
    @patch('src.storage.mcode_memory_storage.McodeMemoryStorage')
    def test_clinician_workflow_trial_matching(self, mock_memory_storage_class, mock_llm_service_class,
                                             sample_patient_data, sample_trial_criteria, tmp_path):
        """Test the trial matching phase of clinician workflow."""
        from src.shared.models import McodeElement

        # Setup mocks
        mock_memory = MagicMock()
        mock_memory.store_processed_patient.return_value = True
        mock_memory_storage_class.return_value = mock_memory

        mock_llm = MagicMock()
        async def mock_map_to_mcode(text):
            return [
                McodeElement(
                    element_type="PrimaryCancerCondition",
                    code="254837009",
                    system="http://snomed.info/sct",
                    display="Breast Cancer",
                    confidence_score=0.95,
                    evidence_text="Patient has malignant neoplasm of breast"
                )
            ]
        mock_llm.map_to_mcode = mock_map_to_mcode
        mock_llm_service_class.return_value = mock_llm

        # Create input files
        patients_file = tmp_path / "patients.ndjson"
        trials_file = tmp_path / "trials.ndjson"
        output_file = tmp_path / "mcode_patients.ndjson"

        # Write patient data
        with open(patients_file, 'w') as f:
            json.dump(sample_patient_data, f)
            f.write('\n')

        # Write trial criteria data
        with open(trials_file, 'w') as f:
            json.dump(sample_trial_criteria, f)
            f.write('\n')

        # Test CLI execution
        import argparse
        args = argparse.Namespace(
            input_file=str(patients_file),
            output_file=str(output_file),
            trials=str(trials_file),
            ingest=True,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            verbose=False,
            log_level="INFO",
            config=None
        )

        patients_processor_main(args)

        # Verify output file was created
        assert output_file.exists()

        # Verify output contains expected mCODE data
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert content
            mcode_data = json.loads(content)
            assert "patient_id" in mcode_data
            assert "mcode_elements" in mcode_data
            assert len(mcode_data["mcode_elements"]) > 0

    @patch('src.services.summarizer.McodeSummarizer')
    @patch('src.storage.mcode_memory_storage.McodeMemoryStorage')
    def test_clinician_workflow_treatment_recommendations(self, mock_memory_storage_class, mock_summarizer_class,
                                                        sample_patient_data, tmp_path):
        """Test the treatment recommendations phase of clinician workflow."""
        # Setup mocks
        mock_memory = MagicMock()
        mock_memory.store_patient_summary.return_value = True
        mock_memory_storage_class.return_value = mock_memory

        mock_summarizer = MagicMock()
        mock_summarizer.create_patient_summary.return_value = "Patient Jane Smith is a 49-year-old female with active breast cancer. She has estrogen receptor positive disease. Based on her clinical profile, she may be eligible for clinical trials targeting hormone receptor positive breast cancer."
        mock_summarizer_class.return_value = mock_summarizer

        # Create input file with mCODE patient data
        input_file = tmp_path / "mcode_patients.ndjson"
        mcode_patient_data = {
            "patient_id": "12345",
            "mcode_elements": [
                {
                    "resource_type": "Patient",
                    "id": "12345",
                    "name": {"family": "Smith", "given": ["Jane"]}
                },
                {
                    "resource_type": "Condition",
                    "id": "condition-1",
                    "clinical_data": {
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": "254837009",
                                "display": "Malignant neoplasm of breast"
                            }]
                        }
                    }
                }
            ],
            "original_patient_data": sample_patient_data
        }
        with open(input_file, 'w') as f:
            json.dump(mcode_patient_data, f)
            f.write('\n')

        # Create output file
        output_file = tmp_path / "patient_summaries.ndjson"

        # Test CLI execution
        import argparse
        args = argparse.Namespace(
            input_file=str(input_file),
            output_file=str(output_file),
            ingest=True,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            dry_run=False,
            verbose=False,
            log_level="INFO",
            config=None
        )

        patients_summarizer_main(args)

        # Verify output file was created
        assert output_file.exists()

        # Verify output contains expected summary data
        with open(output_file, 'r') as f:
            content = f.read().strip()
            assert content
            summary_data = json.loads(content)
            assert summary_data["patient_id"] == "12345"
            assert "summary" in summary_data
            assert "breast" in summary_data["summary"].lower()  # Basic validation that summary contains expected content

    @patch('src.cli.patients_fetcher.PatientsFetcherWorkflow')
    @patch('src.workflows.patients_summarizer_workflow.McodeSummarizer')
    @patch('src.storage.mcode_memory_storage.McodeMemoryStorage')
    def test_complete_clinician_workflow_integration(self, mock_memory_storage_class, mock_summarizer_class,
                                                   mock_fetcher_workflow_class,
                                                   sample_patient_data, sample_trial_criteria, tmp_path):
        """Test the complete end-to-end clinician workflow integration."""
        from src.shared.models import McodeElement

        # Setup mocks
        mock_memory = MagicMock()
        mock_memory.store_processed_patient.return_value = True
        mock_memory.store_patient_summary.return_value = True
        mock_memory_storage_class.return_value = mock_memory

        # Setup fetcher workflow mock
        mock_fetcher_workflow = MagicMock()
        mock_fetcher_workflow.execute.return_value = WorkflowResult(
            success=True,
            data=[sample_patient_data],
            metadata={"total_fetched": 1, "archive_path": "breast_cancer_10_years"}
        )
        mock_fetcher_workflow_class.return_value = mock_fetcher_workflow

        # Note: LLM service is mocked at the McodeSummarizer level

        # Setup summarizer mock
        mock_summarizer = MagicMock()
        mock_summarizer.create_patient_summary.return_value = "Patient Jane Smith is a 49-year-old female with active breast cancer. She has estrogen receptor positive disease. Based on her clinical profile, she may be eligible for clinical trials targeting hormone receptor positive breast cancer."
        mock_summarizer_class.return_value = mock_summarizer

        # Create temporary files
        patients_file = tmp_path / "patients.ndjson"
        trials_file = tmp_path / "trials.ndjson"
        mcode_file = tmp_path / "mcode_patients.ndjson"
        summary_file = tmp_path / "patient_summaries.ndjson"

        # Step 1: Patient Assessment
        import argparse
        fetch_args = argparse.Namespace(
            archive="breast_cancer_10_years", patient_id=None, limit=10,
            list_archives=False, output_file=str(patients_file),
            verbose=False, log_level="INFO", config=None
        )

        import io
        import sys
        from contextlib import redirect_stdout
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            patients_fetcher_main(fetch_args)

        # Verify fetch step completed successfully
        output = stdout_capture.getvalue()
        assert "✅ Patients fetch completed successfully!" in output

        # Step 2: Trial Matching (create mock patient data file since workflow is mocked)
        with open(patients_file, 'w') as f:
            json.dump(sample_patient_data, f)
            f.write('\n')

        with open(trials_file, 'w') as f:
            json.dump(sample_trial_criteria, f)
            f.write('\n')

        process_args = argparse.Namespace(
            input_file=str(patients_file), output_file=str(mcode_file), trials=str(trials_file),
            ingest=True, memory_source="test", model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise", workers=1,
            verbose=False, log_level="INFO", config=None
        )

        patients_processor_main(process_args)

        # Step 3: Treatment Recommendations (create mock mCODE data file)
        mcode_patient_data = {
            "patient_id": "12345",
            "mcode_elements": [
                {
                    "resource_type": "Patient",
                    "id": "12345",
                    "name": {"family": "Smith", "given": ["Jane"]}
                },
                {
                    "resource_type": "Condition",
                    "id": "condition-1",
                    "clinical_data": {
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": "254837009",
                                "display": "Malignant neoplasm of breast"
                            }]
                        }
                    }
                }
            ],
            "original_patient_data": sample_patient_data
        }
        with open(mcode_file, 'w') as f:
            json.dump(mcode_patient_data, f)
            f.write('\n')

        summary_args = argparse.Namespace(
            input_file=str(mcode_file), output_file=str(summary_file), ingest=True,
            memory_source="test", model="deepseek-coder", prompt="direct_mcode_evidence_based_concise",
            workers=1, dry_run=False, verbose=False, log_level="INFO", config=None
        )

        patients_summarizer_main(summary_args)

        # Verify workflows were called correctly
        mock_fetcher_workflow_class.assert_called_once()
        mock_summarizer_class.assert_called_once()

    def test_clinician_workflow_error_handling(self, tmp_path):
        """Test error handling in clinician workflow."""
        # Test with non-existent input file for processor
        nonexistent_file = tmp_path / "nonexistent.ndjson"

        import argparse
        args = argparse.Namespace(
            input_file=str(nonexistent_file), output_file=None, trials=None,
            ingest=False, memory_source="test", model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise", workers=1,
            verbose=False, log_level="INFO", config=None
        )

        with pytest.raises(SystemExit):
            patients_processor_main(args)

    @patch('src.cli.patients_fetcher.PatientsFetcherWorkflow')
    def test_clinician_workflow_invalid_archive(self, mock_workflow_class):
        """Test handling of invalid patient archive."""
        # Setup mock workflow to return failure
        mock_workflow = MagicMock()
        mock_workflow.execute.return_value = WorkflowResult(
            success=False,
            error_message="Archive 'invalid_archive' not found",
            metadata={"fetch_type": "archive"}
        )
        mock_workflow_class.return_value = mock_workflow

        import argparse
        args = argparse.Namespace(
            archive="invalid_archive", patient_id=None, limit=10,
            list_archives=False, output_file=None,
            verbose=False, log_level="INFO", config=None
        )

        import io
        import sys
        from contextlib import redirect_stdout
        stdout_capture = io.StringIO()
        with redirect_stdout(stdout_capture):
            with pytest.raises(SystemExit) as exc_info:
                patients_fetcher_main(args)

        # Verify exit code is 1 (error)
        assert exc_info.value.code == 1

        # Verify error was handled
        output = stdout_capture.getvalue()
        assert "❌ Patients fetch failed" in output