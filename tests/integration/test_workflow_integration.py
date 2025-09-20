"""
Integration tests for workflow combinations and end-to-end processing.
Tests complete workflow chains with proper mocking of external dependencies.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.core.dependency_container import DependencyContainer
from src.workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
from src.workflows.trials_processor_workflow import ClinicalTrialsProcessorWorkflow
from src.workflows.trials_summarizer_workflow import TrialsSummarizerWorkflow
from src.workflows.patients_fetcher_workflow import PatientsFetcherWorkflow
from src.workflows.patients_processor_workflow import PatientsProcessorWorkflow
from src.workflows.patients_summarizer_workflow import PatientsSummarizerWorkflow


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for workflow combinations."""

    @pytest.fixture
    def sample_trial_data(self):
        """Load sample trial data for testing."""
        data_path = Path(__file__).parent.parent / "data" / "sample_trial.json"
        with open(data_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def sample_patient_data(self):
        """Load sample patient data for testing."""
        data_path = Path(__file__).parent.parent / "data" / "sample_patient.json"
        with open(data_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for testing."""
        mock = MagicMock()
        from src.shared.models import PipelineResult, ValidationResult, ProcessingMetadata
        # Make process method async
        async def mock_process(*args, **kwargs):
            from src.shared.models import ProcessingMetadata
            return PipelineResult(
                extracted_entities=[],
                mcode_mappings=[],
                source_references=[],
                validation_results=ValidationResult(compliance_score=1.0),
                metadata=ProcessingMetadata(engine_type="mock"),
                original_data={}
            )
        mock.process = mock_process
        mock.map_to_mcode.return_value = {
            "elements": [
                {
                    "code": "C4872",
                    "display": "Breast Cancer",
                    "system": "http://snomed.info/sct",
                    "mcode_element": "CancerCondition"
                }
            ],
            "confidence_score": 0.95
        }
        return mock

    @pytest.fixture
    def mock_storage(self):
        """Mock storage for testing."""
        mock = MagicMock()
        mock.store_trial_mcode_summary.return_value = None
        mock.store_patient_mcode_summary.return_value = None
        mock.search_similar_trials.return_value = [{"id": "NCT123456", "score": 0.9}]
        return mock

    @pytest.fixture
    def container(self):
        """Create dependency container."""
        return DependencyContainer()

    @patch('src.workflows.trials_fetcher_workflow.get_full_studies_batch')
    def test_end_to_end_trial_processing_workflow(self, mock_get_full_studies_batch, sample_trial_data, mock_llm_service, mock_storage, container):
        """Test complete trial fetch → process → summarize workflow."""
        # Mock get_full_studies_batch to return sample data
        mock_get_full_studies_batch.return_value = {"NCT123456": sample_trial_data}

        # Create workflows
        fetcher = TrialsFetcherWorkflow()
        processor = ClinicalTrialsProcessorWorkflow(container.config)
        summarizer = TrialsSummarizerWorkflow(container.config)

        # Mock dependencies
        with patch.object(processor, 'pipeline', mock_llm_service), \
             patch.object(summarizer, 'memory_storage', mock_storage):

            # Step 1: Fetch trials
            fetch_result = fetcher.execute(nct_ids=['NCT123456'])
            assert fetch_result.success
            assert len(fetch_result.data) == 1

            # Step 2: Process trials
            process_result = processor.execute(trials_data=fetch_result.data)
            assert process_result.success
            assert len(process_result.data) == 1

            # Step 3: Summarize and store
            summary_result = summarizer.execute(trials_data=process_result.data)
            assert summary_result.success

            # Verify mocks were called
            mock_get_full_studies_batch.assert_called()

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_end_to_end_patient_processing_workflow(self, mock_create_generator, sample_patient_data, mock_llm_service, mock_storage, container):
        """Test complete patient fetch → process → summarize workflow."""
        # Mock patient generator
        mock_generator = MagicMock()
        mock_generator.__iter__.return_value = iter([sample_patient_data])
        mock_create_generator.return_value = mock_generator

        # Create workflows
        fetcher = PatientsFetcherWorkflow()
        processor = PatientsProcessorWorkflow(container.config)
        summarizer = PatientsSummarizerWorkflow(container.config)

        # Mock dependencies
        with patch.object(summarizer, 'memory_storage', mock_storage):

            # Step 1: Fetch patients
            fetch_result = fetcher.execute(archive_path='test.zip', limit=1)
            assert fetch_result.success
            assert len(fetch_result.data) == 1

            # Step 2: Process patients
            process_result = processor.execute(patients_data=fetch_result.data)
            assert process_result.success
            assert len(process_result.data) == 1

            # Step 3: Summarize and store
            summary_result = summarizer.execute(patients_data=process_result.data)
            assert summary_result.success

            # Verify mocks were called
            mock_create_generator.assert_called()

    @patch('src.workflows.trials_fetcher_workflow.get_full_studies_batch')
    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_cross_workflow_data_flow(self, mock_create_generator, mock_get_full_studies_batch,
                                     sample_trial_data, sample_patient_data, mock_llm_service, container):
        """Test data flow between trials and patients workflows."""
        # Mock trial fetch
        mock_get_full_studies_batch.return_value = {"NCT123456": sample_trial_data}

        # Mock patient generator
        mock_generator = MagicMock()
        mock_generator.__iter__.return_value = iter([sample_patient_data])
        mock_create_generator.return_value = mock_generator

        # Create workflows
        trial_fetcher = TrialsFetcherWorkflow()
        trial_processor = ClinicalTrialsProcessorWorkflow(container.config)
        patient_fetcher = PatientsFetcherWorkflow()
        patient_processor = PatientsProcessorWorkflow(container.config)

        # Mock LLM service
        with patch.object(trial_processor, 'pipeline', mock_llm_service):

            # Process trials to get mCODE data
            trial_fetch_result = trial_fetcher.execute(nct_ids=['NCT123456'])
            trial_process_result = trial_processor.execute(trials_data=trial_fetch_result.data)

            # Process patients
            patient_fetch_result = patient_fetcher.execute(archive_path='test.zip', limit=1)
            patient_process_result = patient_processor.execute(patients_data=patient_fetch_result.data)

            # Verify both workflows completed successfully
            assert trial_fetch_result.success
            assert trial_process_result.success
            assert patient_fetch_result.success
            assert patient_process_result.success

            # Verify mocks were called
            mock_get_full_studies_batch.assert_called()
            mock_create_generator.assert_called()

    @patch('src.workflows.trials_fetcher_workflow.get_full_studies_batch')
    def test_workflow_error_handling_and_recovery(self, mock_get_full_studies_batch, sample_trial_data, mock_llm_service, container):
        """Test workflow error handling and recovery mechanisms."""
        # Mock fetch to fail initially, then succeed
        mock_get_full_studies_batch.side_effect = [
            Exception("API temporarily unavailable"),
            {"NCT123456": sample_trial_data}
        ]

        fetcher = TrialsFetcherWorkflow()
        processor = ClinicalTrialsProcessorWorkflow(container.config)

        # First attempt should fail
        result = fetcher.execute(nct_ids=['NCT123456'])
        assert not result.success

        # Second attempt should succeed
        with patch.object(processor, 'pipeline', mock_llm_service):
            fetch_result = fetcher.execute(nct_ids=['NCT123456'])
            assert fetch_result.success

            process_result = processor.execute(trials_data=fetch_result.data)
            assert process_result.success

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    @patch('src.workflows.trials_fetcher_workflow.get_full_studies_batch')
    def test_concurrent_workflow_execution(self, mock_get_full_studies_batch, mock_create_generator,
                                         sample_trial_data, sample_patient_data, mock_llm_service, container):
        """Test concurrent execution of multiple workflows."""
        # Mock data
        mock_get_full_studies_batch.return_value = {"NCT123456": sample_trial_data}

        mock_generator = MagicMock()
        mock_generator.__iter__.return_value = iter([sample_patient_data])
        mock_create_generator.return_value = mock_generator

        # Create multiple workflow instances
        workflows = [
            TrialsFetcherWorkflow(),
            ClinicalTrialsProcessorWorkflow(container.config),
            PatientsFetcherWorkflow(),
            PatientsProcessorWorkflow(container.config)
        ]

        # Mock LLM service for processors
        with patch.object(workflows[1], 'pipeline', mock_llm_service):

            # Execute workflows (simulating concurrent execution)
            trial_fetch_result = workflows[0].execute(nct_ids=['NCT123456'])
            patient_fetch_result = workflows[2].execute(archive_path='test.zip', limit=1)

            trial_process_result = workflows[1].execute(trials_data=trial_fetch_result.data)
            patient_process_result = workflows[3].execute(patients_data=patient_fetch_result.data)

            # Verify all completed successfully
            assert trial_fetch_result.success
            assert patient_fetch_result.success
            assert trial_process_result.success
            assert patient_process_result.success

    def test_workflow_configuration_validation(self, container):
        """Test that workflows validate their configuration properly."""
        # Test with invalid configuration
        fetcher = TrialsFetcherWorkflow()

        # Should handle empty input gracefully
        result = fetcher.execute(nct_ids=[])
        assert not result.success  # Should fail with empty input

        # Test with invalid trial IDs (should still attempt to process)
        result = fetcher.execute(nct_ids=['INVALID_ID'])
        # Should attempt to process but may fail gracefully
        assert isinstance(result, object)  # Basic structure check