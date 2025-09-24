"""
Integration tests for cross-module interactions.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.core.data_flow_coordinator import DataFlowCoordinator
from src.core.dependency_container import DependencyContainer
from src.workflows.patients_fetcher_workflow import PatientsFetcherWorkflow


@pytest.mark.integration
class TestCrossModuleIntegration:
    """Integration tests for cross-module interactions."""

    @pytest.fixture
    def sample_trial_data(self):
        """Load sample trial data for testing."""
        return {"protocolSection": {"identificationModule": {"nctId": "NCT123456"}}}

    @pytest.fixture
    def sample_patient_data(self):
        """Load sample patient data for testing."""
        return {
            "entry": [{"resource": {"resourceType": "Patient", "id": "patient_123"}}]
        }

    @pytest.fixture
    def container(self):
        """Create dependency container."""
        return DependencyContainer()

    @patch("src.core.data_fetcher.get_full_studies_batch")
    def test_data_flow_coordinator_with_workflows(
        self,
        mock_get_full_studies_batch,
        container,
        sample_trial_data,
        sample_patient_data,
    ):
        """Test DataFlowCoordinator with mocked workflows."""
        mock_get_full_studies_batch.return_value = {"NCT123456": sample_trial_data}
        with patch.object(
            PatientsFetcherWorkflow,
            "execute",
            return_value=MagicMock(success=True, data=[sample_patient_data]),
        ):

            coordinator = DataFlowCoordinator()
            coordinator.container = container

            # Test trial processing flow
            trial_result = coordinator.process_clinical_trials_complete_flow(
                trial_ids=["NCT123456"]
            )
            assert trial_result is not None
            mock_get_full_studies_batch.assert_called_once()

            # Test patient processing flow
            # patient_result = coordinator.process_patients_complete_flow(archive_path="test.zip", limit=1)
            # assert patient_result is not None
            # mock_patient_fetch.assert_called_once()

    def test_dependency_container_with_real_components(self, container):
        """Test DependencyContainer with real components."""
        # Test creating a real component
        storage = container.create_memory_storage()
        assert storage is not None

        # Test that the same instance is returned for singletons
        storage2 = container.create_memory_storage()
        assert storage is storage2

        # Test creating a new instance
        pipeline1 = container.create_clinical_trial_pipeline()
        pipeline2 = container.create_clinical_trial_pipeline()
        assert pipeline1 is not pipeline2
