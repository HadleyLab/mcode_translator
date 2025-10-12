"""
Integration tests for storage interactions with CORE Memory and comprehensive error handling.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.storage.mcode_memory_storage import McodeMemoryStorage


@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests for storage interactions."""

    @pytest.fixture
    def mock_onco_core_memory(self):
        """Mock CORE Memory client."""
        mock = MagicMock()
        mock.create_memory.return_value = {"id": "test_id"}
        mock.fetch_memory.return_value = {"data": "test_data"}
        mock.search_memory.return_value = [{"id": "test_id", "data": "test_data"}]
        return mock

    @pytest.fixture
    def sample_trial_data(self):
        """Sample trial data for testing."""
        return {
            "nct_id": "NCT123456",
            "title": "Test Trial",
            "mcode_elements": [
                {
                    "code": "C4872",
                    "display": "Breast Cancer",
                    "system": "http://snomed.info/sct",
                }
            ],
            "metadata": {
                "processing_timestamp": "2024-01-01T00:00:00Z",
                "confidence_score": 0.95,
            },
        }

    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing."""
        return {
            "original_patient_data": {
                "entry": [
                    {"resource": {"resourceType": "Patient", "id": "patient_123"}}
                ]
            },
            "mcode_mappings": [
                {
                    "code": "C4872",
                    "display": "Breast Cancer",
                    "system": "http://snomed.info/sct",
                }
            ],
            "demographics": {"age": 45, "gender": "female"},
            "metadata": {"processing_timestamp": "2024-01-01T00:00:00Z"},
        }

    def test_store_trial_summary(
        self, mock_onco_core_memory, sample_trial_data
    ):
        """Test storing trial summary."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            success = storage.store_trial_summary("NCT123456", str(sample_trial_data))
            assert success
            mock_onco_core_memory.ingest.assert_called_once()

    def test_store_patient_summary(
        self, mock_onco_core_memory, sample_patient_data
    ):
        """Test storing patient summary."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            success = storage.store_patient_summary(
                "patient_123", str(sample_patient_data)
            )
            assert success
            mock_onco_core_memory.ingest.assert_called_once()

    def test_search_trials(self, mock_onco_core_memory):
        """Test searching for similar trials."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            mock_onco_core_memory.search.return_value = [
                {"id": "test_id", "data": "test_data"}
            ]
            results = storage.search_trials("breast cancer")
            assert results is not None
            assert len(results) == 1
            assert results[0]["id"] == "test_id"
            mock_onco_core_memory.search.assert_called_once()

    def test_store_trial_summary_with_empty_data(self, mock_onco_core_memory):
        """Test storing trial with empty data."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            success = storage.store_trial_summary("NCT123456", "{}")
            assert success
            mock_onco_core_memory.ingest.assert_called_once()

    def test_store_patient_summary_with_minimal_data(
        self, mock_onco_core_memory
    ):
        """Test storing patient with minimal required data."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            minimal_data = {
                "original_patient_data": {"entry": []},
                "mcode_mappings": [],
                "demographics": {},
                "metadata": {},
            }
            success = storage.store_patient_summary("patient_123", str(minimal_data))
            assert success
            mock_onco_core_memory.ingest.assert_called_once()

    def test_search_trials_empty_results(self, mock_onco_core_memory):
        """Test searching with no results."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            mock_onco_core_memory.search.return_value = []
            results = storage.search_trials("nonexistent condition")
            assert results == []
            mock_onco_core_memory.search.assert_called_once()

    def test_storage_error_handling_connection_failure(self, mock_onco_core_memory):
        """Test handling of storage connection failures."""
        mock_onco_core_memory.ingest.side_effect = Exception("Connection failed")

        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            success = storage.store_trial_summary("NCT123456", {"data": "test"})
            assert not success  # Should handle error gracefully

    def test_storage_error_handling_search_failure(self, mock_onco_core_memory):
        """Test handling of search failures."""
        from src.services.heysol_client import HeySolError

        mock_onco_core_memory.search.side_effect = HeySolError("Search failed")

        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()
            results = storage.search_trials("breast cancer")
            assert results == {
                "episodes": [],
                "facts": [],
            }  # Should return empty results on error

    def test_batch_storage_operations(
        self, mock_onco_core_memory, sample_trial_data, sample_patient_data
    ):
        """Test batch storage operations."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()

            # Store multiple trials
            trials = [sample_trial_data, {**sample_trial_data, "nct_id": "NCT789012"}]
            for trial in trials:
                success = storage.store_trial_summary(trial["nct_id"], trial)
                assert success

            # Store multiple patients
            patients = [
                sample_patient_data,
                {
                    **sample_patient_data,
                    "original_patient_data": {
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Patient",
                                    "id": "patient_456",
                                }
                            }
                        ]
                    },
                },
            ]
            for patient in patients:
                success = storage.store_patient_summary(
                    patient["original_patient_data"]["entry"][0]["resource"]["id"],
                    patient,
                )
                assert success

            # Verify all operations were called
            assert mock_onco_core_memory.ingest.call_count == 4

    def test_storage_data_validation(self, mock_onco_core_memory):
        """Test data validation before storage."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()

            # Test with None data
            success = storage.store_trial_summary("NCT123456", None)
            assert not success

            # Test with invalid patient data structure
            success = storage.store_patient_summary(
                "patient_123", {"invalid": "structure"}
            )
            assert not success

    def test_concurrent_storage_access(
        self, mock_onco_core_memory, sample_trial_data
    ):
        """Test concurrent storage access patterns."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()

            # Simulate concurrent operations
            operations = []
            for i in range(5):
                trial_data = {**sample_trial_data, "nct_id": f"NCT{i}"}
                success = storage.store_trial_summary(
                    trial_data["nct_id"], trial_data
                )
                operations.append(success)

            # All operations should succeed
            assert all(operations)
            assert mock_onco_core_memory.ingest.call_count == 5

    def test_storage_memory_limits(self, mock_onco_core_memory):
        """Test handling of large data sets."""
        with patch(
            "src.storage.mcode_memory_storage.OncoCoreClient",
            return_value=mock_onco_core_memory,
        ):
            storage = McodeMemoryStorage()

            # Create large trial data
            large_trial_data = {
                "nct_id": "NCT123456",
                "large_field": "x" * 10000,  # 10KB string
                "metadata": {"processing_timestamp": "2024-01-01T00:00:00Z"},
            }

            success = storage.store_trial_summary("NCT123456", large_trial_data)
            assert success
            mock_onco_core_memory.ingest.assert_called_once()
