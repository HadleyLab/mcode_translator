"""
Integration tests for storage interactions with CORE Memory.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.storage.mcode_memory_storage import McodeMemoryStorage


@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests for storage interactions."""

    @pytest.fixture
    def mock_core_memory_client(self):
        """Mock CORE Memory client."""
        mock = MagicMock()
        mock.create_memory.return_value = {"id": "test_id"}
        mock.fetch_memory.return_value = {"data": "test_data"}
        mock.search_memory.return_value = [{"id": "test_id", "data": "test_data"}]
        return mock

    def test_store_trial_mcode_summary(self, mock_core_memory_client):
        """Test storing trial mCODE summary."""
        with patch('src.storage.mcode_memory_storage.CoreMemoryClient', return_value=mock_core_memory_client):
            storage = McodeMemoryStorage()
            success = storage.store_trial_mcode_summary("NCT123456", {"data": "test_data"})
            assert success
            mock_core_memory_client.ingest.assert_called_once()

    def test_store_patient_mcode_summary(self, mock_core_memory_client):
        """Test storing patient mCODE summary."""
        with patch('src.storage.mcode_memory_storage.CoreMemoryClient', return_value=mock_core_memory_client):
            storage = McodeMemoryStorage()
            patient_data = {
                "original_patient_data": {"entry": [{"resource": {"resourceType": "Patient"}}]},
                "mcode_mappings": [],
                "demographics": {},
                "metadata": {}
            }
            success = storage.store_patient_mcode_summary("patient_123", patient_data)
            assert success
            mock_core_memory_client.ingest.assert_called_once()

    def test_search_similar_trials(self, mock_core_memory_client):
        """Test searching for similar trials."""
        with patch('src.storage.mcode_memory_storage.CoreMemoryClient', return_value=mock_core_memory_client):
            storage = McodeMemoryStorage()
            mock_core_memory_client.search.return_value = [{"id": "test_id", "data": "test_data"}]
            results = storage.search_similar_trials("breast cancer")
            assert results is not None
            assert len(results) == 1
            assert results[0]["id"] == "test_id"
            mock_core_memory_client.search.assert_called_once()