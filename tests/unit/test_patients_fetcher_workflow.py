"""
Unit tests for PatientsFetcherWorkflow.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.workflows.patients_fetcher_workflow import PatientsFetcherWorkflow
from src.utils.config import Config


class TestPatientsFetcherWorkflow:
    """Test cases for PatientsFetcherWorkflow."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config()

    @pytest.fixture
    def workflow(self, config):
        """Create a test workflow instance."""
        return PatientsFetcherWorkflow(config)

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
                        "name": [{"given": ["John"], "family": "Doe"}]
                    }
                }
            ]
        }

    def test_workflow_initialization(self, workflow):
        """Test workflow initializes correctly."""
        assert workflow is not None
        assert hasattr(workflow, 'execute')
        assert hasattr(workflow, 'list_available_archives')

    def test_execute_missing_archive_path(self, workflow):
        """Test execute fails with missing archive path."""
        result = workflow.execute()

        assert result.success is False
        assert "Archive path is required" in result.error_message

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_execute_single_patient_success(self, mock_create_generator, workflow, mock_patient_data):
        """Test successful single patient fetch."""
        # Mock generator
        mock_generator = Mock()
        mock_generator.get_patient_by_id.return_value = mock_patient_data
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="breast_cancer_10_years",
            patient_id="patient_123"
        )

        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0] == mock_patient_data
        assert result.metadata["fetch_type"] == "single_patient"
        assert result.metadata["total_fetched"] == 1

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_execute_single_patient_not_found(self, mock_create_generator, workflow):
        """Test single patient fetch when patient not found."""
        # Mock generator
        mock_generator = Mock()
        mock_generator.get_patient_by_id.return_value = None
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="breast_cancer_10_years",
            patient_id="nonexistent_patient"
        )

        assert result.success is False
        assert "not found in archive" in result.error_message

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_execute_multiple_patients_success(self, mock_create_generator, workflow, mock_patient_data):
        """Test successful multiple patients fetch."""
        # Mock generator
        mock_generator = Mock()
        # Configure the mock to be iterable
        mock_generator.__iter__ = Mock(return_value=iter([mock_patient_data, mock_patient_data]))
        mock_generator.__len__ = Mock(return_value=100)
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="breast_cancer_10_years",
            limit=2
        )

        assert result.success is True
        assert len(result.data) == 2
        assert result.metadata["fetch_type"] == "multiple_patients"
        assert result.metadata["total_fetched"] == 2
        # Note: requested_limit is not stored in metadata, only in the internal method

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_execute_multiple_patients_empty_archive(self, mock_create_generator, workflow):
        """Test multiple patients fetch from empty archive."""
        # Mock generator
        mock_generator = Mock()
        # Configure the mock to be iterable but empty
        mock_generator.__iter__ = Mock(return_value=iter([]))
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="empty_archive",
            limit=5
        )

        assert result.success is False
        assert "No patients found in archive" in result.error_message

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_execute_with_output_file(self, mock_file, mock_path_exists, mock_create_generator, workflow, mock_patient_data):
        """Test execute with output file saves results."""
        # Mock generator
        mock_generator = Mock()
        mock_generator.get_patient_by_id.return_value = mock_patient_data
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="breast_cancer_10_years",
            patient_id="patient_123",
            output_path="output.ndjson"
        )

        assert result.success is True
        assert str(result.metadata["output_path"]) == "output.ndjson"
        # Verify file was opened for writing
        mock_file.assert_called_once_with(Path("output.ndjson"), "w", encoding="utf-8")

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    @patch('sys.stdout')
    def test_execute_stdout_output(self, mock_stdout, mock_create_generator, workflow, mock_patient_data):
        """Test execute outputs to stdout when no output file."""
        # Mock generator
        mock_generator = Mock()
        mock_generator.get_patient_by_id.return_value = mock_patient_data
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="breast_cancer_10_years",
            patient_id="patient_123"
        )

        assert result.success is True
        # Verify stdout was used
        assert mock_stdout.write.called

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_fetch_single_patient_generator_error(self, mock_create_generator, workflow):
        """Test single patient fetch handles generator errors."""
        mock_create_generator.side_effect = Exception("Generator initialization failed")

        result = workflow.execute(
            archive_path="invalid_archive",
            patient_id="patient_123"
        )

        assert result.success is False
        assert "Failed to fetch patient" in result.error_message

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_fetch_multiple_patients_generator_error(self, mock_create_generator, workflow):
        """Test multiple patients fetch handles generator errors."""
        mock_create_generator.side_effect = Exception("Generator initialization failed")

        result = workflow.execute(
            archive_path="invalid_archive",
            limit=5
        )

        assert result.success is False
        assert "Failed to fetch patients" in result.error_message

    @patch('pathlib.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_results_creates_directory(self, mock_file, mock_mkdir, workflow, mock_patient_data):
        """Test save results creates output directory."""
        workflow._save_results([mock_patient_data], "new_dir/output.ndjson")

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_results_ndjson_format(self, mock_file, workflow, mock_patient_data):
        """Test save results writes NDJSON format."""
        mock_file_handle = Mock()
        mock_file.return_value.__enter__.return_value = mock_file_handle

        workflow._save_results([mock_patient_data, mock_patient_data], "output.ndjson")

        # Verify write was called (json.dump writes the full JSON string + newline)
        assert mock_file_handle.write.call_count >= 2  # At least 2 writes per patient
        # Verify newline was written after each JSON object
        mock_file_handle.write.assert_any_call('\n')

    @patch('sys.stdout')
    def test_output_to_stdout_ndjson_format(self, mock_stdout, workflow, mock_patient_data):
        """Test stdout output writes NDJSON format."""
        workflow._output_to_stdout([mock_patient_data, mock_patient_data])

        # Verify write was called (json.dump writes full JSON + newlines + flush)
        assert mock_stdout.write.call_count >= 3  # JSON + newlines + flush

    def test_list_available_archives(self, workflow):
        """Test listing available archives."""
        archives = workflow.list_available_archives()

        assert isinstance(archives, list)
        assert len(archives) > 0
        assert "breast_cancer_10_years" in archives
        assert "mixed_cancer_lifetime" in archives

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_get_archive_info_success(self, mock_create_generator, workflow):
        """Test getting archive info successfully."""
        # Mock generator
        mock_generator = Mock()
        # Configure the mock to have __len__
        mock_generator.__len__ = Mock(return_value=150)
        mock_create_generator.return_value = mock_generator

        info = workflow.get_archive_info("breast_cancer_10_years")

        assert info["archive_path"] == "breast_cancer_10_years"
        assert info["total_patients"] == 150
        assert "patient_generator_type" in info

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_get_archive_info_error(self, mock_create_generator, workflow):
        """Test getting archive info handles errors."""
        mock_create_generator.side_effect = Exception("Archive not accessible")

        info = workflow.get_archive_info("invalid_archive")

        assert info["archive_path"] == "invalid_archive"
        assert "error" in info

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_execute_with_limit_zero(self, mock_create_generator, workflow):
        """Test execute with limit of zero returns no patients found."""
        # Mock generator with some patients but limit=0 should return empty
        mock_generator = Mock()
        mock_generator.__iter__ = Mock(return_value=iter([{"id": "p1"}, {"id": "p2"}]))
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="breast_cancer_10_years",
            limit=0
        )

        # Should fail with "No patients found" since limit=0 fetches no patients
        assert result.success is False
        assert "No patients found in archive" in result.error_message
        assert len(result.data) == 0
        assert result.metadata["total_fetched"] == 0

    @patch('src.workflows.patients_fetcher_workflow.create_patient_generator')
    def test_execute_limit_exceeds_available(self, mock_create_generator, workflow, mock_patient_data):
        """Test execute when requested limit exceeds available patients."""
        # Mock generator with only 3 patients
        mock_generator = Mock()
        mock_generator.__iter__ = Mock(return_value=iter([mock_patient_data] * 3))
        mock_generator.__len__ = Mock(return_value=3)
        mock_create_generator.return_value = mock_generator

        result = workflow.execute(
            archive_path="small_archive",
            limit=10  # Request more than available
        )

        assert result.success is True
        assert len(result.data) == 3  # Should get all available
        # Note: requested_limit is stored in internal method metadata, not main metadata

    def test_execute_unexpected_error(self, workflow):
        """Test execute handles unexpected errors."""
        # Force an unexpected error by mocking the internal fetch method to raise
        with patch.object(workflow, '_fetch_multiple_patients') as mock_fetch:
            mock_fetch.side_effect = Exception("Unexpected error")

            result = workflow.execute(archive_path="test_archive")

            # Should handle the error gracefully
            assert result.success is False
            assert "Unexpected error" in result.error_message