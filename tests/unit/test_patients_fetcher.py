#!/usr/bin/env python3
"""
Unit tests for PatientsFetcherWorkflow class.

Tests the workflow for fetching synthetic patient data,
including execution, data validation, and error handling.
"""

from unittest.mock import MagicMock, patch
from src.workflows.patients_fetcher import PatientsFetcherWorkflow


class TestPatientsFetcherWorkflow:
    """Test the PatientsFetcherWorkflow class."""

    @patch("src.workflows.patients_fetcher.create_patient_generator")
    def test_execute_single_patient_success(self, mock_create_generator):
        """Test successful single patient fetching."""
        # Mock patient generator
        mock_generator = MagicMock()
        mock_generator.get_patient_by_id.return_value = {"id": "P001", "name": "Test Patient"}
        mock_create_generator.return_value = mock_generator

        # Create workflow
        workflow = PatientsFetcherWorkflow()

        # Execute workflow
        result = workflow.execute(archive_path="test_archive", patient_id="P001")

        # Verify result
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["id"] == "P001"
        assert result.metadata["fetch_type"] == "single_patient"
        assert result.metadata["archive_path"] == "test_archive"

    @patch("src.workflows.patients_fetcher.create_patient_generator")
    def test_execute_single_patient_not_found(self, mock_create_generator):
        """Test single patient fetching when patient not found."""
        # Mock patient generator
        mock_generator = MagicMock()
        mock_generator.get_patient_by_id.return_value = None
        mock_create_generator.return_value = mock_generator

        # Create workflow
        workflow = PatientsFetcherWorkflow()

        # Execute workflow
        result = workflow.execute(archive_path="test_archive", patient_id="P999")

        # Verify result
        assert result.success is False
        assert result.error_message == "Patient P999 not found in archive"
        assert len(result.data) == 0

    @patch("src.workflows.patients_fetcher.create_patient_generator")
    def test_execute_multiple_patients_success(self, mock_create_generator):
        """Test successful multiple patients fetching."""
        # Mock patient generator
        mock_generator = MagicMock()
        mock_patients = [
            {"id": "P001", "name": "Patient 1"},
            {"id": "P002", "name": "Patient 2"},
            {"id": "P003", "name": "Patient 3"},
        ]
        mock_generator.__iter__.return_value = iter(mock_patients)
        mock_generator.__len__.return_value = 3
        mock_create_generator.return_value = mock_generator

        # Create workflow
        workflow = PatientsFetcherWorkflow()

        # Execute workflow
        result = workflow.execute(archive_path="test_archive", limit=2)

        # Verify result
        assert result.success is True
        assert len(result.data) == 2  # Limited to 2
        assert result.metadata["fetch_type"] == "multiple_patients"
        assert result.metadata["archive_path"] == "test_archive"

    @patch("src.workflows.patients_fetcher.create_patient_generator")
    def test_execute_multiple_patients_no_patients(self, mock_create_generator):
        """Test multiple patients fetching when no patients found."""
        # Mock patient generator
        mock_generator = MagicMock()
        mock_generator.__iter__.return_value = iter([])
        mock_generator.__len__.return_value = 0
        mock_create_generator.return_value = mock_generator

        # Create workflow
        workflow = PatientsFetcherWorkflow()

        # Execute workflow
        result = workflow.execute(archive_path="empty_archive", limit=10)

        # Verify result
        assert result.success is False
        assert result.error_message == "No patients found in archive: empty_archive"
        assert len(result.data) == 0

    def test_execute_missing_archive_path(self):
        """Test execution with missing archive path."""
        workflow = PatientsFetcherWorkflow()

        result = workflow.execute()

        assert result.success is False
        assert result.error_message == "Archive path is required for patient fetching."

    def test_list_available_archives(self):
        """Test listing available archives."""
        workflow = PatientsFetcherWorkflow()

        archives = workflow.list_available_archives()

        # Verify returns expected archive types
        expected_archives = [
            "breast_cancer_10_years",
            "breast_cancer_lifetime",
            "mixed_cancer_10_years",
            "mixed_cancer_lifetime",
        ]
        assert archives == expected_archives

    @patch("src.workflows.patients_fetcher.create_patient_generator")
    def test_get_archive_info_success(self, mock_create_generator):
        """Test getting archive information successfully."""
        # Mock patient generator
        mock_generator = MagicMock()
        mock_generator.__len__.return_value = 150
        mock_create_generator.return_value = mock_generator

        workflow = PatientsFetcherWorkflow()

        info = workflow.get_archive_info("test_archive")

        assert info["archive_path"] == "test_archive"
        assert info["total_patients"] == 150
        assert "patient_generator_type" in info

    @patch("src.workflows.patients_fetcher.create_patient_generator")
    def test_get_archive_info_error(self, mock_create_generator):
        """Test getting archive information with error."""
        # Mock patient generator to raise exception
        mock_create_generator.side_effect = Exception("Archive not found")

        workflow = PatientsFetcherWorkflow()

        info = workflow.get_archive_info("invalid_archive")

        assert info["archive_path"] == "invalid_archive"
        assert "error" in info
        assert info["error"] == "Archive not found"
