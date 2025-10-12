#!/usr/bin/env python3
"""
Unit tests for patients_summarizer CLI module.

Tests the command-line interface for generating natural language summaries from mCODE patient data,
including file handling, data parsing, workflow execution, summary extraction, and CORE Memory integration.
"""

from unittest.mock import MagicMock

from src.workflows.patients_summarizer import PatientsSummarizerWorkflow


class TestPatientsSummarizerWorkflow:
    """Test the PatientsSummarizerWorkflow class."""

    def test_execute_successful_summarization(self):
        """Test successful patient summarization workflow."""
        # Mock config
        mock_config = MagicMock()

        # Create workflow
        workflow = PatientsSummarizerWorkflow(config=mock_config)

        # Mock patient data
        patients_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": [{"element_type": "CancerCondition"}]},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}],
            }
        ]

        # Execute workflow
        result = workflow.execute(patients_data=patients_data, store_in_memory=False, workers=0)

        # Verify result
        assert result.success is True
        assert len(result.data) == 1
        assert result.metadata["total_patients"] == 1
        assert result.metadata["successful"] == 1
        assert result.metadata["failed"] == 0

    def test_execute_with_memory_storage(self):
        """Test execution with memory storage enabled."""
        # Mock config and memory storage
        mock_config = MagicMock()
        mock_memory = MagicMock()

        # Create workflow with memory storage
        workflow = PatientsSummarizerWorkflow(config=mock_config, memory_storage=mock_memory)

        # Mock patient data
        patients_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}],
            }
        ]

        # Execute workflow
        result = workflow.execute(patients_data=patients_data, store_in_memory=True, workers=0)

        # Verify result
        assert result.success is True
        assert len(result.data) == 1
        assert result.metadata["stored_in_memory"] is True

    def test_execute_missing_patients_data(self):
        """Test execution with missing patients data."""
        # Mock config
        mock_config = MagicMock()

        # Create workflow
        workflow = PatientsSummarizerWorkflow(config=mock_config)

        # Execute workflow without patients_data
        result = workflow.execute()

        # Verify result
        assert result.success is False
        assert result.error_message == "No patient data provided for summarization."

    def test_execute_dry_run_mode(self):
        """Test dry run mode prevents storage in CORE Memory."""
        # Mock config and memory storage
        mock_config = MagicMock()
        mock_memory = MagicMock()

        # Create workflow with memory storage
        workflow = PatientsSummarizerWorkflow(config=mock_config, memory_storage=mock_memory)

        # Mock patient data
        patients_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}],
            }
        ]

        # Execute workflow with dry run (store_in_memory=False)
        result = workflow.execute(patients_data=patients_data, store_in_memory=False, workers=0)

        # Verify result
        assert result.success is True
        assert len(result.data) == 1
        assert result.metadata["stored_in_memory"] is False

    def test_execute_with_concurrency(self):
        """Test execution with concurrency settings."""
        # Mock config
        mock_config = MagicMock()

        # Create workflow
        workflow = PatientsSummarizerWorkflow(config=mock_config)

        # Mock patient data
        patients_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}],
            }
        ]

        # Execute workflow with workers
        result = workflow.execute(patients_data=patients_data, store_in_memory=False, workers=4)

        # Verify result
        assert result.success is True
        assert len(result.data) == 1

    # Additional workflow tests can be added here as needed

    # Additional workflow tests can be added here as needed

    # Additional workflow tests can be added here as needed

    # Additional workflow tests can be added here as needed
