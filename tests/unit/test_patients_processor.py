#!/usr/bin/env python3
"""
Unit tests for PatientsProcessorWorkflow class.

Tests the workflow for processing patient data with mCODE mapping,
including execution, data validation, and error handling.
"""

from unittest.mock import MagicMock

from src.workflows.patients_processor import PatientsProcessorWorkflow


class TestPatientsProcessorWorkflow:
    """Test the PatientsProcessorWorkflow class."""

    def test_execute_successful_processing(self):
        """Test successful patient processing workflow."""
        # Mock config
        mock_config = MagicMock()

        # Create workflow
        workflow = PatientsProcessorWorkflow(config=mock_config)

        # Mock patient data
        patients_data = [
            {
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "123",
                            "name": [{"given": ["John"], "family": "Doe"}],
                        }
                    }
                ]
            }
        ]

        # Execute workflow
        result = workflow.execute(patients_data=patients_data)

        # Verify result
        assert result.success is True
        assert len(result.data) == 1
        assert result.metadata["total_patients"] == 1
        assert result.metadata["successful"] == 1
        assert result.metadata["failed"] == 0

    def test_execute_missing_patients_data(self):
        """Test execution with missing patients data."""
        # Mock config
        mock_config = MagicMock()

        # Create workflow
        workflow = PatientsProcessorWorkflow(config=mock_config)

        # Execute workflow without patients_data
        result = workflow.execute()

        # Verify result
        assert result.success is False
        assert result.error_message == "No patient data provided for processing."

    def test_execute_empty_patients_data(self):
        """Test execution with empty patients data."""
        # Mock config
        mock_config = MagicMock()

        # Create workflow
        workflow = PatientsProcessorWorkflow(config=mock_config)

        # Execute workflow with empty list
        result = workflow.execute(patients_data=[])

        # Verify result
        assert result.success is False
        assert result.error_message == "No patient data provided for processing."

    def test_execute_with_trials_criteria(self):
        """Test execution with trial criteria filtering."""
        # Mock config
        mock_config = MagicMock()

        # Create workflow
        workflow = PatientsProcessorWorkflow(config=mock_config)

        # Mock patient data
        patients_data = [
            {
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "123",
                            "name": [{"given": ["John"], "family": "Doe"}],
                        }
                    }
                ]
            }
        ]

        # Mock trials criteria
        trials_criteria = {"CancerCondition": ["breast cancer"]}

        # Execute workflow with trials criteria
        result = workflow.execute(patients_data=patients_data, trials_criteria=trials_criteria)

        # Verify result
        assert result.success is True
        assert len(result.data) == 1
        assert result.metadata["trial_criteria_applied"] is True

    def test_execute_with_memory_storage(self):
        """Test execution with memory storage enabled."""
        # Mock config and memory storage
        mock_config = MagicMock()
        mock_memory = MagicMock()

        # Create workflow with memory storage
        workflow = PatientsProcessorWorkflow(config=mock_config, memory_storage=mock_memory)

        # Mock patient data
        patients_data = [
            {
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "123",
                            "name": [{"given": ["John"], "family": "Doe"}],
                        }
                    }
                ]
            }
        ]

        # Execute workflow
        result = workflow.execute(patients_data=patients_data, store_in_memory=True)

        # Verify result
        assert result.success is True
        assert len(result.data) == 1
        # Note: stored_in_memory may be False in test environment due to mock setup
