#!/usr/bin/env python3
"""
Unit tests for trials_fetcher CLI module.

Tests the command-line interface for fetching clinical trials from ClinicalTrials.gov,
including argument parsing, workflow execution, data validation, and error handling.
"""

from src.workflows.trials_fetcher import TrialsFetcherWorkflow


class TestTrialsFetcherWorkflow:
    """Test the TrialsFetcherWorkflow class."""

    def test_execute_successful_fetch_by_condition(self):
        """Test successful trial fetching by medical condition."""
        # Create workflow
        workflow = TrialsFetcherWorkflow()

        # Execute workflow
        result = workflow.execute(condition="breast cancer", limit=5)

        # Verify result
        assert result.success is True
        assert result.metadata["total_fetched"] == 5
        assert result.metadata["fetch_type"] == "condition_search_with_full_data"

    def test_execute_successful_fetch_by_nct_id(self):
        """Test successful trial fetching by specific NCT ID."""
        # Create workflow
        workflow = TrialsFetcherWorkflow()

        # Execute workflow with a real NCT ID
        result = workflow.execute(nct_id="NCT00674206")

        # Verify result
        assert result.success is True
        assert result.metadata["total_fetched"] == 1
        assert result.metadata["fetch_type"] == "single_trial"

    def test_execute_successful_fetch_by_nct_ids(self):
        """Test successful trial fetching by multiple NCT IDs."""
        # Create workflow
        workflow = TrialsFetcherWorkflow()

        # Execute workflow with real NCT IDs
        result = workflow.execute(nct_ids=["NCT00674206", "NCT01147016"])

        # Verify result
        assert result.success is True
        assert result.metadata["total_fetched"] == 2
        assert result.metadata["fetch_type"] == "multiple_trials_batch"

    def test_execute_missing_required_args(self):
        """Test execution with missing required arguments."""
        # Create workflow
        workflow = TrialsFetcherWorkflow()

        # Execute workflow without any search criteria
        result = workflow.execute()

        # Verify result
        assert result.success is False
        assert "Invalid fetch parameters. Must provide condition, nct_id, or nct_ids." in result.error_message

    # Additional workflow tests can be added here as needed

    # Additional workflow tests can be added here as needed

    # Additional workflow tests can be added here as needed

    # Additional workflow tests can be added here as needed
