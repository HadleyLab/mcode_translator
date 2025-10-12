#!/usr/bin/env python3
"""
Unit tests for TrialsOptimizerWorkflow class.
"""

from src.workflows.trials_optimizer import TrialsOptimizerWorkflow


class TestTrialsOptimizerWorkflow:
    """Test the TrialsOptimizerWorkflow class."""

    def test_workflow_instantiation(self):
        """Test that the workflow can be instantiated."""
        workflow = TrialsOptimizerWorkflow()
        assert workflow is not None

    def test_summarize_benchmark_validations(self):
        """Test benchmark validation summarization."""
        workflow = TrialsOptimizerWorkflow()
        # Test that the method exists
        assert hasattr(workflow, "summarize_benchmark_validations")
