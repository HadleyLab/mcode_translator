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

    def test_get_available_prompts(self):
        """Test getting available prompts."""
        workflow = TrialsOptimizerWorkflow()
        prompts = workflow.get_available_prompts()
        assert isinstance(prompts, list)

    def test_get_available_models(self):
        """Test getting available models."""
        workflow = TrialsOptimizerWorkflow()
        models = workflow.get_available_models()
        assert isinstance(models, list)
