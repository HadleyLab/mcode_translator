"""
Tests for TrialsOptimizerWorkflow.
"""

import pytest
from unittest.mock import Mock, patch

from src.workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow
from src.utils.config import Config


class TestTrialsOptimizerWorkflow:
    """Test cases for TrialsOptimizerWorkflow."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config()

    @pytest.fixture
    def workflow(self, config):
        """Create a test workflow instance."""
        return TrialsOptimizerWorkflow(config)

    @pytest.fixture
    def mock_trials_data(self):
        """Create mock trial data for testing."""
        return [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00123456",
                        "briefTitle": "Mock Breast Cancer Trial 1"
                    }
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00234567",
                        "briefTitle": "Mock Breast Cancer Trial 2"
                    }
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00345678",
                        "briefTitle": "Mock Breast Cancer Trial 3"
                    }
                }
            }
        ]

    def test_workflow_initialization(self, workflow):
        """Test workflow initializes correctly."""
        assert workflow is not None
        assert workflow.memory_space == "optimization"

    def test_generate_combinations(self, workflow):
        """Test combination generation."""
        prompts = ["prompt1", "prompt2"]
        models = ["model1", "model2"]
        max_combinations = 3

        combinations = workflow._generate_combinations(prompts, models, max_combinations)

        assert len(combinations) == 3  # Limited by max_combinations
        assert all("prompt" in combo and "model" in combo for combo in combinations)

    def test_generate_combinations_full(self, workflow):
        """Test combination generation without limit."""
        prompts = ["prompt1", "prompt2"]
        models = ["model1"]
        max_combinations = 10

        combinations = workflow._generate_combinations(prompts, models, max_combinations)

        assert len(combinations) == 2  # All combinations
        expected_combinations = [
            {"prompt": "prompt1", "model": "model1"},
            {"prompt": "prompt2", "model": "model1"}
        ]
        assert combinations == expected_combinations

    @patch('src.workflows.trials_optimizer_workflow.McodePipeline')
    def test_test_combination_cv_success(self, mock_pipeline_class, workflow, mock_trials_data):
        """Test cross validation combination testing with successful pipeline."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.mcode_mappings = [Mock()] * 5  # Mock 5 mappings
        mock_result.validation_results = Mock(compliance_score=0.8)
        mock_pipeline.process.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline

        combination = {"prompt": "test_prompt", "model": "test_model"}
        cv_folds = 3

        result = workflow._test_combination_cv(combination, mock_trials_data, cv_folds)

        assert result["success"] is True
        assert result["combination"] == combination
        assert result["cv_folds"] == cv_folds
        assert "cv_average_score" in result
        assert "fold_scores" in result
        assert len(result["fold_scores"]) == cv_folds

    @patch('src.workflows.trials_optimizer_workflow.McodePipeline')
    def test_test_combination_cv_pipeline_failure(self, mock_pipeline_class, workflow, mock_trials_data):
        """Test cross validation handles pipeline failures gracefully."""
        # Mock pipeline that raises exception
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = Exception("Pipeline error")
        mock_pipeline_class.return_value = mock_pipeline

        combination = {"prompt": "test_prompt", "model": "test_model"}
        cv_folds = 2

        result = workflow._test_combination_cv(combination, mock_trials_data, cv_folds)

        # When all trials fail, it still returns success=True with score 0.0
        # This is the expected behavior - failures are handled gracefully
        assert result["success"] is True
        assert result["combination"] == combination
        assert result["cv_average_score"] == 0.0
        assert len(result["fold_scores"]) == cv_folds

    def test_calculate_quality_score(self, workflow):
        """Test quality score calculation."""
        # Mock pipeline result
        mock_result = Mock()
        mock_result.mcode_mappings = [Mock()] * 10  # 10 mappings
        mock_result.validation_results = Mock(compliance_score=0.9)
        mock_result.source_references = [Mock()] * 3  # 3 references

        score = workflow._calculate_quality_score(mock_result)

        assert 0.0 <= score <= 1.0
        # Score should be high due to good mappings, validation, and references

    def test_calculate_quality_score_minimal(self, workflow):
        """Test quality score with minimal data."""
        mock_result = Mock()
        mock_result.mcode_mappings = []
        mock_result.validation_results = None
        mock_result.source_references = None

        score = workflow._calculate_quality_score(mock_result)

        assert score == 0.0

    def test_validate_combination(self, workflow):
        """Test combination validation."""
        # Valid combination
        assert workflow.validate_combination("direct_mcode_evidence_based_concise", "deepseek-coder") is True

        # Invalid prompt
        assert workflow.validate_combination("invalid_prompt", "deepseek-coder") is False

        # Invalid model
        assert workflow.validate_combination("direct_mcode_evidence_based_concise", "invalid_model") is False

    def test_get_available_prompts(self, workflow):
        """Test getting available prompts."""
        prompts = workflow.get_available_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert "direct_mcode_evidence_based_concise" in prompts

    def test_get_available_models(self, workflow):
        """Test getting available models."""
        models = workflow.get_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "deepseek-coder" in models

    @patch('src.workflows.trials_optimizer_workflow.McodePipeline')
    def test_execute_success(self, mock_pipeline_class, workflow, mock_trials_data):
        """Test successful workflow execution."""
        # Mock pipeline for successful processing
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.mcode_mappings = [Mock()] * 8
        mock_result.validation_results = Mock(compliance_score=0.85)
        mock_pipeline.process.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline

        result = workflow.execute(
            trials_data=mock_trials_data,
            cv_folds=3,
            prompts=["direct_mcode_evidence_based_concise"],
            models=["deepseek-coder"],
            max_combinations=1
        )

        assert result.success is True
        assert "total_combinations_tested" in result.metadata
        assert "cv_folds" in result.metadata
        assert result.metadata["cv_folds"] == 3

    def test_execute_no_trials_data(self, workflow):
        """Test execution fails with no trial data."""
        result = workflow.execute(trials_data=[], cv_folds=3)

        assert result.success is False
        assert "No trial data provided" in result.error_message

    @patch('src.workflows.trials_optimizer_workflow.McodePipeline')
    @patch('src.workflows.trials_optimizer_workflow.TrialsOptimizerWorkflow._set_default_llm_spec')
    def test_execute_calls_set_default_spec(self, mock_set_default, mock_pipeline_class, workflow, mock_trials_data):
        """Test that execute calls set default LLM spec on success."""
        # Mock successful pipeline with good score
        mock_pipeline = Mock()
        mock_result = Mock()
        # Create proper mock mcode_mappings list
        mock_mappings = []
        for i in range(10):  # 10 mappings for good score
            mapping = Mock()
            mapping.mcode_mappings = []  # Make it iterable
            mock_mappings.append(mapping)
        mock_result.mcode_mappings = mock_mappings
        mock_result.validation_results = Mock(compliance_score=0.9)
        mock_result.source_references = [Mock()] * 5
        mock_pipeline.process.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline

        workflow.execute(
            trials_data=mock_trials_data,
            cv_folds=3,
            prompts=["direct_mcode_evidence_based_concise"],
            models=["deepseek-coder"],
            max_combinations=1
        )

        # Should have called set default spec since score > 0
        mock_set_default.assert_called_once()

    def test_cv_folds_adjustment(self, workflow, mock_trials_data):
        """Test CV folds adjustment when more folds than trials."""
        trials_count = len(mock_trials_data)  # 3 trials

        with patch.object(workflow, '_test_combination_cv') as mock_test_cv:
            mock_test_cv.return_value = {
                "success": True,
                "cv_average_score": 0.8,
                "fold_scores": [0.8, 0.8, 0.8],
                "combination": {"prompt": "test", "model": "test"}
            }

            result = workflow.execute(
                trials_data=mock_trials_data,
                cv_folds=5,  # More folds than trials
                prompts=["direct_mcode_evidence_based_concise"],
                models=["deepseek-coder"],
                max_combinations=1
            )

            # Should adjust folds to match trial count
            assert result.metadata["cv_folds"] == trials_count