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
    def test_test_single_trial_success(self, mock_pipeline_class, workflow, mock_trials_data):
        """Test individual trial processing with successful pipeline."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_result = Mock()
        # Create proper mock structure for quality score calculation
        mock_result.mcode_mappings = []  # Will be treated as list
        mock_result.validation_results = Mock(compliance_score=0.8)
        mock_result.source_references = [Mock()] * 3  # Add source references for higher score
        mock_pipeline.process.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline

        combination = {"prompt": "test_prompt", "model": "test_model"}
        trial = mock_trials_data[0]
        fold = 0
        combo_idx = 0

        result = workflow._test_single_trial(combination, trial, fold, combo_idx)

        assert result["success"] is True
        assert result["combination"] == combination
        assert result["combo_idx"] == combo_idx
        assert result["fold"] == fold
        assert "score" in result
        assert "trial_score" in result
        assert isinstance(result["score"], float)
        assert result["score"] > 0  # Should have positive score

    @patch('src.workflows.trials_optimizer_workflow.McodePipeline')
    def test_test_single_trial_pipeline_failure(self, mock_pipeline_class, workflow, mock_trials_data):
        """Test individual trial processing handles pipeline failures gracefully."""
        # Mock pipeline that raises exception
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = Exception("Pipeline error")
        mock_pipeline_class.return_value = mock_pipeline

        combination = {"prompt": "test_prompt", "model": "test_model"}
        trial = mock_trials_data[0]
        fold = 0
        combo_idx = 0

        result = workflow._test_single_trial(combination, trial, fold, combo_idx)

        assert result["success"] is False
        assert result["combination"] == combination
        assert result["combo_idx"] == combo_idx
        assert result["fold"] == fold
        assert result["score"] == 0.0  # Failed trials get 0 score
        assert "error" in result

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

    @patch('src.workflows.trials_optimizer_workflow.create_task_queue_from_args')
    @patch('src.workflows.trials_optimizer_workflow.create_task')
    @patch('src.workflows.trials_optimizer_workflow.TaskQueue')
    def test_execute_fully_asynchronous_task_breakdown(self, mock_task_queue_class, mock_create_task, mock_create_queue_from_args, workflow, mock_trials_data):
        """Test that execute creates individual trial-level tasks for maximum parallelism."""
        # Mock task queue
        mock_task_queue = Mock()
        mock_task_queue.worker_pool.max_workers = 4
        mock_task_queue_class.return_value = mock_task_queue

        # Mock CLI args for concurrency
        mock_cli_args = Mock()
        mock_create_queue_from_args.return_value = mock_task_queue

        # Mock successful task results - simulate individual trial results
        mock_task_results = []
        task_id = 0
        for combo_idx in range(2):  # 2 combinations
            for fold in range(3):  # 3 folds
                for trial_idx in range(1):  # 1 trial per fold (simplified)
                    mock_result = Mock()
                    mock_result.success = True
                    mock_result.result = {
                        "combo_idx": combo_idx,
                        "fold": fold,
                        "score": 0.8,
                        "trial_score": 0.8,
                        "success": True
                    }
                    mock_task = Mock()
                    mock_task.task_id = f"trial_{task_id}"
                    mock_task_result = Mock()
                    mock_task_result.success = True
                    mock_task_result.result = mock_result.result
                    mock_task_results.append(mock_task_result)
                    task_id += 1

        mock_task_queue.execute_tasks.return_value = mock_task_results

        # Mock create_task to track calls
        mock_create_task.side_effect = lambda **kwargs: Mock(kwargs=kwargs)

        result = workflow.execute(
            trials_data=mock_trials_data,
            cv_folds=3,
            prompts=["prompt1", "prompt2"],
            models=["model1"],
            max_combinations=2,
            cli_args=mock_cli_args
        )

        # Verify individual trial tasks were created (not combination tasks)
        assert mock_create_task.call_count == 6  # 2 combos × 3 folds × 1 trial each

        # Verify task creation calls were for _test_single_trial
        for call in mock_create_task.call_args_list:
            assert call[1]['func'] == workflow._test_single_trial
            assert 'combination' in call[1]
            assert 'trial' in call[1]
            assert 'fold' in call[1]
            assert 'combo_idx' in call[1]

        assert result.success is True
        assert len(result.data) == 2  # 2 combinations with results

    @patch('src.workflows.trials_optimizer_workflow.create_task_queue_from_args')
    @patch('src.workflows.trials_optimizer_workflow.TaskQueue')
    def test_execute_result_aggregation_from_trial_scores(self, mock_task_queue_class, mock_create_queue_from_args, workflow, mock_trials_data):
        """Test that results are properly aggregated from individual trial scores."""
        # Mock task queue
        mock_task_queue = Mock()
        mock_task_queue.worker_pool.max_workers = 2
        mock_task_queue_class.return_value = mock_task_queue
        mock_create_queue_from_args.return_value = mock_task_queue

        # Mock CLI args
        mock_cli_args = Mock()

        # Track progress callback calls to verify aggregation
        progress_calls = []

        def mock_execute_tasks(tasks, progress_callback=None):
            # Simulate calling progress callback for each task
            for i, task in enumerate(tasks):
                mock_result = Mock()
                mock_result.success = True
                # Extract combo_idx from task kwargs
                combo_idx = task.kwargs.get('combo_idx', 0)
                fold = task.kwargs.get('fold', 0)
                score = 0.7 + (combo_idx * 0.1) + (fold * 0.05)
                mock_result.result = {
                    "combo_idx": combo_idx,
                    "fold": fold,
                    "score": score,
                    "trial_score": score,
                    "success": True
                }
                progress_calls.append((i + 1, len(tasks), mock_result))
                if progress_callback:
                    progress_callback(i + 1, len(tasks), mock_result)
            return []

        mock_task_queue.execute_tasks.side_effect = mock_execute_tasks

        result = workflow.execute(
            trials_data=mock_trials_data,
            cv_folds=3,
            prompts=["prompt1", "prompt2"],
            models=["model1"],
            max_combinations=2,
            cli_args=mock_cli_args
        )

        assert result.success is True
        assert len(result.data) == 2  # 2 combinations

        # Verify progress callback was called
        assert len(progress_calls) == 6  # 2 combos × 3 folds × 1 trial each

        # Verify CV statistics are computed correctly
        for combo_result in result.data:
            combo_idx = result.data.index(combo_result)
            # Each combination should have 3 trial scores
            assert combo_result["total_trials"] == 3
            assert len(combo_result["fold_scores"]) == 3
            assert "cv_average_score" in combo_result
            assert "cv_std_score" in combo_result
            assert combo_result["cv_average_score"] > 0

    @patch('src.workflows.trials_optimizer_workflow.create_task_queue_from_args')
    @patch('src.workflows.trials_optimizer_workflow.TaskQueue')
    def test_execute_handles_trial_failures_in_aggregation(self, mock_task_queue_class, mock_create_queue_from_args, workflow, mock_trials_data):
        """Test that failed trials are handled correctly in result aggregation."""
        # Mock task queue
        mock_task_queue = Mock()
        mock_task_queue.worker_pool.max_workers = 2
        mock_task_queue_class.return_value = mock_task_queue
        mock_create_queue_from_args.return_value = mock_task_queue

        mock_cli_args = Mock()

        # Track progress callback calls
        progress_calls = []

        def mock_execute_tasks(tasks, progress_callback=None):
            # Simulate mix of successful and failed trials
            results = []
            for i, task in enumerate(tasks):
                combo_idx = task.kwargs.get('combo_idx', 0)
                fold = task.kwargs.get('fold', 0)

                if combo_idx == 0 and fold == 2:  # Make one trial fail
                    mock_result = Mock()
                    mock_result.success = False
                    mock_result.error = "Trial failed"
                    mock_result.result = None
                else:
                    score = 0.7 + (combo_idx * 0.1) + (fold * 0.05)
                    mock_result = Mock()
                    mock_result.success = True
                    mock_result.result = {
                        "combo_idx": combo_idx,
                        "fold": fold,
                        "score": score,
                        "success": True
                    }

                progress_calls.append((i + 1, len(tasks), mock_result))
                if progress_callback:
                    progress_callback(i + 1, len(tasks), mock_result)
                results.append(mock_result)
            return results

        mock_task_queue.execute_tasks.side_effect = mock_execute_tasks

        result = workflow.execute(
            trials_data=mock_trials_data,
            cv_folds=3,
            prompts=["prompt1", "prompt2"],
            models=["model1"],
            max_combinations=2,
            cli_args=mock_cli_args
        )

        assert result.success is True
        assert len(result.data) == 2

        # Combo 0 should have 2 successful trials (1 failed)
        combo0_result = result.data[0]
        assert combo0_result["total_trials"] == 2  # Only successful trials counted
        assert combo0_result["cv_average_score"] > 0

        # Combo 1 should have 3 successful trials
        combo1_result = result.data[1]
        assert combo1_result["total_trials"] == 3
        assert combo1_result["cv_average_score"] > 0

    @patch('src.workflows.trials_optimizer_workflow.create_task_queue_from_args')
    @patch('src.workflows.trials_optimizer_workflow.TaskQueue')
    def test_execute_maximum_parallelism_calculation(self, mock_task_queue_class, mock_create_queue_from_args, workflow, mock_trials_data):
        """Test that the maximum parallelism is calculated correctly."""
        # Mock task queue
        mock_task_queue = Mock()
        mock_task_queue.worker_pool.max_workers = 8
        mock_task_queue_class.return_value = mock_task_queue
        mock_create_queue_from_args.return_value = mock_task_queue

        mock_cli_args = Mock()

        # Mock successful results
        mock_task_results = []
        for i in range(12):  # 2 combos × 3 folds × 2 trials = 12 tasks
            mock_result = Mock()
            mock_result.success = True
            mock_result.result = {
                "combo_idx": i % 2,
                "fold": (i // 2) % 3,
                "score": 0.8,
                "success": True
            }
            mock_task_result = Mock()
            mock_task_result.success = True
            mock_task_result.result = mock_result.result
            mock_task_results.append(mock_task_result)

        mock_task_queue.execute_tasks.return_value = mock_task_results

        # Mock logging to capture the asynchronous message
        with patch.object(workflow.logger, 'info') as mock_logger:
            result = workflow.execute(
                trials_data=mock_trials_data * 2,  # 6 trials total
                cv_folds=3,
                prompts=["prompt1", "prompt2"],
                models=["model1"],
                max_combinations=2,
                cli_args=mock_cli_args
            )

            # Verify the asynchronous logging message was called
            async_log_calls = [call for call in mock_logger.call_args_list
                             if "Fully asynchronous optimization" in str(call)]
            assert len(async_log_calls) == 1

            # Verify task count calculation: 2 combos × 3 folds × 2 trials = 12 tasks
            log_message = str(async_log_calls[0])
            assert "12 concurrent tasks" in log_message

        assert result.success is True

    @patch('src.workflows.trials_optimizer_workflow.create_task_queue_from_args')
    @patch('src.workflows.trials_optimizer_workflow.TaskQueue')
    def test_execute_concurrency_args_integration(self, mock_task_queue_class, mock_create_queue_from_args, workflow, mock_trials_data):
        """Test that concurrency arguments are properly integrated."""
        # Mock task queue with specific worker count
        mock_task_queue = Mock()
        mock_task_queue.worker_pool.max_workers = 6
        mock_task_queue_class.return_value = mock_task_queue

        # Mock CLI args with concurrency settings
        mock_cli_args = Mock()
        mock_cli_args.workers = 6
        mock_cli_args.worker_pool = "optimizer"
        mock_create_queue_from_args.return_value = mock_task_queue

        # Mock successful results
        mock_task_results = [Mock(success=True, result={"combo_idx": 0, "score": 0.8, "success": True})]
        mock_task_queue.execute_tasks.return_value = mock_task_results

        with patch.object(workflow.logger, 'info') as mock_logger:
            result = workflow.execute(
                trials_data=mock_trials_data,
                cv_folds=3,
                prompts=["prompt1"],
                models=["model1"],
                max_combinations=1,
                cli_args=mock_cli_args
            )

            # Verify concurrency logging
            worker_log_calls = [call for call in mock_logger.call_args_list
                              if "concurrent workers" in str(call)]
            assert len(worker_log_calls) == 1
            assert "6 concurrent workers" in str(worker_log_calls[0])

        assert result.success is True