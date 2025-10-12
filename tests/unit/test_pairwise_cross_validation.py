"""
Unit tests for pairwise_cross_validation module.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.optimization.pairwise_cross_validation import (
    PairwiseComparisonTask,
    PairwiseCrossValidator,
)
from src.shared.types import TaskStatus


class TestPairwiseComparisonTask:
    """Test PairwiseComparisonTask dataclass."""

    def test_task_creation(self):
        """Test creating a pairwise comparison task."""
        task = PairwiseComparisonTask(
            trial_id="trial_123",
            gold_prompt="prompt1",
            gold_model="model1",
            comp_prompt="prompt2",
            comp_model="model2",
        )

        assert task.trial_id == "trial_123"
        assert task.gold_prompt == "prompt1"
        assert task.gold_model == "model1"
        assert task.comp_prompt == "prompt2"
        assert task.comp_model == "model2"
        assert task.status.value == "Pending"  # Default status

    def test_task_default_values(self):
        """Test default values in PairwiseComparisonTask."""
        task = PairwiseComparisonTask()

        assert task.task_id != ""  # Should be auto-generated UUID
        assert task.trial_id == ""
        assert task.gold_prompt == ""
        assert task.gold_model == ""
        assert task.comp_prompt == ""
        assert task.comp_model == ""
        assert task.status.value == "Pending"
        assert task.error_message == ""
        assert task.comparison_metrics == {}
        assert task.start_time == 0.0
        assert task.end_time == 0.0
        assert task.duration_ms == 0.0


class TestPairwiseCrossValidatorInit:
    """Test PairwiseCrossValidator initialization."""

    @patch("src.optimization.pairwise_cross_validation.PromptLoader")
    @patch("src.optimization.pairwise_cross_validation.LLMLoader")
    def test_init_default(self, mock_llm_loader, mock_prompt_loader):
        """Test default initialization."""
        validator = PairwiseCrossValidator()

        assert validator.output_dir == Path("pairwise_optimization_results")
        assert validator.pairwise_results == []
        assert validator.combination_cache == {}
        assert validator.summary_stats == {}
        assert validator.task_queue is None

        # Verify loaders are created
        mock_prompt_loader.assert_called_once()
        mock_llm_loader.assert_called_once()

    @patch("src.optimization.pairwise_cross_validation.PromptLoader")
    @patch("src.optimization.pairwise_cross_validation.LLMLoader")
    def test_init_custom_output_dir(self, mock_llm_loader, mock_prompt_loader):
        """Test initialization with custom output directory."""
        validator = PairwiseCrossValidator("custom_results")

        assert validator.output_dir == Path("custom_results")


class TestPairwiseCrossValidatorMethods:
    """Test PairwiseCrossValidator methods."""

    @pytest.fixture
    @patch("src.optimization.pairwise_cross_validation.PromptLoader")
    @patch("src.optimization.pairwise_cross_validation.LLMLoader")
    def setup_validator(self, mock_llm_loader, mock_prompt_loader):
        """Set up validator with mocked loaders."""
        mock_prompt_instance = Mock()
        mock_llm_instance = Mock()
        mock_prompt_loader.return_value = mock_prompt_instance
        mock_llm_loader.return_value = mock_llm_instance

        validator = PairwiseCrossValidator()
        return validator, mock_prompt_instance, mock_llm_instance

    def test_get_available_prompts(self, setup_validator):
        """Test getting available prompts."""
        validator, mock_prompt, mock_llm = setup_validator

        mock_prompt.list_available_prompts.return_value = {"prompt1": {}, "prompt2": {}}

        result = validator.get_available_prompts()

        assert result == ["prompt1", "prompt2"]
        mock_prompt.list_available_prompts.assert_called_once()

    def test_get_available_models(self, setup_validator):
        """Test getting available models."""
        validator, mock_prompt, mock_llm = setup_validator

        mock_llm.list_available_llms.return_value = {"model1": {}, "model2": {}}

        result = validator.get_available_models()

        assert result == ["model1", "model2"]
        mock_llm.list_available_llms.assert_called_once()

    def test_load_trials_successful_trials(self, setup_validator):
        """Test loading trials with successful_trials format."""
        validator, _, _ = setup_validator

        trial_data = {
            "successful_trials": [
                {"id": "trial1", "data": "test1"},
                {"id": "trial2", "data": "test2"},
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(trial_data, tmp)
            tmp.flush()

            try:
                result = validator.load_trials(tmp.name)

                assert len(result) == 2
                assert result[0]["id"] == "trial1"
                assert result[1]["id"] == "trial2"
            finally:
                Path(tmp.name).unlink()

    def test_load_trials_list_format(self, setup_validator):
        """Test loading trials with list format."""
        validator, _, _ = setup_validator

        trial_data = [
            {"id": "trial1", "data": "test1"},
            {"id": "trial2", "data": "test2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(trial_data, tmp)
            tmp.flush()

            try:
                result = validator.load_trials(tmp.name)

                assert len(result) == 2
                assert result[0]["id"] == "trial1"
                assert result[1]["id"] == "trial2"
            finally:
                Path(tmp.name).unlink()

    def test_load_trials_file_not_found(self, setup_validator):
        """Test loading trials from non-existent file."""
        validator, _, _ = setup_validator

        with pytest.raises(FileNotFoundError):
            validator.load_trials("nonexistent.json")

    def test_load_trials_invalid_format(self, setup_validator):
        """Test loading trials with invalid format."""
        validator, _, _ = setup_validator

        invalid_data = {"invalid": "format"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(invalid_data, tmp)
            tmp.flush()

            try:
                with pytest.raises(ValueError, match="Invalid trials file format"):
                    validator.load_trials(tmp.name)
            finally:
                Path(tmp.name).unlink()

    def test_extract_trial_id_with_nct_id(self, setup_validator):
        """Test extracting trial ID with NCT ID present."""
        validator, _, _ = setup_validator

        trial_data = {"protocolSection": {"identificationModule": {"nctId": "NCT123456"}}}

        result = validator._extract_trial_id(trial_data, 0)

        assert result == "NCT123456"

    def test_extract_trial_id_fallback(self, setup_validator):
        """Test extracting trial ID with fallback to index."""
        validator, _, _ = setup_validator

        trial_data = {"no_nct_id": True}

        result = validator._extract_trial_id(trial_data, 5)

        assert result == "trial_5"

    def test_extract_trial_id_key_error(self, setup_validator):
        """Test extracting trial ID with KeyError."""
        validator, _, _ = setup_validator

        trial_data = {}  # Missing protocolSection

        result = validator._extract_trial_id(trial_data, 3)

        assert result == "trial_3"

    def test_analyze_pairwise_results_empty(self, setup_validator):
        """Test analyzing empty pairwise results."""
        validator, _, _ = setup_validator

        result = validator.analyze_pairwise_results()

        assert result == {}

    def test_analyze_pairwise_results_with_data(self, setup_validator):
        """Test analyzing pairwise results with mock data."""
        validator, _, _ = setup_validator

        # Create mock successful task
        mock_task = Mock()
        mock_task.status = TaskStatus.SUCCESS
        mock_task.comparison_metrics = {
            "mapping_f1_score": 0.85,
            "mapping_precision": 0.9,
            "mapping_recall": 0.8,
            "gold_mappings_count": 10,
            "comp_mappings_count": 9,
        }

        validator.pairwise_results = [mock_task]

        result = validator.analyze_pairwise_results()

        assert "summary" in result
        assert "configuration_analysis" in result
        assert "overall_metrics" in result
        assert result["summary"]["total_comparisons"] == 1
        assert result["summary"]["successful_comparisons"] == 1
        assert result["summary"]["success_rate"] == 1.0

    def test_print_summary_no_stats(self, setup_validator):
        """Test printing summary with no statistics."""
        validator, _, _ = setup_validator

        # Should not raise exception
        validator.print_summary()

    def test_print_summary_with_stats(self, setup_validator):
        """Test printing summary with statistics."""
        validator, _, _ = setup_validator

        mock_logger = Mock()
        validator.logger = mock_logger

        # Set up mock summary stats
        validator.summary_stats = {
            "summary": {
                "total_comparisons": 10,
                "successful_comparisons": 8,
                "success_rate": 0.8,
                "unique_config_pairs": 5,
            },
            "overall_metrics": {
                "mapping_f1_score": {"mean": 0.75},
                "mapping_jaccard_similarity": {"mean": 0.65},
            },
        }

        validator.print_summary()

        # Verify logger calls
        mock_logger.info.assert_called()

    def test_initialize_and_shutdown(self, setup_validator):
        """Test initialize and shutdown methods."""
        validator, _, _ = setup_validator

        mock_logger = Mock()
        validator.logger = mock_logger

        # Test initialize
        validator.initialize()
        mock_logger.info.assert_called_with("🤖 Initializing pairwise validator")

        # Test shutdown
        validator.shutdown()
        mock_logger.info.assert_called_with("🛑 Shutting down pairwise validator")

    @patch("src.optimization.pairwise_cross_validation.TaskQueue")
    @patch("src.optimization.pairwise_cross_validation.McodePipeline")
    def test_run_pairwise_validation(
        self, mock_pipeline_class, mock_task_queue_class, setup_validator
    ):
        """Test running pairwise validation."""
        validator, _, _ = setup_validator

        # Mock task queue
        mock_task_queue = Mock()
        mock_task_queue_class.return_value = mock_task_queue
        # Mock task result objects with success attribute
        mock_result = Mock()
        mock_result.success = True
        mock_task_queue.execute_tasks.return_value = [mock_result]

        # Create mock tasks
        mock_task = Mock()
        mock_task.task_id = "test_task"
        tasks = [mock_task]

        # Mock logger
        mock_logger = Mock()
        validator.logger = mock_logger

        # Run validation
        validator.run_pairwise_validation(tasks, max_workers=2)

        # Verify task queue was created and used
        mock_task_queue_class.assert_called_once_with(max_workers=2, name="PairwiseValidator")
        mock_task_queue.execute_tasks.assert_called_once()

    @patch("src.optimization.pairwise_cross_validation.McodePipeline")
    @patch("src.optimization.pairwise_cross_validation.BenchmarkResult")
    def test_process_pairwise_task(
        self, mock_benchmark_class, mock_pipeline_class, setup_validator
    ):
        """Test processing a single pairwise task."""
        validator, _, _ = setup_validator

        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline_result = Mock()
        mock_pipeline_result.mcode_mappings = []
        mock_pipeline_result.validation_results.compliance_score = 0.8
        mock_pipeline.process = AsyncMock(return_value=mock_pipeline_result)

        # Mock benchmark result
        mock_benchmark = Mock()
        mock_benchmark.pipeline_result = mock_pipeline_result
        mock_benchmark_class.return_value = mock_benchmark

        # Create task
        task = PairwiseComparisonTask(
            trial_id="NCT123",
            trial_data={"test": "data"},
            gold_prompt="prompt1",
            gold_model="model1",
            comp_prompt="prompt2",
            comp_model="model2",
        )

        # Process task
        asyncio.run(validator._process_pairwise_task(task))

        # Verify pipeline was created and called
        assert mock_pipeline_class.call_count == 2  # Gold and comp pipelines
        assert task.status == TaskStatus.SUCCESS
        assert task.gold_result is not None
        assert task.comp_result is not None

    def test_calculate_comparison_metrics(self, setup_validator):
        """Test calculating comparison metrics."""
        validator, _, _ = setup_validator

        # Create mock benchmark results
        mock_gold_result = Mock()
        mock_gold_result.pipeline_result.mcode_mappings = [
            Mock(element_type="Condition", code="C50", confidence_score=0.9)
        ]
        mock_gold_result.pipeline_result.validation_results.compliance_score = 0.85

        mock_comp_result = Mock()
        mock_comp_result.pipeline_result.mcode_mappings = [
            Mock(element_type="Condition", code="C50", confidence_score=0.8)
        ]
        mock_comp_result.pipeline_result.validation_results.compliance_score = 0.82

        # Create task
        task = PairwiseComparisonTask()
        task.gold_result = mock_gold_result
        task.comp_result = mock_comp_result

        # Calculate metrics
        validator._calculate_comparison_metrics(task)

        # Verify metrics were calculated
        assert "mapping_jaccard_similarity" in task.comparison_metrics
        assert "mapping_f1_score" in task.comparison_metrics
        assert "gold_mappings_count" in task.comparison_metrics
        assert "comp_mappings_count" in task.comparison_metrics

    def test_calculate_mapping_metrics(self, setup_validator):
        """Test calculating mapping-level metrics."""
        validator, _, _ = setup_validator

        # Create mock mappings
        gold_mappings = [
            Mock(element_type="Condition", code="C50", confidence_score=0.9),
            Mock(element_type="Treatment", code="T123", confidence_score=0.8),
        ]
        comp_mappings = [
            Mock(element_type="Condition", code="C50", confidence_score=0.85),
            Mock(element_type="Treatment", code="T456", confidence_score=0.7),
        ]

        # Calculate metrics
        result = validator._calculate_mapping_metrics(gold_mappings, comp_mappings)

        # Verify all expected metrics are present
        expected_metrics = [
            "mapping_jaccard_similarity",
            "mapping_precision",
            "mapping_recall",
            "mapping_f1_score",
            "mapping_true_positives",
            "mapping_false_positives",
            "mapping_false_negatives",
            "gold_mappings_count",
            "comp_mappings_count",
            "gold_avg_confidence",
            "comp_avg_confidence",
            "true_positive_examples",
            "false_positive_examples",
            "false_negative_examples",
        ]

        for metric in expected_metrics:
            assert metric in result

    def test_calculate_mapping_metrics_edge_cases(self, setup_validator):
        """Test mapping metrics calculation with edge cases."""
        validator, _, _ = setup_validator

        # Test with empty mappings
        result = validator._calculate_mapping_metrics([], [])
        assert result["mapping_jaccard_similarity"] == 0.0

        # Test with invalid mappings (None element_type or code)
        gold_mappings = [Mock(element_type=None, code="C50")]
        comp_mappings = [Mock(element_type="Condition", code=None)]

        result = validator._calculate_mapping_metrics(gold_mappings, comp_mappings)
        assert isinstance(result, dict)

    @patch("src.optimization.pairwise_cross_validation.json.dump")
    @patch("src.optimization.pairwise_cross_validation.datetime")
    def test_save_results(self, mock_datetime, mock_json_dump, setup_validator):
        """Test saving results."""
        validator, _, _ = setup_validator

        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"

        # Add mock results
        mock_task = Mock()
        mock_task.task_id = "test_task"
        mock_task.trial_id = "NCT123"
        mock_task.gold_prompt = "prompt1"
        mock_task.gold_model = "model1"
        mock_task.comp_prompt = "prompt2"
        mock_task.comp_model = "model2"
        mock_task.comparison_metrics = {"test": "metric"}
        mock_task.status = TaskStatus.SUCCESS
        mock_task.duration_ms = 100.0

        validator.pairwise_results = [mock_task]

        # Save results
        validator.save_results()

        # Verify json.dump was called
        assert mock_json_dump.call_count >= 2  # Results and summary files

    def test_generate_detailed_report(self, setup_validator):
        """Test generating detailed markdown report."""
        validator, _, _ = setup_validator

        # Set up mock summary stats
        validator.summary_stats = {
            "summary": {
                "total_comparisons": 10,
                "successful_comparisons": 8,
                "success_rate": 0.8,
                "unique_config_pairs": 5,
            },
            "overall_metrics": {
                "mapping_f1_score": {
                    "mean": 0.75,
                    "median": 0.7,
                    "stdev": 0.1,
                    "min": 0.6,
                    "max": 0.9,
                }
            },
        }

        # Generate report
        validator.generate_detailed_report("test_timestamp")

        # Verify report file was created
        report_file = validator.output_dir / "pairwise_report_test_timestamp.md"
        assert report_file.exists()

        # Check report content
        with open(report_file, "r") as f:
            content = f.read()

        assert "# Pairwise Cross-Validation Report" in content
        assert "Executive Summary" in content
        assert "Overall Metrics" in content

    def test_analyze_prompt_performance(self, setup_validator):
        """Test analyzing prompt performance."""
        validator, _, _ = setup_validator

        mock_logger = Mock()
        validator.logger = mock_logger

        # Set up mock config analysis
        validator.summary_stats = {
            "configuration_analysis": {
                "evidence_based_with_codes_model1_vs_evidence_based_model2": {
                    "mapping_f1_score": {"mean": 0.85}
                },
                "evidence_based_model1_vs_other_model2": {"mapping_f1_score": {"mean": 0.75}},
            }
        }

        # Analyze performance
        validator._analyze_prompt_performance()

        # Verify logger calls for prompt analysis
        mock_logger.info.assert_called()


class TestPairwiseCrossValidatorIntegration:
    """Integration tests for PairwiseCrossValidator."""

    @patch("src.optimization.pairwise_cross_validation.PromptLoader")
    @patch("src.optimization.pairwise_cross_validation.LLMLoader")
    def test_generate_pairwise_tasks(self, mock_llm_loader, mock_prompt_loader):
        """Test generating pairwise tasks."""
        mock_prompt_instance = Mock()
        mock_llm_instance = Mock()
        mock_prompt_loader.return_value = mock_prompt_instance
        mock_llm_loader.return_value = mock_llm_instance

        mock_prompt_instance.list_available_prompts.return_value = {
            "prompt1": {},
            "prompt2": {},
        }
        mock_llm_instance.list_available_llms.return_value = {
            "model1": {},
            "model2": {},
        }

        validator = PairwiseCrossValidator()

        trials = [
            {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}},
            {"protocolSection": {"identificationModule": {"nctId": "NCT2"}}},
        ]

        tasks = validator.generate_pairwise_tasks(
            prompts=["prompt1", "prompt2"], models=["model1", "model2"], trials=trials
        )

        # Should generate tasks for all combinations
        # For 2 prompts × 2 models × 2 trials = 8 combinations
        # Each combination compared against others with same trial = complex calculation
        assert len(tasks) > 0
        assert all(isinstance(task, PairwiseComparisonTask) for task in tasks)

        # Check that tasks have proper structure
        task = tasks[0]
        assert task.trial_id in ["NCT1", "NCT2"]
        assert task.gold_prompt in ["prompt1", "prompt2"]
        assert task.gold_model in ["model1", "model2"]


if __name__ == "__main__":
    pytest.main([__file__])
