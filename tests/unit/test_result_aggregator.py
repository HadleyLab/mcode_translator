"""
Unit tests for result_aggregator module.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.optimization.result_aggregator import OptimizationResultAggregator


class TestOptimizationResultAggregator:
    """Test OptimizationResultAggregator class."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def aggregator(self, mock_logger):
        """Create OptimizationResultAggregator instance."""
        with patch('src.optimization.result_aggregator.PerformanceAnalyzer'), \
             patch('src.optimization.result_aggregator.ReportGenerator'), \
             patch('src.optimization.result_aggregator.BiologicalAnalyzer'):
            return OptimizationResultAggregator(mock_logger)

    @pytest.fixture
    def sample_combo_results(self):
        """Sample combo results for testing."""
        return {
            0: {
                "scores": [0.85, 0.82, 0.88],
                "errors": [],
                "metrics": [
                    {"precision": 0.9, "recall": 0.8, "f1_score": 0.85},
                    {"precision": 0.85, "recall": 0.85, "f1_score": 0.85},
                    {"precision": 0.95, "recall": 0.82, "f1_score": 0.88}
                ],
                "mcode_elements": ["element1", "element2", "element3"]
            },
            1: {
                "scores": [],
                "errors": ["Error 1", "Error 2", "Error 3"],
                "metrics": [],
                "mcode_elements": []
            }
        }

    @pytest.fixture
    def sample_combinations(self):
        """Sample combinations for testing."""
        return [
            {"model": "gpt-4", "prompt": "direct_mcode"},
            {"model": "claude-3", "prompt": "evidence_based"}
        ]

    @pytest.fixture
    def sample_trials_data(self):
        """Sample trials data for testing."""
        return [
            {"id": "NCT123", "title": "Trial 1"},
            {"id": "NCT456", "title": "Trial 2"}
        ]

    def test_init(self, mock_logger):
        """Test initialization."""
        with patch('src.optimization.result_aggregator.PerformanceAnalyzer') as mock_perf, \
             patch('src.optimization.result_aggregator.ReportGenerator') as mock_report, \
             patch('src.optimization.result_aggregator.BiologicalAnalyzer') as mock_bio:

            aggregator = OptimizationResultAggregator(mock_logger)

            assert aggregator.logger == mock_logger
            assert aggregator.performance_analyzer == mock_perf.return_value
            assert aggregator.report_generator == mock_report.return_value
            assert aggregator.biological_analyzer == mock_bio.return_value

    def test_aggregate_results_successful_combinations(self, aggregator, sample_combo_results, sample_combinations, sample_trials_data):
        """Test aggregating results with successful combinations."""
        with patch('src.optimization.result_aggregator.datetime') as mock_datetime, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            # Mock datetime
            mock_datetime.now.return_value.strftime.return_value = "20231201_120000"
            mock_datetime.now.return_value.isoformat.return_value = "2023-12-01T12:00:00"

            result = aggregator.aggregate_results(
                sample_combo_results, sample_combinations, sample_trials_data, cv_folds=3
            )

            # Verify structure
            assert "optimization_results" in result
            assert "best_result" in result
            assert "best_score" in result

            # Verify successful result
            successful_results = [r for r in result["optimization_results"] if r["success"]]
            assert len(successful_results) == 1

            success_result = successful_results[0]
            assert success_result["combination"] == {"model": "gpt-4", "prompt": "direct_mcode"}
            assert success_result["cv_average_score"] == pytest.approx(0.85, rel=1e-2)
            assert success_result["cv_std_score"] > 0
            assert success_result["fold_scores"] == [0.85, 0.82, 0.88]
            assert success_result["cv_folds"] == 3
            assert success_result["total_trials"] == 3
            assert success_result["total_elements"] == 3
            assert success_result["metrics"]["precision"] == pytest.approx(0.9, rel=1e-2)
            assert success_result["metrics"]["recall"] == pytest.approx(0.823, rel=1e-2)
            assert success_result["metrics"]["f1_score"] == pytest.approx(0.86, rel=1e-2)

            # Verify best result
            assert result["best_result"] == success_result
            assert result["best_score"] == success_result["cv_average_score"]

    def test_aggregate_results_failed_combinations(self, aggregator, sample_combo_results, sample_combinations, sample_trials_data):
        """Test aggregating results with failed combinations."""
        with patch('src.optimization.result_aggregator.datetime') as mock_datetime, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            # Mock datetime
            mock_datetime.now.return_value.strftime.return_value = "20231201_120000"
            mock_datetime.now.return_value.isoformat.return_value = "2023-12-01T12:00:00"

            result = aggregator.aggregate_results(
                sample_combo_results, sample_combinations, sample_trials_data, cv_folds=3
            )

            # Verify failed result
            failed_results = [r for r in result["optimization_results"] if not r["success"]]
            assert len(failed_results) == 1

            failed_result = failed_results[0]
            assert failed_result["combination"] == {"model": "claude-3", "prompt": "evidence_based"}
            assert "error" in failed_result
            assert "All 3 trials failed" in failed_result["error"]

    def test_aggregate_results_creates_run_files(self, aggregator, sample_combo_results, sample_combinations, sample_trials_data):
        """Test that aggregate_results creates run files."""
        with patch('src.optimization.result_aggregator.datetime') as mock_datetime, \
             patch('pathlib.Path.mkdir'), \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            # Mock datetime
            mock_datetime.now.return_value.strftime.return_value = "20231201_120000"
            mock_datetime.now.return_value.isoformat.return_value = "2023-12-01T12:00:00"

            aggregator.aggregate_results(
                sample_combo_results, sample_combinations, sample_trials_data, cv_folds=3
            )

            # Verify files were created (should be called for both success and failure)
            assert mock_open.call_count == 2  # One for success, one for failure
            assert mock_json_dump.call_count == 2

    def test_aggregate_results_empty_scores(self, aggregator, sample_combinations, sample_trials_data):
        """Test aggregating results with empty scores."""
        combo_results = {
            0: {
                "scores": [],
                "errors": ["Error"],
                "metrics": [],
                "mcode_elements": []
            }
        }

        with patch('src.optimization.result_aggregator.datetime') as mock_datetime, \
             patch('builtins.open', create=True), \
             patch('json.dump'):

            mock_datetime.now.return_value.strftime.return_value = "20231201_120000"
            mock_datetime.now.return_value.isoformat.return_value = "2023-12-01T12:00:00"

            result = aggregator.aggregate_results(
                combo_results, [sample_combinations[0]], sample_trials_data, cv_folds=3
            )

            # Should have one failed result
            assert len(result["optimization_results"]) == 1
            assert not result["optimization_results"][0]["success"]
            assert result["best_result"] is None
            assert result["best_score"] == 0

    def test_generate_reports(self, aggregator, sample_combo_results, sample_combinations, sample_trials_data):
        """Test generating comprehensive reports."""
        optimization_results = [
            {
                "combination": {"model": "gpt-4", "prompt": "direct_mcode"},
                "success": True,
                "cv_average_score": 0.85,
                "timestamp": "2023-12-01T12:00:00"
            }
        ]

        with patch.object(aggregator.biological_analyzer, 'generate_biological_analysis_report') as mock_bio, \
             patch.object(aggregator.performance_analyzer, 'analyze_by_category') as mock_perf, \
             patch.object(aggregator.performance_analyzer, 'analyze_by_provider') as mock_provider, \
             patch.object(aggregator.performance_analyzer, 'summarize_errors') as mock_errors, \
             patch.object(aggregator.report_generator, 'generate_mega_report') as mock_mega, \
             patch('src.optimization.result_aggregator.datetime') as mock_datetime, \
             patch('builtins.open', create=True), \
             patch('pathlib.Path.mkdir'):

            # Setup mocks
            mock_perf.return_value = {"model_stats": "data"}
            mock_provider.return_value = {"provider_stats": "data"}
            mock_errors.return_value = {"error_analysis": "data"}
            mock_mega.return_value = "# Mega Report Content"
            mock_datetime.now.return_value.strftime.return_value = "20231201_120000"

            result = aggregator.generate_reports(
                optimization_results, sample_trials_data, sample_combo_results, sample_combinations
            )

            # Verify analyzers were called
            mock_bio.assert_called_once_with(sample_combo_results, sample_combinations, sample_trials_data)
            mock_perf.assert_called_once_with(optimization_results, "model")
            mock_provider.assert_called_once_with(optimization_results)
            mock_errors.assert_called_once_with(optimization_results)

            # Verify result structure
            assert "model_stats" in result
            assert "provider_stats" in result
            assert "error_analysis" in result
            assert result["total_runs"] == 1
            assert result["successful_runs"] == 1
            assert "time_range" in result

    def test_generate_reports_mega_report_failure(self, aggregator):
        """Test generate_reports handles mega report generation failure."""
        optimization_results = [{"success": True, "timestamp": "2023-12-01T12:00:00"}]

        with patch.object(aggregator.biological_analyzer, 'generate_biological_analysis_report'), \
             patch.object(aggregator.performance_analyzer, 'analyze_by_category'), \
             patch.object(aggregator.performance_analyzer, 'analyze_by_provider'), \
             patch.object(aggregator.performance_analyzer, 'summarize_errors'), \
             patch.object(aggregator.report_generator, 'generate_mega_report', side_effect=Exception("Report failed")):

            # Should not raise exception
            result = aggregator.generate_reports(optimization_results, [], {}, [])

            assert "model_stats" in result

    def test_log_performance_analysis_provider_rankings(self, aggregator):
        """Test logging performance analysis with provider rankings."""
        model_analysis = {"gpt-4": {"success_rate": 0.9, "avg_score": 0.85, "avg_processing_time": 2.5, "avg_cost": 0.02, "runs": 10}}
        prompt_analysis = {"direct_mcode": {"success_rate": 0.8, "avg_score": 0.82, "avg_processing_time": 3.0, "runs": 8}}
        provider_analysis = {
            "openai": {"success_rate": 0.95, "avg_score": 0.88, "avg_processing_time": 2.0, "avg_cost": 0.015, "models": ["gpt-4"]},
            "anthropic": {"success_rate": 0.85, "avg_score": 0.78, "avg_processing_time": 3.5, "avg_cost": 0.025, "models": ["claude-3"]}
        }
        all_results = [{"success": True}]

        aggregator.log_performance_analysis(model_analysis, prompt_analysis, provider_analysis, all_results)

        # Verify logger calls
        aggregator.logger.info.assert_called()

        # Check that provider rankings were logged
        calls = [call.args[0] for call in aggregator.logger.info.call_args_list]
        assert any("PROVIDER RANKINGS" in call for call in calls)
        assert any("MODEL RANKINGS" in call for call in calls)
        assert any("PROMPT RANKINGS" in call for call in calls)

    def test_log_performance_analysis_empty_data(self, aggregator):
        """Test logging performance analysis with empty data."""
        aggregator.log_performance_analysis({}, {}, {}, [])

        # Should not raise exception
        aggregator.logger.info.assert_called()

    def test_log_performance_analysis_error_analysis(self, aggregator):
        """Test logging performance analysis with error analysis."""
        model_analysis = {"gpt-4": {"success_rate": 0.8, "error_count": 2, "error_types": {"timeout": 1, "api_error": 1}, "runs": 10}}
        prompt_analysis = {}
        provider_analysis = {"openai": {"success_rate": 0.9}}
        all_results = [{"success": False, "error": "timeout"}]

        with patch.object(aggregator.performance_analyzer, 'summarize_errors', return_value={"timeout": 5, "api_error": 3}):
            aggregator.log_performance_analysis(model_analysis, prompt_analysis, provider_analysis, all_results)

            # Verify error analysis was logged
            calls = [call.args[0] for call in aggregator.logger.info.call_args_list]
            assert any("COMPREHENSIVE ERROR ANALYSIS" in call for call in calls)
            assert any("Model Reliability" in call for call in calls)

    def test_log_performance_analysis_performance_insights(self, aggregator):
        """Test logging performance analysis with performance insights."""
        model_analysis = {}
        prompt_analysis = {}
        provider_analysis = {
            "fast_provider": {"success_rate": 0.9, "avg_processing_time": 1.0, "avg_cost": 0.01},
            "slow_provider": {"success_rate": 0.8, "avg_processing_time": 5.0, "avg_cost": 0.05}
        }
        all_results = []

        aggregator.log_performance_analysis(model_analysis, prompt_analysis, provider_analysis, all_results)

        # Verify performance insights were logged
        calls = [call.args[0] for call in aggregator.logger.info.call_args_list]
        assert any("PERFORMANCE INSIGHTS" in call for call in calls)
        assert any("Fastest provider" in call for call in calls)
        assert any("Slowest provider" in call for call in calls)

    def test_log_performance_analysis_cost_analysis(self, aggregator):
        """Test logging performance analysis with cost analysis."""
        model_analysis = {
            "cheap_model": {"avg_cost": 0.005, "success_rate": 0.8},
            "expensive_model": {"avg_cost": 0.05, "success_rate": 0.9}
        }
        prompt_analysis = {}
        provider_analysis = {}
        all_results = []

        aggregator.log_performance_analysis(model_analysis, prompt_analysis, provider_analysis, all_results)

        # Verify cost analysis was logged
        calls = [call.args[0] for call in aggregator.logger.info.call_args_list]
        assert any("Cheapest model" in call for call in calls)
        assert any("Most expensive model" in call for call in calls)

    def test_log_performance_analysis_reliability_insights(self, aggregator):
        """Test logging performance analysis with reliability insights."""
        model_analysis = {}
        prompt_analysis = {}
        provider_analysis = {
            "reliable_provider": {"success_rate": 0.95},
            "unreliable_provider": {"success_rate": 0.6}
        }
        all_results = []

        aggregator.log_performance_analysis(model_analysis, prompt_analysis, provider_analysis, all_results)

        # Verify reliability insights were logged
        calls = [call.args[0] for call in aggregator.logger.info.call_args_list]
        assert any("Most reliable provider" in call for call in calls)
        assert any("Least reliable provider" in call for call in calls)


if __name__ == "__main__":
    pytest.main([__file__])