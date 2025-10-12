"""
Unit tests for metrics module.
"""

from unittest.mock import patch

import pytest

from src.utils.metrics import BenchmarkMetrics, MatchingMetrics, PerformanceMetrics


class TestMatchingMetrics:
    """Test MatchingMetrics functionality."""

    def test_init_and_reset(self):
        """Test initialization and reset."""
        metrics = MatchingMetrics()
        assert metrics.total_patients == 0
        assert metrics.total_trials == 0
        assert metrics.total_matches == 0

        # Add some data
        metrics.total_patients = 10
        metrics.total_matches = 5

        # Reset
        metrics.reset()
        assert metrics.total_patients == 0
        assert metrics.total_matches == 0

    def test_record_match_biomarker(self):
        """Test recording biomarker matches."""
        metrics = MatchingMetrics()

        match_reasons = ["Biomarker match: HER2 (positive)"]
        genomic_variants = []

        metrics.record_match(match_reasons, genomic_variants)

        assert metrics.total_matches == 1
        assert metrics.match_reasons["Biomarker match: HER2 (positive)"] == 1
        assert metrics.biomarker_match_counts["HER2"] == 1

    def test_record_match_variant(self):
        """Test recording variant matches."""
        metrics = MatchingMetrics()

        match_reasons = ["Variant match: BRCA1 c.68_69delAG"]
        genomic_variants = []

        metrics.record_match(match_reasons, genomic_variants)

        assert metrics.total_matches == 1
        assert metrics.match_reasons["Variant match: BRCA1 c.68_69delAG"] == 1
        assert metrics.gene_match_counts["BRCA1"] == 1

    def test_record_match_stage(self):
        """Test recording stage matches."""
        metrics = MatchingMetrics()

        match_reasons = ["Stage match: T2N0M0", "Stage compatible: Early stage"]
        genomic_variants = []

        metrics.record_match(match_reasons, genomic_variants)

        assert metrics.total_matches == 1
        assert metrics.match_reasons["Stage match: T2N0M0"] == 1
        assert metrics.match_reasons["Stage compatible: Early stage"] == 1
        assert metrics.stage_match_counts["T2N0M0"] == 1
        assert metrics.stage_match_counts["Early"] == 1

    def test_record_match_treatments(self):
        """Test recording treatment matches."""
        metrics = MatchingMetrics()

        match_reasons = ["Shared treatments: chemotherapy, hormone therapy"]
        genomic_variants = []

        metrics.record_match(match_reasons, genomic_variants)

        assert metrics.total_matches == 1
        assert metrics.match_reasons["Shared treatments: chemotherapy, hormone therapy"] == 1
        assert metrics.treatment_match_counts["chemotherapy"] == 1
        assert metrics.treatment_match_counts["hormone therapy"] == 1

    def test_get_summary(self):
        """Test getting metrics summary."""
        metrics = MatchingMetrics()
        metrics.total_patients = 100
        metrics.total_trials = 50
        metrics.total_matches = 25

        # Add some match data
        metrics.record_match(["Biomarker match: HER2 (positive)"], [])
        metrics.record_match(["Variant match: BRCA1 mutation"], [])
        metrics.record_match(["Stage match: T1N0M0"], [])

        summary = metrics.get_summary()

        assert summary["total_patients"] == 100
        assert summary["total_trials"] == 50
        assert summary["total_matches"] == 28  # 25 + 3 recorded
        assert summary["match_rate"] == 28 / (100 * 50)

        # Check top items are sorted
        assert len(summary["top_match_reasons"]) <= 5
        assert len(summary["top_genes"]) <= 5
        assert len(summary["top_biomarkers"]) <= 5

    @patch("src.utils.metrics.get_logger")
    def test_log_summary(self, mock_get_logger):
        """Test logging summary."""
        mock_logger = mock_get_logger.return_value
        metrics = MatchingMetrics()
        metrics.total_patients = 10
        metrics.total_trials = 5
        metrics.total_matches = 3

        metrics.log_summary()

        # Verify logger calls
        mock_logger.info.assert_called()


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics functionality."""

    def test_init(self):
        """Test initialization."""
        metrics = BenchmarkMetrics()
        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 0

        metrics = BenchmarkMetrics(5, 2, 1)
        assert metrics.tp == 5
        assert metrics.fp == 2
        assert metrics.fn == 1

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = BenchmarkMetrics(10, 2, 3)  # tp, fp, fn

        result = metrics.calculate_metrics()

        expected_precision = 10 / (10 + 2)  # 10/12
        expected_recall = 10 / (10 + 3)  # 10/13
        expected_f1 = (
            2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        )

        assert result["precision"] == expected_precision
        assert result["recall"] == expected_recall
        assert result["f1_score"] == expected_f1

    def test_calculate_metrics_edge_cases(self):
        """Test metrics calculation edge cases."""
        # No true positives
        metrics = BenchmarkMetrics(0, 5, 5)  # tp, fp, fn
        result = metrics.calculate_metrics()
        assert result["precision"] == 0
        assert result["recall"] == 0
        assert result["f1_score"] == 0

        # No false positives
        metrics = BenchmarkMetrics(5, 0, 5)  # tp, fp, fn
        result = metrics.calculate_metrics()
        assert result["precision"] == 1.0
        assert result["recall"] == 5 / 10
        assert result["f1_score"] > 0

        # No false negatives
        metrics = BenchmarkMetrics(5, 5, 0)  # tp, fp, fn
        result = metrics.calculate_metrics()
        assert result["precision"] == 5 / 10
        assert result["recall"] == 1.0
        assert result["f1_score"] > 0

    def test_compare_mcode_elements(self):
        """Test comparing mCODE elements."""
        predicted = [
            {"element_type": "CancerCondition", "code": "C123"},
            {"element_type": "TNMStageGroup", "code": "T2N0M0"},
            {"element_type": "Procedure", "code": "P456"},
        ]

        ground_truth = [
            {"element_type": "CancerCondition", "code": "C123"},
            {"element_type": "TNMStageGroup", "code": "T2N0M0"},
            {"element_type": "MedicationRequest", "code": "M789"},
        ]

        metrics = BenchmarkMetrics.compare_mcode_elements(predicted, ground_truth)

        # 2 matches (CancerCondition, TNMStageGroup), 1 FP (Procedure), 1 FN (MedicationRequest)
        assert metrics.tp == 2
        assert metrics.fp == 1
        assert metrics.fn == 1

    def test_compare_mcode_elements_empty(self):
        """Test comparing with empty lists."""
        predicted = []
        ground_truth = [{"element_type": "CancerCondition", "code": "C123"}]

        metrics = BenchmarkMetrics.compare_mcode_elements(predicted, ground_truth)

        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 1

    def test_compare_mcode_elements_identical(self):
        """Test comparing identical lists."""
        elements = [{"element_type": "CancerCondition", "code": "C123"}]

        metrics = BenchmarkMetrics.compare_mcode_elements(elements, elements)

        assert metrics.tp == 1
        assert metrics.fp == 0
        assert metrics.fn == 0


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""

    def test_init_and_reset(self):
        """Test initialization and reset."""
        metrics = PerformanceMetrics()
        assert metrics.start_time is None
        assert metrics.end_time is None
        assert metrics.processing_time == 0.0
        assert metrics.tokens_used == 0
        assert metrics.estimated_cost == 0.0
        assert metrics.elements_processed == 0

    def test_start_stop_tracking(self):
        """Test start and stop tracking."""
        metrics = PerformanceMetrics()

        metrics.start_tracking()
        assert metrics.start_time is not None

        import time

        time.sleep(0.01)  # Small delay

        metrics.stop_tracking(tokens_used=1000, elements_processed=10)

        assert metrics.end_time is not None
        assert metrics.processing_time > 0
        assert metrics.tokens_used == 1000
        assert metrics.elements_processed == 10
        assert metrics.estimated_cost == (1000 / 1000) * 0.01  # 0.01 per 1K tokens

    def test_stop_tracking_without_start(self):
        """Test stop tracking without starting."""
        metrics = PerformanceMetrics()

        # Should not crash
        metrics.stop_tracking(500, 5)

        assert metrics.processing_time == 0.0

    def test_get_metrics(self):
        """Test getting metrics."""
        metrics = PerformanceMetrics()
        metrics.start_tracking()
        import time

        time.sleep(0.01)
        metrics.stop_tracking(tokens_used=2000, elements_processed=20)

        result = metrics.get_metrics()

        assert "processing_time_seconds" in result
        assert "tokens_used" in result
        assert "estimated_cost_usd" in result
        assert "elements_processed" in result
        assert "start_time" in result
        assert "end_time" in result

        # Derived metrics
        assert "processing_time_per_element" in result
        assert "tokens_per_element" in result
        assert "cost_per_element_usd" in result
        assert "elements_per_second" in result
        assert "tokens_per_second" in result

        # Check derived calculations
        assert result["processing_time_per_element"] == result["processing_time_seconds"] / 20
        assert result["tokens_per_element"] == 2000 / 20
        assert result["cost_per_element_usd"] == result["estimated_cost_usd"] / 20
        assert result["elements_per_second"] == 20 / result["processing_time_seconds"]
        assert result["tokens_per_second"] == 2000 / result["processing_time_seconds"]

    def test_get_metrics_no_elements(self):
        """Test getting metrics with no elements processed."""
        metrics = PerformanceMetrics()
        metrics.start_tracking()
        import time

        time.sleep(0.01)
        metrics.stop_tracking(tokens_used=1000, elements_processed=0)

        result = metrics.get_metrics()

        # Should handle division by zero gracefully
        assert result["processing_time_per_element"] == result["processing_time_seconds"] / 1
        assert result["tokens_per_element"] == 1000 / 1
        assert result["cost_per_element_usd"] == result["estimated_cost_usd"] / 1
        assert result["elements_per_second"] == 0 / result["processing_time_seconds"]
        assert result["tokens_per_second"] == 1000 / result["processing_time_seconds"]

    def test_to_dict(self):
        """Test to_dict method."""
        metrics = PerformanceMetrics()
        metrics.start_tracking()
        metrics.stop_tracking(500, 5)

        result = metrics.to_dict()
        expected = metrics.get_metrics()

        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__])
