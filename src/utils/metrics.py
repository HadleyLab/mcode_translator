from collections import defaultdict
from typing import Any, Dict, List
import json
import time
from datetime import datetime

from src.utils.logging_config import get_logger


class MatchingMetrics:
    """
    Tracks metrics for breast cancer patient-trial matching
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.total_patients = 0
        self.total_trials = 0
        self.total_matches = 0
        self.match_reasons = defaultdict(int)
        self.gene_match_counts = defaultdict(int)
        self.biomarker_match_counts = defaultdict(int)
        self.stage_match_counts = defaultdict(int)
        self.treatment_match_counts = defaultdict(int)

    def record_match(self, match_reasons: List[str], genomic_variants: List[Dict]):
        """Record metrics for a successful match"""
        self.total_matches += 1

        for reason in match_reasons:
            self.match_reasons[reason] += 1

            # Track specific match types
            if reason.startswith("Biomarker match:"):
                bio_name = reason.split(":")[1].split("(")[0].strip()
                self.biomarker_match_counts[bio_name] += 1
            elif reason.startswith("Variant match:"):
                gene = reason.split(":")[1].split()[0].strip()
                self.gene_match_counts[gene] += 1
            elif reason.startswith("Stage match:") or reason.startswith(
                "Stage compatible:"
            ):
                stage = reason.split(":")[1].split()[0].strip()
                self.stage_match_counts[stage] += 1
            elif reason.startswith("Shared treatments:"):
                treatments = reason.split(":")[1].strip()
                for treatment in treatments.split(","):
                    self.treatment_match_counts[treatment.strip()] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of metrics"""
        return {
            "total_patients": self.total_patients,
            "total_trials": self.total_trials,
            "total_matches": self.total_matches,
            "match_rate": self.total_matches
            / max(1, self.total_patients * self.total_trials),
            "top_match_reasons": sorted(
                self.match_reasons.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "top_genes": sorted(
                self.gene_match_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "top_biomarkers": sorted(
                self.biomarker_match_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "top_stages": sorted(
                self.stage_match_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "top_treatments": sorted(
                self.treatment_match_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def log_summary(self):
        """Log metrics summary"""
        summary = self.get_summary()
        self.logger.info("Matching Metrics Summary:")
        self.logger.info(f"Patients processed: {summary['total_patients']}")
        self.logger.info(f"Trials processed: {summary['total_trials']}")
        self.logger.info(f"Total matches: {summary['total_matches']}")
        self.logger.info(f"Match rate: {summary['match_rate']:.2%}")

        self.logger.info("Top match reasons:")
        for reason, count in summary["top_match_reasons"]:
            self.logger.info(f"  - {reason}: {count}")

        self.logger.info("Top matching genes:")
        for gene, count in summary["top_genes"]:
            self.logger.info(f"  - {gene}: {count}")

        self.logger.info("Top matching biomarkers:")
        for biomarker, count in summary["top_biomarkers"]:
            self.logger.info(f"  - {biomarker}: {count}")

        self.logger.info("Top matching stages:")
        for stage, count in summary["top_stages"]:
            self.logger.info(f"  - {stage}: {count}")

        self.logger.info("Top matching treatments:")
        for treatment, count in summary["top_treatments"]:
            self.logger.info(f"  - {treatment}: {count}")


class BenchmarkMetrics:
    """Calculates and stores precision, recall, and F1-score for benchmark validation."""

    def __init__(
        self,
        true_positives: int = 0,
        false_positives: int = 0,
        false_negatives: int = 0,
    ):
        self.tp = true_positives
        self.fp = false_positives
        self.fn = false_negatives

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculates precision, recall, and F1-score.

        Returns:
            A dictionary containing precision, recall, and F1-score.
        """
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {"precision": precision, "recall": recall, "f1_score": f1}

    @staticmethod
    def compare_mcode_elements(
        predicted: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> "BenchmarkMetrics":
        """
        Compares predicted mCODE elements against ground truth and calculates metrics.

        Args:
            predicted: A list of predicted mCODE elements.
            ground_truth: A list of ground truth mCODE elements.

        Returns:
            A BenchmarkMetrics object with calculated TP, FP, and FN.
        """
        predicted_set = {json.dumps(d, sort_keys=True) for d in predicted}
        ground_truth_set = {json.dumps(d, sort_keys=True) for d in ground_truth}

        tp = len(predicted_set.intersection(ground_truth_set))
        fp = len(predicted_set - ground_truth_set)
        fn = len(ground_truth_set - predicted_set)

        return BenchmarkMetrics(tp, fp, fn)


class PerformanceMetrics:
    """
    Tracks performance metrics including time, tokens, and cost for optimization workflows.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all performance metrics"""
        self.start_time = None
        self.end_time = None
        self.processing_time = 0.0
        self.tokens_used = 0
        self.estimated_cost = 0.0
        self.elements_processed = 0

    def start_tracking(self):
        """Start performance tracking"""
        self.start_time = time.time()

    def stop_tracking(self, tokens_used: int = 0, elements_processed: int = 0):
        """Stop performance tracking and record metrics"""
        if self.start_time is None:
            return

        self.end_time = time.time()
        self.processing_time = self.end_time - self.start_time
        self.tokens_used = tokens_used
        self.elements_processed = elements_processed

        # Calculate estimated cost (rough approximation)
        # OpenAI pricing: ~$0.01 per 1K tokens for GPT-4, ~$0.002 per 1K for GPT-3.5
        cost_per_1k_tokens = 0.01  # Default to GPT-4 pricing
        self.estimated_cost = (tokens_used / 1000) * cost_per_1k_tokens

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "processing_time_seconds": self.processing_time,
            "tokens_used": self.tokens_used,
            "estimated_cost_usd": self.estimated_cost,
            "elements_processed": self.elements_processed,
            "start_time": (
                datetime.fromtimestamp(self.start_time).isoformat()
                if self.start_time
                else None
            ),
            "end_time": (
                datetime.fromtimestamp(self.end_time).isoformat()
                if self.end_time
                else None
            ),
            # Derived metrics
            "processing_time_per_element": self.processing_time
            / max(self.elements_processed, 1),
            "tokens_per_element": self.tokens_used / max(self.elements_processed, 1),
            "cost_per_element_usd": self.estimated_cost
            / max(self.elements_processed, 1),
            "elements_per_second": self.elements_processed
            / max(self.processing_time, 0.001),
            "tokens_per_second": self.tokens_used / max(self.processing_time, 0.001),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.get_metrics()
