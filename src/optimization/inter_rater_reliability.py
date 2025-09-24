#!/usr/bin/env python3
"""
Inter-Rater Reliability Analysis for mCODE Elements

This module implements comprehensive inter-rater reliability analysis for mCODE element extraction,
measuring agreement between different LLM models and prompts on the same clinical trial data.

Key metrics:
- Cohen's Kappa (pairwise agreement)
- Fleiss' Kappa (multi-rater agreement)
- Percentage agreement
- Element presence/absence agreement
- Values/codes agreement
- Confidence score agreement
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import pearsonr

from src.pipeline import McodePipeline
from src.shared.models import McodeElement
from src.utils.logging_config import get_logger, setup_logging


@dataclass
class RaterResult:
    """Result from a single rater (model+prompt combination) on a trial."""

    rater_id: str  # e.g., "deepseek-coder_direct_mcode_evidence_based_concise"
    trial_id: str
    mcode_elements: List[McodeElement] = field(default_factory=list)
    success: bool = False
    error_message: str = ""
    execution_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgreementMetrics:
    """Agreement metrics between raters for a specific element or aspect."""

    # Basic agreement
    percentage_agreement: float = 0.0
    cohens_kappa: float = 0.0
    fleiss_kappa: float = 0.0

    # Detailed breakdown
    total_items: int = 0
    agreed_items: int = 0
    disagreed_items: int = 0

    # Confidence intervals (if calculated)
    kappa_ci_lower: Optional[float] = None
    kappa_ci_upper: Optional[float] = None

    # Additional stats
    prevalence: float = 0.0  # Proportion of positive cases
    bias: float = 0.0  # Agreement bias


@dataclass
class InterRaterAnalysis:
    """Complete inter-rater reliability analysis for a set of trials."""

    trial_analyses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_metrics: Dict[str, AgreementMetrics] = field(default_factory=dict)
    rater_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    element_reliability: Dict[str, AgreementMetrics] = field(default_factory=dict)

    # Metadata
    num_raters: int = 0
    num_trials: int = 0
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class InterRaterReliabilityAnalyzer:
    """
    Comprehensive inter-rater reliability analyzer for mCODE elements.

    Measures agreement between different LLM models/prompts on:
    1. Presence/absence of mCODE elements
    2. Specific values and codes
    3. Confidence scores
    """

    def __init__(self, output_dir: str = "inter_rater_analysis_results"):
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Results storage
        self.rater_results: Dict[str, Dict[str, RaterResult]] = defaultdict(
            dict
        )  # trial_id -> rater_id -> result
        self.analysis_results: Optional[InterRaterAnalysis] = None

    def initialize(self) -> None:
        """Initialize the analyzer."""
        setup_logging("INFO")
        self.logger.info("🤝 Initializing Inter-Rater Reliability Analyzer")

    async def run_analysis(
        self,
        trials_data: List[Dict[str, Any]],
        rater_configs: List[Dict[str, str]],
        max_concurrent: int = 3,
    ) -> InterRaterAnalysis:
        """
        Run complete inter-rater reliability analysis.

        Args:
            trials_data: List of trial data dictionaries
            rater_configs: List of rater configurations [{'model': str, 'prompt': str}, ...]
            max_concurrent: Maximum concurrent API calls

        Returns:
            Complete inter-rater analysis results
        """
        self.logger.info("🔬 Starting inter-rater reliability analysis")
        self.logger.info(
            f"   📊 {len(trials_data)} trials × {len(rater_configs)} raters = {len(trials_data) * len(rater_configs)} total evaluations"
        )

        start_time = time.time()

        # Collect data from all raters
        await self._collect_rater_data(trials_data, rater_configs, max_concurrent)

        # Perform agreement analysis
        analysis = self._analyze_agreement()

        duration = time.time() - start_time
        self.logger.info(f"✅ Inter-rater analysis completed in {duration:.2f} seconds")

        self.analysis_results = analysis
        return analysis

    async def _collect_rater_data(
        self,
        trials_data: List[Dict[str, Any]],
        rater_configs: List[Dict[str, str]],
        max_concurrent: int,
    ) -> None:
        """Collect mCODE extraction results from all raters on all trials."""
        from src.utils.concurrency import AsyncQueue, create_task

        self.logger.info("📥 Collecting rater data...")

        # Create tasks for all rater-trial combinations
        tasks = []
        for trial in trials_data:
            trial_id = self._extract_trial_id(trial)
            for rater_config in rater_configs:
                rater_id = f"{rater_config['model']}_{rater_config['prompt']}"
                task = create_task(
                    task_id=f"{trial_id}_{rater_id}",
                    func=self._run_single_rater,
                    trial_data=trial,
                    rater_config=rater_config,
                    rater_id=rater_id,
                )
                tasks.append(task)

        # Create async queue and execute tasks
        queue = AsyncQueue(
            max_concurrent=max_concurrent, name="InterRaterDataCollection"
        )

        def progress_callback(completed, total, result):
            if completed % 5 == 0:
                self.logger.info(
                    f"📊 Progress: {completed}/{total} evaluations completed"
                )

        # Execute all tasks
        task_results = await queue.execute_tasks(
            tasks, progress_callback=progress_callback
        )

        # Process results
        for task_result in task_results:
            if task_result.success:
                # Extract trial_id and rater_id from task_id
                task_id_parts = task_result.task_id.split("_", 1)
                if len(task_id_parts) == 2:
                    trial_id, rater_id = task_id_parts
                    self.rater_results[trial_id][rater_id] = task_result.result
            else:
                self.logger.warning(
                    f"❌ Task {task_result.task_id} failed: {task_result.error}"
                )

        self.logger.info(
            f"📊 Collected data from {len(self.rater_results)} trials × {len(rater_configs)} raters"
        )

    def _run_single_rater(
        self, trial_data: Dict[str, Any], rater_config: Dict[str, str], rater_id: str
    ) -> RaterResult:
        """Run a single rater on a trial and collect results."""

        start_time = time.time()

        result = RaterResult(
            rater_id=rater_id, trial_id=self._extract_trial_id(trial_data)
        )

        try:
            # Create event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Initialize pipeline
            pipeline = McodePipeline(
                prompt_name=rater_config["prompt"], model_name=rater_config["model"]
            )

            # Run extraction
            pipeline_result = loop.run_until_complete(pipeline.process(trial_data))

            # Store results
            result.mcode_elements = pipeline_result.mcode_mappings
            result.success = True

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.debug(f"Failed rater {rater_id}: {e}")

        result.execution_time_seconds = time.time() - start_time
        return result

    def _analyze_agreement(self) -> InterRaterAnalysis:
        """Analyze agreement between raters across all collected data."""
        self.logger.info("🔍 Analyzing inter-rater agreement...")

        analysis = InterRaterAnalysis(
            num_raters=len(
                set(
                    r.rater_id
                    for trial_results in self.rater_results.values()
                    for r in trial_results.values()
                )
            ),
            num_trials=len(self.rater_results),
        )

        # Check if we have any data to analyze
        if not self.rater_results:
            self.logger.warning("No rater results available for analysis")
            return analysis

        # Analyze each trial
        for trial_id, trial_results in self.rater_results.items():
            trial_analysis = self._analyze_trial_agreement(trial_id, trial_results)
            analysis.trial_analyses[trial_id] = trial_analysis

        # Calculate overall metrics
        analysis.overall_metrics = self._calculate_overall_metrics(analysis)

        # Analyze rater performance
        analysis.rater_performance = self._analyze_rater_performance()

        # Analyze element-level reliability
        analysis.element_reliability = self._analyze_element_reliability(analysis)

        return analysis

    def _analyze_trial_agreement(
        self, trial_id: str, trial_results: Dict[str, RaterResult]
    ) -> Dict[str, Any]:
        """Analyze agreement for a single trial."""
        successful_raters = {
            rid: result for rid, result in trial_results.items() if result.success
        }

        if len(successful_raters) < 2:
            return {"error": "Insufficient successful raters for agreement analysis"}

        # Presence/absence agreement
        presence_agreement = self._calculate_presence_agreement(successful_raters)

        # Values/codes agreement
        values_agreement = self._calculate_values_agreement(successful_raters)

        # Confidence agreement
        confidence_agreement = self._calculate_confidence_agreement(successful_raters)

        return {
            "num_raters": len(successful_raters),
            "presence_agreement": presence_agreement,
            "values_agreement": values_agreement,
            "confidence_agreement": confidence_agreement,
            "rater_ids": list(successful_raters.keys()),
        }

    def _calculate_presence_agreement(
        self, rater_results: Dict[str, RaterResult]
    ) -> AgreementMetrics:
        """Calculate agreement on element presence/absence."""
        rater_ids = list(rater_results.keys())

        # Get all unique element types across all raters
        all_element_types = set()
        for result in rater_results.values():
            all_element_types.update(
                elem.element_type for elem in result.mcode_elements
            )

        element_types = sorted(all_element_types)

        if len(element_types) == 0:
            return AgreementMetrics()

        # Create presence/absence matrix: rows = elements, columns = raters
        presence_matrix = np.zeros((len(element_types), len(rater_ids)))

        for elem_idx, elem_type in enumerate(element_types):
            for rater_idx, rater_id in enumerate(rater_ids):
                result = rater_results[rater_id]
                has_element = any(
                    elem.element_type == elem_type for elem in result.mcode_elements
                )
                presence_matrix[elem_idx, rater_idx] = 1 if has_element else 0

        # Calculate agreement metrics
        metrics = AgreementMetrics()

        # Percentage agreement
        total_agreements = 0
        total_comparisons = 0

        for i in range(len(element_types)):
            row = presence_matrix[i, :]
            if np.sum(row) == 0 or np.sum(row) == len(rater_ids):
                total_agreements += 1
            total_comparisons += 1

        metrics.percentage_agreement = (
            total_agreements / total_comparisons if total_comparisons > 0 else 0
        )

        # Fleiss' Kappa for multiple raters
        if len(rater_ids) > 1:
            metrics.fleiss_kappa = self._calculate_fleiss_kappa(presence_matrix)

        # Cohen's Kappa (pairwise average)
        if len(rater_ids) == 2:
            metrics.cohens_kappa = self._calculate_cohens_kappa(
                presence_matrix[:, 0], presence_matrix[:, 1]
            )

        metrics.total_items = len(element_types)
        metrics.agreed_items = total_agreements
        metrics.disagreed_items = total_comparisons - total_agreements

        return metrics

    def _calculate_values_agreement(
        self, rater_results: Dict[str, RaterResult]
    ) -> Dict[str, float]:
        """Calculate agreement on element values and codes."""
        # Group elements by type
        elements_by_type: Dict[str, Dict[str, List[McodeElement]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for rater_id, result in rater_results.items():
            for elem in result.mcode_elements:
                elements_by_type[elem.element_type][rater_id].append(elem)

        agreement_scores = {}

        for elem_type, rater_elements in elements_by_type.items():
            if len(rater_elements) < 2:
                continue

            # For each element type, compare values across raters
            # This is simplified - in practice, we'd need to match elements better
            exact_matches = 0
            total_comparisons = 0

            # Compare all pairs of raters
            rater_ids = list(rater_elements.keys())
            for i in range(len(rater_ids)):
                for j in range(i + 1, len(rater_ids)):
                    rater_i = rater_ids[i]
                    rater_j = rater_ids[j]

                    elems_i = rater_elements[rater_i]
                    elems_j = rater_elements[rater_j]

                    # Simple comparison: check if any elements match exactly
                    for elem_i in elems_i:
                        for elem_j in elems_j:
                            total_comparisons += 1
                            if (
                                elem_i.code == elem_j.code
                                and elem_i.display == elem_j.display
                            ):
                                exact_matches += 1
                                break

            agreement_scores[elem_type] = (
                exact_matches / total_comparisons if total_comparisons > 0 else 0
            )

        # Overall values agreement
        if agreement_scores:
            overall_agreement = sum(agreement_scores.values()) / len(agreement_scores)
        else:
            overall_agreement = 0.0

        return {
            "overall_values_agreement": overall_agreement,
            "element_type_agreements": agreement_scores,
        }

    def _calculate_confidence_agreement(
        self, rater_results: Dict[str, RaterResult]
    ) -> Dict[str, Any]:
        """Calculate agreement on confidence scores."""
        # Group confidence scores by element type
        confidence_by_type: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for rater_id, result in rater_results.items():
            for elem in result.mcode_elements:
                confidence_by_type[elem.element_type][rater_id].append(
                    elem.confidence_score or 0.0
                )

        agreement_stats = {}

        for elem_type, rater_confidences in confidence_by_type.items():
            if len(rater_confidences) < 2:
                continue

            # Calculate correlation between rater confidence scores
            rater_ids = list(rater_confidences.keys())
            correlations = []

            for i in range(len(rater_ids)):
                for j in range(i + 1, len(rater_ids)):
                    conf_i = rater_confidences[rater_ids[i]]
                    conf_j = rater_confidences[rater_ids[j]]

                    if len(conf_i) > 1 and len(conf_j) > 1:
                        try:
                            corr, _ = pearsonr(conf_i, conf_j)
                            correlations.append(corr)
                        except:
                            pass

            if correlations:
                agreement_stats[elem_type] = {
                    "mean_correlation": statistics.mean(correlations),
                    "correlations": correlations,
                }

        # Overall confidence agreement
        if agreement_stats:
            overall_corr = statistics.mean(
                stats["mean_correlation"] for stats in agreement_stats.values()
            )
        else:
            overall_corr = 0.0

        return {
            "overall_confidence_agreement": overall_corr,
            "element_type_stats": agreement_stats,
        }

    def _calculate_cohens_kappa(
        self, ratings1: np.ndarray, ratings2: np.ndarray
    ) -> float:
        """Calculate Cohen's Kappa for two raters."""
        if len(ratings1) != len(ratings2):
            return 0.0

        # Confusion matrix
        a = np.sum((ratings1 == 1) & (ratings2 == 1))  # Both positive
        b = np.sum(
            (ratings1 == 1) & (ratings2 == 0)
        )  # Rater1 positive, Rater2 negative
        c = np.sum(
            (ratings1 == 0) & (ratings2 == 1)
        )  # Rater1 negative, Rater2 positive
        d = np.sum((ratings1 == 0) & (ratings2 == 0))  # Both negative

        total = a + b + c + d
        if total == 0:
            return 0.0

        # Observed agreement
        p_o = (a + d) / total

        # Expected agreement
        p_e = ((a + b) * (a + c) + (c + d) * (b + d)) / (total * total)

        if p_e == 1.0:
            return 0.0

        kappa = (p_o - p_e) / (1 - p_e)
        return kappa

    def _calculate_fleiss_kappa(self, rating_matrix: np.ndarray) -> float:
        """
        Calculate Fleiss' Kappa for multiple raters.

        Args:
            rating_matrix: Matrix of shape (n_items, n_raters) with 0/1 values

        Returns:
            Fleiss' Kappa value
        """
        n_items, n_raters = rating_matrix.shape

        if n_raters < 2 or n_items == 0:
            return 0.0

        # Calculate P_i (agreement for each item)
        P_i = np.zeros(n_items)
        for i in range(n_items):
            ratings = rating_matrix[i, :]
            P_i[i] = (np.sum(ratings) / n_raters) ** 2 + (
                (n_raters - np.sum(ratings)) / n_raters
            ) ** 2

        # Overall observed agreement
        P_bar = np.mean(P_i)

        # Expected agreement
        p_bar = np.mean(rating_matrix)  # Overall proportion of positive ratings
        P_e_bar = p_bar**2 + (1 - p_bar) ** 2

        if P_e_bar == 1.0:
            return 0.0

        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        return kappa

    def _calculate_overall_metrics(
        self, analysis: InterRaterAnalysis
    ) -> Dict[str, AgreementMetrics]:
        """Calculate overall agreement metrics across all trials."""
        # Aggregate presence/absence metrics
        presence_metrics = self._aggregate_presence_metrics(analysis)

        # Aggregate values agreement
        values_metrics = self._aggregate_values_metrics(analysis)

        # Aggregate confidence agreement
        confidence_metrics = self._aggregate_confidence_metrics(analysis)

        return {
            "presence_agreement": presence_metrics,
            "values_agreement": AgreementMetrics(
                percentage_agreement=values_metrics.get("overall", 0.0)
            ),
            "confidence_agreement": AgreementMetrics(
                percentage_agreement=confidence_metrics.get("overall", 0.0)
            ),
        }

    def _aggregate_presence_metrics(
        self, analysis: InterRaterAnalysis
    ) -> AgreementMetrics:
        """Aggregate presence/absence agreement across all trials."""
        all_kappas = []
        all_percentages = []

        for trial_analysis in analysis.trial_analyses.values():
            if "presence_agreement" in trial_analysis:
                metrics = trial_analysis["presence_agreement"]
                if hasattr(metrics, "fleiss_kappa") and metrics.fleiss_kappa != 0:
                    all_kappas.append(metrics.fleiss_kappa)
                elif hasattr(metrics, "cohens_kappa") and metrics.cohens_kappa != 0:
                    all_kappas.append(metrics.cohens_kappa)

                if metrics.percentage_agreement > 0:
                    all_percentages.append(metrics.percentage_agreement)

        aggregated = AgreementMetrics()

        if all_kappas:
            aggregated.fleiss_kappa = statistics.mean(all_kappas)
        if all_percentages:
            aggregated.percentage_agreement = statistics.mean(all_percentages)

        return aggregated

    def _aggregate_values_metrics(
        self, analysis: InterRaterAnalysis
    ) -> Dict[str, float]:
        """Aggregate values agreement across all trials."""
        all_overall = []

        for trial_analysis in analysis.trial_analyses.values():
            if "values_agreement" in trial_analysis:
                overall = trial_analysis["values_agreement"].get(
                    "overall_values_agreement", 0
                )
                if overall > 0:
                    all_overall.append(overall)

        return {"overall": statistics.mean(all_overall) if all_overall else 0.0}

    def _aggregate_confidence_metrics(
        self, analysis: InterRaterAnalysis
    ) -> Dict[str, float]:
        """Aggregate confidence agreement across all trials."""
        all_overall = []

        for trial_analysis in analysis.trial_analyses.values():
            if "confidence_agreement" in trial_analysis:
                overall = trial_analysis["confidence_agreement"].get(
                    "overall_confidence_agreement", 0
                )
                if not np.isnan(overall):
                    all_overall.append(overall)

        return {"overall": statistics.mean(all_overall) if all_overall else 0.0}

    def _analyze_rater_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance characteristics of each rater."""
        rater_stats = defaultdict(
            lambda: {
                "trials_processed": 0,
                "success_rate": 0.0,
                "avg_elements": 0.0,
                "avg_execution_time": 0.0,
                "errors": [],
            }
        )

        # Collect stats for each rater
        for trial_results in self.rater_results.values():
            for rater_id, result in trial_results.items():
                stats = rater_stats[rater_id]
                stats["trials_processed"] += 1

                if result.success:
                    stats["success_rate"] = (
                        (stats["success_rate"] * (stats["trials_processed"] - 1)) + 1
                    ) / stats["trials_processed"]
                    stats["avg_elements"] = (
                        (stats["avg_elements"] * (stats["trials_processed"] - 1))
                        + len(result.mcode_elements)
                    ) / stats["trials_processed"]
                    stats["avg_execution_time"] = (
                        (stats["avg_execution_time"] * (stats["trials_processed"] - 1))
                        + result.execution_time_seconds
                    ) / stats["trials_processed"]
                else:
                    stats["success_rate"] = (
                        stats["success_rate"] * (stats["trials_processed"] - 1)
                    ) / stats["trials_processed"]
                    stats["errors"].append(result.error_message)

        return dict(rater_stats)

    def _analyze_element_reliability(
        self, analysis: InterRaterAnalysis
    ) -> Dict[str, AgreementMetrics]:
        """Analyze reliability for each element type across all trials."""
        defaultdict(list)

        # Collect agreement data for each element type
        for trial_analysis in analysis.trial_analyses.values():
            if "presence_agreement" in trial_analysis:
                # This is simplified - in practice, we'd track per-element agreement
                pass

        # For now, return empty dict - this would need more sophisticated tracking
        return {}

    def _extract_trial_id(self, trial_data: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        try:
            return trial_data["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"trial_{hash(str(trial_data)) % 10000}"

    def save_results(self) -> None:
        """Save inter-rater reliability analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw rater results
        rater_results_file = (
            self.output_dir / f"inter_rater_rater_results_{timestamp}.json"
        )
        with open(rater_results_file, "w") as f:
            # Convert to serializable format
            serializable_results = {}
            for trial_id, trial_results in self.rater_results.items():
                serializable_results[trial_id] = {}
                for rater_id, result in trial_results.items():
                    serializable_results[trial_id][rater_id] = {
                        "rater_id": result.rater_id,
                        "trial_id": result.trial_id,
                        "mcode_elements": [
                            elem.model_dump() for elem in result.mcode_elements
                        ],
                        "success": result.success,
                        "error_message": result.error_message,
                        "execution_time_seconds": result.execution_time_seconds,
                        "timestamp": result.timestamp,
                    }

            json.dump(serializable_results, f, indent=2)

        # Save analysis results
        analysis_file = self.output_dir / f"inter_rater_analysis_{timestamp}.json"
        with open(analysis_file, "w") as f:
            # Convert to serializable format
            serializable_analysis = {
                "trial_analyses": {},
                "overall_metrics": {},
                "rater_performance": self.analysis_results.rater_performance,
                "element_reliability": {},
                "num_raters": self.analysis_results.num_raters,
                "num_trials": self.analysis_results.num_trials,
                "analysis_timestamp": self.analysis_results.analysis_timestamp,
            }

            # Convert trial analyses to serializable format
            for (
                trial_id,
                trial_analysis,
            ) in self.analysis_results.trial_analyses.items():
                serializable_trial = {}
                for key, value in trial_analysis.items():
                    if hasattr(value, "__dict__"):  # AgreementMetrics object
                        serializable_trial[key] = {
                            "percentage_agreement": value.percentage_agreement,
                            "cohens_kappa": value.cohens_kappa,
                            "fleiss_kappa": value.fleiss_kappa,
                            "total_items": value.total_items,
                            "agreed_items": value.agreed_items,
                            "disagreed_items": value.disagreed_items,
                        }
                    else:
                        serializable_trial[key] = value
                serializable_analysis["trial_analyses"][trial_id] = serializable_trial

            # Convert AgreementMetrics to dict
            for key, metrics in self.analysis_results.overall_metrics.items():
                serializable_analysis["overall_metrics"][key] = {
                    "percentage_agreement": metrics.percentage_agreement,
                    "cohens_kappa": metrics.cohens_kappa,
                    "fleiss_kappa": metrics.fleiss_kappa,
                    "total_items": metrics.total_items,
                    "agreed_items": metrics.agreed_items,
                    "disagreed_items": metrics.disagreed_items,
                }

            json.dump(serializable_analysis, f, indent=2)

        self.logger.info(f"💾 Inter-rater results saved to {self.output_dir}")

    def generate_report(self) -> str:
        """Generate a comprehensive markdown report."""
        if not self.analysis_results:
            return "No analysis results available"

        report = f"""# Inter-Rater Reliability Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Trials Analyzed:** {self.analysis_results.num_trials}
- **Raters Compared:** {self.analysis_results.num_raters}
- **Total Evaluations:** {self.analysis_results.num_trials * self.analysis_results.num_raters}

## Overall Agreement Metrics

"""

        # Overall metrics
        for metric_name, metrics in self.analysis_results.overall_metrics.items():
            report += f"### {metric_name.replace('_', ' ').title()}\n"
            report += (
                f"- **Percentage Agreement:** {metrics.percentage_agreement:.3f}\n"
            )
            if metrics.fleiss_kappa != 0:
                report += f"- **Fleiss' Kappa:** {metrics.fleiss_kappa:.3f}\n"
            if metrics.cohens_kappa != 0:
                report += f"- **Cohen's Kappa:** {metrics.cohens_kappa:.3f}\n"
            report += f"- **Items Analyzed:** {metrics.total_items}\n"
            report += f"- **Agreed Items:** {metrics.agreed_items}\n"
            report += f"- **Disagreed Items:** {metrics.disagreed_items}\n\n"

        # Rater performance
        report += "## Rater Performance\n\n"
        report += "| Rater | Success Rate | Avg Elements | Avg Time (s) |\n"
        report += "|-------|-------------|--------------|--------------|\n"

        for rater_id, stats in self.analysis_results.rater_performance.items():
            report += f"| {rater_id} | {stats['success_rate']:.1%} | {stats['avg_elements']:.1f} | {stats['avg_execution_time']:.1f} |\n"

        report += "\n## Trial-Level Analysis\n\n"

        for trial_id, trial_analysis in self.analysis_results.trial_analyses.items():
            report += f"### Trial {trial_id}\n"
            report += f"- **Raters:** {trial_analysis.get('num_raters', 0)}\n"

            if "presence_agreement" in trial_analysis:
                presence = trial_analysis["presence_agreement"]
                report += (
                    f"- **Presence Agreement:** {presence.percentage_agreement:.3f}\n"
                )
                if presence.fleiss_kappa != 0:
                    report += f"- **Fleiss' Kappa:** {presence.fleiss_kappa:.3f}\n"

            report += "\n"

        report += "---\n*Generated by Inter-Rater Reliability Analyzer*"

        return report
