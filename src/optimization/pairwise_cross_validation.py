#!/usr/bin/env python3
"""
Standardized Pairwise Cross-Validation for mCODE Optimization
Implements full pairwise comparisons: prompts × models × trials
Each combination serves as both gold standard and comparator.
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import kendalltau, ttest_ind

from src.pipeline import McodePipeline
from src.shared.models import BenchmarkResult, McodeElement, PipelineResult
from src.shared.types import TaskStatus
from src.utils.concurrency import TaskQueue
from src.utils.llm_loader import LLMLoader
from src.utils.logging_config import get_logger, setup_logging
from src.utils.prompt_loader import PromptLoader


@dataclass
class PairwiseComparisonTask:
    """Data structure for pairwise comparison tasks."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trial_id: str = ""
    trial_data: Dict[str, Any] = field(default_factory=dict)

    # Gold standard configuration
    gold_prompt: str = ""
    gold_model: str = ""

    # Comparator configuration
    comp_prompt: str = ""
    comp_model: str = ""

    # Results storage
    gold_result: Optional[BenchmarkResult] = None
    comp_result: Optional[BenchmarkResult] = None
    status: TaskStatus = TaskStatus.PENDING
    error_message: str = ""

    # Comparison metrics (mapping performance only)
    comparison_metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0


class PairwiseCrossValidator:
    """
    Standardized cross-validation with full pairwise comparisons.
    Tests all prompt × model × trial combinations against each other.
    """

    def __init__(self, output_dir: str = "pairwise_optimization_results"):
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize loaders
        self.prompt_loader = PromptLoader()
        self.llm_loader = LLMLoader()

        # Results storage
        self.pairwise_results: List[PairwiseComparisonTask] = []
        self.combination_cache: Dict[str, BenchmarkResult] = (
            {}
        )  # prompt_model_trial -> result
        self.summary_stats: Dict[str, Any] = {}

        # Task queue for async processing
        self.task_queue: Optional[TaskQueue] = None

    def initialize(self) -> None:
        """Initialize the validator."""
        # Ensure logging is configured
        setup_logging("INFO")
        self.logger.info("🤖 Initializing pairwise validator")

    def shutdown(self) -> None:
        """Shutdown the validator."""
        self.logger.info("🛑 Shutting down pairwise validator")

    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt names."""
        return list(self.prompt_loader.list_available_prompts().keys())

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.llm_loader.list_available_llms().keys())

    def load_trials(self, trials_file: str) -> List[Dict[str, Any]]:
        """Load trial data from file."""
        trials_path = Path(trials_file)

        if not trials_path.exists():
            raise FileNotFoundError(f"Trials file not found: {trials_path}")

        with open(trials_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "successful_trials" in data:
            trials = data["successful_trials"]
        elif isinstance(data, list):
            trials = data
        else:
            raise ValueError("Invalid trials file format")

        self.logger.info(f"📥 Loaded {len(trials)} trials for pairwise validation")
        return trials

    def generate_pairwise_tasks(
        self,
        prompts: List[str],
        models: List[str],
        trials: List[Dict[str, Any]],
        max_comparisons: Optional[int] = None,
    ) -> List[PairwiseComparisonTask]:
        """Generate all possible pairwise comparison tasks."""
        tasks = []

        # First, generate all unique combinations for caching
        all_combinations = []
        for prompt in prompts:
            for model in models:
                for trial_idx, trial in enumerate(trials):
                    trial_id = self._extract_trial_id(trial, trial_idx)
                    all_combinations.append((prompt, model, trial_id, trial))

        # Generate pairwise comparisons
        for i, (gold_prompt, gold_model, gold_trial_id, gold_trial) in enumerate(
            all_combinations
        ):
            for j, (comp_prompt, comp_model, comp_trial_id, comp_trial) in enumerate(
                all_combinations
            ):
                if i != j:  # Avoid self-comparison
                    # Only compare same trial across different configurations
                    if gold_trial_id == comp_trial_id:
                        task = PairwiseComparisonTask(
                            trial_id=gold_trial_id,
                            trial_data=gold_trial,
                            gold_prompt=gold_prompt,
                            gold_model=gold_model,
                            comp_prompt=comp_prompt,
                            comp_model=comp_model,
                        )
                        tasks.append(task)

        # Limit comparisons if requested
        if max_comparisons and len(tasks) > max_comparisons:
            step = len(tasks) // max_comparisons
            tasks = tasks[::step][:max_comparisons]
            self.logger.info(f"🎯 Limited to {len(tasks)} pairwise comparisons")

        self.logger.info(f"🔄 Generated {len(tasks)} pairwise comparison tasks")
        return tasks

    def run_pairwise_validation(self, tasks: List[PairwiseComparisonTask]) -> None:
        """Run pairwise validation."""
        self.logger.info("🔬 Running pairwise validation")
        start_time = time.time()

        # Process all tasks
        processed_tasks = 0
        total_tasks = len(tasks)

        for task in tasks:
            try:
                self._process_pairwise_task(task)
                processed_tasks += 1

                # Progress logging
                if processed_tasks % 10 == 0:
                    progress = (processed_tasks / total_tasks) * 100
                    self.logger.info(
                        f"📊 Progress: {processed_tasks}/{total_tasks} ({progress:.1f}%)"
                    )

            except Exception as e:
                self.logger.error(f"❌ Failed to process task {task.task_id}: {e}")
                task.status = TaskStatus.FAILED
                task.error_message = str(e)

        duration = time.time() - start_time
        self.logger.info(f"🏁 Pairwise validation completed in {duration:.2f} seconds")

    def _process_pairwise_task(self, task: PairwiseComparisonTask) -> None:
        """Process a single pairwise comparison task."""
        task.start_time = time.time()
        task.status = TaskStatus.PROCESSING

        try:
            # Get or generate gold standard result
            gold_key = f"{task.gold_prompt}_{task.gold_model}_{task.trial_id}"
            if gold_key in self.combination_cache:
                task.gold_result = self.combination_cache[gold_key]
            else:
                gold_pipeline = McodePipeline(
                    prompt_name=task.gold_prompt, model_name=task.gold_model
                )
                pipeline_result = gold_pipeline.process(task.trial_data)

                # Convert to BenchmarkResult
                benchmark = BenchmarkResult(
                    task_id=f"{task.gold_prompt}_{task.gold_model}_{task.trial_id}",
                    trial_id=task.trial_id,
                    pipeline_result=pipeline_result,
                    execution_time_seconds=0.0,  # Will be set later
                    status="success"
                )

                task.gold_result = benchmark
                self.combination_cache[gold_key] = benchmark

            # Get or generate comparator result
            comp_key = f"{task.comp_prompt}_{task.comp_model}_{task.trial_id}"
            if comp_key in self.combination_cache:
                task.comp_result = self.combination_cache[comp_key]
            else:
                comp_pipeline = McodePipeline(
                    prompt_name=task.comp_prompt, model_name=task.comp_model
                )
                pipeline_result = comp_pipeline.process(task.trial_data)

                benchmark = BenchmarkResult(
                    task_id=f"{task.comp_prompt}_{task.comp_model}_{task.trial_id}",
                    trial_id=task.trial_id,
                    pipeline_result=pipeline_result,
                    execution_time_seconds=0.0,  # Will be set later
                    status="success"
                )

                task.comp_result = benchmark
                self.combination_cache[comp_key] = benchmark

            # Calculate comparison metrics
            self._calculate_comparison_metrics(task)

            task.status = TaskStatus.SUCCESS
            task.end_time = time.time()
            task.duration_ms = (task.end_time - task.start_time) * 1000

            self.pairwise_results.append(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            raise

    def _calculate_comparison_metrics(self, task: PairwiseComparisonTask) -> None:
        """Calculate comprehensive comparison metrics between gold and comparator."""
        if not task.gold_result or not task.comp_result:
            return

        gold_mappings = task.gold_result.pipeline_result.mcode_mappings
        comp_mappings = task.comp_result.pipeline_result.mcode_mappings

        # Mapping-level metrics only
        mapping_metrics = self._calculate_mapping_metrics(gold_mappings, comp_mappings)

        # Combined metrics
        task.comparison_metrics = {
            **mapping_metrics,
            "gold_mappings_count": len(gold_mappings),
            "comp_mappings_count": len(comp_mappings),
            "gold_compliance_score": task.gold_result.pipeline_result.validation_results.compliance_score,
            "comp_compliance_score": task.comp_result.pipeline_result.validation_results.compliance_score,
        }

    def _calculate_mapping_metrics(
        self, gold_mappings: List[McodeElement], comp_mappings: List[McodeElement]
    ) -> Dict[str, Any]:
        """Calculate mapping-level comparison metrics with detailed examples and edge case handling."""
        try:
            # Convert mappings to comparable strings with validation
            gold_strings = []
            for m in gold_mappings:
                if m.element_type and m.code:
                    gold_strings.append(f"{m.element_type}={json.dumps(m.code)}")
                else:
                    self.logger.warning(f"Invalid gold mapping: element_type={m.element_type}, code={m.code}")

            comp_strings = []
            for m in comp_mappings:
                if m.element_type and m.code:
                    comp_strings.append(f"{m.element_type}={json.dumps(m.code)}")
                else:
                    self.logger.warning(f"Invalid comp mapping: element_type={m.element_type}, code={m.code}")

            gold_set = set(gold_strings)
            comp_set = set(comp_strings)
            intersection = gold_set.intersection(comp_set)
            union = gold_set.union(comp_set)

            # Jaccard similarity
            jaccard = len(intersection) / len(union) if union else 0

            # Precision/recall metrics
            true_positives = len(intersection)
            false_positives = len(comp_set - gold_set)
            false_negatives = len(gold_set - comp_set)

            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Get detailed examples with better formatting
            true_positive_examples = list(intersection)[:5]  # Limit to 5 examples
            false_positive_examples = list(comp_set - gold_set)[:5]  # Limit to 5 examples
            false_negative_examples = list(gold_set - comp_set)[:5]  # Limit to 5 examples

            # Additional metrics for quality assessment
            gold_confidence_avg = sum(m.confidence_score or 0 for m in gold_mappings) / len(gold_mappings) if gold_mappings else 0
            comp_confidence_avg = sum(m.confidence_score or 0 for m in comp_mappings) / len(comp_mappings) if comp_mappings else 0

            return {
                "mapping_jaccard_similarity": jaccard,
                "mapping_precision": precision,
                "mapping_recall": recall,
                "mapping_f1_score": f1,
                "mapping_true_positives": true_positives,
                "mapping_false_positives": false_positives,
                "mapping_false_negatives": false_negatives,
                "gold_mappings_count": len(gold_mappings),
                "comp_mappings_count": len(comp_mappings),
                "gold_avg_confidence": gold_confidence_avg,
                "comp_avg_confidence": comp_confidence_avg,
                "true_positive_examples": true_positive_examples,
                "false_positive_examples": false_positive_examples,
                "false_negative_examples": false_negative_examples,
            }

        except Exception as e:
            self.logger.error(f"Error calculating mapping metrics: {e}")
            return {
                "mapping_jaccard_similarity": 0.0,
                "mapping_precision": 0.0,
                "mapping_recall": 0.0,
                "mapping_f1_score": 0.0,
                "mapping_true_positives": 0,
                "mapping_false_positives": 0,
                "mapping_false_negatives": 0,
                "gold_mappings_count": len(gold_mappings),
                "comp_mappings_count": len(comp_mappings),
                "gold_avg_confidence": 0.0,
                "comp_avg_confidence": 0.0,
                "true_positive_examples": [],
                "false_positive_examples": [],
                "false_negative_examples": [],
                "error": str(e)
            }

    def _extract_trial_id(self, trial_data: Dict[str, Any], index: int) -> str:
        """Extract trial ID from trial data."""
        try:
            return trial_data["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"trial_{index}"

    def analyze_pairwise_results(self) -> Dict[str, Any]:
        """Analyze pairwise comparison results."""
        if not self.pairwise_results:
            return {}

        successful_results = [
            r for r in self.pairwise_results if r.status == TaskStatus.SUCCESS
        ]

        # Group by configuration pairs
        config_stats = defaultdict(list)
        for result in successful_results:
            config_key = f"{result.gold_prompt}_{result.gold_model}_vs_{result.comp_prompt}_{result.comp_model}"
            config_stats[config_key].append(result.comparison_metrics)

        # Calculate aggregate statistics
        analysis = {
            "summary": {
                "total_comparisons": len(self.pairwise_results),
                "successful_comparisons": len(successful_results),
                "success_rate": (
                    len(successful_results) / len(self.pairwise_results)
                    if self.pairwise_results
                    else 0
                ),
                "unique_config_pairs": len(config_stats),
            },
            "configuration_analysis": {},
            "overall_metrics": {},
        }

        # Calculate overall metrics
        all_metrics = [r.comparison_metrics for r in successful_results]
        for metric_name in successful_results[0].comparison_metrics.keys():
            values = [
                m.get(metric_name, 0)
                for m in all_metrics
                if m.get(metric_name) is not None
            ]
            if values:
                analysis["overall_metrics"][metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                }

        # Calculate per-configuration statistics
        for config_key, metrics_list in config_stats.items():
            config_metrics = {}
            for metric_name in metrics_list[0].keys():
                values = [
                    m.get(metric_name, 0)
                    for m in metrics_list
                    if m.get(metric_name) is not None
                ]
                if values:
                    config_metrics[metric_name] = {
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                        "count": len(values),
                    }

            analysis["configuration_analysis"][config_key] = config_metrics

        self.summary_stats = analysis
        return analysis

    def save_results(self, detailed_report: bool = False) -> None:
        """Save pairwise validation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_file = self.output_dir / f"pairwise_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "total_tasks": len(self.pairwise_results),
                        "successful_tasks": len(
                            [
                                r
                                for r in self.pairwise_results
                                if r.status == TaskStatus.SUCCESS
                            ]
                        ),
                    },
                    "results": [
                        {
                            "task_id": r.task_id,
                            "trial_id": r.trial_id,
                            "gold_config": f"{r.gold_prompt}_{r.gold_model}",
                            "comp_config": f"{r.comp_prompt}_{r.comp_model}",
                            "metrics": r.comparison_metrics,
                            "status": r.status.value,
                            "duration_ms": r.duration_ms,
                        }
                        for r in self.pairwise_results
                    ],
                    "analysis": self.summary_stats,
                },
                f,
                indent=2,
            )

        self.logger.info(f"💾 Pairwise results saved to {results_file}")

        # Save summary
        summary_file = self.output_dir / f"pairwise_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(self.summary_stats, f, indent=2)

        self.logger.info(f"📊 Summary saved to {summary_file}")

        # Save detailed examples report
        examples_file = self.output_dir / f"pairwise_examples_{timestamp}.json"
        detailed_examples = []
        for result in self.pairwise_results:
            if result.status == TaskStatus.SUCCESS:
                detailed_examples.append(
                    {
                        "task_id": result.task_id,
                        "trial_id": result.trial_id,
                        "gold_config": f"{result.gold_prompt}_{result.gold_model}",
                        "comp_config": f"{result.comp_prompt}_{result.comp_model}",
                        "true_positive_examples": result.comparison_metrics.get(
                            "true_positive_examples", []
                        ),
                        "false_positive_examples": result.comparison_metrics.get(
                            "false_positive_examples", []
                        ),
                        "false_negative_examples": result.comparison_metrics.get(
                            "false_negative_examples", []
                        ),
                    }
                )

        with open(examples_file, "w") as f:
            json.dump(detailed_examples, f, indent=2)

        self.logger.info(f"📋 Detailed examples saved to {examples_file}")

        if detailed_report:
            self.generate_detailed_report(timestamp)

    def generate_detailed_report(self, timestamp: str) -> None:
        """Generate detailed markdown report."""
        report_file = self.output_dir / f"pairwise_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write("# Pairwise Cross-Validation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary section
            summary = self.summary_stats.get("summary", {})
            f.write("## Summary\n\n")
            f.write(f"- **Total Comparisons**: {summary.get('total_comparisons', 0)}\n")
            f.write(
                f"- **Successful Comparisons**: {summary.get('successful_comparisons', 0)}\n"
            )
            f.write(f"- **Success Rate**: {summary.get('success_rate', 0):.1%}\n")
            f.write(
                f"- **Unique Configuration Pairs**: {summary.get('unique_config_pairs', 0)}\n\n"
            )

            # Overall metrics
            overall = self.summary_stats.get("overall_metrics", {})
            f.write("## Overall Metrics\n\n")
            f.write("| Metric | Mean | Median | Std Dev | Min | Max |\n")
            f.write("|--------|------|--------|---------|-----|-----|\n")

            for metric_name, stats in overall.items():
                f.write(
                    f"| {metric_name} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['stdev']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |\n"
                )

            f.write("\n")

    def print_summary(self) -> None:
        """Print summary of pairwise validation results to console."""
        if not self.summary_stats:
            self.logger.warning("No summary statistics available")
            return

        summary = self.summary_stats.get("summary", {})
        overall_metrics = self.summary_stats.get("overall_metrics", {})

        self.logger.info("🎯 Pairwise Cross-Validation Summary")
        self.logger.info(f"📊 Total Comparisons: {summary.get('total_comparisons', 0)}")
        self.logger.info(
            f"✅ Successful Comparisons: {summary.get('successful_comparisons', 0)}"
        )
        self.logger.info(f"📈 Success Rate: {summary.get('success_rate', 0):.1%}")
        self.logger.info(
            f"🔄 Unique Configuration Pairs: {summary.get('unique_config_pairs', 0)}"
        )

        # Log key metrics
        if overall_metrics:
            self.logger.info("\n📋 Key Performance Metrics:")
            for metric_name, stats in overall_metrics.items():
                if metric_name in ["mapping_f1_score", "mapping_jaccard_similarity"]:
                    self.logger.info(
                        f"   {metric_name.replace('_', ' ').title()}: {stats['mean']:.3f} (mean)"
                    )

        # Show top configuration pairs by mapping F1 score
        config_analysis = self.summary_stats.get("configuration_analysis", {})
        if config_analysis:
            # Find best configuration pairs by mapping F1 score
            best_configs = sorted(
                [
                    (config_key, metrics.get("mapping_f1_score", {}).get("mean", 0))
                    for config_key, metrics in config_analysis.items()
                    if metrics.get("mapping_f1_score")
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            if best_configs:
                self.logger.info("\n🏆 Top Configuration Pairs by Mapping F1-Score:")
                for config_key, f1_score in best_configs:
                    self.logger.info(f"   {config_key}: {f1_score:.3f}")
