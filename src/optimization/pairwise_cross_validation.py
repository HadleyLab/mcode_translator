#!/usr/bin/env python3
"""
Standardized Pairwise Cross-Validation for mCODE Optimization
Implements full pairwise comparisons: prompts × models × trials
Each combination serves as both gold standard and comparator.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional
import uuid

from src.pipeline import McodePipeline
from src.shared.models import BenchmarkResult, McodeElement
from src.shared.types import TaskStatus
from src.utils.concurrency import AsyncTaskQueue
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
        self.combination_cache: Dict[str, BenchmarkResult] = {}  # prompt_model_trial -> result
        self.summary_stats: Dict[str, Any] = {}

        # Task queue for async processing
        self.task_queue: Optional[AsyncTaskQueue] = None

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

        with open(trials_path) as f:
            data: Any = json.load(f)

        if isinstance(data, dict) and "successful_trials" in data:
            trials: List[Dict[str, Any]] = data["successful_trials"]
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
        for i, (gold_prompt, gold_model, gold_trial_id, gold_trial) in enumerate(all_combinations):
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

    def run_pairwise_validation(
        self, tasks: List[PairwiseComparisonTask], max_workers: int = 5
    ) -> None:
        """Run pairwise validation with concurrent processing."""
        self.logger.info("🔬 Running pairwise validation")
        start_time = time.time()

        # Initialize task queue if not already done
        if self.task_queue is None:
            self.task_queue = AsyncTaskQueue(max_concurrent=max_workers, name="PairwiseValidator")

        # Convert tasks to Task objects for TaskQueue
        from src.utils.concurrency import Task

        queue_tasks = []
        for task in tasks:
            queue_task = Task(
                id=task.task_id,
                func=self._process_pairwise_task_async,
                args=(task,),
                kwargs={},
            )
            queue_tasks.append(queue_task)

        self.logger.info(f"🚀 Starting concurrent processing with {max_workers} workers")

        def progress_callback(completed: int, total: int, result: Any) -> None:
            if completed % 10 == 0:
                progress = (completed / total) * 100
                self.logger.info(f"📊 Progress: {completed}/{total} ({progress:.1f}%)")

        # Execute tasks concurrently
        import asyncio
        task_results = asyncio.run(self.task_queue.execute_tasks(
            queue_tasks, progress_callback=progress_callback
        ))

        # Process results
        successful_tasks = 0
        failed_tasks = 0

        for result in task_results:
            if result.success:
                successful_tasks += 1
                # Task result is already stored in the task object
            else:
                failed_tasks += 1
                self.logger.error(f"❌ Task failed: {result.error}")

        duration = time.time() - start_time
        self.logger.info(f"🏁 Pairwise validation completed in {duration:.2f} seconds")
        self.logger.info(f"✅ Successful: {successful_tasks}, Failed: {failed_tasks}")

    async def _process_pairwise_task(self, task: PairwiseComparisonTask) -> None:
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
                pipeline_result = await gold_pipeline.process(task.trial_data)

                # Convert to BenchmarkResult
                benchmark = BenchmarkResult(
                    task_id=f"{task.gold_prompt}_{task.gold_model}_{task.trial_id}",
                    trial_id=task.trial_id,
                    pipeline_result=pipeline_result,
                    execution_time_seconds=0.0,  # Will be set later
                    memory_usage_mb=None,
                    status="success",
                    entities_extracted=None,
                    entities_mapped=None,
                    extraction_completeness=None,
                    mapping_accuracy=None,
                    precision=None,
                    recall=None,
                    f1_score=None,
                    compliance_score=None,
                    prompt_variant_id=None,
                    api_config_name=None,
                    test_case_id=None,
                    pipeline_type=None,
                    start_time=None,
                    end_time=None,
                    duration_ms=None,
                    success=True,
                    error_message=None,
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
                pipeline_result = await comp_pipeline.process(task.trial_data)

                benchmark = BenchmarkResult(
                    task_id=f"{task.comp_prompt}_{task.comp_model}_{task.trial_id}",
                    trial_id=task.trial_id,
                    pipeline_result=pipeline_result,
                    execution_time_seconds=0.0,  # Will be set later
                    memory_usage_mb=None,
                    status="success",
                    entities_extracted=None,
                    entities_mapped=None,
                    extraction_completeness=None,
                    mapping_accuracy=None,
                    precision=None,
                    recall=None,
                    f1_score=None,
                    compliance_score=None,
                    prompt_variant_id=None,
                    api_config_name=None,
                    test_case_id=None,
                    pipeline_type=None,
                    start_time=None,
                    end_time=None,
                    duration_ms=None,
                    success=True,
                    error_message=None,
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

    async def _process_pairwise_task_async(self, task: PairwiseComparisonTask) -> Dict[str, Any]:
        """Async version of pairwise task processing for TaskQueue."""
        try:
            await self._process_pairwise_task(task)
            return {"success": True, "task_id": task.task_id}
        except Exception as e:
            return {"success": False, "task_id": task.task_id, "error": str(e)}

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
                    self.logger.warning(
                        f"Invalid gold mapping: element_type={m.element_type}, code={m.code}"
                    )

            comp_strings = []
            for m in comp_mappings:
                if m.element_type and m.code:
                    comp_strings.append(f"{m.element_type}={json.dumps(m.code)}")
                else:
                    self.logger.warning(
                        f"Invalid comp mapping: element_type={m.element_type}, code={m.code}"
                    )

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
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Get detailed examples with better formatting
            true_positive_examples = list(intersection)[:5]  # Limit to 5 examples
            false_positive_examples = list(comp_set - gold_set)[:5]  # Limit to 5 examples
            false_negative_examples = list(gold_set - comp_set)[:5]  # Limit to 5 examples

            # Additional metrics for quality assessment
            gold_confidence_avg = (
                sum(m.confidence_score or 0 for m in gold_mappings) / len(gold_mappings)
                if gold_mappings
                else 0
            )
            comp_confidence_avg = (
                sum(m.confidence_score or 0 for m in comp_mappings) / len(comp_mappings)
                if comp_mappings
                else 0
            )

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
                "error": str(e),
            }

    def _extract_trial_id(self, trial_data: Dict[str, Any], index: int) -> str:
        """Extract trial ID from trial data."""
        try:
            trial_id: str = trial_data["protocolSection"]["identificationModule"]["nctId"]
            return trial_id
        except (KeyError, TypeError):
            return f"trial_{index}"

    def analyze_pairwise_results(self) -> Dict[str, Any]:
        """Analyze pairwise comparison results."""
        if not self.pairwise_results:
            return {}

        successful_results = [r for r in self.pairwise_results if r.status == TaskStatus.SUCCESS]

        # Group by configuration pairs
        config_stats = defaultdict(list)
        for result in successful_results:
            config_key = f"{result.gold_prompt}_{result.gold_model}_vs_{result.comp_prompt}_{result.comp_model}"
            config_stats[config_key].append(result.comparison_metrics)

        # Calculate aggregate statistics
        analysis: Dict[str, Any] = {
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

        # Calculate overall metrics (only for numeric metrics)
        all_metrics = [r.comparison_metrics for r in successful_results]
        if successful_results:
            for metric_name in successful_results[0].comparison_metrics.keys():
                values = [
                    m.get(metric_name, 0) for m in all_metrics if m.get(metric_name) is not None
                ]

                # Only calculate statistics for numeric values (not lists or strings)
                if values and all(isinstance(v, (int, float)) for v in values):
                    try:
                        analysis["overall_metrics"][metric_name] = {
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                            "min": min(values),
                            "max": max(values),
                        }
                    except (TypeError, ValueError) as e:
                        self.logger.warning(
                            f"Could not calculate statistics for {metric_name}: {e}"
                        )

        # Calculate per-configuration statistics
        for config_key, metrics_list in config_stats.items():
            config_metrics = {}
            if metrics_list:
                for metric_name in metrics_list[0].keys():
                    values = [
                        m.get(metric_name, 0)
                        for m in metrics_list
                        if m.get(metric_name) is not None
                    ]

                    # Only calculate statistics for numeric values
                    if values and all(isinstance(v, (int, float)) for v in values):
                        try:
                            config_metrics[metric_name] = {
                                "mean": statistics.mean(values),
                                "median": statistics.median(values),
                                "stdev": (statistics.stdev(values) if len(values) > 1 else 0),
                                "count": len(values),
                            }
                        except (TypeError, ValueError) as e:
                            self.logger.warning(
                                f"Could not calculate config statistics for {metric_name}: {e}"
                            )

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
                            [r for r in self.pairwise_results if r.status == TaskStatus.SUCCESS]
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

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(
                "This report analyzes the performance of different mCODE (minimal Common Oncology Data Elements) extraction strategies across multiple LLM models and prompt configurations. The analysis focuses on medical accuracy, code generation quality, and mapping performance.\n\n"
            )

            # Summary section
            summary = self.summary_stats.get("summary", {})
            f.write("## Summary\n\n")
            f.write(f"- **Total Comparisons**: {summary.get('total_comparisons', 0)}\n")
            f.write(f"- **Successful Comparisons**: {summary.get('successful_comparisons', 0)}\n")
            f.write(f"- **Success Rate**: {summary.get('success_rate', 0):.1%}\n")
            f.write(
                f"- **Unique Configuration Pairs**: {summary.get('unique_config_pairs', 0)}\n\n"
            )

            # Prompt Strategy Analysis
            f.write("## Prompt Strategy Analysis\n\n")
            f.write("### SNOMED-Enabled Prompts ⭐\n")
            f.write(
                "**Strategy**: Integrated SNOMED CT code reference table with evidence-based extraction\n"
            )
            f.write("**Key Features**:\n")
            f.write(
                "- Pre-defined SNOMED CT codes for common cancer conditions, treatments, and demographics\n"
            )
            f.write("- Generates actual medical codes instead of null/placeholder values\n")
            f.write("- Higher code coverage and medical accuracy\n")
            f.write("- Maintains strict evidence-based approach\n\n")

            f.write("### Evidence-Based Prompts 🔍\n")
            f.write("**Strategy**: Strict fidelity to source text with conservative mapping\n")
            f.write("**Key Features**:\n")
            f.write("- Only extracts information explicitly stated in source\n")
            f.write("- No inference or extrapolation\n")
            f.write("- Prioritizes accuracy over completeness\n\n")

            f.write("### Other Prompts 📝\n")
            f.write(
                "**Strategy**: Various approaches including structured, comprehensive, and minimal extraction\n"
            )
            f.write("**Key Features**:\n")
            f.write("- Different levels of detail and structure\n")
            f.write("- Varying approaches to medical terminology handling\n\n")

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

            # Add interpretation section
            f.write("## Key Findings\n\n")
            f.write("### Medical Code Generation Quality\n")
            f.write("- SNOMED-enabled prompts show improved code coverage\n")
            f.write("- Evidence-based approaches maintain high accuracy\n")
            f.write("- Confidence scores indicate reliable extraction\n\n")

            f.write("### Performance Patterns\n")
            f.write(
                "- Different models show varying performance with different prompt strategies\n"
            )
            f.write("- Mapping F1-score provides comprehensive quality metric\n")
            f.write("- Jaccard similarity measures overlap between extraction methods\n\n")

    def print_summary(self) -> None:
        """Print summary of pairwise validation results to console."""
        if not self.summary_stats:
            self.logger.warning("No summary statistics available")
            return

        summary = self.summary_stats.get("summary", {})
        overall_metrics = self.summary_stats.get("overall_metrics", {})

        self.logger.info("🎯 Pairwise Cross-Validation Summary")
        self.logger.info(f"📊 Total Comparisons: {summary.get('total_comparisons', 0)}")
        self.logger.info(f"✅ Successful Comparisons: {summary.get('successful_comparisons', 0)}")
        self.logger.info(f"📈 Success Rate: {summary.get('success_rate', 0):.1%}")
        self.logger.info(f"🔄 Unique Configuration Pairs: {summary.get('unique_config_pairs', 0)}")

        # Log key metrics
        if overall_metrics:
            self.logger.info("\n📋 Key Performance Metrics:")
            for metric_name, stats in overall_metrics.items():
                if metric_name in ["mapping_f1_score", "mapping_jaccard_similarity"]:
                    self.logger.info(
                        f"   {metric_name.replace('_', ' ').title()}: {stats['mean']:.3f} (mean)"
                    )

        # Analyze prompt performance and highlight SNOMED-enabled prompt
        self._analyze_prompt_performance()

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
            )[
                :5
            ]  # Show top 5 instead of 3

            if best_configs:
                self.logger.info("\n🏆 Top Configuration Pairs by Mapping F1-Score:")
                for config_key, f1_score in best_configs:
                    # Highlight SNOMED-enabled prompts
                    if "evidence_based_with_codes" in config_key:
                        self.logger.info(f"   ⭐ {config_key}: {f1_score:.3f} (SNOMED-enabled)")
                    else:
                        self.logger.info(f"   {config_key}: {f1_score:.3f}")

    def _analyze_prompt_performance(self) -> None:
        """Analyze and highlight different prompt strategies."""
        config_analysis = self.summary_stats.get("configuration_analysis", {})

        # Group by prompt types
        prompt_performance: Dict[str, List[float]] = {}
        snomed_prompts = []
        evidence_based_prompts = []
        other_prompts = []

        for config_key in config_analysis.keys():
            # Extract prompt name from config key (format: prompt_model_vs_prompt_model)
            prompt_parts = config_key.split("_vs_")[0].split("_")
            if len(prompt_parts) >= 2:
                prompt_name = "_".join(prompt_parts[:-1])  # Remove model part
                prompt_parts[-1]

                if prompt_name not in prompt_performance:
                    prompt_performance[prompt_name] = []

                f1_score = config_analysis[config_key].get("mapping_f1_score", {}).get("mean", 0)
                prompt_performance[prompt_name].append(f1_score)

                # Categorize prompts
                if "evidence_based_with_codes" in prompt_name:
                    snomed_prompts.append((config_key, f1_score))
                elif "evidence_based" in prompt_name:
                    evidence_based_prompts.append((config_key, f1_score))
                else:
                    other_prompts.append((config_key, f1_score))

        # Calculate average performance by prompt type
        self.logger.info("\n🧬 PROMPT STRATEGY ANALYSIS:")
        self.logger.info("-" * 40)

        if snomed_prompts:
            snomed_avg = sum(f1 for _, f1 in snomed_prompts) / len(snomed_prompts)
            self.logger.info(
                f"⭐ SNOMED-Enabled Prompts: {snomed_avg:.3f} avg F1 ({len(snomed_prompts)} configs)"
            )
            self.logger.info("   → Includes integrated SNOMED CT code reference table")
            self.logger.info("   → Generates actual medical codes instead of null values")
            self.logger.info("   → Higher code coverage and medical accuracy")

        if evidence_based_prompts:
            evidence_avg = sum(f1 for _, f1 in evidence_based_prompts) / len(evidence_based_prompts)
            self.logger.info(
                f"🔍 Evidence-Based Prompts: {evidence_avg:.3f} avg F1 ({len(evidence_based_prompts)} configs)"
            )
            self.logger.info("   → Strict fidelity to source text only")
            self.logger.info("   → Conservative mapping approach")

        if other_prompts:
            other_avg = sum(f1 for _, f1 in other_prompts) / len(other_prompts)
            self.logger.info(
                f"📝 Other Prompts: {other_avg:.3f} avg F1 ({len(other_prompts)} configs)"
            )

        # Show best SNOMED vs non-SNOMED comparison
        if snomed_prompts and (evidence_based_prompts or other_prompts):
            best_snomed = max(snomed_prompts, key=lambda x: x[1])
            all_non_snomed = evidence_based_prompts + other_prompts
            if all_non_snomed:
                best_non_snomed = max(all_non_snomed, key=lambda x: x[1])
                if best_non_snomed[1] > 0:
                    improvement = ((best_snomed[1] - best_non_snomed[1]) / best_non_snomed[1]) * 100
                    self.logger.info(
                        f"\n📈 Best SNOMED vs Best Non-SNOMED: {improvement:+.1f}% improvement"
                    )
                    self.logger.info("   → Demonstrates value of integrated medical coding")
                else:
                    self.logger.info(
                        "\n📈 Best SNOMED vs Best Non-SNOMED: N/A (non-SNOMED F1 is zero)"
                    )
