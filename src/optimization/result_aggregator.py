"""
Optimization Result Aggregator - Handles result processing and aggregation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.optimization.biological_analyzer import BiologicalAnalyzer
from src.optimization.performance_analyzer import PerformanceAnalyzer
from src.optimization.report_generator import ReportGenerator


class OptimizationResultAggregator:
    """Aggregates and processes optimization results."""

    def __init__(self, logger):
        self.logger = logger
        self.performance_analyzer = PerformanceAnalyzer()
        self.report_generator = ReportGenerator()
        self.biological_analyzer = BiologicalAnalyzer()

    def aggregate_results(
        self,
        combo_results: Dict[int, Dict],
        combinations: List[Dict[str, str]],
        trials_data: List[Dict[str, Any]],
        cv_folds: int,
    ) -> Dict[str, Any]:
        """Aggregate optimization results by combination."""
        optimization_results = []
        best_result = None
        best_score = 0

        # Create runs directory
        runs_dir = Path("optimization_runs")
        runs_dir.mkdir(exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for combo_idx, combo in enumerate(combinations):
            scores = combo_results[combo_idx]["scores"]
            errors = combo_results[combo_idx]["errors"]
            all_metrics = combo_results[combo_idx]["metrics"]

            if scores:
                cv_average_score = sum(scores) / len(scores)
                cv_std = (
                    sum((s - cv_average_score) ** 2 for s in scores) / len(scores)
                ) ** 0.5

                avg_precision = sum(m.get("precision", 0) for m in all_metrics) / len(
                    all_metrics
                )
                avg_recall = sum(m.get("recall", 0) for m in all_metrics) / len(
                    all_metrics
                )
                avg_f1 = sum(m.get("f1_score", 0) for m in all_metrics) / len(
                    all_metrics
                )

                result = {
                    "combination": combo,
                    "success": True,
                    "cv_average_score": cv_average_score,
                    "cv_std_score": cv_std,
                    "fold_scores": scores,
                    "cv_folds": cv_folds,
                    "total_trials": len(scores),
                    "total_elements": len(combo_results[combo_idx]["mcode_elements"]),
                    "errors": errors,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {
                        "precision": avg_precision,
                        "recall": avg_recall,
                        "f1_score": avg_f1,
                    },
                    "predicted_mcode": combo_results[combo_idx]["mcode_elements"],
                }
                optimization_results.append(result)

                # Save individual run
                run_filename = f"run_{run_timestamp}_{combo['model']}_{combo['prompt'].replace('/', '_')}.json"
                run_path = runs_dir / run_filename
                with open(run_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 Saved run result to: {run_path}")

                if cv_average_score > best_score:
                    best_score = cv_average_score
                    best_result = result

                self.logger.info(
                    f"✅ Completed combination: {combo['model']} + {combo['prompt']} (CV score: {cv_average_score:.3f} ± {cv_std:.3f})"
                )
            else:
                error_result = {
                    "combination": combo,
                    "success": False,
                    "error": f"All {len(errors)} trials failed: {errors[:3]}...",
                    "timestamp": datetime.now().isoformat(),
                }
                optimization_results.append(error_result)

                run_filename = f"run_{run_timestamp}_{combo['model']}_{combo['prompt'].replace('/', '_')}_FAILED.json"
                run_path = runs_dir / run_filename
                with open(run_path, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 Saved failed run result to: {run_path}")

                self.logger.error(
                    f"❌ Failed combination {combo['model']} + {combo['prompt']}: All trials failed"
                )

        return {
            "optimization_results": optimization_results,
            "best_result": best_result,
            "best_score": best_score,
        }

    def generate_reports(
        self,
        optimization_results: List[Dict[str, Any]],
        trials_data: List[Dict[str, Any]],
        combo_results: Dict[int, Dict],
        combinations: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis reports."""
        # Biological analysis
        self.biological_analyzer.generate_biological_analysis_report(
            combo_results, combinations, trials_data
        )

        # Performance analysis
        analysis_summary = self.performance_analyzer.analyze_by_category(
            optimization_results, "model"
        )
        provider_analysis = self.performance_analyzer.analyze_by_provider(
            optimization_results
        )
        error_analysis = self.performance_analyzer.summarize_errors(
            optimization_results
        )

        mega_analysis = {
            "model_stats": analysis_summary,
            "provider_stats": provider_analysis,
            "error_analysis": error_analysis,
            "total_runs": len(optimization_results),
            "successful_runs": len(
                [r for r in optimization_results if r.get("success", False)]
            ),
            "time_range": {
                "earliest": min(
                    (
                        r.get("timestamp")
                        for r in optimization_results
                        if r.get("timestamp")
                    ),
                    default=None,
                ),
                "latest": max(
                    (
                        r.get("timestamp")
                        for r in optimization_results
                        if r.get("timestamp")
                    ),
                    default=None,
                ),
            },
        }

        # Generate mega report
        try:
            mega_report = self.report_generator.generate_mega_report(
                mega_analysis, "", ""
            )
            mega_report_path = (
                Path("optimization_runs")
                / f"mega_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(mega_report_path, "w", encoding="utf-8") as f:
                f.write(mega_report)
            self.logger.info(
                f"📊 Mega optimization report saved to: {mega_report_path}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate mega report: {e}")

        return mega_analysis

    def log_performance_analysis(
        self,
        model_analysis: Dict,
        prompt_analysis: Dict,
        provider_analysis: Dict,
        all_results: List[Dict[str, Any]],
    ):
        """Log detailed reliability and performance analysis."""
        self.logger.info("🔍 RELIABILITY & PERFORMANCE ANALYSIS:")

        # Provider rankings
        self.logger.info("🏢 PROVIDER RANKINGS:")
        sorted_providers = sorted(
            provider_analysis.items(),
            key=lambda x: (x[1].get("success_rate", 0), -x[1].get("avg_score", 0)),
            reverse=True,
        )
        for provider, stats in sorted_providers:
            success_rate = stats.get("success_rate", 0) * 100
            avg_score = stats.get("avg_score", 0)
            avg_time = stats.get("avg_processing_time", 0)
            avg_cost = stats.get("avg_cost", 0)
            models = stats.get("models", [])
            self.logger.info(
                f"   🏆 {provider}: {success_rate:.1f}% success, score: {avg_score:.3f}, {avg_time:.1f}s, ${avg_cost:.4f} (models: {', '.join(models)})"
            )

        # Model rankings
        self.logger.info("🤖 MODEL RANKINGS:")
        sorted_models = sorted(
            model_analysis.items(),
            key=lambda x: (x[1].get("success_rate", 0), -x[1].get("avg_score", 0)),
            reverse=True,
        )
        for model, stats in sorted_models:
            success_rate = stats.get("success_rate", 0) * 100
            avg_score = stats.get("avg_score", 0)
            avg_time = stats.get("avg_processing_time", 0)
            avg_cost = stats.get("avg_cost", 0)
            runs = stats.get("runs", 0)
            self.logger.info(
                f"   🏆 {model}: {success_rate:.1f}% success ({runs} runs), score: {avg_score:.3f}, {avg_time:.1f}s, ${avg_cost:.4f}"
            )

        # Prompt rankings
        self.logger.info("📝 PROMPT RANKINGS:")
        sorted_prompts = sorted(
            prompt_analysis.items(),
            key=lambda x: (x[1].get("success_rate", 0), -x[1].get("avg_score", 0)),
            reverse=True,
        )
        for prompt, stats in sorted_prompts:
            success_rate = stats.get("success_rate", 0) * 100
            avg_score = stats.get("avg_score", 0)
            avg_time = stats.get("avg_processing_time", 0)
            runs = stats.get("runs", 0)
            self.logger.info(
                f"   🏆 {prompt}: {success_rate:.1f}% success ({runs} runs), score: {avg_score:.3f}, {avg_time:.1f}s"
            )

        # Performance insights
        self.logger.info("⚡ PERFORMANCE INSIGHTS:")
        if sorted_providers:
            fastest_provider = min(
                sorted_providers,
                key=lambda x: x[1].get("avg_processing_time", float("inf")),
            )
            slowest_provider = max(
                sorted_providers, key=lambda x: x[1].get("avg_processing_time", 0)
            )
            self.logger.info(
                f"   🏃‍♂️ Fastest provider: {fastest_provider[0]} ({fastest_provider[1].get('avg_processing_time', 0):.1f}s avg)"
            )
            self.logger.info(
                f"   🐌 Slowest provider: {slowest_provider[0]} ({slowest_provider[1].get('avg_processing_time', 0):.1f}s avg)"
            )

        if sorted_models:
            cheapest_model = min(
                sorted_models, key=lambda x: x[1].get("avg_cost", float("inf"))
            )
            most_expensive_model = max(
                sorted_models, key=lambda x: x[1].get("avg_cost", 0)
            )
            self.logger.info(
                f"   💰 Cheapest model: {cheapest_model[0]} (${cheapest_model[1].get('avg_cost', 0):.4f} avg)"
            )
            self.logger.info(
                f"   💸 Most expensive model: {most_expensive_model[0]} (${most_expensive_model[1].get('avg_cost', 0):.4f} avg)"
            )

        # Error analysis
        self.logger.info("🔍 COMPREHENSIVE ERROR ANALYSIS:")
        error_summary = self.performance_analyzer.summarize_errors(all_results)
        total_errors = sum(error_summary.values())
        if total_errors > 0:
            self.logger.info("   📈 Overall Error Distribution:")
            for error_type, count in sorted(
                error_summary.items(), key=lambda x: x[1], reverse=True
            ):
                if count > 0:
                    percentage = (count / total_errors) * 100
                    self.logger.info(f"      {error_type}: {count} ({percentage:.1f}%)")

        # Model reliability
        self.logger.info("   🤖 Model Reliability:")
        model_reliability = []
        for model, stats in model_analysis.items():
            runs = stats.get("runs", 0)
            success_rate = stats.get("success_rate", 0) * 100
            error_count = stats.get("error_count", 0)
            primary_error = None
            if error_count > 0:
                error_types = stats.get("error_types", {})
                primary_error = (
                    max(error_types.items(), key=lambda x: x[1])
                    if error_types
                    else None
                )

            model_reliability.append(
                (model, success_rate, error_count, primary_error, runs)
            )

        model_reliability.sort(key=lambda x: (x[1], -x[2]))
        for model, success_rate, error_count, primary_error, runs in model_reliability:
            status = "✅" if success_rate >= 90 else "⚠️" if success_rate >= 50 else "❌"
            error_info = (
                f" ({primary_error[0]}: {primary_error[1]})"
                if primary_error and primary_error[1] > 0
                else ""
            )
            self.logger.info(
                f"      {status} {model}: {success_rate:.1f}% success ({runs} runs){error_info}"
            )

        # Reliability insights
        if sorted_providers:
            most_reliable_provider = max(
                sorted_providers, key=lambda x: x[1].get("success_rate", 0)
            )
            least_reliable_provider = min(
                sorted_providers, key=lambda x: x[1].get("success_rate", 0)
            )
            self.logger.info(
                f"   🛡️ Most reliable provider: {most_reliable_provider[0]} ({most_reliable_provider[1].get('success_rate', 0)*100:.1f}% success)"
            )
            if least_reliable_provider[1].get("success_rate", 1) < 1.0:
                self.logger.info(
                    f"   ⚠️ Least reliable provider: {least_reliable_provider[0]} ({least_reliable_provider[1].get('success_rate', 0)*100:.1f}% success)"
                )
