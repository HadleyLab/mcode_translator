"""
Trials Optimizer Workflow - Optimize mCODE translation parameters.

This workflow handles testing different combinations of prompts and models
to find optimal settings for mCODE translation. Results are saved to
configuration files, not to CORE Memory.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

from optimization.cross_validation import CrossValidator
from optimization.execution_manager import OptimizationExecutionManager
from optimization.result_aggregator import OptimizationResultAggregator
from shared.extractors import DataExtractor
from storage.mcode_memory_storage import OncoCoreMemory
from utils.config import Config
from utils.logging_config import Loggable

from .base_workflow import BaseWorkflow, WorkflowResult


class TrialsOptimizerWorkflow(BaseWorkflow, Loggable):
    """
    Workflow for optimizing mCODE translation parameters.

    Tests different combinations of prompts and models to find optimal
    settings for mCODE processing. Results are saved to config files.
    """

    def __init__(
        self,
        config: Config,
        memory_storage: OncoCoreMemory,
    ) -> None:
        super().__init__(config, memory_storage)
        Loggable.__init__(self)
        self.cross_validator = CrossValidator()
        self.execution_manager = OptimizationExecutionManager(self.logger)
        self.result_aggregator = OptimizationResultAggregator(self.logger)
        self.extractor = DataExtractor()

    @property
    def memory_space(self) -> str:
        """Optimizer workflows use 'optimization' space."""
        return "optimization"

    async def execute_async(self, **kwargs: Any) -> WorkflowResult:
        """
        Execute the optimization workflow with cross validation asynchronously.

        Args:
            **kwargs: Workflow parameters including:
                 - trials_data: List of trial data for testing
                 - prompts: List of prompt templates to test
                 - models: List of LLM models to test
                 - max_combinations: Maximum combinations to test
                 - cv_folds: Number of cross validation folds
                 - output_config: Where to save optimal settings
                 - cli_args: CLI arguments for concurrency configuration

        Returns:
            WorkflowResult: Optimization results
        """
        trials_data = kwargs["trials_data"]
        prompts = kwargs["prompts"]
        models = kwargs["models"]
        max_combinations = kwargs["max_combinations"]
        cv_folds = kwargs["cv_folds"]
        output_config = kwargs.get("output_config")
        cli_args = kwargs.get("cli_args")

        if not trials_data:
            raise ValueError("No trial data provided for optimization.")

        combinations = self.cross_validator.generate_combinations(prompts, models, max_combinations)

        self.logger.info(f"🧪 Generated {len(combinations)} combinations to test:")
        for i, combo in enumerate(combinations, 1):
            self.logger.info(f"  {i}. {combo['model']} + {combo['prompt']}")

        combo_results = await self.execution_manager.execute_optimization(
            trials_data, combinations, cv_folds, cli_args
        )

        aggregation_result = self.result_aggregator.aggregate_results(
            combo_results, combinations, trials_data, cv_folds
        )
        optimization_results = aggregation_result["optimization_results"]
        best_result = aggregation_result["best_result"]
        best_score = aggregation_result["best_score"]

        if best_result:
            self._set_default_llm_spec(best_result)

        if output_config and best_result:
            self._save_optimal_config(best_result, output_config)

        self.result_aggregator.generate_reports(
            optimization_results, trials_data, combo_results, combinations
        )

        successful_combinations = len([r for r in optimization_results if r.get("success", False)])
        total_combinations = len(combinations)
        total_trials_processed = sum(
            len(r.get("fold_scores", [])) for r in optimization_results if r.get("success", False)
        )

        self.logger.info("📊 Optimization complete!")
        self.logger.info(
            f"   ✅ Successful combinations: {successful_combinations}/{total_combinations}"
        )
        self.logger.info(f"   📈 Total trials processed: {total_trials_processed}")
        if best_result:
            model = best_result["combination"]["model"]
            prompt = best_result["combination"]["prompt"]
            self.logger.info(f"   🏆 Best CV score: {best_score:.3f} ({model} + {prompt})")

        inter_rater_analysis = None
        if kwargs.get("run_inter_rater_reliability", False):
            inter_rater_analysis = await self._run_inter_rater_reliability_analysis(
                trials_data,
                combinations,
                kwargs.get("inter_rater_max_concurrent", 3),
            )

        return self._create_result(
            success=len(optimization_results) > 0,
            data=optimization_results,
            metadata={
                "total_combinations_tested": len(combinations),
                "cv_folds": cv_folds,
                "successful_tests": len(
                    [r for r in optimization_results if r.get("success", False)]
                ),
                "best_score": best_score,
                "best_combination": (best_result.get("combination") if best_result else None),
                "config_saved": output_config is not None,
                "inter_rater_reliability": inter_rater_analysis is not None,
            },
        )

    def execute(self, **kwargs: Any) -> WorkflowResult:
        return asyncio.run(self.execute_async(**kwargs))

    def _save_optimal_config(self, best_result: Dict[str, Any], output_path: str) -> None:
        config_data = {
            "optimal_settings": {
                "model": best_result["combination"]["model"],
                "prompt": best_result["combination"]["prompt"],
                "cv_score": best_result.get(
                    "cv_average_score", best_result.get("average_score", 0)
                ),
                "cv_std": best_result.get("cv_std_score", 0),
                "optimization_timestamp": datetime.now().isoformat(),
                "optimizer_version": "2.0.0",
            },
            "metadata": {
                "combinations_tested": best_result.get("combinations_tested", 0),
                "cv_folds": best_result.get("cv_folds", 3),
                "total_trials": best_result.get("total_trials", 0),
                "fold_scores": best_result.get("fold_scores", []),
                "metrics": best_result.get("metrics", {}),
            },
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 Saved optimal config to: {output_file}")

    def _set_default_llm_spec(self, best_result: Dict[str, Any]) -> None:
        best_model = best_result["combination"]["model"]
        best_prompt = best_result["combination"]["prompt"]
        best_score = best_result.get("cv_average_score", best_result.get("average_score", 0))

        llm_config_path = Path("src/config/llms_config.json")
        with open(llm_config_path, encoding="utf-8") as f:
            llm_config = json.load(f)

        llm_config["models"]["default"] = best_model

        if "optimization" not in llm_config["models"]:
            llm_config["models"]["optimization"] = {}

        llm_config["models"]["optimization"]["optimized_model"] = best_model
        llm_config["models"]["optimization"]["optimized_prompt"] = best_prompt
        llm_config["models"]["optimization"]["optimization_score"] = best_score
        llm_config["models"]["optimization"]["last_optimized"] = datetime.now().isoformat()

        with open(llm_config_path, "w", encoding="utf-8") as f:
            json.dump(llm_config, f, indent=2, ensure_ascii=False)

        prompts_config_path = Path("src/config/prompts_config.json")
        with open(prompts_config_path, encoding="utf-8") as f:
            prompts_config = json.load(f)

        prompts_config["prompts"]["default"] = best_prompt

        if "optimization" not in prompts_config["prompts"]:
            prompts_config["prompts"]["optimization"] = {}

        prompts_config["prompts"]["optimization"]["optimized_prompt"] = best_prompt
        prompts_config["prompts"]["optimization"]["optimization_score"] = best_score
        prompts_config["prompts"]["optimization"]["last_optimized"] = datetime.now().isoformat()

        with open(prompts_config_path, "w", encoding="utf-8") as f:
            json.dump(prompts_config, f, indent=2, ensure_ascii=False)

        msg = (
            f"🔧 Updated defaults - Model: {best_model} (in {llm_config_path}), "
            f"Prompt: {best_prompt} (in {prompts_config_path}), CV score: {best_score:.3f}"
        )
        self.logger.info(msg)

    def get_available_prompts(self) -> List[str]:
        return [
            "direct_mcode_evidence_based_concise",
            "direct_mcode_evidence_based",
            "direct_mcode_minimal",
            "direct_mcode_structured",
        ]

    def get_available_models(self) -> List[str]:
        return [
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "deepseek-coder",
            "deepseek-chat",
            "deepseek-reasoner",
            "claude-3",
            "llama-3",
        ]

    def validate_combination(self, prompt: str, model: str) -> bool:
        available_prompts = self.get_available_prompts()
        available_models = self.get_available_models()
        return prompt in available_prompts and model in available_models

    def summarize_benchmark_validations(self) -> None:
        self.logger.info("Benchmark validation summary not implemented yet")

    async def _run_inter_rater_reliability_analysis(
        self,
        trials_data: List[Dict[str, Any]],
        combinations: List[Dict[str, str]],
        max_concurrent: int = 3,
    ) -> Dict[str, Any]:
        from optimization.inter_rater_reliability import InterRaterReliabilityAnalyzer

        self.logger.info("🤝 Starting inter-rater reliability analysis...")

        rater_configs = [
            {"model": combo["model"], "prompt": combo["prompt"]} for combo in combinations
        ]

        analyzer = InterRaterReliabilityAnalyzer()
        analyzer.initialize()

        analysis = await analyzer.run_analysis(trials_data, rater_configs, max_concurrent)

        analyzer.save_results()

        report = analyzer.generate_report()
        report_path = (
            Path("optimization_runs")
            / f"inter_rater_reliability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_path, "w") as f:
            f.write(report)

        self.logger.info(f"📊 Inter-rater reliability report saved to: {report_path}")

        return {
            "num_raters": analysis.num_raters,
            "num_trials": analysis.num_trials,
            "overall_agreement": {
                "presence_agreement": cast(
                    Any, analysis.overall_metrics.get("presence_agreement", {})
                ).percentage_agreement,
                "values_agreement": cast(
                    Any, analysis.overall_metrics.get("values_agreement", {})
                ).percentage_agreement,
                "confidence_agreement": cast(
                    Any, analysis.overall_metrics.get("confidence_agreement", {})
                ).percentage_agreement,
            },
            "report_path": str(report_path),
        }
