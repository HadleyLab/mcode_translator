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
from typing import Any, Dict, List, Optional

from src.optimization.cross_validation import CrossValidator
from src.optimization.execution_manager import OptimizationExecutionManager
from src.optimization.result_aggregator import OptimizationResultAggregator
from src.shared.extractors import DataExtractor

from .base_workflow import BaseWorkflow, WorkflowResult


class TrialsOptimizerWorkflow(BaseWorkflow):
    """
    Workflow for optimizing mCODE translation parameters.

    Tests different combinations of prompts and models to find optimal
    settings for mCODE processing. Results are saved to config files.
    """

    def __init__(self):
        super().__init__()
        self.cross_validator = CrossValidator()
        self.execution_manager = OptimizationExecutionManager(self.logger)
        self.result_aggregator = OptimizationResultAggregator(self.logger)
        self.extractor = DataExtractor()

    @property
    def memory_space(self) -> str:
        """Optimizer workflows use 'optimization' space."""
        return "optimization"

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the optimization workflow with cross validation.

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
        try:
            # Extract parameters
            trials_data = kwargs["trials_data"]
            prompts = kwargs.get("prompts", ["direct_mcode_evidence_based_concise"])
            models = kwargs.get("models", ["deepseek-coder"])
            max_combinations = kwargs.get("max_combinations", 5)
            cv_folds = kwargs["cv_folds"]
            output_config = kwargs.get("output_config")
            cli_args = kwargs.get("cli_args")

            if not trials_data:
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for optimization.",
                )

            if len(trials_data) < cv_folds:
                self.logger.warning(
                    f"Only {len(trials_data)} trials available, reducing CV folds to {len(trials_data)}"
                )
                cv_folds = len(trials_data)

            # Log optimization scope for clarity
            self.logger.info("🔬 OPTIMIZATION SCOPE:")
            self.logger.info(f"   📊 Prompts: {len(prompts)} ({', '.join(prompts)})")
            self.logger.info(f"   🤖 Models: {len(models)} ({', '.join(models)})")
            self.logger.info(f"   📈 Max combinations: {max_combinations}")
            self.logger.info(f"   📋 Trials: {len(trials_data)}")
            self.logger.info(f"   🔄 CV folds: {cv_folds}")

            # Generate combinations to test
            combinations = self.cross_validator.generate_combinations(
                prompts, models, max_combinations
            )

            # Log actual combinations generated
            total_possible = len(prompts) * len(models)
            self.logger.info("🎯 COMBINATIONS GENERATED:")
            self.logger.info(f"   📊 Total possible: {total_possible}")
            self.logger.info(f"   ✅ Actually testing: {len(combinations)}")
            if len(combinations) < total_possible:
                self.logger.info(
                    f"   ✂️  Limited by max_combinations={max_combinations}"
                )
            else:
                self.logger.info("   🎉 Testing ALL possible combinations!")

            self.logger.info(f"🧪 Generated {len(combinations)} combinations to test:")
            for i, combo in enumerate(combinations, 1):
                self.logger.info(f"  {i}. {combo['model']} + {combo['prompt']}")

            # Execute optimization using the execution manager
            combo_results = asyncio.run(
                self.execution_manager.execute_optimization(
                    trials_data, combinations, cv_folds, cli_args
                )
            )

            # Create directory for saving individual runs
            runs_dir = Path("optimization_runs")
            runs_dir.mkdir(exist_ok=True)
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Aggregate results using the result aggregator
            aggregation_result = self.result_aggregator.aggregate_results(
                combo_results, combinations, trials_data, cv_folds
            )
            optimization_results = aggregation_result["optimization_results"]
            best_result = aggregation_result["best_result"]
            best_score = aggregation_result["best_score"]

            # Set default LLM specification based on best result
            if best_result:
                self._set_default_llm_spec(best_result)

            # Save optimal settings if requested
            if output_config and best_result:
                self._save_optimal_config(best_result, output_config)

            # Generate reports using the result aggregator
            mega_analysis = self.result_aggregator.generate_reports(
                optimization_results, trials_data, combo_results, combinations
            )

            # Final summary
            successful_combinations = len(
                [r for r in optimization_results if r.get("success", False)]
            )
            total_combinations = len(combinations)
            total_trials_processed = sum(
                len(r.get("fold_scores", []))
                for r in optimization_results
                if r.get("success", False)
            )

            self.logger.info("📊 Optimization complete!")
            self.logger.info(
                f"   ✅ Successful combinations: {successful_combinations}/{total_combinations}"
            )
            self.logger.info(f"   📈 Total trials processed: {total_trials_processed}")
            if best_result:
                self.logger.info(
                    f"   🏆 Best CV score: {best_score:.3f} ({best_result['combination']['model']} + {best_result['combination']['prompt']})"
                )

            # Run inter-rater reliability analysis if requested
            inter_rater_analysis = None
            if kwargs.get("run_inter_rater_reliability", False):
                try:
                    inter_rater_analysis = asyncio.run(
                        self._run_inter_rater_reliability_analysis(
                            trials_data,
                            combinations,
                            kwargs.get("inter_rater_max_concurrent", 3),
                        )
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to run inter-rater reliability analysis: {e}"
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
                    "best_combination": (
                        best_result.get("combination") if best_result else None
                    ),
                    "config_saved": output_config is not None,
                    "inter_rater_reliability": inter_rater_analysis is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials optimization")



    def _save_optimal_config(
        self, best_result: Dict[str, Any], output_path: str
    ) -> None:
        """Save optimal settings to configuration file."""
        try:
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

        except Exception as e:
            self.logger.error(f"Failed to save optimal config: {e}")
            raise

    def _set_default_llm_spec(self, best_result: Dict[str, Any]) -> None:
        """Set the default LLM specification based on optimization results."""
        best_model = best_result["combination"]["model"]
        best_prompt = best_result["combination"]["prompt"]
        best_score = best_result.get(
            "cv_average_score", best_result.get("average_score", 0)
        )

        # Update the LLM config file
        llm_config_path = Path("src/config/llms_config.json")
        if not llm_config_path.exists():
            raise FileNotFoundError(f"LLM config file not found: {llm_config_path}")

        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_config = json.load(f)

        # Update the default model
        llm_config["models"]["default"] = best_model

        # Add optimization metadata
        if "optimization" not in llm_config["models"]:
            llm_config["models"]["optimization"] = {}

        llm_config["models"]["optimization"]["optimized_model"] = best_model
        llm_config["models"]["optimization"]["optimized_prompt"] = best_prompt
        llm_config["models"]["optimization"]["optimization_score"] = best_score
        llm_config["models"]["optimization"][
            "last_optimized"
        ] = datetime.now().isoformat()

        with open(llm_config_path, "w", encoding="utf-8") as f:
            json.dump(llm_config, f, indent=2, ensure_ascii=False)

        # Update the prompts config file
        prompts_config_path = Path("src/config/prompts_config.json")
        if not prompts_config_path.exists():
            raise FileNotFoundError(
                f"Prompts config file not found: {prompts_config_path}"
            )

        with open(prompts_config_path, "r", encoding="utf-8") as f:
            prompts_config = json.load(f)

        # Update the default prompt
        prompts_config["prompts"]["default"] = best_prompt

        # Add optimization metadata to prompts config
        if "optimization" not in prompts_config["prompts"]:
            prompts_config["prompts"]["optimization"] = {}

        prompts_config["prompts"]["optimization"]["optimized_prompt"] = best_prompt
        prompts_config["prompts"]["optimization"]["optimization_score"] = best_score
        prompts_config["prompts"]["optimization"][
            "last_optimized"
        ] = datetime.now().isoformat()

        with open(prompts_config_path, "w", encoding="utf-8") as f:
            json.dump(prompts_config, f, indent=2, ensure_ascii=False)

        self.logger.info(
            f"🔧 Updated defaults - Model: {best_model} (in {llm_config_path}), Prompt: {best_prompt} (in {prompts_config_path}), CV score: {best_score:.3f}"
        )

    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt templates."""
        # This would typically scan the prompts directory
        return [
            "direct_mcode_evidence_based_concise",
            "direct_mcode_evidence_based",
            "direct_mcode_minimal",
            "direct_mcode_structured",
        ]

    def get_available_models(self) -> List[str]:
        """Get list of available LLM models."""
        # This would typically come from configuration
        return [
            # OpenAI models
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            # DeepSeek models
            "deepseek-coder",
            "deepseek-chat",
            "deepseek-reasoner",
            # Other models (for compatibility)
            "claude-3",
            "llama-3",
        ]

    def validate_combination(self, prompt: str, model: str) -> bool:
        """
        Validate that a prompt×model combination is valid.

        Args:
            prompt: Prompt template name
            model: Model name

        Returns:
            bool: True if combination is valid
        """
        available_prompts = self.get_available_prompts()
        available_models = self.get_available_models()

        return prompt in available_prompts and model in available_models



    async def _run_inter_rater_reliability_analysis(
        self,
        trials_data: List[Dict[str, Any]],
        combinations: List[Dict[str, str]],
        max_concurrent: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Run inter-rater reliability analysis on optimization results."""
        try:
            from src.optimization.inter_rater_reliability import (
                InterRaterReliabilityAnalyzer,
            )

            self.logger.info("🤝 Starting inter-rater reliability analysis...")

            # Convert combinations to rater configs
            rater_configs = [
                {"model": combo["model"], "prompt": combo["prompt"]}
                for combo in combinations
            ]

            # Initialize analyzer
            analyzer = InterRaterReliabilityAnalyzer()
            analyzer.initialize()

            # Run analysis
            analysis = await analyzer.run_analysis(
                trials_data, rater_configs, max_concurrent
            )

            # Save results
            analyzer.save_results()

            # Generate and save report
            report = analyzer.generate_report()
            report_path = (
                Path("optimization_runs")
                / f"inter_rater_reliability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(report_path, "w") as f:
                f.write(report)

            self.logger.info(
                f"📊 Inter-rater reliability report saved to: {report_path}"
            )

            # Return summary for metadata
            return {
                "num_raters": analysis.num_raters,
                "num_trials": analysis.num_trials,
                "overall_agreement": {
                    "presence_agreement": analysis.overall_metrics.get(
                        "presence_agreement", {}
                    ).percentage_agreement,
                    "values_agreement": analysis.overall_metrics.get(
                        "values_agreement", {}
                    ).percentage_agreement,
                    "confidence_agreement": analysis.overall_metrics.get(
                        "confidence_agreement", {}
                    ).percentage_agreement,
                },
                "report_path": str(report_path),
            }

        except Exception as e:
            self.logger.error(f"Failed to run inter-rater reliability analysis: {e}")
            return None
