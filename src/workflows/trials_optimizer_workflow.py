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

from src.pipeline import McodePipeline
from src.utils.logging_config import get_logger

from .base_workflow import BaseWorkflow, WorkflowResult


class TrialsOptimizerWorkflow(BaseWorkflow):
    """
    Workflow for optimizing mCODE translation parameters.

    Tests different combinations of prompts and models to find optimal
    settings for mCODE processing. Results are saved to config files.
    """

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the optimization workflow.

        Args:
            **kwargs: Workflow parameters including:
                - trials_data: List of trial data for testing
                - prompts: List of prompt templates to test
                - models: List of LLM models to test
                - max_combinations: Maximum combinations to test
                - output_config: Where to save optimal settings

        Returns:
            WorkflowResult: Optimization results
        """
        try:
            # Extract parameters
            trials_data = kwargs.get("trials_data", [])
            prompts = kwargs.get("prompts", ["direct_mcode_evidence_based_concise"])
            models = kwargs.get("models", ["deepseek-coder"])
            max_combinations = kwargs.get("max_combinations", 5)
            output_config = kwargs.get("output_config")

            if not trials_data:
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for optimization.",
                )

            # Generate combinations to test
            combinations = self._generate_combinations(
                prompts, models, max_combinations
            )

            self.logger.info(
                f"🔬 Testing {len(combinations)} prompt×model combinations"
            )

            # Run optimization
            optimization_results = []
            best_result = None
            best_score = 0

            for i, combo in enumerate(combinations):
                try:
                    self.logger.info(
                        f"Testing combination {i+1}/{len(combinations)}: {combo['model']} + {combo['prompt']}"
                    )

                    # Test combination
                    result = self._test_combination(combo, trials_data)

                    optimization_results.append(result)

                    # Track best result
                    score = result.get("average_score", 0)
                    if score > best_score:
                        best_score = score
                        best_result = result

                except Exception as e:
                    self.logger.error(f"Failed to test combination {combo}: {e}")
                    error_result = {
                        "combination": combo,
                        "error": str(e),
                        "success": False,
                    }
                    optimization_results.append(error_result)

            # Save optimal settings if requested
            if output_config and best_result:
                self._save_optimal_config(best_result, output_config)

            self.logger.info(f"📊 Optimization complete. Best score: {best_score:.3f}")

            return self._create_result(
                success=len(optimization_results) > 0,
                data=optimization_results,
                metadata={
                    "total_combinations_tested": len(combinations),
                    "successful_tests": len(
                        [r for r in optimization_results if r.get("success", False)]
                    ),
                    "best_score": best_score,
                    "best_combination": (
                        best_result.get("combination") if best_result else None
                    ),
                    "config_saved": output_config is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials optimization")

    def _generate_combinations(
        self, prompts: List[str], models: List[str], max_combinations: int
    ) -> List[Dict[str, str]]:
        """Generate combinations of prompts and models to test."""
        combinations = []

        for prompt in prompts:
            for model in models:
                if len(combinations) >= max_combinations:
                    break
                combinations.append({"prompt": prompt, "model": model})

        return combinations

    def _test_combination(
        self, combination: Dict[str, str], trials_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test a specific prompt×model combination."""
        prompt_name = combination["prompt"]
        model_name = combination["model"]

        try:
            # Initialize pipeline with this combination
            pipeline = McodePipeline(prompt_name=prompt_name, model_name=model_name)

            # Test on sample trials (limit for performance)
            test_trials = trials_data[:3]  # Test on first 3 trials
            scores = []

            for trial in test_trials:
                try:
                    # Process trial
                    result = pipeline.process_clinical_trial(trial)

                    # Calculate quality score
                    score = self._calculate_quality_score(result)
                    scores.append(score)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process trial with {model_name}: {e}"
                    )
                    scores.append(0.0)

            # Calculate average score
            average_score = sum(scores) / len(scores) if scores else 0.0

            return {
                "combination": combination,
                "success": True,
                "average_score": average_score,
                "individual_scores": scores,
                "trials_tested": len(test_trials),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "combination": combination,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_quality_score(self, result) -> float:
        """
        Calculate a quality score for mCODE processing results.

        This is a simplified scoring function. In practice, this would
        use more sophisticated metrics.
        """
        try:
            # Base score from number of mappings
            mappings_count = (
                len(result.mcode_mappings) if hasattr(result, "mcode_mappings") else 0
            )

            # Bonus for validation results
            validation_bonus = 0
            if hasattr(result, "validation_results") and result.validation_results:
                compliance = result.validation_results.get("compliance_score", 0)
                validation_bonus = compliance * 0.2

            # Bonus for source references
            reference_bonus = 0
            if hasattr(result, "source_references") and result.source_references:
                reference_bonus = min(len(result.source_references) * 0.1, 0.3)

            # Calculate total score (0-1 scale)
            base_score = min(
                mappings_count / 20.0, 1.0
            )  # Expect ~20 mappings for good quality
            total_score = min(base_score + validation_bonus + reference_bonus, 1.0)

            return total_score

        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {e}")
            return 0.0

    def _save_optimal_config(
        self, best_result: Dict[str, Any], output_path: str
    ) -> None:
        """Save optimal settings to configuration file."""
        try:
            config_data = {
                "optimal_settings": {
                    "model": best_result["combination"]["model"],
                    "prompt": best_result["combination"]["prompt"],
                    "score": best_result["average_score"],
                    "optimization_timestamp": datetime.now().isoformat(),
                    "optimizer_version": "1.0.0",
                },
                "metadata": {
                    "combinations_tested": best_result.get("combinations_tested", 0),
                    "trials_used_for_testing": best_result.get("trials_tested", 0),
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
        return ["deepseek-coder", "gpt-4", "claude-3", "llama-3"]

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
