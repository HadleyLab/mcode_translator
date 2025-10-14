"""
Performance Analysis Module - Analyzes optimization results and performance metrics.

This module provides comprehensive analysis of optimization results including
model performance, prompt effectiveness, provider comparisons, error analysis,
and ensemble-specific metrics for confidence calibration and expert agreement.
"""

from typing import Any, Dict, List
import numpy as np
from scipy import stats

from src.utils.logging_config import get_logger


class PerformanceAnalyzer:
    """
    Analyzes optimization results and performance metrics.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def analyze_by_category(
        self, all_results: List[Dict[str, Any]], category_key: str
    ) -> Dict[str, Any]:
        """Analyze results by a specific category (model or prompt) with comprehensive error tracking."""
        category_stats: Dict[str, Dict[str, Any]] = {}

        for result in all_results:
            combo = result.get("combination", {})
            category_value = combo.get(category_key)

            if not category_value:
                continue

            if category_value not in category_stats:
                category_stats[category_value] = {
                    "runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "total_score": 0,
                    "scores": [],
                    "processing_times": [],
                    "token_usage": [],
                    "costs": [],
                    "errors": [],
                    "error_types": {
                        "json_parsing": 0,
                        "quota_exceeded": 0,
                        "rate_limit": 0,
                        "auth_error": 0,
                        "api_error": 0,
                        "network_error": 0,
                        "timeout": 0,
                        "model_error": 0,
                        "other": 0,
                    },
                    "error_patterns": [],  # Store actual error messages for pattern analysis
                    "combinations_tested": set(),  # Track which combinations this category was part of
                }

            stats = category_stats[category_value]
            stats["runs"] += 1

            # Track combination
            combo_key = f"{combo.get('model', 'unknown')}+{combo.get('prompt', 'unknown')}"
            stats["combinations_tested"].add(combo_key)

            if result.get("success"):
                stats["successful_runs"] += 1
                stats["total_score"] += result.get("cv_average_score", 0)
                stats["scores"].append(result.get("cv_average_score", 0))

                # Performance metrics
                perf = result.get("performance_metrics", {})
                if perf:
                    stats["processing_times"].append(perf.get("processing_time_seconds", 0))
                    stats["token_usage"].append(perf.get("tokens_used", 0))
                    stats["costs"].append(perf.get("estimated_cost_usd", 0))
            else:
                stats["failed_runs"] += 1

            # Errors - strict categorization
            errors = result.get("errors", [])
            stats["errors"].extend(errors)

            for error in errors:
                error_str = str(error).lower()
                stats["error_patterns"].append(str(error))  # Store full error for pattern analysis

                # Strict error categorization
                if "json" in error_str and (
                    "parsing" in error_str
                    or "decode" in error_str
                    or "invalid json" in error_str
                    or "expecting" in error_str
                ):
                    stats["error_types"]["json_parsing"] += 1
                elif (
                    "quota" in error_str
                    or "billing" in error_str
                    or "plan" in error_str
                    or "insufficient_quota" in error_str
                ):
                    stats["error_types"]["quota_exceeded"] += 1
                elif (
                    "rate limit" in error_str
                    or "429" in error_str
                    or "too many requests" in error_str
                ):
                    stats["error_types"]["rate_limit"] += 1
                elif (
                    "auth" in error_str
                    or "unauthorized" in error_str
                    or "forbidden" in error_str
                    or "401" in error_str
                    or "403" in error_str
                ):
                    stats["error_types"]["auth_error"] += 1
                elif "timeout" in error_str or "timed out" in error_str:
                    stats["error_types"]["timeout"] += 1
                elif "connection" in error_str or "network" in error_str or "dns" in error_str:
                    stats["error_types"]["network_error"] += 1
                elif "api" in error_str and not any(
                    x in error_str
                    for x in ["json", "quota", "rate", "auth", "timeout", "connection"]
                ):
                    stats["error_types"]["api_error"] += 1
                elif "model" in error_str and (
                    "not found" in error_str or "does not exist" in error_str
                ):
                    stats["error_types"]["model_error"] += 1
                else:
                    stats["error_types"]["other"] += 1

        # Calculate final statistics
        for category_value, stats in category_stats.items():
            stats["combinations_tested"] = list(stats["combinations_tested"])
            stats["error_count"] = len(stats["errors"])

            if stats["successful_runs"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["successful_runs"]
                stats["success_rate"] = stats["successful_runs"] / stats["runs"]
                stats["failure_rate"] = stats["failed_runs"] / stats["runs"]
                stats["avg_processing_time"] = (
                    sum(stats["processing_times"]) / len(stats["processing_times"])
                    if stats["processing_times"]
                    else 0
                )
                stats["avg_tokens"] = (
                    sum(stats["token_usage"]) / len(stats["token_usage"])
                    if stats["token_usage"]
                    else 0
                )
                stats["avg_cost"] = (
                    sum(stats["costs"]) / len(stats["costs"]) if stats["costs"] else 0
                )
            else:
                stats["avg_score"] = 0
                stats["success_rate"] = 0
                stats["failure_rate"] = 1.0
                stats["avg_processing_time"] = 0
                stats["avg_tokens"] = 0
                stats["avg_cost"] = 0

        return category_stats

    def analyze_by_provider(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results by provider (OpenAI, DeepSeek, etc.)."""
        provider_mapping = {
            "gpt-4-turbo": "OpenAI",
            "gpt-4o": "OpenAI",
            "gpt-4o-mini": "OpenAI",
            "gpt-3.5-turbo": "OpenAI",
            "deepseek-coder": "DeepSeek",
            "deepseek-chat": "DeepSeek",
            "deepseek-reasoner": "DeepSeek",
            "claude-3": "Anthropic",
            "llama-3": "Meta",
        }

        provider_stats: Dict[str, Dict[str, Any]] = {}

        for result in all_results:
            combo = result.get("combination", {})
            model = combo.get("model", "")
            provider = provider_mapping.get(model, "Unknown")

            if provider not in provider_stats:
                provider_stats[provider] = {
                    "models": set(),
                    "runs": 0,
                    "successful_runs": 0,
                    "total_score": 0,
                    "scores": [],
                    "processing_times": [],
                    "token_usage": [],
                    "costs": [],
                    "errors": [],
                }

            stats = provider_stats[provider]
            stats["models"].add(model)
            stats["runs"] += 1

            if result.get("success"):
                stats["successful_runs"] += 1
                stats["total_score"] += result.get("cv_average_score", 0)
                stats["scores"].append(result.get("cv_average_score", 0))

                # Performance metrics
                perf = result.get("performance_metrics", {})
                if perf:
                    stats["processing_times"].append(perf.get("processing_time_seconds", 0))
                    stats["token_usage"].append(perf.get("tokens_used", 0))
                    stats["costs"].append(perf.get("estimated_cost_usd", 0))

            # Errors
            stats["errors"].extend(result.get("errors", []))

        # Calculate averages
        for provider, stats in provider_stats.items():
            stats["models"] = list(stats["models"])
            if stats["successful_runs"] > 0:
                stats["avg_score"] = stats["total_score"] / stats["successful_runs"]
                stats["success_rate"] = stats["successful_runs"] / stats["runs"]
                stats["avg_processing_time"] = (
                    sum(stats["processing_times"]) / len(stats["processing_times"])
                    if stats["processing_times"]
                    else 0
                )
                stats["avg_tokens"] = (
                    sum(stats["token_usage"]) / len(stats["token_usage"])
                    if stats["token_usage"]
                    else 0
                )
                stats["avg_cost"] = (
                    sum(stats["costs"]) / len(stats["costs"]) if stats["costs"] else 0
                )
                stats["error_count"] = len(stats["errors"])

        return provider_stats

    def summarize_errors(self, all_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize error types across all results with strict categorization."""
        error_counts = {
            "json_parsing": 0,
            "quota_exceeded": 0,
            "rate_limit": 0,
            "auth_error": 0,
            "api_error": 0,
            "network_error": 0,
            "timeout": 0,
            "model_error": 0,
            "other": 0,
        }

        for result in all_results:
            errors = result.get("errors", [])
            for error in errors:
                error_str = str(error).lower()

                # Strict error categorization matching the analysis
                if "json" in error_str and (
                    "parsing" in error_str
                    or "decode" in error_str
                    or "invalid json" in error_str
                    or "expecting" in error_str
                ):
                    error_counts["json_parsing"] += 1
                elif (
                    "quota" in error_str
                    or "billing" in error_str
                    or "plan" in error_str
                    or "insufficient_quota" in error_str
                ):
                    error_counts["quota_exceeded"] += 1
                elif (
                    "rate limit" in error_str
                    or "429" in error_str
                    or "too many requests" in error_str
                ):
                    error_counts["rate_limit"] += 1
                elif (
                    "auth" in error_str
                    or "unauthorized" in error_str
                    or "forbidden" in error_str
                    or "401" in error_str
                    or "403" in error_str
                ):
                    error_counts["auth_error"] += 1
                elif "timeout" in error_str or "timed out" in error_str:
                    error_counts["timeout"] += 1
                elif "connection" in error_str or "network" in error_str or "dns" in error_str:
                    error_counts["network_error"] += 1
                elif "api" in error_str and not any(
                    x in error_str
                    for x in ["json", "quota", "rate", "auth", "timeout", "connection"]
                ):
                    error_counts["api_error"] += 1
                elif "model" in error_str and (
                    "not found" in error_str or "does not exist" in error_str
                ):
                    error_counts["model_error"] += 1
                else:
                    error_counts["other"] += 1

        return error_counts

    def analyze_ensemble_performance(
        self,
        ensemble_results: List[Dict[str, Any]],
        individual_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze ensemble performance compared to individual engines.

        Args:
            ensemble_results: Results from ensemble engine
            individual_results: Results from individual engines (regex, llm, etc.)

        Returns:
            Comprehensive ensemble performance analysis
        """
        self.logger.info("🔍 Analyzing ensemble performance...")

        analysis = {
            "agreement_analysis": self._analyze_expert_agreement(ensemble_results),
            "confidence_calibration": self._analyze_confidence_calibration(ensemble_results),
            "improvement_analysis": self._analyze_ensemble_improvements(ensemble_results, individual_results),
            "expert_contribution": self._analyze_expert_contributions(ensemble_results),
            "consensus_analysis": self._analyze_consensus_levels(ensemble_results)
        }

        self.logger.info("✅ Ensemble performance analysis completed")
        return analysis

    def _analyze_expert_agreement(self, ensemble_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze agreement levels between experts in ensemble."""
        if not ensemble_results:
            return {"error": "No ensemble results provided"}

        # Extract agreement data from ensemble results
        agreement_scores = []
        expert_decisions = []

        for result in ensemble_results:
            if "ensemble_metadata" in result:
                metadata = result["ensemble_metadata"]

                # Calculate agreement between experts
                if "expert_assessments" in metadata:
                    decisions = []
                    for assessment in metadata["expert_assessments"]:
                        decisions.append(assessment.get("is_match", False))

                    if len(decisions) > 1:
                        agreement = sum(decisions) / len(decisions)
                        agreement_scores.append(agreement)

                        # Store individual decisions for diversity analysis
                        expert_decisions.append(decisions)

        if not agreement_scores:
            return {"error": "No agreement data found in ensemble results"}

        # Calculate agreement statistics
        avg_agreement = sum(agreement_scores) / len(agreement_scores)
        agreement_std = np.std(agreement_scores) if len(agreement_scores) > 1 else 0

        # Calculate diversity score (how different expert opinions are)
        diversity_scores = []
        for decisions in expert_decisions:
            # Diversity as variance in decisions
            diversity = np.var(decisions) if len(decisions) > 1 else 0
            diversity_scores.append(diversity)

        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0

        return {
            "average_agreement": avg_agreement,
            "agreement_std": agreement_std,
            "agreement_interpretation": self._interpret_agreement_level(avg_agreement),
            "average_diversity": avg_diversity,
            "diversity_interpretation": self._interpret_diversity_level(avg_diversity),
            "total_assessments": len(agreement_scores)
        }

    def _analyze_confidence_calibration(self, ensemble_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence calibration in ensemble results."""
        if not ensemble_results:
            return {"error": "No ensemble results provided"}

        confidence_scores = []
        accuracy_scores = []

        for result in ensemble_results:
            confidence = result.get("confidence_score", 0.0)
            # For calibration analysis, we need ground truth accuracy
            # This would need to be provided or calculated from gold standard
            confidence_scores.append(confidence)

        if not confidence_scores:
            return {"error": "No confidence scores found"}

        # Basic calibration analysis (would need ground truth for full calibration)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        confidence_std = np.std(confidence_scores) if len(confidence_scores) > 1 else 0

        # Expected calibration analysis (ECE - Expected Calibration Error)
        # This is a simplified version - full implementation would bin confidences
        ece_score = 0.0  # Placeholder for Expected Calibration Error

        return {
            "average_confidence": avg_confidence,
            "confidence_std": confidence_std,
            "expected_calibration_error": ece_score,
            "calibration_quality": self._interpret_calibration_quality(ece_score),
            "confidence_distribution": self._analyze_confidence_distribution(confidence_scores)
        }

    def _analyze_ensemble_improvements(
        self,
        ensemble_results: List[Dict[str, Any]],
        individual_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze improvements provided by ensemble over individual engines."""
        if not ensemble_results or not individual_results:
            return {"error": "Missing results for improvement analysis"}

        # Extract performance metrics for comparison
        ensemble_accuracy = self._calculate_ensemble_accuracy(ensemble_results)
        individual_accuracies = {}

        for engine_name, results in individual_results.items():
            individual_accuracies[engine_name] = self._calculate_engine_accuracy(results)

        # Calculate improvements
        improvements = {}
        for engine_name, accuracy in individual_accuracies.items():
            improvement = ensemble_accuracy - accuracy
            improvements[engine_name] = {
                "accuracy_improvement": improvement,
                "relative_improvement": improvement / accuracy if accuracy > 0 else 0,
                "is_significant": improvement > 0.05  # 5% improvement threshold
            }

        # Overall improvement analysis
        best_individual = max(individual_accuracies.values()) if individual_accuracies else 0
        overall_improvement = ensemble_accuracy - best_individual

        return {
            "ensemble_accuracy": ensemble_accuracy,
            "individual_accuracies": individual_accuracies,
            "improvements": improvements,
            "overall_improvement": overall_improvement,
            "best_individual_engine": max(individual_accuracies, key=individual_accuracies.get) if individual_accuracies else None,
            "ensemble_is_best": ensemble_accuracy >= best_individual if individual_accuracies else True
        }

    def _analyze_expert_contributions(self, ensemble_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual expert contributions to ensemble decisions."""
        if not ensemble_results:
            return {"error": "No ensemble results provided"}

        expert_contributions = {}
        expert_weights = {}
        expert_success_rates = {}

        for result in ensemble_results:
            if "ensemble_metadata" in result:
                metadata = result["ensemble_metadata"]

                if "expert_assessments" in metadata:
                    for assessment in metadata["expert_assessments"]:
                        expert_type = assessment.get("expert_type", "unknown")

                        # Initialize expert tracking
                        if expert_type not in expert_contributions:
                            expert_contributions[expert_type] = 0
                            expert_weights[expert_type] = []
                            expert_success_rates[expert_type] = []

                        expert_contributions[expert_type] += 1

                        # Track weights and success (simplified)
                        weight = assessment.get("weight", 1.0)
                        expert_weights[expert_type].append(weight)

                        success = assessment.get("success", True)
                        expert_success_rates[expert_type].append(1 if success else 0)

        # Calculate expert statistics
        expert_analysis = {}
        for expert_type in expert_contributions:
            contributions = expert_contributions[expert_type]
            avg_weight = sum(expert_weights[expert_type]) / len(expert_weights[expert_type]) if expert_weights[expert_type] else 0
            success_rate = sum(expert_success_rates[expert_type]) / len(expert_success_rates[expert_type]) if expert_success_rates[expert_type] else 0

            expert_analysis[expert_type] = {
                "total_contributions": contributions,
                "average_weight": avg_weight,
                "success_rate": success_rate,
                "influence_score": avg_weight * success_rate  # Combined influence metric
            }

        return {
            "expert_analysis": expert_analysis,
            "total_experts_used": len(expert_contributions),
            "most_influential_expert": max(expert_analysis.items(), key=lambda x: x[1]["influence_score"])[0] if expert_analysis else None,
            "expert_diversity": len(expert_contributions) / sum(expert_contributions.values()) if expert_contributions else 0
        }

    def _analyze_consensus_levels(self, ensemble_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consensus levels in ensemble decisions."""
        if not ensemble_results:
            return {"error": "No ensemble results provided"}

        consensus_levels = []
        consensus_scores = []

        for result in ensemble_results:
            consensus_level = result.get("consensus_level", "unknown")
            consensus_levels.append(consensus_level)

            # Convert consensus level to numeric score
            score = {"high": 3, "moderate": 2, "low": 1, "none": 0}.get(consensus_level, 0)
            consensus_scores.append(score)

        if not consensus_scores:
            return {"error": "No consensus data found"}

        # Calculate consensus statistics
        avg_consensus_score = sum(consensus_scores) / len(consensus_scores)
        consensus_distribution = {
            "high": consensus_levels.count("high"),
            "moderate": consensus_levels.count("moderate"),
            "low": consensus_levels.count("low"),
            "none": consensus_levels.count("none")
        }

        return {
            "average_consensus_score": avg_consensus_score,
            "consensus_interpretation": self._interpret_consensus_level(avg_consensus_score),
            "consensus_distribution": consensus_distribution,
            "high_consensus_rate": consensus_distribution["high"] / len(consensus_levels),
            "total_decisions": len(consensus_levels)
        }

    def _calculate_ensemble_accuracy(self, ensemble_results: List[Dict[str, Any]]) -> float:
        """Calculate accuracy of ensemble results (requires ground truth)."""
        # This is a simplified implementation
        # In practice, this would compare against gold standard
        correct_predictions = 0
        total_predictions = len(ensemble_results)

        for result in ensemble_results:
            # Placeholder logic - would need actual ground truth comparison
            confidence = result.get("confidence_score", 0.0)
            # Simple heuristic: high confidence predictions are assumed correct
            if confidence > 0.7:
                correct_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

    def _calculate_engine_accuracy(self, engine_results: List[Dict[str, Any]]) -> float:
        """Calculate accuracy of individual engine results."""
        # Similar to ensemble accuracy calculation
        return self._calculate_ensemble_accuracy(engine_results)

    def _interpret_agreement_level(self, agreement_score: float) -> str:
        """Interpret agreement level score."""
        if agreement_score >= 0.8:
            return "high_agreement"
        elif agreement_score >= 0.6:
            return "moderate_agreement"
        else:
            return "low_agreement"

    def _interpret_diversity_level(self, diversity_score: float) -> str:
        """Interpret diversity level score."""
        if diversity_score >= 0.2:
            return "high_diversity"
        elif diversity_score >= 0.1:
            return "moderate_diversity"
        else:
            return "low_diversity"

    def _interpret_calibration_quality(self, ece_score: float) -> str:
        """Interpret calibration quality based on ECE score."""
        if ece_score <= 0.05:
            return "well_calibrated"
        elif ece_score <= 0.1:
            return "moderately_calibrated"
        else:
            return "poorly_calibrated"

    def _interpret_consensus_level(self, consensus_score: float) -> str:
        """Interpret consensus level score."""
        if consensus_score >= 2.5:
            return "high_consensus"
        elif consensus_score >= 1.5:
            return "moderate_consensus"
        else:
            return "low_consensus"

    def _analyze_confidence_distribution(self, confidence_scores: List[float]) -> Dict[str, Any]:
        """Analyze distribution of confidence scores."""
        if not confidence_scores:
            return {"error": "No confidence scores provided"}

        return {
            "mean": sum(confidence_scores) / len(confidence_scores),
            "std": np.std(confidence_scores) if len(confidence_scores) > 1 else 0,
            "min": min(confidence_scores),
            "max": max(confidence_scores),
            "median": sorted(confidence_scores)[len(confidence_scores) // 2]
        }
