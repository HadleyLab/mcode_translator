"""
Performance Analysis Module - Analyzes optimization results and performance metrics.

This module provides comprehensive analysis of optimization results including
model performance, prompt effectiveness, provider comparisons, and error analysis.
"""

from typing import Any, Dict, List

from src.utils.logging_config import get_logger


class PerformanceAnalyzer:
    """
    Analyzes optimization results and performance metrics.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def analyze_by_category(
        self, all_results: List[Dict], category_key: str
    ) -> Dict[str, Any]:
        """Analyze results by a specific category (model or prompt) with comprehensive error tracking."""
        category_stats = {}

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
            combo_key = (
                f"{combo.get('model', 'unknown')}+{combo.get('prompt', 'unknown')}"
            )
            stats["combinations_tested"].add(combo_key)

            if result.get("success"):
                stats["successful_runs"] += 1
                stats["total_score"] += result.get("cv_average_score", 0)
                stats["scores"].append(result.get("cv_average_score", 0))

                # Performance metrics
                perf = result.get("performance_metrics", {})
                if perf:
                    stats["processing_times"].append(
                        perf.get("processing_time_seconds", 0)
                    )
                    stats["token_usage"].append(perf.get("tokens_used", 0))
                    stats["costs"].append(perf.get("estimated_cost_usd", 0))
            else:
                stats["failed_runs"] += 1

            # Errors - strict categorization
            errors = result.get("errors", [])
            stats["errors"].extend(errors)

            for error in errors:
                error_str = str(error).lower()
                stats["error_patterns"].append(
                    str(error)
                )  # Store full error for pattern analysis

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
                elif (
                    "connection" in error_str
                    or "network" in error_str
                    or "dns" in error_str
                ):
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

    def analyze_by_provider(self, all_results: List[Dict]) -> Dict[str, Any]:
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

        provider_stats = {}

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
                    stats["processing_times"].append(
                        perf.get("processing_time_seconds", 0)
                    )
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

    def summarize_errors(self, all_results: List[Dict]) -> Dict[str, int]:
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
                elif (
                    "connection" in error_str
                    or "network" in error_str
                    or "dns" in error_str
                ):
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
