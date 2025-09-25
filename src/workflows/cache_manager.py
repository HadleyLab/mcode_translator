"""
Cache Manager - Handle caching for trial processing workflows.

This module provides caching functionality for processed trials,
mCODE extractions, and summaries to improve performance.
"""

import hashlib
from typing import Any, Dict, Optional

from src.utils.api_manager import APIManager


class TrialCacheManager:
    """Manage caching for trial processing operations."""

    def __init__(self):
        """Initialize the cache manager."""
        api_manager = APIManager()
        self.workflow_cache = api_manager.get_cache(
            "trials_processor"
        )  # LLM processing results
        self.summary_cache = api_manager.get_cache(
            "trial_summaries"
        )  # LLM summary results

    def get_cached_trial_result(
        self, trial: Dict[str, Any], model: str, prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached processed trial result if available."""
        trial_id = self._extract_trial_id(trial)
        cache_key_data = {
            "function": "processed_trial",
            "trial_id": trial_id,
            "model": model,
            "prompt": prompt,
            "trial_hash": hashlib.md5(str(trial).encode("utf-8")).hexdigest()[:8],
        }

        cached_result = self.workflow_cache.get_by_key(cache_key_data)
        if cached_result is not None:
            print(f"Cache HIT for processed trial {trial_id}")
            return cached_result
        return None

    def cache_trial_result(
        self, processed_trial: Dict[str, Any], model: str, prompt: str
    ) -> None:
        """Cache processed trial result."""
        trial_id = self._extract_trial_id(processed_trial)
        cache_key_data = {
            "function": "processed_trial",
            "trial_id": trial_id,
            "model": model,
            "prompt": prompt,
            "trial_hash": hashlib.md5(str(processed_trial).encode("utf-8")).hexdigest()[
                :8
            ],
        }

        # Convert McodeElement objects to dicts for JSON serialization
        serializable_trial = self._make_trial_serializable(processed_trial)
        self.workflow_cache.set_by_key(serializable_trial, cache_key_data)
        print(f"Cached processed trial {trial_id}")

    # Removed mCODE extraction caching - it's pure computation, not API calls

    def get_cached_natural_language_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> Optional[str]:
        """Generate natural language summary with caching."""
        cache_key_data = {
            "function": "natural_language_summary",
            "trial_id": trial_id,
            "mcode_hash": hashlib.md5(str(mcode_elements).encode("utf-8")).hexdigest()[
                :8
            ],
            "trial_hash": hashlib.md5(str(trial_data).encode("utf-8")).hexdigest()[:8],
        }

        cached_result = self.summary_cache.get_by_key(cache_key_data)
        if cached_result is not None:
            print(f"Cache HIT for natural language summary {trial_id}")
            return cached_result
        return None

    def cache_natural_language_summary(
        self,
        summary: str,
        trial_id: str,
        mcode_elements: Dict[str, Any],
        trial_data: Dict[str, Any],
    ) -> None:
        """Cache natural language summary."""
        cache_key_data = {
            "function": "natural_language_summary",
            "trial_id": trial_id,
            "mcode_hash": hashlib.md5(str(mcode_elements).encode("utf-8")).hexdigest()[
                :8
            ],
            "trial_hash": hashlib.md5(str(trial_data).encode("utf-8")).hexdigest()[:8],
        }

        self.summary_cache.set_by_key(summary, cache_key_data)
        print(f"Cached natural language summary for {trial_id}")

    def clear_all_caches(self) -> None:
        """Clear all workflow-related caches (only API calls)."""
        self.workflow_cache.clear_cache()
        self.summary_cache.clear_cache()
        print("Cleared all workflow caches (API calls only)")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics for all workflow caches."""
        return {
            "workflow_cache": self.workflow_cache.get_stats(),  # LLM processing results
            "summary_cache": self.summary_cache.get_stats(),  # LLM summary results
        }

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        try:
            return trial["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"unknown_trial_{hashlib.md5(str(trial).encode('utf-8')).hexdigest()[:8]}"

    def _make_trial_serializable(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Convert McodeElement objects to dictionaries for JSON serialization."""
        import copy

        serializable_trial = copy.deepcopy(trial)

        # Convert McodeResults if present
        if "McodeResults" in serializable_trial:
            mcode_results = serializable_trial["McodeResults"]
            if "mcode_mappings" in mcode_results:
                # Convert McodeElement objects to dicts
                mappings = []
                for mapping in mcode_results["mcode_mappings"]:
                    if hasattr(mapping, "model_dump"):  # Pydantic model
                        mappings.append(mapping.model_dump())
                    else:  # Already a dict
                        mappings.append(mapping)
                mcode_results["mcode_mappings"] = mappings

        return serializable_trial
