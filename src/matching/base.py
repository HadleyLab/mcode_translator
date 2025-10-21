"""
Base classes for the triple-engine matching system.

Defines the abstract `MatchingEngineBase` class to ensure a consistent
interface for all matching engines (Regex, LLM, CoreMemory).
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional
import asyncio
import json
import hashlib
from datetime import datetime

from shared.models import McodeElement


class MatchingResult:
    """Container for matching results with metadata."""

    def __init__(self, is_match: bool, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.is_match = is_match
        self.error = error
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.cache_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "is_match": self.is_match,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "cache_key": self.cache_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchingResult":
        """Create from dictionary (for cache loading)."""
        is_match = data.get("is_match", False)
        result = cls(is_match, data.get("error"), data.get("metadata", {}))
        result.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        result.cache_key = data.get("cache_key")
        return result


class MatchingEngineBase(ABC):
    """
    Abstract base class for patient-trial matching engines with robust error handling and caching.

    Provides common functionality for all matching engines including
    async streaming support, batch processing, caching, and error recovery.
    """

    def __init__(self, cache_enabled: bool = True, max_retries: int = 3):
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries
        self.cache_manager = None
        if cache_enabled:
            from utils.api_manager import APIManager
            self.cache_manager = APIManager()

    def _generate_cache_key(self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]) -> str:
        """Generate a unique cache key for the matching operation."""
        # Create a deterministic key based on patient and trial data
        patient_str = json.dumps(patient_data, sort_keys=True)
        trial_str = json.dumps(trial_criteria, sort_keys=True)
        combined = f"{patient_str}|{trial_str}|{self.__class__.__name__}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[MatchingResult]:
        """Retrieve cached result if available."""
        if not self.cache_enabled or not self.cache_manager:
            return None

        try:
            cache = self.cache_manager.get_cache("matching")
            cached_data = cache.get_by_key(cache_key)
            if cached_data:
                result = MatchingResult.from_dict(cached_data)
                result.cache_key = cache_key
                return result
        except Exception:
            # Cache miss or error - continue without cache
            pass
        return None

    def _cache_result(self, result: MatchingResult) -> None:
        """Cache the matching result."""
        if not self.cache_enabled or not self.cache_manager or not result.cache_key:
            return

        try:
            cache = self.cache_manager.get_cache("matching")
            cache.set_by_key(result.to_dict(), result.cache_key)
        except Exception:
            # Cache write failure - continue without caching
            pass

    @abstractmethod
    async def match(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> bool:
        """
        Match patient data against trial criteria.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary

        Returns:
            A boolean indicating if a match was found.
        """
        pass

    async def match_with_recovery(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> MatchingResult:
        """
        Match with automatic error recovery and caching.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary

        Returns:
            MatchingResult with elements, error info, and metadata
        """
        cache_key = self._generate_cache_key(patient_data, trial_criteria)

        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # Perform matching with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                is_match = await self.match(patient_data, trial_criteria)
                result = MatchingResult(is_match, metadata={"attempts": attempt + 1, "success": True})
                result.cache_key = cache_key
                self._cache_result(result)
                return result

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    continue

        # All retries failed
        result = MatchingResult(
            False,
            error=last_error,
            metadata={"attempts": self.max_retries + 1, "success": False}
        )
        result.cache_key = cache_key
        self._cache_result(result)
        return result

    async def match_streaming(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async streaming version of patient-trial matching with recovery.

        Yields intermediate results and final matching decision.
        """
        # Start matching process
        yield {"status": "starting", "engine": self.__class__.__name__}

        try:
            # Perform matching with recovery
            result = await self.match_with_recovery(patient_data, trial_criteria)

            if result.error:
                yield {
                    "status": "error",
                    "error": result.error,
                    "metadata": result.metadata,
                }
            else:
                # Yield final result
                yield {
                    "status": "completed",
                    "is_match": result.is_match,
                    "metadata": result.metadata,
                }

        except Exception as e:
            yield {
                "status": "error",
                "error": str(e),
            }

    async def compare_matches(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[MatchingResult]:
        """
        Batch matching for comparison and experimentation with recovery.

        Processes multiple patient-trial pairs concurrently with error recovery.
        """
        tasks = [self.match_with_recovery(req["patient"], req["trial"]) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return only MatchingResult instances
        return [r for r in results if isinstance(r, MatchingResult)]