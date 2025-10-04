"""
Unified Processor for mCODE Trial Processing

Provides a single interface to choose between RegexEngine and LLMEngine processing methods.
Uses ProcessingService for pipeline orchestration and engine-agnostic processing.
Users can select their preferred approach based on speed, cost, and accuracy requirements.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.services.processing_service import ProcessingService, ProcessingEngine
from src.services.llm.engine import LLMEngine
from src.services.regex.engine import RegexEngine
from src.services.summarizer import McodeSummarizer
from src.utils.logging_config import get_logger


@dataclass
class ProcessingResult:
    """Result of trial processing with metadata about the engine used."""

    success: bool
    data: Any
    engine: str
    processing_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TrialProcessor(ABC):
    """Abstract base class for trial processing engines."""

    @abstractmethod
    async def process_trial(self, trial_data: Dict[str, Any]) -> ProcessingResult:
        """Process a single trial and return the result."""
        pass

    @abstractmethod
    async def process_batch(self, trials_data: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple trials and return the results."""
        pass


class BaseProcessor(TrialProcessor):
    """Base processor that handles common functionality for both Regex and LLM processing."""

    def __init__(self, engine: ProcessingEngine, engine_name: str, include_codes: bool = True, **engine_kwargs: Any):
        self.logger = get_logger(__name__)
        self.engine = engine
        self.processing_service = ProcessingService(engine=self.engine)
        self.summarizer = McodeSummarizer(detail_level="full", include_mcode=include_codes)
        self.engine_name = engine_name

    async def process_trial(self, trial_data: Dict[str, Any]) -> ProcessingResult:
        """Process a single trial using the configured engine."""
        start_time = time.time()

        try:
            # Process trial using ProcessingService (handles pipeline orchestration)
            result = await self.processing_service.process_trial(trial_data)

            if result.error:
                raise Exception(result.error or "Processing failed")

            # Generate summary using UnifiedSummarizer
            nct_id = self._extract_nct_id(trial_data)
            summary = self.summarizer.create_trial_summary(trial_data)

            processing_time = time.time() - start_time

            return ProcessingResult(
                success=True,
                data=summary,
                engine=self.engine_name,
                processing_time=processing_time,
                metadata=self._get_engine_metadata(result, summary)
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"{self.engine_name.upper()} processing failed: {e}")

            return ProcessingResult(
                success=False,
                data=None,
                engine=self.engine_name,
                processing_time=processing_time,
                error_message=str(e),
            )

    async def process_batch(self, trials_data: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """Process multiple trials using the configured engine."""
        results: List[ProcessingResult] = []
        for trial_data in trials_data:
            result = await self.process_trial(trial_data)
            results.append(result)
        return results

    def _extract_nct_id(self, trial_data: Dict[str, Any]) -> str:
        """Extract NCT ID from trial data for use as subject."""
        protocol_section = trial_data.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        return identification.get("nctId", "Unknown")

    def _get_engine_metadata(self, result: Any, summary: str) -> Dict[str, Any]:
        """Get engine-specific metadata - to be overridden by subclasses."""
        return {
            "elements_extracted": len(result.mcode_mappings),
            "summary_length": len(summary)
        }


class RegexProcessor(BaseProcessor):
    """Regex-based processor using ProcessingService with RegexEngine for fast, deterministic processing."""

    def __init__(self, include_codes: bool = True) -> None:
        super().__init__(engine=RegexEngine(), engine_name="regex", include_codes=include_codes)

    def _get_engine_metadata(self, result: Any, summary: str) -> Dict[str, Any]:
        """Get regex-specific metadata."""
        return {
            **super()._get_engine_metadata(result, summary),
            "approach": "structured_extraction",
            "llm_calls": False,
            "deterministic": True,
            "fast_processing": True
        }


class LLMProcessor(BaseProcessor):
    """LLM-based processor using ProcessingService with LLMEngine for flexible, intelligent processing."""

    def __init__(self, model_name: str = "deepseek-coder", prompt_name: str = "direct_mcode_evidence_based_concise", include_codes: bool = True):
        super().__init__(
            engine=LLMEngine(model_name=model_name, prompt_name=prompt_name),
            engine_name="llm",
            include_codes=include_codes
        )

    def _get_engine_metadata(self, result: Any, summary: str) -> Dict[str, Any]:
        """Get LLM-specific metadata."""
        # Cast to LLMEngine to access specific attributes
        llm_engine = self.engine  # type: ignore
        return {
            **super()._get_engine_metadata(result, summary),
            "approach": "llm_enhanced",
            "model": getattr(llm_engine, 'model_name', 'unknown'),
            "prompt": getattr(llm_engine, 'prompt_name', 'unknown'),
            "llm_calls": True,
            "flexible": True,
            "intelligent_processing": True
        }


class UnifiedTrialProcessor:
    """
    Unified processor that allows users to choose between RegexEngine and LLMEngine processing.

    Provides a single interface with engine selection and automatic fallbacks.
    """

    def __init__(
        self,
        default_engine: str = "regex",
        llm_model: str = "deepseek-coder",
        llm_prompt: str = "direct_mcode_evidence_based_concise",
        include_codes: bool = True
    ):
        """
        Initialize the unified processor.

        Args:
            default_engine: Default processing engine ('regex' or 'llm')
            llm_model: LLM model name for LLM-based processing
            llm_prompt: Prompt name for LLM-based processing
            include_codes: Whether to include mCODE codes in summaries
        """
        self.logger = get_logger(__name__)
        self.default_engine = default_engine

        # Initialize processors
        self.regex_processor = RegexProcessor(include_codes=include_codes)
        self.llm_processor = LLMProcessor(model_name=llm_model, prompt_name=llm_prompt, include_codes=include_codes)

        self.logger.info(f"✅ Unified processor initialized with default engine: {default_engine}")

    def get_processor(self, engine: Optional[str] = None) -> TrialProcessor:
        """Get the appropriate processor based on engine selection."""
        engine = engine or self.default_engine

        if engine == "regex":
            return self.regex_processor
        elif engine == "llm":
            return self.llm_processor
        else:
            raise ValueError(f"Unknown processing engine: {engine}. Use 'regex' or 'llm'.")

    async def process_trial(
        self,
        trial_data: Dict[str, Any],
        engine: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single trial using the specified engine.

        Args:
            trial_data: Clinical trial data dictionary
            engine: Processing engine ('regex' or 'llm')

        Returns:
            ProcessingResult with the result and metadata
        """
        processor = self.get_processor(engine)
        return await processor.process_trial(trial_data)

    async def process_batch(
        self,
        trials_data: List[Dict[str, Any]],
        engine: Optional[str] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple trials using the specified engine.

        Args:
            trials_data: List of clinical trial data dictionaries
            engine: Processing engine ('regex' or 'llm')

        Returns:
            List of ProcessingResult instances
        """
        processor = self.get_processor(engine)
        return await processor.process_batch(trials_data)

    def recommend_engine(self, trial_data: Dict[str, Any]) -> str:
        """
        Recommend the best processing engine based on trial characteristics.

        Args:
            trial_data: Clinical trial data dictionary

        Returns:
            Recommended engine ('regex' or 'llm')
        """
        # Simple heuristic: use regex for structured data, LLM for complex cases

        # Check if trial has well-structured data
        protocol_section = trial_data.get("protocolSection", {})
        has_design_module = "designModule" in protocol_section
        has_status_module = "statusModule" in protocol_section

        # If data is well-structured, regex is usually sufficient and faster
        if has_design_module and has_status_module:
            return "regex"

        # For complex or unstructured data, LLM might be better
        return "llm"

    async def compare_engines_async(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare both engines for a given trial to help users choose.
        This is the async version that should be used in async contexts.

        Args:
            trial_data: Clinical trial data dictionary

        Returns:
            Comparison results with performance metrics
        """
        # Run both engines and compare performance
        regex_result = None
        llm_result = None

        try:
            # Test regex engine
            regex_result = await self.regex_processor.process_trial(trial_data)
        except Exception as e:
            self.logger.warning(f"Regex engine failed in comparison: {e}")

        try:
            # Test LLM engine
            llm_result = await self.llm_processor.process_trial(trial_data)
        except Exception as e:
            self.logger.warning(f"LLM engine failed in comparison: {e}")

        # Build comparison results
        comparison: Dict[str, Any] = {
            "recommendation": self.recommend_engine(trial_data),
            "regex": {
                "available": regex_result is not None,
                "success": regex_result.success if regex_result else False,
                "processing_time": regex_result.processing_time if regex_result else None,
                "advantages": ["fast_processing", "cost_effective", "deterministic", "reliable"],
                "characteristics": ["structured_data", "rule_based_extraction", "consistent_results"]
            },
            "llm": {
                "available": llm_result is not None,
                "success": llm_result.success if llm_result else False,
                "processing_time": llm_result.processing_time if llm_result else None,
                "advantages": ["flexible", "handles_complexity", "intelligent", "unstructured_data"],
                "characteristics": ["ai_powered", "adaptive_processing", "advanced_patterns"]
            },
        }

        # Add performance analysis
        if regex_result and llm_result and regex_result.success and llm_result.success:
            time_diff = llm_result.processing_time - regex_result.processing_time
            comparison["performance_analysis"] = {
                "time_difference": time_diff,
                "speedup_factor": regex_result.processing_time / llm_result.processing_time if llm_result.processing_time > 0 else float('inf'),
                "winner": "regex" if regex_result.processing_time < llm_result.processing_time else "llm"
            }

        return comparison

    def compare_engines(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare both engines for a given trial to help users choose.
        This is the sync version that handles async internally.

        Args:
            trial_data: Clinical trial data dictionary

        Returns:
            Comparison results with performance metrics
        """
        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # If we're in an async context, we need to create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(self.compare_engines_async(trial_data)))
                return future.result(timeout=60)  # 60 second timeout
        except RuntimeError:
            # No event loop running, we can use asyncio.run safely
            return asyncio.run(self.compare_engines_async(trial_data))

    async def benchmark_engines(self, trial_data: Dict[str, Any], iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark both engines with multiple iterations for accurate performance comparison.

        Args:
            trial_data: Clinical trial data dictionary
            iterations: Number of iterations for benchmarking

        Returns:
            Detailed benchmark results
        """
        import statistics

        regex_times = []
        llm_times = []
        regex_successes = 0
        llm_successes = 0

        # Benchmark regex engine
        for i in range(iterations):
            try:
                result = await self.regex_processor.process_trial(trial_data)
                regex_times.append(result.processing_time)
                if result.success:
                    regex_successes += 1
            except Exception as e:
                self.logger.warning(f"Regex iteration {i+1} failed: {e}")

        # Benchmark LLM engine
        for i in range(iterations):
            try:
                result = await self.llm_processor.process_trial(trial_data)
                llm_times.append(result.processing_time)
                if result.success:
                    llm_successes += 1
            except Exception as e:
                self.logger.warning(f"LLM iteration {i+1} failed: {e}")

        # Calculate statistics
        benchmark_results: Dict[str, Any] = {
            "iterations": iterations,
            "regex": {
                "success_rate": regex_successes / iterations,
                "avg_time": statistics.mean(regex_times) if regex_times else None,
                "min_time": min(regex_times) if regex_times else None,
                "max_time": max(regex_times) if regex_times else None,
                "stdev_time": statistics.stdev(regex_times) if len(regex_times) > 1 else None,
            },
            "llm": {
                "success_rate": llm_successes / iterations,
                "avg_time": statistics.mean(llm_times) if llm_times else None,
                "min_time": min(llm_times) if llm_times else None,
                "max_time": max(llm_times) if llm_times else None,
                "stdev_time": statistics.stdev(llm_times) if len(llm_times) > 1 else None,
            },
            "recommendation": self.recommend_engine(trial_data)
        }

        # Add relative performance comparison
        if regex_times and llm_times:
            avg_regex_time = statistics.mean(regex_times)
            avg_llm_time = statistics.mean(llm_times)

            benchmark_results["relative_performance"] = {
                "speedup_factor": avg_llm_time / avg_regex_time if avg_regex_time > 0 else float('inf'),
                "faster_engine": "regex" if avg_regex_time < avg_llm_time else "llm",
                "time_savings": abs(avg_llm_time - avg_regex_time)
            }

        return benchmark_results