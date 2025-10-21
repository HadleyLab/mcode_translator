"""
Ultra-Lean McodePipeline

Zero redundancy, maximum performance. Leverages existing infrastructure.
"""

import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Union

from pipeline.document_ingestor import DocumentIngestor

if TYPE_CHECKING:
    from services.regex.service import RegexService
from services.llm.service import LLMService
from shared.models import (
    ClinicalTrialData,
    McodeElement,
    McodeMappingResponse,
    PipelineResult,
    ProcessingMetadata,
    ValidationResult,
)
from utils.config import Config
from utils.logging_config import get_logger

# Sentinel for default values
_PROMPT_DEFAULT = object()


class McodePipeline:
    """
    Ultra-lean mCODE pipeline with zero redundancy.

    Direct data flow: Raw Dict → Existing Models → Existing PipelineResult
    """

    service: Union["RegexService", LLMService]

    def __init__(
        self,
        model_name: Optional[str] = None,
        prompt_name: Any = _PROMPT_DEFAULT,
        config: Optional[Config] = None,
        engine: str = "llm",
    ):
        """
        Initialize with existing infrastructure.

        Args:
            model_name: LLM model name (uses existing llm_loader) or "regex" for regex engine
            prompt_name: Prompt template name (uses existing prompt_loader)
            config: Existing Config instance
            engine: Processing engine ("llm" or "regex")
        """
        self.logger = get_logger(__name__)
        self.logger.info(
            f"🔧 McodePipeline initializing with engine: {engine}, model: {model_name}, "
            f"prompt: {prompt_name}"
        )

        self.config = config or Config()
        self.engine = engine

        # Validate model_name
        if model_name is not None and (not isinstance(model_name, str) or model_name.strip() == ""):
            raise ValueError("model_name cannot be empty or None if provided")
        self.model_name = model_name or ("regex" if engine == "regex" else "deepseek-coder")

        # Handle prompt_name
        if prompt_name is _PROMPT_DEFAULT:
            self.prompt_name = "direct_mcode_evidence_based_concise"
        else:
            self.prompt_name = prompt_name  # Could be None if explicitly passed

        # Leverage existing components
        self.document_ingestor = DocumentIngestor()

        # Initialize appropriate service based on engine
        if engine == "regex":
            # Lazy import to avoid circular dependency
            from services.regex.service import RegexService
            self.service = RegexService(self.config, self.model_name, self.prompt_name)
        else:
            self.service = LLMService(self.config, self.model_name, self.prompt_name)

        self.logger.info(f"✅ McodePipeline initialized with {engine} engine")

    async def process(self, trial_data: ClinicalTrialData) -> PipelineResult:
        """
        Process clinical trial data with ultra-lean async data flow.

        Args:
            trial_data: Raw clinical trial data dictionary

        Returns:
            PipelineResult with existing validated models
        """
        # Log trial data information
        if isinstance(trial_data, dict):
            self.logger.info(
                f"🚀 Pipeline.process called with trial data keys: " f"{list(trial_data.keys())[:5]}..."
            )
        else:
            self.logger.info(
                "🚀 Pipeline.process called with ClinicalTrialData model"
            )

        # Input is already a validated ClinicalTrialData model
        self.logger.info(f"Processing trial: {trial_data.nct_id}")

        # Stage 1: Document processing (existing component) - STRICT: No fallback, fail fast
        sections = self.document_ingestor.ingest_clinical_trial_document(trial_data.model_dump())
        self.logger.info(f"📄 Document ingestor returned {len(sections)} sections")

        # Stage 2: Processing using configured service - STRICT: No fallback, fail fast
        all_elements: List[McodeElement] = []
        for i, section in enumerate(sections):
            self.logger.info(
                f"🔍 Processing section {i+1}/{len(sections)}: "
                f"'{section.name}' (content length: "
                f"{len(section.content) if section.content else 0})"
            )
            if section.content and section.content.strip():
                if self.engine == "regex":
                    # For regex, we need full trial data, not sections
                    # Use the service's trial mapping method
                    elements = self.service.map_trial_to_mcode(trial_data.model_dump())  # type: ignore
                    self.logger.info(f"✅ REGEX service returned {len(elements)} elements")
                    all_elements.extend(elements)
                    break  # Regex processes the whole trial at once, not per section
                else:
                    # LLM processing per section
                    self.logger.info(f"🚀 Calling LLM service for section {i+1}")
                    response = await self.service.map_to_mcode(section.content)
                    elements = response.mcode_elements if isinstance(response, McodeMappingResponse) else []
                    self.logger.info(
                        f"✅ LLM service returned {len(elements)} elements " f"for section {i+1}"
                    )
                    all_elements.extend(elements)
            else:
                self.logger.info(f"⚠️ Skipping empty section {i+1}")

        # Stage 3: Return existing PipelineResult (no wrapper) - STRICT: No fallback, fail fast
        return PipelineResult(
            extracted_entities=[],  # Direct mapping, no intermediate entities
            mcode_mappings=all_elements,
            source_references=[],  # Could populate if needed
            validation_results=ValidationResult(
                compliance_score=self._calculate_compliance_score(all_elements)
            ),
            metadata=ProcessingMetadata(
                engine_type=self.engine.upper(),
                entities_count=0,
                mapped_count=len(all_elements),
                model_used=self.model_name if self.engine == "llm" else None,
                prompt_used=self.prompt_name if self.engine == "llm" else None,
                processing_time_seconds=None,
                token_usage=None,
            ),
            original_data=trial_data.model_dump(),
            error=None,
        )

    async def process_batch(self, trials_data: List[ClinicalTrialData]) -> List[PipelineResult]:
        """
        Process multiple trials efficiently using asyncio.gather for concurrent batch processing.

        Args:
            trials_data: List of raw clinical trial data dictionaries

        Returns:
            List of PipelineResult instances
        """
        self.logger.info(f"🚀 Starting concurrent batch processing of {len(trials_data)} trials")

        # Use asyncio.gather for concurrent processing instead of sequential loop
        tasks = [self.process(trial_data) for trial_data in trials_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during processing
        processed_results: List[PipelineResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                self.logger.error(f"❌ Batch processing failed for trial {i}: {result}")
                # Create error result for failed trial
                error_result = PipelineResult(
                    extracted_entities=[],
                    mcode_mappings=[],
                    source_references=[],
                    validation_results=ValidationResult(compliance_score=0.0),
                    metadata=ProcessingMetadata(
                        engine_type=self.engine.upper(),
                        entities_count=0,
                        mapped_count=0,
                        model_used=self.model_name if self.engine == "llm" else None,
                        prompt_used=self.prompt_name if self.engine == "llm" else None,
                        processing_time_seconds=None,
                        token_usage=None,
                    ),
                    original_data={},  # Empty dict for error cases
                    error=str(result),
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        self.logger.info(f"✅ Batch processing completed: {len(processed_results)} trials")
        return processed_results

    def _calculate_compliance_score(self, elements: List[McodeElement]) -> float:
        """
        Calculate basic compliance score.

        Args:
            elements: List of mapped mCODE elements

        Returns:
            Compliance score between 0.0 and 1.0
        """
        if not elements:
            return 0.0

        # Basic scoring: presence of required element types
        required_types = {"CancerCondition", "CancerTreatment", "TumorMarker"}
        found_types = {elem.element_type for elem in elements}

        compliance = len(found_types.intersection(required_types)) / len(required_types)
        return min(compliance, 1.0)  # Cap at 1.0
