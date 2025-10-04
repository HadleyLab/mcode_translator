"""
Ultra-Lean McodePipeline

Zero redundancy, maximum performance. Leverages existing infrastructure.
"""

from typing import Any, Dict, List, Optional

from src.pipeline.document_ingestor import DocumentIngestor
from src.services.llm.service import LLMService
from src.shared.models import (ClinicalTrialData, McodeElement, PipelineResult,
                               ProcessingMetadata, ValidationResult)
from src.utils.config import Config
from src.utils.logging_config import get_logger


class McodePipeline:
    """
    Ultra-lean mCODE pipeline with zero redundancy.

    Direct data flow: Raw Dict → Existing Models → Existing PipelineResult
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
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
        self.model_name = model_name or ("regex" if engine == "regex" else "deepseek-coder")
        self.prompt_name = prompt_name or "direct_mcode_evidence_based_concise"

        # Leverage existing components
        self.document_ingestor = DocumentIngestor()

        # Initialize appropriate service based on engine
        if engine == "regex":
            from src.services.regex.service import RegexService
            self.service = RegexService(self.config, self.model_name, self.prompt_name)
        else:
            self.service = LLMService(self.config, self.model_name, self.prompt_name)

        self.logger.info(f"✅ McodePipeline initialized with {engine} engine")

    async def process(self, trial_data: Dict[str, Any]) -> PipelineResult:
        """
        Process clinical trial data with ultra-lean async data flow.

        Args:
            trial_data: Raw clinical trial data dictionary

        Returns:
            PipelineResult with existing validated models
        """
        self.logger.info(
            f"🚀 Pipeline.process called with trial data keys: "
            f"{list(trial_data.keys())[:5]}..."
        )

        # Validate input using existing ClinicalTrialData model - STRICT: No fallback, fail fast
        validated_trial = ClinicalTrialData(**trial_data)
        self.logger.info(f"Processing trial: {validated_trial.nct_id}")

        # Stage 1: Document processing (existing component) - STRICT: No fallback, fail fast
        sections = self.document_ingestor.ingest_clinical_trial_document(trial_data)
        self.logger.info(f"📄 Document ingestor returned {len(sections)} sections")

        # Stage 2: Processing using configured service - STRICT: No fallback, fail fast
        all_elements = []
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
                    elements = self.service.map_trial_to_mcode(trial_data)
                    self.logger.info(
                        f"✅ REGEX service returned {len(elements)} elements"
                    )
                    all_elements.extend(elements)
                    break  # Regex processes the whole trial at once, not per section
                else:
                    # LLM processing per section
                    self.logger.info(f"🚀 Calling LLM service for section {i+1}")
                    elements = await self.service.map_to_mcode(section.content)
                    self.logger.info(
                        f"✅ LLM service returned {len(elements)} elements "
                        f"for section {i+1}"
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
            original_data=trial_data,
            error=None,
        )

    async def process_batch(
        self, trials_data: List[Dict[str, Any]]
    ) -> List[PipelineResult]:
        """
        Process multiple trials efficiently.

        Args:
            trials_data: List of raw clinical trial data dictionaries

        Returns:
            List of PipelineResult instances
        """
        results = []
        for trial_data in trials_data:
            result = await self.process(trial_data)
            results.append(result)

        self.logger.info(f"Batch processing completed: {len(results)} trials")
        return results

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
