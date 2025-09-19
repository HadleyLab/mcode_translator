"""
Ultra-Lean McodePipeline

Zero redundancy, maximum performance. Leverages existing infrastructure.
"""

from typing import Any, Dict, List

from src.pipeline.document_ingestor import DocumentIngestor
from src.pipeline.llm_service import LLMService
from src.shared.models import (
    ClinicalTrialData,
    McodeElement,
    PipelineResult,
    ProcessingMetadata,
    ValidationResult,
)
from src.utils.config import Config
from src.utils.logging_config import get_logger


class McodePipeline:
    """
    Ultra-lean mCODE pipeline with zero redundancy.

    Direct data flow: Raw Dict → Existing Models → Existing PipelineResult
    """

    def __init__(
        self,
        model_name: str = None,
        prompt_name: str = None,
        config: Config = None
    ):
        """
        Initialize with existing infrastructure.

        Args:
            model_name: LLM model name (uses existing llm_loader)
            prompt_name: Prompt template name (uses existing prompt_loader)
            config: Existing Config instance
        """
        self.config = config or Config()
        self.model_name = model_name or "deepseek-coder"
        self.prompt_name = prompt_name or "direct_mcode_evidence_based_concise"
        self.logger = get_logger(__name__)

        # Leverage existing components
        self.document_ingestor = DocumentIngestor()
        self.llm_service = LLMService(
            self.config, self.model_name, self.prompt_name
        )

    async def process(self, trial_data: Dict[str, Any]) -> PipelineResult:
        """
        Process clinical trial data with ultra-lean async data flow.

        Args:
            trial_data: Raw clinical trial data dictionary

        Returns:
            PipelineResult with existing validated models
        """
        try:
            # Validate input using existing ClinicalTrialData model
            validated_trial = ClinicalTrialData(**trial_data)
            self.logger.info(f"Processing trial: {validated_trial.nct_id}")

            # Stage 1: Document processing (existing component)
            sections = self.document_ingestor.ingest_clinical_trial_document(trial_data)

            # Stage 2: Async LLM processing (existing utils)
            all_elements = []
            for section in sections:
                if section.content and section.content.strip():
                    elements = await self.llm_service.map_to_mcode(section.content)
                    all_elements.extend(elements)

            # Stage 3: Return existing PipelineResult (no wrapper)
            return PipelineResult(
                extracted_entities=[],  # Direct mapping, no intermediate entities
                mcode_mappings=all_elements,
                source_references=[],  # Could populate if needed
                validation_results=ValidationResult(
                    compliance_score=self._calculate_compliance_score(all_elements)
                ),
                metadata=ProcessingMetadata(
                    engine_type="LLM",
                    entities_count=0,
                    mapped_count=len(all_elements),
                    model_used=self.model_name,
                    prompt_used=self.prompt_name,
                ),
                original_data=trial_data,
                error=None
            )

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            return PipelineResult(
                extracted_entities=[],
                mcode_mappings=[],
                source_references=[],
                validation_results=ValidationResult(compliance_score=0.0),
                metadata=ProcessingMetadata(engine_type="LLM"),
                original_data=trial_data,
                error=str(e)
            )

    def process_batch(self, trials_data: List[Dict[str, Any]]) -> List[PipelineResult]:
        """
        Process multiple trials efficiently.

        Args:
            trials_data: List of raw clinical trial data dictionaries

        Returns:
            List of PipelineResult instances
        """
        results = []
        for trial_data in trials_data:
            result = self.process(trial_data)
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
