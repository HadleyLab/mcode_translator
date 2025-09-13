"""Direct mCODE Pipeline Module.

This module provides a high-performance single-step pipeline that maps clinical text
directly to standardized mCODE elements using evidence-based LLM processing.

Classes:
    McodePipeline: Main pipeline class for direct clinical text to mCODE mapping.

Example:
    >>> from src.pipeline import McodePipeline
    >>> pipeline = McodePipeline(prompt_name="direct_mcode_evidence_based_concise")
    >>> result = pipeline.process_clinical_trial(trial_data)
    >>> print(f"Generated {len(result.mcode_mappings)} mCODE mappings")
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from .document_ingestor import DocumentIngestor, DocumentSection
from .mcode_llm import McodeConfigurationError, McodeMapper, McodeMappingError
from .pipeline_base import ProcessingPipeline
from src.shared.models import (
    McodeElement,
    PipelineResult,
    ProcessingMetadata,
    SourceReference,
    TokenUsage,
    ValidationResult,
    clinical_trial_from_dict,
)

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils import Loggable, PromptLoader, global_token_tracker


class McodePipeline(ProcessingPipeline, Loggable):
    """High-performance pipeline for direct clinical text to mCODE mapping.

    This pipeline provides a single-step process that maps clinical trial text
    directly to standardized mCODE elements without intermediate entity extraction.
    Optimized for accuracy and evidence-based mappings.

    Attributes:
        prompt_name (str): Name of the prompt template for mCODE mapping.
        model_name (str): Name of the LLM model for processing.
        document_ingestor (DocumentIngestor): Component for processing trial documents.
        llm_mapper (McodeMapper): LLM-based mCODE mapping component.
    """

    def __init__(
        self,
        prompt_name: str = "direct_mcode_evidence_based_concise",
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize the mCODE pipeline with specified configuration.

        Args:
            prompt_name: Name of the prompt template for mCODE mapping.
                Defaults to evidence-based concise prompt for optimal quality.
            model_name: Name of the model to use for LLM operations.
                If None, uses default from configuration.

        Raises:
            ValueError: If McodeMapper initialization fails due to configuration issues.
        """
        super().__init__()
        self.prompt_name = prompt_name
        self.model_name = model_name
        self.document_ingestor = DocumentIngestor()
        try:
            self.llm_mapper = McodeMapper(
                prompt_name=self.prompt_name, model_name=self.model_name
            )
        except McodeConfigurationError as e:
            raise ValueError(f"Failed to initialize McodeMapper: {str(e)}")

    def process(
        self, data: Dict[str, Any], **kwargs
    ) -> PipelineResult:
        """
        Process clinical trial data (DataProcessor protocol implementation).

        Args:
            data: Clinical trial data dictionary
            **kwargs: Additional processing parameters

        Returns:
            PipelineResult containing processing results
        """
        task_id = kwargs.get('task_id')
        return self.process_clinical_trial(data, task_id)

    def process_clinical_trial(
        self, trial_data: Dict[str, Any], task_id: Optional[str] = None
    ) -> PipelineResult:
        """Process clinical trial data to extract mCODE mappings.

        Args:
            trial_data: Raw clinical trial data containing protocol sections.
            task_id: Optional task identifier for benchmarking and tracking.

        Returns:
            PipelineResult containing:
                - mcode_mappings: List of standardized mCODE elements
                - source_references: Provenance tracking for mappings
                - validation_results: Quality metrics and compliance scores
                - metadata: Processing statistics and configuration info

        Raises:
            McodeMappingError: If LLM-based mapping fails.
            ValueError: If trial data format is invalid.
        """
        try:
            self.logger.info("🚀 Starting mCODE pipeline processing")
            self.logger.info(
                f"   📄 Trial ID: {trial_data.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')}"
            )
            self.logger.info(
                f"   📋 Title: {trial_data.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'unknown')}"
            )

            global_token_tracker.reset()
            document_sections = self.document_ingestor.ingest_clinical_trial_document(
                trial_data
            )

            all_mappings = []
            source_references = []
            mapping_result = None

            for section in document_sections:
                if not section.content or not section.content.strip():
                    continue

                section_context = {
                    "name": section.name,
                    "type": section.source_type,
                    "position": section.position,
                }

                mapping_result = self.llm_mapper.map_to_mcode(
                    entities=[],  # No entities, we are mapping directly from text
                    trial_context=trial_data,
                    source_references=[],
                    clinical_text=section.content,
                )

                all_mappings.extend(mapping_result["mapped_elements"])
                source_references.extend(mapping_result.get("source_references", []))

            # Get the validation results from the last mapping result
            validation_results = {}
            if mapping_result is not None:
                validation_results = mapping_result["validation_results"]

            # Convert mappings to McodeElement instances
            mcode_elements = []
            for mapping in all_mappings:
                try:
                    mcode_elements.append(McodeElement(**mapping))
                except Exception as e:
                    self.logger.warning(f"Failed to convert mapping to McodeElement: {e}")
                    # Keep original if conversion fails
                    mcode_elements.append(mapping)

            # Convert source references to SourceReference instances
            source_refs = []
            for ref in source_references:
                try:
                    source_refs.append(SourceReference(**ref))
                except Exception as e:
                    self.logger.warning(f"Failed to convert source reference: {e}")
                    # Keep original if conversion fails
                    source_refs.append(ref)

            # Create validation result
            try:
                validation_result = ValidationResult(**validation_results)
            except Exception as e:
                self.logger.warning(f"Failed to convert validation results: {e}")
                validation_result = ValidationResult(
                    compliance_score=validation_results.get("compliance_score", 0.0)
                )

            # Create token usage
            token_usage = None
            if mapping_result is not None:
                token_data = mapping_result["metadata"].get("token_usage", {})
                try:
                    token_usage = TokenUsage(**token_data)
                except Exception as e:
                    self.logger.warning(f"Failed to convert token usage: {e}")

            # Create processing metadata
            metadata = ProcessingMetadata(
                engine_type="LLM",
                entities_count=0,
                mapped_count=len(all_mappings),
                compliance_score=validation_result.compliance_score,
                token_usage=token_usage,
                aggregate_token_usage=global_token_tracker.get_total_usage().to_dict(),
            )

            return PipelineResult(
                extracted_entities=[],  # No entities extracted in direct mapping
                mcode_mappings=mcode_elements,
                source_references=source_refs,
                validation_results=validation_result,
                metadata=metadata,
                original_data=trial_data,
                error=None,
            )

        except (McodeMappingError, ValueError) as e:
            self.logger.error(f"mCODE pipeline processing FAILED: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error in mCODE pipeline processing: {str(e)}"
            )
            raise RuntimeError(f"Unexpected pipeline error: {str(e)}")
