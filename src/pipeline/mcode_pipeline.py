"""
A single-step pipeline to map clinical text directly to mCODE entities.
"""

import json
from typing import Dict, List, Any, Optional

from .pipeline_base import ProcessingPipeline, PipelineResult
from .mcode_mapper import McodeMapper, McodeMappingError, McodeConfigurationError
from .document_ingestor import DocumentIngestor, DocumentSection
import sys
import os

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import (
    Loggable,
    PromptLoader,
    global_token_tracker
)

class McodePipeline(ProcessingPipeline, Loggable):
    """
    A single-step pipeline that maps clinical text directly to mCODE entities.
    """

    def __init__(self, prompt_name: str = "direct_mcode"):
        """
        Initialize the mCODE pipeline.

        Args:
            prompt_name: Name of the prompt template for mCODE mapping.
        """
        super().__init__()
        self.prompt_name = prompt_name
        self.document_ingestor = DocumentIngestor()
        try:
            self.llm_mapper = McodeMapper(prompt_name=self.prompt_name)
        except McodeConfigurationError as e:
            raise ValueError(f"Failed to initialize McodeMapper: {str(e)}")

    def process_clinical_trial(self, trial_data: Dict[str, Any], task_id: Optional[str] = None) -> PipelineResult:
        """
        Process complete clinical trial data through the mCODE pipeline.

        Args:
            trial_data: Raw clinical trial data from API or source.
            task_id: Optional task ID for associating with a BenchmarkTask.

        Returns:
            PipelineResult with mCODE mappings and source tracking.
        """
        try:
            self.logger.info("🚀 Starting mCODE pipeline processing")
            self.logger.info(f"   📄 Trial ID: {trial_data.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')}")
            self.logger.info(f"   📋 Title: {trial_data.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'unknown')}")

            global_token_tracker.reset()
            document_sections = self.document_ingestor.ingest_clinical_trial_document(trial_data)
            
            all_mappings = []
            source_references = []
            mapping_result = None

            for section in document_sections:
                if not section.content or not section.content.strip():
                    continue

                section_context = {
                    'name': section.name,
                    'type': section.source_type,
                    'position': section.position
                }
                
                mapping_result = self.llm_mapper.map_to_mcode(
                    entities=[],  # No entities, we are mapping directly from text
                    trial_context=trial_data,
                    source_references=[],
                    clinical_text=section.content
                )
                
                all_mappings.extend(mapping_result['mapped_elements'])
                source_references.extend(mapping_result.get('source_references', []))

            # Get the validation results from the last mapping result
            validation_results = {}
            if mapping_result is not None:
                validation_results = mapping_result['validation_results']

            return PipelineResult(
                extracted_entities=[],
                mcode_mappings=all_mappings,
                source_references=source_references,
                validation_results=validation_results,
                metadata={
                    'pipeline_version': 'Mcode_pipeline_v1',
                    'engine_type': 'LLM',
                    'entities_count': 0,
                    'mapped_count': len(all_mappings),
                    'compliance_score': validation_results.get('compliance_score', 0.0),
                    'token_usage': mapping_result['metadata'].get('token_usage', {}) if mapping_result is not None else {},
                    'aggregate_token_usage': global_token_tracker.get_total_usage().to_dict()
                },
                original_data=trial_data,
                error=None
            )

        except (McodeMappingError, ValueError) as e:
            self.logger.error(f"mCODE pipeline processing FAILED: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in mCODE pipeline processing: {str(e)}")
            raise RuntimeError(f"Unexpected pipeline error: {str(e)}")