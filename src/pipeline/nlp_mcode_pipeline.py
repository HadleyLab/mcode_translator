"""
NLP Extraction to Mcode Mapping Pipeline - A two-step pipeline that first extracts NLP entities and then maps them to Mcode.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from .pipeline_base import ProcessingPipeline, PipelineResult
from .mcode_mapper import McodeMapper, SourceReference, McodeMappingError, McodeConfigurationError
from .nlp_extractor import NlpLlm, NlpExtractionError, NlpConfigurationError
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

class NlpMcodePipeline(ProcessingPipeline, Loggable):
    """
    A two-step pipeline that first extracts NLP entities from clinical text and then
    maps those entities to the Mcode standard.
    """

    def __init__(self, extraction_prompt_name: str = "generic_extraction", mapping_prompt_name: str = "generic_mapping"):
        """
        Initialize the NLP Extraction to Mcode Mapping Pipeline.

        Args:
            extraction_prompt_name: Name of the prompt template for entity extraction.
            mapping_prompt_name: Name of the prompt template for Mcode mapping.
        """
        super().__init__()
        self.extraction_prompt_name = extraction_prompt_name
        self.mapping_prompt_name = mapping_prompt_name

        self.document_ingestor = DocumentIngestor()
        try:
            self.nlp_extractor = NlpLlm(prompt_name=self.extraction_prompt_name)
            self.llm_mapper = McodeMapper(prompt_name=self.mapping_prompt_name)
        except (NlpConfigurationError, McodeConfigurationError) as e:
            raise ValueError(f"Failed to initialize pipeline components: {str(e)}")
    
    def process_clinical_text(self, clinical_text: str, context: Dict[str, Any] = None, task_id: Optional[str] = None) -> PipelineResult:
        """
        ATOMIC processor for clinical text - the most fundamental processing unit
        
        Args:
            clinical_text: Raw clinical text to process
            context: Optional context about the text source (section name, type, position, etc.)
            task_id: Optional task ID for associating with a BenchmarkTask
            
        Returns:
            PipelineResult with extracted entities and Mcode mappings
        """
        try:
            self.logger.info("🧬 Processing clinical text with ATOMIC processor")
            self.logger.info(f"   📝 Text length: {len(clinical_text)} characters")
            if context:
                self.logger.info(f"   📋 Section: {context.get('name', 'unknown')} ({context.get('type', 'unknown')})")
            
            # Reset token tracker for clean tracking
            global_token_tracker.reset()
            
            # Step 1: Extract entities from clinical text
            extraction_result = self.nlp_extractor.extract_entities(
                clinical_text,
                section_context=context
            )
            entities = extraction_result.entities
            self.logger.info(f"✅ Extracted {len(entities)} entities from clinical text")
            if entities:
                entity_types = {}
                for entity in entities:
                    entity_type = entity.get('type', 'unknown')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                type_summary = ", ".join([f"{count} {etype}" for etype, count in entity_types.items()])
                self.logger.info(f"   📊 Entity types: {type_summary}")
            
            # Step 2: Map entities to Mcode elements
            mapping_result = self.llm_mapper.map_to_mcode(
                entities=entities,
                trial_context=context or {}
            )
            
            # Step 3: Prepare result
            return PipelineResult(
                extracted_entities=entities,
                mcode_mappings=mapping_result['mapped_elements'],
                source_references=mapping_result.get('source_references', []),
                validation_results=mapping_result['validation_results'],
                metadata={
                    'pipeline_version': 'strict_dynamic_v1',
                    'engine_type': 'LLM',
                    'content_type': 'clinical_text',
                    'text_length': len(clinical_text),
                    'entities_count': len(entities),
                    'mapped_count': len(mapping_result['mapped_elements']),
                    'token_usage': extraction_result.metadata.get('token_usage', {}),
                    'aggregate_token_usage': global_token_tracker.get_total_usage().to_dict()
                },
                original_data={'clinical_text': clinical_text, 'context': context},
                error=None
            )
            
        except (NlpExtractionError, McodeMappingError, ValueError) as e:
            self.logger.error(f"ATOMIC clinical text processing FAILED: {str(e)}")
            raise  # Re-raise the exception for proper error handling
        except Exception as e:
            self.logger.error(f"Unexpected error in ATOMIC clinical text processing: {str(e)}")
            raise RuntimeError(f"Unexpected pipeline error: {str(e)}")
    
    def process_clinical_trial(self, trial_data: Dict[str, Any], task_id: Optional[str] = None) -> PipelineResult:
        """
        Process complete clinical trial data through the strict pipeline
        
        Args:
            trial_data: Raw clinical trial data from API or source
            task_id: Optional task ID for associating with a BenchmarkTask
            
        Returns:
            PipelineResult with extracted entities, Mcode mappings, and source tracking
        """
        try:
            self.logger.info("🚀 Starting STRICT clinical trial processing")
            self.logger.info(f"   📄 Trial ID: {trial_data.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')}")
            self.logger.info(f"   📋 Title: {trial_data.get('protocolSection', {}).get('identificationModule', {}).get('briefTitle', 'unknown')}")
            
            # Reset token tracker for clean tracking
            global_token_tracker.reset()
            
            # Step 1: Ingest and extract document sections
            document_sections = self.document_ingestor.ingest_clinical_trial_document(trial_data)
            self.logger.info(f"📋 Extracted {len(document_sections)} document sections")
            
            # Log section details
            section_types = {}
            for section in document_sections:
                section_type = section.source_type
                section_types[section_type] = section_types.get(section_type, 0) + 1
            type_summary = ", ".join([f"{count} {stype}" for stype, count in section_types.items()])
            self.logger.info(f"   📊 Section types: {type_summary}")
            
            # Step 2: Process each section using the atomic processor
            all_entities = []
            source_references = []
            
            # Process sections with progress tracking
            total_sections = len(document_sections)
            processed_sections = 0
            total_entities = 0
            
            for i, section in enumerate(document_sections, 1):
                if not section.content or not section.content.strip():
                    self.logger.info(f"   ⏭️  Skipping empty section: {section.name}")
                    continue
                
                self.logger.info(f"   🔄 Processing section {i}/{total_sections}: {section.name} ({len(section.content)} chars)")
                    
                # Use atomic processor for each section
                section_context = {
                    'name': section.name,
                    'type': section.source_type,
                    'position': section.position
                }
                
                section_result = self.process_clinical_text(section.content, section_context, task_id=task_id)
                section_entities = len(section_result.extracted_entities)
                all_entities.extend(section_result.extracted_entities)
                total_entities += section_entities
                processed_sections += 1
                
                self.logger.info(f"   ✅ Section {i}/{total_sections} complete: {section_entities} entities extracted")
                
                # Create source references for provenance
                for entity in section_result.extracted_entities:
                    source_ref = SourceReference(
                        section_name=section.name,
                        section_type=section.source_type,
                        text_fragment=entity.get('text', ''),
                        position_range=entity.get('source_context', {}).get('position_range', {}),
                        extraction_method="llm_based",
                        confidence=entity.get('confidence', 0.8),
                        provenance_chain=[{
                            'step': 'entity_extraction',
                            'section': section.name,
                            'timestamp': 'current'
                        }]
                    )
                    source_references.append(source_ref)
            
            self.logger.info(f"✅ Extracted {total_entities} total entities from {processed_sections} sections")
            if all_entities:
                entity_types = {}
                for entity in all_entities:
                    entity_type = entity.get('type', 'unknown')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                type_summary = ", ".join([f"{count} {etype}" for etype, count in entity_types.items()])
                self.logger.info(f"   📊 Total entity types: {type_summary}")
            
            # Step 3: Map entities to Mcode elements with source tracking
            self.logger.info("🗺️  Starting Mcode mapping phase...")
            self.logger.info(f"   📊 Input: {len(all_entities)} entities to map")
            self.logger.info(f"   📋 Source references: {len(source_references)}")
            
            mapping_result = self.llm_mapper.map_to_mcode(
                entities=all_entities,
                trial_context=trial_data,
                source_references=source_references
            )
            
            self.logger.info(f"✅ Mapped {len(mapping_result['mapped_elements'])} Mcode elements")
            self.logger.info(f"   🎯 Compliance score: {mapping_result['validation_results'].get('compliance_score', 0):.2%}")
            
            # Step 4: Prepare comprehensive result
            return PipelineResult(
                extracted_entities=all_entities,
                mcode_mappings=mapping_result['mapped_elements'],
                source_references=mapping_result.get('source_references', []),
                validation_results=mapping_result['validation_results'],
                metadata={
                    'pipeline_version': 'strict_dynamic_v1',
                    'engine_type': 'LLM',
                    'entities_count': len(all_entities),
                    'mapped_count': len(mapping_result['mapped_elements']),
                    'compliance_score': mapping_result['validation_results']['compliance_score'],
                    'token_usage': mapping_result['metadata'].get('token_usage', {}),
                    'aggregate_token_usage': global_token_tracker.get_total_usage().to_dict()
                },
                original_data=trial_data,
                error=None
            )
            
        except (NlpConfigurationError, McodeConfigurationError, NlpExtractionError,
                McodeMappingError, ValueError) as e:
            self.logger.error(f"STRICT clinical trial processing FAILED: {str(e)}")
            raise  # Re-raise the exception for proper error handling
        except Exception as e:
            self.logger.error(f"Unexpected error in STRICT clinical trial processing: {str(e)}")
            raise RuntimeError(f"Unexpected pipeline error: {str(e)}")
    
    def process_eligibility_criteria(self, criteria_text: str,
                                   section_context: Dict[str, Any] = None) -> PipelineResult:
        """
        Process eligibility criteria text through the strict pipeline
        
        Args:
            criteria_text: Eligibility criteria text to process
            section_context: Optional context about the criteria section
            
        Returns:
            PipelineResult with extracted entities and Mcode mappings
        """
        try:
            self.logger.info("📋 Processing eligibility criteria with STRICT pipeline")
            self.logger.info(f"   📝 Criteria length: {len(criteria_text)} characters")
            if section_context:
                self.logger.info(f"   📋 Section: {section_context.get('name', 'unknown')}")
            
            # Use the atomic processor for eligibility criteria
            result = self.process_clinical_text(criteria_text, section_context)
            
            # Update metadata to reflect eligibility criteria context
            result.metadata['content_type'] = 'eligibility_criteria'
            result.original_data = {'criteria_text': criteria_text, 'section_context': section_context}
            
            return result
            
        except (NlpExtractionError, McodeMappingError, ValueError) as e:
            self.logger.error(f"STRICT criteria processing FAILED: {str(e)}")
            raise  # Re-raise the exception for proper error handling
        except Exception as e:
            self.logger.error(f"Unexpected error in STRICT criteria processing: {str(e)}")
            raise RuntimeError(f"Unexpected pipeline error: {str(e)}")
    
    def _extract_entities_from_section(self, section: DocumentSection) -> List[Dict[str, Any]]:
        """
        Extract entities from a document section with source context
        
        Args:
            section: DocumentSection to process
            
        Returns:
            List of extracted entities with source tracking
        """
        try:
            if not section.content or not section.content.strip():
                return []
            
            # Extract entities with section context using atomic processor
            section_context = {
                'name': section.name,
                'type': section.source_type,
                'position': section.position
            }
            
            # Use the atomic processor for consistency
            result = self.process_clinical_text(section.content, section_context)
            
            # Enhance entities with section context
            enhanced_entities = []
            for entity in result.extracted_entities:
                enhanced_entity = entity.copy()
                if 'source_context' not in enhanced_entity:
                    enhanced_entity['source_context'] = {}
                
                enhanced_entity['source_context'].update({
                    'section_name': section.name,
                    'section_type': section.source_type,
                    'section_position': section.position
                })
                
                enhanced_entities.append(enhanced_entity)
            
            return enhanced_entities
            
        except NlpExtractionError as e:
            self.logger.error(f"STRICT entity extraction from section {section.name} FAILED: {str(e)}")
            raise  # Re-raise to propagate the error
        except Exception as e:
            self.logger.error(f"Unexpected error in section extraction {section.name}: {str(e)}")
            raise RuntimeError(f"Unexpected section extraction error: {str(e)}")


# Example usage
if __name__ == "__main__":
    from src.utils.logging_config import get_logger
    logger = get_logger(__name__)
    
    # Sample clinical trial data structure
    sample_trial_data = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Sample Clinical Trial"
            },
            "designModule": {
                "designInfo": "Randomized controlled trial"
            },
            "eligibilityModule": {
                "eligibilityCriteria": """
                    INCLUSION CRITERIA:
                    - Histologically confirmed diagnosis of cancer
                    - Must have measurable disease
                    - Age ≥ 18 years
                    
                    EXCLUSION CRITERIA:
                    - Prior malignancy within 5 years
                    - Uncontrolled intercurrent illness
                """
            },
            "conditionsModule": {
                "conditions": ["Cancer", "Neoplasms"]
            }
        }
    }
    
    # Initialize strict pipeline
    pipeline = NlpMcodePipeline()
    
    try:
        # Process complete trial
        result = pipeline.process_clinical_trial(sample_trial_data)
        
        logger.info("STRICT Dynamic Extraction Pipeline Results:")
        logger.info(f"Extracted entities: {len(result.extracted_entities)}")
        logger.info(f"Mapped Mcode elements: {len(result.mcode_mappings)}")
        logger.info(f"Validation valid: {result.validation_results['valid']}")
        logger.info(f"Compliance score: {result.validation_results['compliance_score']}")
        
        # Show sample mappings
        for mapping in result.mcode_mappings[:3]:
            logger.info(f"  - {mapping['resourceType']}: {mapping['element_name']}")
            
    except (NlpConfigurationError, McodeConfigurationError, NlpExtractionError,
            McodeMappingError, ValueError, RuntimeError) as e:
        logger.error(f"❌ STRICT Pipeline FAILED: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}")