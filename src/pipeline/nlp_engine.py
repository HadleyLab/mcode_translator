"""
STRICT NLP Extractor - No fallbacks, exception-based error handling
Uses shared StrictLLMBase for LLM operations
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .nlp_base import ProcessingResult, NLPEngine
from .strict_llm_base import StrictLLMBase, LLMExecutionError, LLMResponseError, LLMCallMetrics
from src.utils.logging_config import Loggable
from src.utils.prompt_loader import load_prompt
from src.utils.config import Config
from src.utils.token_tracker import global_token_tracker


class NLPConfigurationError(Exception):
    """Exception raised for NLP configuration issues"""
    pass


class NPLExtractionError(Exception):
    """Exception raised for NLP extraction failures"""
    pass


@dataclass
class EntityValidationResult:
    """Result of entity validation with detailed error information"""
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class StrictNlpExtractor(NLPEngine, StrictLLMBase):
    """
    STRICT NLP Extractor for medical entity extraction from clinical text
    No fallbacks, explicit error handling, and strict validation
    Inherits from both NLPEngine abstract class and StrictLLMBase
    """
    
    # Default prompt template - can be overridden by pipeline
    ENTITY_EXTRACTION_PROMPT_TEMPLATE = None
    

    def __init__(self,
                 model_name: str = None,
                 temperature: float = None,
                 max_tokens: int = None):
        """
        Initialize strict NLP extractor with explicit configuration validation
        
        Args:
            model_name: LLM model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            
        Raises:
            NLPConfigurationError: If configuration is invalid
            LLMConfigurationError: If LLM configuration fails
        """
        try:
            # Get default values from unified configuration (strict infrastructure - no fallbacks)
            config = Config()
            final_model_name = model_name or config.get_model_name()
            final_temperature = temperature if temperature is not None else config.get_temperature()
            final_max_tokens = max_tokens if max_tokens is not None else config.get_max_tokens()
            
            # Initialize StrictLLMBase first
            StrictLLMBase.__init__(
                self,
                model_name=final_model_name,
                temperature=final_temperature,
                max_tokens=final_max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Initialize NLPEngine
            NLPEngine.__init__(self)
            
            # Initialize Loggable
            Loggable.__init__(self)
            
            self.logger.info("✅ Strict NLP Extractor initialized successfully")
            self.logger.info(f"   🤖 Model: {final_model_name}")
            self.logger.info(f"   🌡️  Temperature: {final_temperature}")
            self.logger.info(f"   📏 Max tokens: {final_max_tokens}")
            
        except Exception as e:
            raise NLPConfigurationError(f"Failed to initialize StrictNlpExtractor: {str(e)}")
    
    def extract_entities(self, clinical_text: str, section_context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Extract medical entities from clinical text using LLM with strict error handling
        
        Args:
            clinical_text: Clinical text to process
            section_context: Optional context about the document section
            
        Returns:
            ProcessingResult with extracted entities and relationships
            
        Raises:
            ValueError: If clinical text is invalid
            NPLExtractionError: If extraction fails
            LLMExecutionError: If LLM API call fails
            LLMResponseError: If LLM response parsing fails
        """
        # Validate input with strict error handling
        self._validate_clinical_text(clinical_text)
        
        try:
            self.logger.info("🔍 Starting entity extraction...")
            
            # Use custom prompt template if set, otherwise load from library
            if self.ENTITY_EXTRACTION_PROMPT_TEMPLATE:
                # Format the template with clinical text
                prompt = self.ENTITY_EXTRACTION_PROMPT_TEMPLATE.format(clinical_text=clinical_text)
                self.logger.info("   📝 Using custom prompt template")
            else:
                # Fallback to loading from file-based library
                prompt = load_prompt("generic_extraction", clinical_text=clinical_text)
                self.logger.info("   📝 Using generic extraction prompt from library")
            
            # Call LLM for entity extraction
            self.logger.info("🤖 Calling LLM for entity extraction...")
            llm_response, metrics = self._call_llm_extraction(prompt)
            self.logger.info("✅ LLM extraction call completed")
            
            # Parse and validate LLM response with strict validation
            self.logger.info("📋 Parsing and validating LLM response...")
            parsed_result = self._parse_llm_response(llm_response, clinical_text, section_context)
            
            
            # Add token usage to processing result
            if hasattr(parsed_result, 'metadata'):
                parsed_result.metadata['token_usage'] = {
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens
                }

            self.logger.info(f"✅ Successfully parsed {len(parsed_result.entities)} entities")
            self.logger.info(f"   📊 Extraction token usage - Prompt: {metrics.prompt_tokens}, Completion: {metrics.completion_tokens}, Total: {metrics.total_tokens}")
            
            return parsed_result
        except (LLMExecutionError, LLMResponseError) as e:
            raise NPLExtractionError(f"LLM-based extraction failed: {str(e)}")
        except Exception as e:
            raise NPLExtractionError(f"Unexpected error during extraction: {str(e)}")
    
    def _validate_clinical_text(self, clinical_text: str) -> None:
        """Validate clinical text with strict error handling"""
        if not clinical_text:
            raise ValueError("Clinical text cannot be empty")
        
        if not isinstance(clinical_text, str):
            raise ValueError("Clinical text must be a string")
        
        if len(clinical_text.strip()) < 10:
            raise ValueError("Clinical text must be at least 10 characters long")
    
    def _call_llm_extraction(self, prompt: str) -> Tuple[str, 'LLMCallMetrics']:
        """
        Call LLM API for entity extraction with strict error handling
        
        Args:
            prompt: Formatted prompt for LLM extraction
            
        Returns:
            A tuple containing the LLM response text and call metrics
            
        Raises:
            LLMExecutionError: If API call fails
        """
        cache_key_data = {
            "prompt": prompt,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_template": self.ENTITY_EXTRACTION_PROMPT_TEMPLATE or "generic_extraction"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            return self._call_llm_api(messages, cache_key_data)
            
        except LLMExecutionError as e:
            self.logger.error(f"❌ LLM extraction call failed: {str(e)}")
            raise
    
    def _parse_llm_response(self, response_text: str, original_text: str,
                          section_context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Parse and validate LLM extraction response with strict error handling
        
        Args:
            response_text: Raw LLM response
            original_text: Original clinical text for context
            section_context: Optional section context
            
        Returns:
            ProcessingResult with parsed entities
            
        Raises:
            LLMResponseError: If JSON parsing fails or response is invalid
        """
        try:
            # Parse and validate JSON response
            parsed = self._parse_and_validate_json_response(response_text)
            
            # Validate required structure
            self._validate_extraction_response_structure(parsed)
            
            # Process entities with strict validation
            processed_entities = self._process_entities_with_validation(
                parsed['entities'], original_text, section_context
            )
            
            # Process relationships with validation
            relationships = parsed.get('relationships', [])
            valid_relationships = self._validate_relationships(
                relationships, len(processed_entities)
            )
            
            # Create features structure
            features = {
                'extracted_entities': processed_entities,
                'relationships': valid_relationships,
                'text_metadata': {
                    'original_length': len(original_text),
                    'section_context': section_context or {}
                }
            }
            
            return ProcessingResult(
                features=features,
                mcode_mappings={},
                metadata=parsed.get('metadata', {}),
                entities=processed_entities,
                error=None
            )
            
        except (LLMResponseError, ValueError) as e:
            self.logger.error(f"❌ Failed to parse LLM response: {str(e)}")
            raise LLMResponseError(f"Parsing failed: {str(e)}")
    
    def _validate_extraction_response_structure(self, parsed_response: Dict[str, Any]) -> None:
        """Validate the structure of the extraction response"""
        if not isinstance(parsed_response, dict):
            raise LLMResponseError("LLM response must be a JSON object")
        
        if 'entities' not in parsed_response:
            raise LLMResponseError("Missing 'entities' field in LLM response")
        
        if not isinstance(parsed_response['entities'], list):
            raise LLMResponseError("'entities' must be an array")
    
    def _process_entities_with_validation(self, entities: List[Dict[str, Any]], 
                                        original_text: str,
                                        section_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process entities with strict validation and enhancement"""
        processed_entities = []
        
        for entity in entities:
            validation_result = self._validate_entity_strict(entity)
            
            if validation_result.valid:
                enhanced_entity = self._enhance_entity_with_context(
                    entity, original_text, section_context
                )
                processed_entities.append(enhanced_entity)
            else:
                raise NPLExtractionError(f"Invalid entity: {validation_result.errors}")
        
        return processed_entities
    
    def _validate_entity_strict(self, entity: Dict[str, Any]) -> EntityValidationResult:
        """Validate a single entity with detailed error reporting"""
        result = EntityValidationResult(valid=True)
        
        # Check required fields
        required_fields = ['text', 'type']
        for field in required_fields:
            if field not in entity:
                result.valid = False
                result.errors.append(f"Missing required field: {field}")
        
        # Validate entity type
        valid_types = [
            'condition', 'biomarker', 'genomic_variant', 'medication',
            'procedure', 'demographic', 'temporal', 'exclusion'
        ]
        if 'type' in entity and entity['type'] not in valid_types:
            result.valid = False
            result.errors.append(f"Invalid entity type: {entity['type']}")
        
        # Validate attributes structure if present
        if 'attributes' in entity and not isinstance(entity['attributes'], dict):
            result.valid = False
            result.errors.append("Attributes must be a dictionary")
        
        # Validate confidence score if present
        if 'confidence' in entity:
            if not isinstance(entity['confidence'], (int, float)):
                result.valid = False
                result.errors.append("Confidence must be a number")
            elif entity['confidence'] < 0 or entity['confidence'] > 1:
                result.warnings.append("Confidence score should be between 0 and 1")
        
        return result
    
    def _enhance_entity_with_context(self, entity: Dict[str, Any], original_text: str,
                                   section_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance entity with additional context and source tracking"""
        enhanced = entity.copy()
        
        # Ensure source_context exists
        if 'source_context' not in enhanced:
            enhanced['source_context'] = {}
        
        # Add section context if available
        if section_context:
            enhanced['source_context'].update({
                'section_name': section_context.get('name', 'unknown'),
                'section_type': section_context.get('type', 'unknown'),
                'document_id': section_context.get('document_id', 'unknown')
            })
        
        # Ensure confidence score
        if 'confidence' not in enhanced:
            enhanced['confidence'] = 0.8  # Default confidence
        
        return enhanced
    
    def _validate_relationships(self, relationships: List[Dict[str, Any]], 
                              max_entity_index: int) -> List[Dict[str, Any]]:
        """Validate relationships with strict bounds checking"""
        valid_relationships = []
        valid_types = [
            'has_status', 'has_value', 'indicates', 'contraindicates',
            'treats', 'diagnoses', 'monitors', 'associated_with'
        ]
        
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
                
            # Check required fields
            required_fields = ['source_entity_index', 'target_entity_index', 'relationship_type']
            if not all(field in rel for field in required_fields):
                continue
            
            # Validate indices are within bounds
            try:
                source_idx = int(rel['source_entity_index'])
                target_idx = int(rel['target_entity_index'])
                
                if (source_idx < 0 or source_idx >= max_entity_index or
                    target_idx < 0 or target_idx >= max_entity_index):
                    continue
                
                # Validate relationship type
                if rel['relationship_type'] not in valid_types:
                    continue
                
                valid_relationships.append(rel)
                
            except (ValueError, TypeError):
                continue
        
        return valid_relationships
    
    def process_text(self, text: str) -> ProcessingResult:
        """
        Process clinical text and extract entities - implements NLPEngine abstract method
        
        Args:
            text: Clinical text to process
            
        Returns:
            ProcessingResult containing extracted entities
            
        Raises:
            ValueError: If text is invalid
            NPLExtractionError: If extraction fails
        """
        return self.extract_entities(text)
    
    def process_request(self, clinical_text: str, section_context: Dict[str, Any] = None) -> ProcessingResult:
        """
        Process LLM request for entity extraction - implements StrictLLMBase abstract method
        
        Args:
            clinical_text: Clinical text to process
            section_context: Optional context about the document section
            
        Returns:
            ProcessingResult with extracted entities and relationships
            
        Raises:
            ValueError: If clinical text is invalid
            NPLExtractionError: If extraction fails
            LLMExecutionError: If LLM API call fails
            LLMResponseError: If LLM response parsing fails
        """
        return self.extract_entities(clinical_text, section_context)


# Factory function for backward compatibility (with deprecation warning)
def create_nlp_engine(model_name: str = None,
                     temperature: float = None,
                     max_tokens: int = None) -> StrictNlpExtractor:
    """Create a strict NLP extractor instance"""
    import warnings
    warnings.warn(
        "create_nlp_engine() is deprecated. Use StrictNlpExtractor() directly for strict error handling.",
        DeprecationWarning,
        stacklevel=2
    )
    return StrictNlpExtractor(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
