"""
STRICT MCode Mapper - No fallbacks, exception-based error handling
Uses shared StrictLLMBase for LLM operations
"""

import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .strict_llm_base import StrictLLMBase, LLMExecutionError, LLMResponseError, LLMCallMetrics
from src.utils.logging_config import Loggable
from src.utils.prompt_loader import PromptLoader
from src.utils.config import Config
from src.utils.token_tracker import global_token_tracker


class MCodeConfigurationError(Exception):
    """Exception raised for MCode mapping configuration issues"""
    pass


class MCodeMappingError(Exception):
    """Exception raised for MCode mapping failures"""
    pass


@dataclass
class SourceReference:
    """Represents source tracking information for mCODE mappings"""
    section_name: str
    section_type: str
    text_fragment: str
    position_range: Dict[str, int]
    extraction_method: str
    confidence: float
    provenance_chain: List[Dict[str, Any]]


@dataclass
class MCodeValidationResult:
    """Result of MCode element validation with detailed error information"""
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    compliance_score: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class StrictMcodeMapper(StrictLLMBase, Loggable):
    """
    STRICT MCode Mapping Engine
    Uses LLMs to dynamically map extracted entities to mCODE elements
    No fallbacks, explicit error handling, and strict validation
    """

    # Prompt template will be loaded from file-based library
    MCODE_MAPPING_PROMPT_TEMPLATE = None
    prompt_name: Optional[str] = None

    def __init__(self,
                 prompt_name: str = "generic_mapping",
                 model_name: str = None,
                 temperature: float = None,
                 max_tokens: int = None):
        """
        Initialize strict MCode mapper with explicit configuration validation
        
        Args:
            model_name: LLM model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            
        Raises:
            MCodeConfigurationError: If configuration is invalid
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
            
            # Initialize Loggable
            Loggable.__init__(self)
            
            # Standard mCODE value sets
            self.mcode_value_sets = {
                'gender': ['male', 'female', 'other', 'unknown'],
                'ethnicity': ['hispanic-or-latino', 'not-hispanic-or-latino', 'unknown'],
                'race': ['american-indian-or-alaska-native', 'asian', 'black-or-african-american',
                        'native-hawaiian-or-other-pacific-islander', 'white', 'other', 'unknown']
            }

            # Initialize prompt loader - don't load prompt template here
            # Prompt template will be set by pipeline or via prompt_key parameter
            self.prompt_loader = PromptLoader()
            self.prompt_name = prompt_name
            self.MCODE_MAPPING_PROMPT_TEMPLATE = self.prompt_loader.get_prompt(self.prompt_name)

            self.logger.info("✅ Strict MCode Mapper initialized successfully")
            self.logger.info(f"   🤖 Model: {final_model_name}")
            self.logger.info(f"   🌡️  Temperature: {final_temperature}")
            self.logger.info(f"   📏 Max tokens: {final_max_tokens}")
            # Log that mapper is initialized without a prompt template
            self.logger.info(f"   📝 Prompt: {self.prompt_name}")
            
        except Exception as e:
            raise MCodeConfigurationError(f"Failed to initialize StrictMcodeMapper: {str(e)}")
    
    def map_to_mcode(self, entities: List[Dict[str, Any]],
                    trial_context: Dict[str, Any] = None,
                    source_references: List[SourceReference] = None,
                    prompt_key: str = "generic_mapping",
                    clinical_text: str = None) -> Dict[str, Any]:
        """
        Map extracted entities to mCODE elements using LLM with strict error handling
        
        Args:
            entities: List of extracted medical entities
            trial_context: Optional trial information for context
            source_references: Optional source tracking information
            prompt_key: Key for the prompt template to use
            
        Returns:
            Dictionary with mCODE mappings and validation results
            
        Raises:
            ValueError: If entities are invalid
            MCodeMappingError: If mapping fails
            LLMExecutionError: If LLM API call fails
            LLMResponseError: If LLM response parsing fails
        """
        # Validate input with strict error handling
        self._validate_entities(entities, clinical_text)
        
        try:
            self.logger.info("🗺️  Starting mCODE mapping process...")
            self.logger.info(f"   📊 Input entities: {len(entities)}")
            
            # Prepare entities for LLM processing
            entities_json = json.dumps(entities, indent=2)
            context_json = json.dumps(trial_context or {}, indent=2)
            
            self.logger.info("   📋 Preparing prompt for LLM mapping...")
            
            prompt_template = self.MCODE_MAPPING_PROMPT_TEMPLATE
            self.logger.info(f"   📝 Using prompt template: {self.prompt_name}")
            
            # Determine which placeholders the template expects
            format_kwargs = {}
            if "{extracted_entities_json}" in prompt_template:
                format_kwargs["extracted_entities_json"] = entities_json
            if "{entities_json}" in prompt_template:
                format_kwargs["entities_json"] = entities_json
            
            if "{clinical_text}" in prompt_template:
                format_kwargs["clinical_text"] = clinical_text or context_json
            if "{trial_context}" in prompt_template:
                format_kwargs["trial_context"] = context_json
            
            # Format the prompt with the appropriate placeholders
            prompt = prompt_template.format(**format_kwargs)
            
            # Call LLM for mapping
            self.logger.info("🤖 Calling LLM for mCODE mapping...")
            llm_response, metrics = self._call_llm_mapping(prompt, prompt_key)
            self.logger.info("✅ LLM mapping call completed")
            
            # Parse and validate LLM response
            self.logger.info("📋 Parsing and validating LLM mapping response...")
            mapped_elements = self._parse_llm_response(llm_response, source_references)
            self.logger.info(f"✅ Successfully parsed {len(mapped_elements)} mCODE elements")
            
            # Validate mCODE compliance with strict validation
            self.logger.info("✅ Validating mCODE compliance...")
            validation_results = self._validate_mcode_compliance_strict(mapped_elements)
            self.logger.info(f"   🎯 Compliance score: {validation_results.get('compliance_score', 0):.2%}")
            
            return {
                'mapped_elements': mapped_elements,
                'validation_results': validation_results,
                'source_references': source_references or [],
                'metadata': {
                    'mapping_method': 'llm_based',
                    'entities_count': len(entities),
                    'mapped_count': len(mapped_elements),
                    'token_usage': {
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.completion_tokens,
                        "total_tokens": metrics.total_tokens
                    }
                }
            }
            
            self.logger.info(f"   📊 Mapping token usage - Prompt: {metrics.prompt_tokens}, Completion: {metrics.completion_tokens}, Total: {metrics.total_tokens}")
            
        except (LLMExecutionError, LLMResponseError) as e:
            raise MCodeMappingError(f"LLM-based mapping failed: {str(e)}")
        except Exception as e:
            raise MCodeMappingError(f"Unexpected error during mapping: {str(e)}")
    
    def _validate_entities(self, entities: List[Dict[str, Any]], clinical_text: str = None) -> None:
        """Validate entities with strict error handling"""
        if entities is None:
            # For direct text-to-mcode, entities can be None
            if clinical_text is None:
                raise ValueError("Entities and clinical_text cannot both be None")
        elif not isinstance(entities, list):
            raise ValueError("Entities must be a list")
        elif len(entities) > 0:
            # Validate each entity has basic structure
            for i, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    raise ValueError(f"Entity at index {i} must be a dictionary")
                
                if 'text' not in entity or not entity['text']:
                    raise ValueError(f"Entity at index {i} must have a 'text' field")
    
    def _call_llm_mapping(self, prompt: str, prompt_key: str = "generic_mapping") -> Tuple[str, 'LLMCallMetrics']:
        """
        Call LLM API for mCODE mapping with strict error handling
        
        Args:
            prompt: Formatted prompt for LLM mapping
            
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
            "prompt_template": self.prompt_name
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            return self._call_llm_api(messages, cache_key_data)
            
        except LLMExecutionError as e:
            self.logger.error(f"❌ LLM mapping call failed: {str(e)}")
            raise
    
    def _parse_llm_response(self, response_text: str,
                          source_references: List[SourceReference] = None) -> List[Dict[str, Any]]:
        """
        Parse and validate LLM mapping response with strict error handling
        
        Args:
            response_text: Raw LLM response
            source_references: List of source references for provenance tracking
            
        Returns:
            List of validated mCODE mappings with proper source references
            
        Raises:
            LLMResponseError: If JSON parsing fails or response is invalid
        """
        try:
            # Parse and validate JSON response
            parsed = self._parse_and_validate_json_response(response_text)
            
            # Validate required structure
            self._validate_mapping_response_structure(parsed)
            
            mapped_elements = parsed['mcode_mappings']
            
            if not isinstance(mapped_elements, list):
                raise LLMResponseError("'mapped_elements' must be an array")
            
            # Transform LLM response format to expected FHIR resource format
            transformed_elements = []
            for mapping in mapped_elements:
                # Convert prompt template format to FHIR resource format
                fhir_element = self._transform_mapping_to_fhir(mapping)
                transformed_elements.append(fhir_element)
            
            # Validate each mapped element and connect with actual source references
            validated_elements = []
            for element in transformed_elements:
                validation_result = self._validate_mcode_element_strict(element)
                
                if validation_result.valid:
                    # Add UUID for tracking
                    element['id'] = str(uuid.uuid4())
                    
                    # Replace LLM-generated minimal source_reference with actual SourceReference objects
                    if source_references:
                        self._connect_source_references_strict(element, source_references)
                    
                    validated_elements.append(element)
                else:
                    raise MCodeMappingError(f"Invalid MCode element: {validation_result.errors}")
            
            return validated_elements
            
        except (LLMResponseError, ValueError) as e:
            self.logger.error(f"❌ Failed to parse LLM mapping response: {str(e)}")
            raise LLMResponseError(f"Mapping response parsing failed: {str(e)}")
    
    def _validate_mapping_response_structure(self, parsed_response: Dict[str, Any]) -> None:
        """Validate the structure of the mapping response"""
        if not isinstance(parsed_response, dict):
            raise LLMResponseError("LLM response must be a JSON object")
        
        if 'mcode_mappings' not in parsed_response:
            raise LLMResponseError("Missing 'mcode_mappings' field in LLM response")
        
        # Validate each mapping has required mcode_element field
        for mapping in parsed_response['mcode_mappings']:
            if 'mcode_element' not in mapping:
                raise LLMResponseError("Each mapping must contain 'mcode_element' field")
    
    def _validate_mcode_element_strict(self, element: Dict[str, Any]) -> MCodeValidationResult:
        """Validate a single mCODE element with detailed error reporting"""
        result = MCodeValidationResult(valid=True)
        
        # Check required fields
        required_fields = ['resourceType', 'mcode_element']
        for field in required_fields:
            if field not in element:
                result.valid = False
                result.errors.append(f"Missing required field: {field}")
        
        # Validate resource type
        valid_resource_types = [
            'Condition', 'Observation', 'MedicationStatement',
            'Procedure', 'GenomicVariant', 'Patient'
        ]
        if 'resourceType' in element and element['resourceType'] not in valid_resource_types:
            result.valid = False
            result.errors.append(f"Invalid resource type: {element['resourceType']}")
        
        # Validate code structure if present
        if 'code' in element:
            code = element['code']
            if not isinstance(code, dict):
                result.valid = False
                result.errors.append("Code must be a dictionary")
            else:
                if 'system' not in code or 'code' not in code:
                    result.valid = False
                    result.errors.append("Code must contain 'system' and 'code' fields")
        
        # Validate source_text_fragment if present
        if 'source_text_fragment' in element and not isinstance(element['source_text_fragment'], str):
            result.warnings.append("source_text_fragment should be a string")
        
        # Validate mapping_confidence if present
        if 'mapping_confidence' in element:
            if not isinstance(element['mapping_confidence'], (int, float)):
                result.valid = False
                result.errors.append("Mapping confidence must be a number")
            elif element['mapping_confidence'] < 0 or element['mapping_confidence'] > 1:
                result.warnings.append("Mapping confidence should be between 0 and 1")
        
        return result
    
    def _connect_source_references_strict(self, element: Dict[str, Any], 
                                        source_references: List[SourceReference]) -> None:
        """
        Connect actual SourceReference objects to mCODE element based on text fragment matching
        with strict validation
        
        Args:
            element: mCODE element to connect source references to
            source_references: List of source references from entity extraction
        """
        if 'source_text_fragment' not in element:
            raise MCodeMappingError("MCode element missing required source_text_fragment field for reference connection")
            
        # Get the text fragment from LLM-generated source reference
        llm_text_fragment = element['source_text_fragment']
        if not llm_text_fragment:
            raise MCodeMappingError("Empty source_text_fragment in MCode element - cannot connect source references")
            
        # Find matching source references by text fragment similarity
        matching_references = []
        for ref in source_references:
            # Simple text matching - could be enhanced with fuzzy matching
            if (llm_text_fragment.lower() in ref.text_fragment.lower() or 
                ref.text_fragment.lower() in llm_text_fragment.lower()):
                matching_references.append(ref)
        
        # Connect actual SourceReference objects to the element
        if matching_references:
            # Convert SourceReference objects to dictionaries for serialization
            element['source_references'] = [{
                'section_name': ref.section_name,
                'section_type': ref.section_type,
                'text_fragment': ref.text_fragment,
                'position_range': ref.position_range,
                'extraction_method': ref.extraction_method,
                'confidence': ref.confidence,
                'provenance_chain': ref.provenance_chain
            } for ref in matching_references]
            
            # Keep the original source_text_fragment for reference
            element['llm_source_text_fragment'] = element['source_text_fragment']
        else:
            # If no matches found, keep the LLM-generated text fragment but mark it as such
            element['llm_source_text_fragment'] = element['source_text_fragment']
            element['source_references'] = []
            # This is not an error - it's expected that some LLM-generated fragments won't match source references
            self.logger.info(f"No source references found for LLM-generated text fragment: {llm_text_fragment[:100]}...")
    
    def _transform_mapping_to_fhir(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform LLM mapping response format to FHIR resource format
        
        Args:
            mapping: LLM mapping in prompt template format
            
        Returns:
            FHIR resource format mapping
        """
        # Map mcode_element to resourceType
        mcode_to_resource = {
            'CancerCondition': 'Condition',
            'CancerDiseaseStatus': 'Observation',
            'TNMClinicalStageGroup': 'Observation',
            'GenomicVariant': 'Observation',
            'CancerRelatedMedication': 'MedicationStatement',
            'CancerRelatedProcedure': 'Procedure',
            'ECOGPerformanceStatus': 'Observation',
            'KarnofskyPerformanceStatus': 'Observation'
        }
        
        resource_type = mcode_to_resource.get(mapping.get('mcode_element', ''), 'Observation')
        
        # Create FHIR-compliant element
        fhir_element = {
            'resourceType': resource_type,
            'mcode_element': mapping.get('mcode_element', ''),
            'code': {
                'system': 'http://hl7.org/fhir/us/mcode',
                'code': mapping.get('mcode_element', '').lower().replace(' ', '-'),
                'display': mapping.get('mcode_element', '')
            },
            'value': mapping.get('value', ''),
            'mapping_confidence': mapping.get('confidence', 0.0),
            'source_text_fragment': f"Entity index {mapping.get('source_entity_index', 'unknown')}",
            'mapping_rationale': mapping.get('mapping_rationale', '')
        }
        
        return fhir_element
    
    def _validate_mcode_compliance_strict(self, mapped_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate mCODE compliance of mapped elements with strict validation
        
        Args:
            mapped_elements: List of mapped mCODE elements
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'compliance_score': 1.0
        }
        
        if not mapped_elements:
            validation_results['valid'] = False
            validation_results['errors'].append("No mCODE elements mapped")
            validation_results['compliance_score'] = 0.0
            return validation_results
        
        # Check required fields for each element type
        compliant_count = 0
        for element in mapped_elements:
            resource_type = element.get('resourceType')
            
            # Basic compliance check
            is_compliant = self._is_fully_compliant_strict(element)
            if is_compliant:
                compliant_count += 1
            
            # Type-specific validation
            if resource_type == 'Condition':
                if 'code' not in element:
                    validation_results['warnings'].append(f"Condition missing code: {element.get('element_name')}")
            
            elif resource_type == 'Observation':
                if 'code' not in element:
                    validation_results['warnings'].append(f"Observation missing code: {element.get('element_name')}")
            
            elif resource_type == 'MedicationStatement':
                if 'code' not in element:
                    validation_results['warnings'].append(f"Medication missing code: {element.get('element_name')}")
        
        # Calculate compliance score
        total_elements = len(mapped_elements)
        validation_results['compliance_score'] = compliant_count / total_elements if total_elements > 0 else 0.0
        
        return validation_results
    
    def _is_fully_compliant_strict(self, element: Dict[str, Any]) -> bool:
        """
        Check if an element is fully mCODE compliant with strict validation
        
        Args:
            element: mCODE element to check
            
        Returns:
            True if fully compliant, False otherwise
        """
        resource_type = element.get('resourceType')
        
        if resource_type in ['Condition', 'Observation', 'MedicationStatement']:
            return ('code' in element and 
                    isinstance(element['code'], dict) and
                    'system' in element['code'] and
                    'code' in element['code'])
        
        return True
    
    def process_request(self, entities: List[Dict[str, Any]],
                       trial_context: Dict[str, Any] = None,
                       source_references: List[SourceReference] = None,
                       prompt_key: str = "generic_mapping") -> Dict[str, Any]:
        """
        Process LLM request for mCODE mapping - implements StrictLLMBase abstract method
        
        Args:
            entities: List of extracted medical entities
            trial_context: Optional trial information for context
            source_references: Optional source tracking information
            prompt_key: Key for the prompt template to use
            
        Returns:
            Dictionary with mCODE mappings and validation results
            
        Raises:
            ValueError: If entities are invalid
            MCodeMappingError: If mapping fails
            LLMExecutionError: If LLM API call fails
            LLMResponseError: If LLM response parsing fails
        """
        return self.map_to_mcode(entities, trial_context, source_references, prompt_key)

    def get_prompt_name(self) -> str:
        """Returns the name of the prompt being used."""
        return self.prompt_name


# Factory function for backward compatibility (with deprecation warning)
def create_mcode_mapper(model_name: str = None,
                       temperature: float = None,
                       max_tokens: int = None) -> StrictMcodeMapper:
    """
    Factory function for creating StrictMcodeMapper instances
    Maintains backward compatibility with existing code
    
    Args:
        model_name: LLM model name
        temperature: Generation temperature
        max_tokens: Maximum response tokens
    
    Returns:
        StrictMcodeMapper instance
    
    Raises:
        MCodeConfigurationError: If configuration is invalid
    """
    import warnings
    warnings.warn(
        "create_mcode_mapper() is deprecated. Use StrictMcodeMapper() directly instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return StrictMcodeMapper(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )