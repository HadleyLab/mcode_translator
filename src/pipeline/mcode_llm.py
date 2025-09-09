"""High-performance mCODE Mapping Engine.

This module provides LLM-based mapping of clinical text to standardized mCODE elements
with strict validation, evidence-based processing, and comprehensive error handling.

Classes:
    McodeMapper: Core engine for clinical text to mCODE transformation.
    SourceReference: Data structure for source provenance tracking.
    McodeValidationResult: Validation results with detailed error reporting.

Exceptions:
    McodeConfigurationError: Configuration validation failures.
    McodeMappingError: Mapping operation failures.

Example:
    >>> from src.pipeline import McodeMapper
    >>> mapper = McodeMapper(prompt_name="direct_mcode_evidence_based_concise")
    >>> result = mapper.map_to_mcode(entities, trial_context, clinical_text)
    >>> print(f"Quality Score: {result['validation_results']['compliance_score']}")
"""

import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .llm_base import LlmBase, LlmExecutionError, LlmResponseError, LLMCallMetrics
from src.utils.logging_config import Loggable
from src.utils.prompt_loader import PromptLoader
from src.utils.config import Config
from src.utils.token_tracker import global_token_tracker


class McodeConfigurationError(Exception):
    """Raised when mCODE mapping configuration is invalid or incomplete."""
    pass


class McodeMappingError(Exception):
    """Raised when mCODE mapping operations fail due to processing errors."""
    pass


@dataclass
class SourceReference:
    """Source provenance tracking for mCODE mappings.
    
    Provides comprehensive tracking of where mCODE elements were extracted from
    in the original clinical text, enabling audit trails and validation.
    
    Attributes:
        section_name: Name of the document section (e.g., 'eligibilityModule').
        section_type: Type classification of the section.
        text_fragment: Original text fragment that generated the mapping.
        position_range: Character positions in source document.
        extraction_method: Method used for extraction (e.g., 'llm_based').
        confidence: Confidence score (0.0-1.0) for the mapping.
        provenance_chain: Chain of processing steps that led to this mapping.
    """
    section_name: str
    section_type: str
    text_fragment: str
    position_range: Dict[str, int]
    extraction_method: str
    confidence: float
    provenance_chain: List[Dict[str, Any]]


@dataclass
class McodeValidationResult:
    """Validation result for mCODE element compliance and quality.
    
    Provides detailed validation information including compliance scores,
    error messages, and warnings for mCODE element validation.
    
    Attributes:
        valid: Whether the mCODE element passes validation.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
        compliance_score: Numeric score (0.0-1.0) indicating compliance level.
    """
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    compliance_score: float = 0.0
    
    def __post_init__(self):
        """Initialize default empty lists for errors and warnings."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class McodeMapper(LlmBase, Loggable):
    """High-performance LLM-based mCODE mapping engine.
    
    Transforms clinical text directly to standardized mCODE elements using
    evidence-based LLM processing with strict validation and quality control.
    
    Features:
        - Evidence-based prompt engineering for accuracy
        - Comprehensive source provenance tracking
        - Strict validation with detailed error reporting
        - Configurable LLM models and parameters
        - Token usage tracking and optimization
    
    Attributes:
        MCODE_MAPPING_PROMPT_TEMPLATE: Loaded prompt template for mapping.
        prompt_name: Name of the active prompt template.
        mcode_value_sets: Standard value sets for mCODE compliance.
        prompt_loader: Component for loading prompt templates.
    """

    # Prompt template will be loaded from file-based library
    MCODE_MAPPING_PROMPT_TEMPLATE = None
    prompt_name: Optional[str] = None

    def __init__(self,
                 prompt_name: str = "direct_mcode_evidence_based_concise",
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> None:
        """Initialize the mCODE mapper with specified configuration.
        
        Args:
            prompt_name: Name of the prompt template to use for mapping.
                Defaults to evidence-based concise prompt for optimal quality.
            model_name: LLM model name. If None, uses configuration default.
            temperature: Temperature for text generation. If None, uses config default.
            max_tokens: Maximum tokens for response. If None, uses config default.
            
        Raises:
            McodeConfigurationError: If configuration validation fails.
            LlmConfigurationError: If LLM initialization fails.
        """
        try:
            # STRICT: No fallback to default configuration - all parameters must be explicitly provided
            if not model_name:
                raise McodeConfigurationError("Model name is required - no fallback to default model allowed in strict mode")
            
            # Get configuration for the specified model only
            config = Config()
            final_model_name = model_name
            final_temperature = temperature if temperature is not None else config.get_temperature(model_name)
            final_max_tokens = max_tokens if max_tokens is not None else config.get_max_tokens(model_name)
            
            # Initialize LlmBase with explicit configuration
            LlmBase.__init__(
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

        except Exception as e:
            raise McodeConfigurationError(f"Failed to initialize McodeMapper: {str(e)}")
    
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
            McodeMappingError: If mapping fails
            LlmExecutionError: If LLM API call fails
            LlmResponseError: If LLM response parsing fails
        """
        # Validate input with strict error handling
        self._validate_entities(entities, clinical_text)
        
        try:
            # Prepare entities for LLM processing
            entities_json = json.dumps(entities, indent=2)
            context_json = json.dumps(trial_context or {}, indent=2)
            
            prompt_template = self.MCODE_MAPPING_PROMPT_TEMPLATE
            
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
            parsed_response, metrics = self._call_llm_mapping(prompt, prompt_key)
            
            # Parse and validate LLM response
            mapped_elements = self._parse_llm_response(parsed_response, source_references)
            
            # Validate mCODE compliance with strict validation
            validation_results = self._validate_mcode_compliance_strict(mapped_elements)
            
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
            
        except (LlmExecutionError, LlmResponseError) as e:
            raise McodeMappingError(f"LLM-based mapping failed: {str(e)}")
        except Exception as e:
            raise McodeMappingError(f"Unexpected error during mapping: {str(e)}")
    
    def _validate_entities(self, entities: List[Dict[str, Any]], clinical_text: str = None) -> None:
        """Validate entities with strict error handling"""
        if entities is None:
            # For direct text-to-mCODE, entities can be None
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
    
    def _call_llm_mapping(self, prompt: str, prompt_key: str = "generic_mapping") -> Tuple[Dict[str, Any], 'LLMCallMetrics']:
        """
        Call LLM API for mCODE mapping with strict error handling
        
        Args:
            prompt: Formatted prompt for LLM mapping
            
        Returns:
            A tuple containing the LLM response text and call metrics
            
        Raises:
            LlmExecutionError: If API call fails
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
            
        except LlmExecutionError as e:
            raise
    
    def _parse_llm_response(self, parsed_response: Dict[str, Any],
                          source_references: List[SourceReference] = None) -> List[Dict[str, Any]]:
        """
        Parse and validate LLM mapping response with strict error handling
        
        Args:
            parsed_response: Parsed JSON response from LLM
            source_references: List of source references for provenance tracking
        
        Returns:
            List of validated mCODE mappings with proper source references
        
        Raises:
            LlmResponseError: If response structure is invalid
        """
        try:
            # Use the already parsed response
            parsed = parsed_response
            
            # Validate required structure
            self._validate_mapping_response_structure(parsed)
            
            mapped_elements = parsed['mcode_mappings']
            
            if not isinstance(mapped_elements, list):
                raise LlmResponseError("'mapped_elements' must be an array")
            
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
                    raise McodeMappingError(f"Invalid mCODE element: {validation_result.errors}")
            
            return validated_elements
            
        except (LlmResponseError, ValueError) as e:
            raise LlmResponseError(f"Mapping response parsing failed: {str(e)}")
    
    def _validate_mapping_response_structure(self, parsed_response: Dict[str, Any]) -> None:
        """Validate the structure of the mapping response"""
        if not isinstance(parsed_response, dict):
            raise LlmResponseError("LLM response must be a JSON object")
        
        if 'mcode_mappings' not in parsed_response:
            raise LlmResponseError("Missing 'mcode_mappings' field in LLM response")
        
        # Validate each mapping has required Mcode_element field
        for mapping in parsed_response['mcode_mappings']:
            if 'Mcode_element' not in mapping:
                raise LlmResponseError("Each mapping must contain 'Mcode_element' field")
    
    def _validate_mcode_element_strict(self, element: Dict[str, Any]) -> McodeValidationResult:
        """Validate a single mCODE element with detailed error reporting"""
        result = McodeValidationResult(valid=True)
        
        # Check required fields
        required_fields = ['resourceType', 'Mcode_element']
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
            raise McodeMappingError("mCODE element missing required source_text_fragment field for reference connection")
            
        # Get the text fragment from LLM-generated source reference
        llm_text_fragment = element['source_text_fragment']
        if not llm_text_fragment:
            raise McodeMappingError("Empty source_text_fragment in mCODE element - cannot connect source references")
            
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
    
    def _transform_mapping_to_fhir(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform LLM mapping response format to FHIR resource format
        
        Args:
            mapping: LLM mapping in prompt template format
            
        Returns:
            FHIR resource format mapping
        """
        # Map Mcode_element to resourceType
        Mcode_to_resource = {
            'CancerCondition': 'Condition',
            'CancerDiseaseStatus': 'Observation',
            'TNMClinicalStageGroup': 'Observation',
            'GenomicVariant': 'Observation',
            'CancerRelatedMedication': 'MedicationStatement',
            'CancerRelatedProcedure': 'Procedure',
            'ECOGPerformanceStatus': 'Observation',
            'KarnofskyPerformanceStatus': 'Observation'
        }
        
        resource_type = Mcode_to_resource.get(mapping.get('Mcode_element', ''), 'Observation')
        
        # Create FHIR-compliant element
        fhir_element = {
            'resourceType': resource_type,
            'Mcode_element': mapping.get('Mcode_element', ''),
            'code': {
                'system': 'http://hl7.org/fhir/us/mCODE',
                'code': mapping.get('Mcode_element', '').lower().replace(' ', '-'),
                'display': mapping.get('Mcode_element', '')
            },
            'value': mapping.get('value', ''),
            'mapping_confidence': mapping.get('mapping_confidence', mapping.get('confidence', 0.0)),
            'source_text_fragment': mapping.get('source_text_fragment', f"Entity index {mapping.get('source_entity_index', 'unknown')}"),
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
        Process LLM request for mCODE mapping - implements LlmBase abstract method
        
        Args:
            entities: List of extracted medical entities
            trial_context: Optional trial information for context
            source_references: Optional source tracking information
            prompt_key: Key for the prompt template to use
            
        Returns:
            Dictionary with mCODE mappings and validation results
            
        Raises:
            ValueError: If entities are invalid
            McodeMappingError: If mapping fails
            LlmExecutionError: If LLM API call fails
            LlmResponseError: If LLM response parsing fails
        """
        return self.map_to_mcode(entities, trial_context, source_references, prompt_key)

    def get_prompt_name(self) -> str:
        """Returns the name of the prompt being used."""
        return self.prompt_name


# Factory function for backward compatibility (with deprecation warning)
def create_mcode_mapper(model_name: str = None,
                       temperature: float = None,
                       max_tokens: int = None) -> McodeMapper:
    """
    Factory function for creating McodeMapper instances
    Maintains backward compatibility with existing code
    
    Args:
        model_name: LLM model name
        temperature: Generation temperature
        max_tokens: Maximum response tokens
    
    Returns:
        McodeMapper instance
    
    Raises:
        McodeConfigurationError: If configuration is invalid
    """
    import warnings
    warnings.warn(
        "create_mcode_mapper() is deprecated. Use McodeMapper() directly instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return McodeMapper(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )