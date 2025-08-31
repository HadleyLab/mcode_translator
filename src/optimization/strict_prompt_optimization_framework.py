"""
STRICT Prompt Optimization Framework - No fallbacks, fails hard on invalid configs
Integrated with file-based prompt library using PromptLoader
"""

import json
import os
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import pandas as pd
from pathlib import Path
from enum import Enum

from src.utils.logging_config import Loggable, get_logger
from src.utils.config import Config
from src.utils.prompt_loader import PromptLoader, prompt_loader
from src.pipeline.mcode_mapper import StrictMcodeMapper
from src.utils.token_tracker import global_token_tracker


class PromptType(Enum):
    """Types of prompts that can be optimized"""
    NLP_EXTRACTION = "nlp_extraction"
    MCODE_MAPPING = "mcode_mapping"


class APIConfig:
    """Configuration for API endpoints and authentication using unified Config - STRICT implementation"""
    
    def __init__(self, name: str, model: Optional[str] = None):
        self.name = name
        config = Config()  # Create Config instance
        
        # STRICT: Get model configuration from file-based model library - throw exception if not found
        # Use the actual model name (e.g., "deepseek-coder") not the config name (e.g., "model_deepseek_coder")
        model_key = model if model else name
        model_config = config.get_model_config(model_key)
        self.base_url = model_config.base_url
        self.model = model_config.model_identifier
        self.temperature = model_config.default_parameters.get('temperature', 0.1)
        self.max_tokens = model_config.default_parameters.get('max_tokens', 4000)
        
        self.timeout = 30  # Default timeout, can be configured if needed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'base_url': self.base_url,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout
        }


@dataclass
class PromptVariant:
    """A specific prompt variant with metadata - now uses prompt library keys"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    prompt_type: PromptType = PromptType.NLP_EXTRACTION
    prompt_key: str = ""  # Key to look up in prompt library
    description: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'prompt_type': self.prompt_type.value,
            'prompt_key': self.prompt_key,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'parameters': self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVariant':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            prompt_type=PromptType(data.get('prompt_type', 'nlp_extraction')),
            prompt_key=data.get('prompt_key', ''),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
            tags=data.get('tags', []),
            parameters=data.get('parameters', {})
        )
    
    def get_prompt_content(self, **format_kwargs) -> str:
        """Get the actual prompt content from the prompt library"""
        if not self.prompt_key:
            raise ValueError(f"Prompt variant '{self.name}' has no prompt_key specified")
        
        try:
            return prompt_loader.get_prompt(self.prompt_key, **format_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load prompt '{self.prompt_key}' for variant '{self.name}': {str(e)}")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_variant_id: str = ""
    api_config_name: str = ""
    test_case_id: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = False
    error_message: str = ""
    
    # Performance metrics
    entities_extracted: int = 0
    entities_mapped: int = 0
    extraction_completeness: float = 0.0
    mapping_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    compliance_score: float = 0.0
    token_usage: int = 0
    
    # Raw results
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    mcode_mappings: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'run_id': self.run_id,
            'prompt_variant_id': self.prompt_variant_id,
            'api_config_name': self.api_config_name,
            'test_case_id': self.test_case_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error_message': self.error_message,
            'entities_extracted': self.entities_extracted,
            'entities_mapped': self.entities_mapped,
            'extraction_completeness': self.extraction_completeness,
            'mapping_accuracy': self.mapping_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'compliance_score': self.compliance_score,
            'token_usage': self.token_usage,
            # Include raw results for validation recalculation
            'extracted_entities': self.extracted_entities,
            'mcode_mappings': self.mcode_mappings,
            'validation_results': self.validation_results
        }
    
    def calculate_metrics(self, expected_entities: List[Dict[str, Any]] = None,
                         expected_mappings: List[Dict[str, Any]] = None,
                         framework: 'StrictPromptOptimizationFramework' = None) -> None:
        """Calculate performance metrics based on results using gold standard validation"""
        framework.logger.debug("calculate_metrics method called")
        if not self.success:
            framework.logger.debug("self.success is False, returning early")
            return
        
        # Debug logging to understand why mapping validation is not being executed
        framework.logger.debug(f"calculate_metrics called with expected_entities={bool(expected_entities)}, expected_mappings={bool(expected_mappings)}")
        if expected_entities:
            framework.logger.debug(f"expected_entities length: {len(expected_entities)}")
        if expected_mappings:
            framework.logger.debug(f"expected_mappings length: {len(expected_mappings)}")
        
        # Basic counts
        self.entities_extracted = len(self.extracted_entities)
        self.entities_mapped = len(self.mcode_mappings)
        
        # Calculate completeness (if expected entities provided)
        if expected_entities:
            expected_count = len(expected_entities)
            if expected_count > 0:
                self.extraction_completeness = self.entities_extracted / expected_count
        
        # Use validation results from mCODE mapping
        if self.validation_results:
            self.compliance_score = self.validation_results.get('compliance_score', 0.0)
        
        # Calculate extraction metrics using text-based matching for NLP entities
        framework.logger.debug(f"Extraction validation: expected_entities={bool(expected_entities)}, extracted_entities={bool(self.extracted_entities)}, len={len(self.extracted_entities)}")
        framework.logger.debug(f"Before extraction validation: expected_entities is None: {expected_entities is None}")
        framework.logger.debug(f"Before extraction validation: self.extracted_entities is None: {self.extracted_entities is None}")
        framework.logger.debug(f"Before extraction validation: expected_entities bool: {bool(expected_entities)}")
        framework.logger.debug(f"Before extraction validation: self.extracted_entities bool: {bool(self.extracted_entities)}")
        if expected_entities and self.extracted_entities:
            framework.logger.debug(f"Calculating extraction metrics with {len(self.extracted_entities)} extracted and {len(expected_entities)} expected entities")
            # Use fuzzy text matching for extraction metrics to handle different text representations
            true_positives_ext, false_positives_ext, false_negatives_ext = BenchmarkResult._calculate_fuzzy_text_matches(
                self.extracted_entities, expected_entities, framework
            )
            
            self.precision = true_positives_ext / (true_positives_ext + false_positives_ext) if (true_positives_ext + false_positives_ext) > 0 else 0
            self.recall = true_positives_ext / (true_positives_ext + false_negatives_ext) if (true_positives_ext + false_negatives_ext) > 0 else 0
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0
            
        # DEBUG: Log validation details to help diagnose 0.000 scores for extraction
        # Trigger debug logging if any key metric is 0.0 to catch edge cases
        # Moved outside the if block to catch cases where validation condition fails
        if (self.f1_score == 0.0 or self.precision == 0.0 or self.recall == 0.0):
            framework.logger.warning(f"⚠️  Extraction validation debug - Zero metrics detected")
            framework.logger.warning(f"   F1={self.f1_score:.6f}, Precision={self.precision:.6f}, Recall={self.recall:.6f}")
            framework.logger.warning(f"   Extracted entities: {len(self.extracted_entities)}")
            framework.logger.warning(f"   Expected entities: {len(expected_entities) if expected_entities else 0}")
            
            if expected_entities and self.extracted_entities:
                true_positives_ext, false_positives_ext, false_negatives_ext = BenchmarkResult._calculate_fuzzy_text_matches(
                    self.extracted_entities, expected_entities, framework, debug=True
                )
                
                framework.logger.warning(f"   True positives: {true_positives_ext}")
                framework.logger.warning(f"   False positives: {false_positives_ext}")
                framework.logger.warning(f"   False negatives: {false_negatives_ext}")
                
                # Log detailed text comparison using exact matching for comparison
                extracted_texts = set(entity.get('text', '') for entity in self.extracted_entities)
                expected_texts = set(entity.get('text', '') for entity in expected_entities)
                exact_true_positives = len(extracted_texts & expected_texts)
                
                framework.logger.warning(f"   Exact text matches: {exact_true_positives}")
                framework.logger.warning(f"   Fuzzy text matches: {true_positives_ext}")
                framework.logger.warning(f"   Improvement with fuzzy matching: {true_positives_ext - exact_true_positives}")
            else:
                framework.logger.warning(f"   Validation condition failed: expected_entities={bool(expected_entities)}, extracted_entities={bool(self.extracted_entities)}")
        
        # Calculate mapping accuracy using mCODE-based matching for mCODE elements
        framework.logger.debug(f"Mapping validation: expected_mappings={bool(expected_mappings)}, mcode_mappings={bool(self.mcode_mappings)}, len={len(self.mcode_mappings) if self.mcode_mappings else 0}")
        framework.logger.debug(f"Mapping validation: expected_mappings type={type(expected_mappings)}, mcode_mappings type={type(self.mcode_mappings)}")
        framework.logger.debug(f"Mapping validation: expected_mappings value={expected_mappings}, mcode_mappings value={self.mcode_mappings}")
        framework.logger.debug(f"Mapping validation: expected_mappings is None: {expected_mappings is None}")
        framework.logger.debug(f"Mapping validation: self.mcode_mappings is None: {self.mcode_mappings is None}")
        framework.logger.debug(f"Mapping validation: expected_mappings bool: {bool(expected_mappings)}")
        framework.logger.debug(f"Mapping validation: self.mcode_mappings bool: {bool(self.mcode_mappings)}")
        if expected_mappings and self.mcode_mappings:
            framework.logger.debug(f"Calculating mapping metrics with {len(self.mcode_mappings)} mapped and {len(expected_mappings)} expected mappings")
            # Use mCODE-based matching for mapping metrics (compare mcode_element + value tuples)
            # Case-insensitive matching for both mcode_element and values to handle case differences
            actual_mappings = set((m.get('mcode_element', '').lower(), m.get('value', '').lower()) for m in self.mcode_mappings)
            expected_mappings_set = set((m.get('mcode_element', '').lower(), m.get('value', '').lower()) for m in expected_mappings)
            
            # Debug logging to understand what's happening
            framework.logger.debug(f"Mapping validation debug info:")
            framework.logger.debug(f"  Actual mappings count: {len(actual_mappings)}")
            framework.logger.debug(f"  Expected mappings count: {len(expected_mappings_set)}")
            
            # Show some examples of both sets
            actual_list = list(actual_mappings)
            expected_list = list(expected_mappings_set)
            framework.logger.debug(f"  Sample actual mappings: {actual_list[:5]}")
            framework.logger.debug(f"  Sample expected mappings: {expected_list[:5]}")
            
            # Calculate intersection and differences
            intersection = actual_mappings & expected_mappings_set
            false_positives = actual_mappings - expected_mappings_set
            false_negatives = expected_mappings_set - actual_mappings
            
            framework.logger.debug(f"  Intersection count: {len(intersection)}")
            framework.logger.debug(f"  False positives count: {len(false_positives)}")
            framework.logger.debug(f"  False negatives count: {len(false_negatives)}")
            
            # Show some examples of intersection if it exists
            if intersection:
                framework.logger.debug(f"  Sample intersection: {list(intersection)[:5]}")
            
            true_positives_map = len(intersection)
            false_positives_map = len(false_positives)
            false_negatives_map = len(false_negatives)
            
            mapping_precision = true_positives_map / (true_positives_map + false_positives_map) if (true_positives_map + false_positives_map) > 0 else 0
            mapping_recall = true_positives_map / (true_positives_map + false_negatives_map) if (true_positives_map + false_negatives_map) > 0 else 0
            
            self.mapping_accuracy = 2 * (mapping_precision * mapping_recall) / (mapping_precision + mapping_recall) if (mapping_precision + mapping_recall) > 0 else 0
            
            framework.logger.debug(f"  Mapping accuracy calculation: precision={mapping_precision:.3f}, recall={mapping_recall:.3f}, accuracy={self.mapping_accuracy:.3f}")
            
        # DEBUG: Log validation details to help diagnose 0.000 scores
        # Trigger debug logging if any key metric is 0.0 to catch edge cases
        # Moved outside the if block to catch cases where validation condition fails
        if (self.f1_score == 0.0 or self.precision == 0.0 or self.recall == 0.0 or
            self.mapping_accuracy == 0.0):
            framework.logger.warning(f"⚠️  Validation debug - Zero metrics detected")
            framework.logger.warning(f"   F1={self.f1_score:.6f}, Precision={self.precision:.6f}, Recall={self.recall:.6f}, Mapping={self.mapping_accuracy:.6f}")
            framework.logger.warning(f"   Actual mappings: {len(self.mcode_mappings)}")
            framework.logger.warning(f"   Expected mappings: {len(expected_mappings) if expected_mappings else 0}")
            
            if expected_mappings and self.mcode_mappings:
                # Use case-insensitive matching for both mcode_element and values in debug logging as well
                actual_mappings = set((m.get('mcode_element', '').lower(), m.get('value', '').lower()) for m in self.mcode_mappings)
                expected_mappings_set = set((m.get('mcode_element', '').lower(), m.get('value', '').lower()) for m in expected_mappings)
                
                true_positives_map = len(actual_mappings & expected_mappings_set)
                false_positives_map = len(actual_mappings - expected_mappings_set)
                false_negatives_map = len(expected_mappings_set - actual_mappings)
                
                framework.logger.warning(f"   Mapping intersection: {len(actual_mappings & expected_mappings_set)}")
                framework.logger.warning(f"   Actual only: {len(actual_mappings - expected_mappings_set)}")
                framework.logger.warning(f"   Expected only: {len(expected_mappings_set - actual_mappings)}")
                
                if actual_mappings and expected_mappings_set:
                    framework.logger.warning(f"   Actual mappings: {sorted(list(actual_mappings))[:5]}")
                    framework.logger.warning(f"   Expected mappings: {sorted(list(expected_mappings_set))[:5]}")
            else:
                framework.logger.warning(f"   Mapping validation condition failed: expected_mappings={bool(expected_mappings)}, mcode_mappings={bool(self.mcode_mappings)}")


    @staticmethod
    def _calculate_fuzzy_text_matches(extracted_entities: List[Dict[str, Any]],
                                    expected_entities: List[Dict[str, Any]],
                                    framework: 'StrictPromptOptimizationFramework' = None,
                                    debug: bool = False) -> Tuple[int, int, int]:
        """
        Calculate text matches using fuzzy matching to handle different text representations
        
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        extracted_texts = [entity.get('text', '') for entity in extracted_entities]
        expected_texts = [entity.get('text', '') for entity in expected_entities]
        
        # Track matches
        matched_extracted = set()
        matched_expected = set()
        
        # First pass: exact matches
        for i, extracted_text in enumerate(extracted_texts):
            for j, expected_text in enumerate(expected_texts):
                if extracted_text == expected_text:
                    matched_extracted.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(f"   ✅ Exact match: '{extracted_text}' -> '{expected_text}'")
        
        # Second pass: fuzzy matches for remaining entities
        for i, extracted_text in enumerate(extracted_texts):
            if i in matched_extracted:
                continue
                
            for j, expected_text in enumerate(expected_texts):
                if j in matched_expected:
                    continue
                
                # Check if extracted text contains expected text (partial match)
                if expected_text.lower() in extracted_text.lower():
                    matched_extracted.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(f"   ✅ Partial match: '{extracted_text}' contains '{expected_text}'")
                    continue
                
                # Check if expected text contains extracted text (partial match)
                if extracted_text.lower() in expected_text.lower():
                    matched_extracted.add(i)
                    matched_expected.add(j)
                    if debug and framework:
                        framework.logger.warning(f"   ✅ Partial match: '{expected_text}' contains '{extracted_text}'")
                    continue
                
                # Check for combined entities (e.g., "Pregnancy or breastfeeding" should match both)
                combined_match = False
                if " or " in extracted_text.lower():
                    parts = [part.strip() for part in extracted_text.lower().split(" or ")]
                    if expected_text.lower() in parts:
                        matched_extracted.add(i)
                        matched_expected.add(j)
                        combined_match = True
                        if debug and framework:
                            framework.logger.warning(f"   ✅ Combined entity match: '{extracted_text}' contains '{expected_text}'")
                
                if not combined_match and " or " in expected_text.lower():
                    parts = [part.strip() for part in expected_text.lower().split(" or ")]
                    if extracted_text.lower() in parts:
                        matched_extracted.add(i)
                        matched_expected.add(j)
                        if debug and framework:
                            framework.logger.warning(f"   ✅ Combined entity match: '{expected_text}' contains '{extracted_text}'")
        
        # Calculate metrics
        true_positives = len(matched_expected)
        false_positives = len(extracted_entities) - len(matched_extracted)
        false_negatives = len(expected_entities) - len(matched_expected)
        
        if debug and framework:
            framework.logger.warning(f"   Total extracted: {len(extracted_entities)}")
            framework.logger.warning(f"   Total expected: {len(expected_entities)}")
            framework.logger.warning(f"   Matched extracted: {len(matched_extracted)}")
            framework.logger.warning(f"   Matched expected: {len(matched_expected)}")
            framework.logger.warning(f"   Unmatched extracted: {false_positives}")
            framework.logger.warning(f"   Unmatched expected: {false_negatives}")
            
            # Log some unmatched examples for debugging
            unmatched_extracted = [extracted_texts[i] for i in range(len(extracted_texts)) if i not in matched_extracted]
            unmatched_expected = [expected_texts[j] for j in range(len(expected_texts)) if j not in matched_expected]
            
            if unmatched_extracted:
                framework.logger.warning(f"   Unmatched extracted examples: {unmatched_extracted[:3]}")
            if unmatched_expected:
                framework.logger.warning(f"   Unmatched expected examples: {unmatched_expected[:3]}")
        
        return true_positives, false_positives, false_negatives

class StrictPromptOptimizationFramework(Loggable):
    """STRICT framework - no fallbacks, fails hard on invalid configs
    Now integrated with file-based prompt library using PromptLoader"""
    
    def __init__(self, results_dir: str = "./optimization_results"):
        super().__init__()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage for configurations and variants
        self.api_configs: Dict[str, APIConfig] = {}
        self.prompt_variants: Dict[str, PromptVariant] = {}
        self.test_cases: Dict[str, Dict[str, Any]] = {}
        self.benchmark_results: List[BenchmarkResult] = []
        
        # Initialize mCODE mapper for validation
        self.mcode_mapper = StrictMcodeMapper()
        
        # Automatically add all models from configuration
        self._add_all_models_from_config()
    
    def _add_all_models_from_config(self) -> None:
        """Automatically add all models from the configuration"""
        try:
            config = Config()
            llm_providers = config.get_llm_providers()
            
            for provider in llm_providers:
                model_name = provider.get('model')
                if model_name:
                    # Create a unique config name for each model
                    config_name = f"model_{model_name.replace('-', '_').replace('.', '_')}"
                    api_config = APIConfig(name=config_name, model=model_name)
                    self.add_api_config(api_config)
                    self.logger.info(f"Automatically added model '{model_name}' from configuration")
                else:
                    self.logger.warning(f"Skipping provider with no model name: {provider}")
        except Exception as e:
            self.logger.error(f"Failed to automatically add models from configuration: {str(e)}")
            raise
    
    def _validate_api_config(self, config: APIConfig) -> None:
        """STRICT validation - fail hard on invalid API configs using unified Config"""
        # The unified Config class already performs validation, so we just need to ensure
        # the configuration is accessible and valid
        try:
            # Test that we can access the configuration values by creating a Config instance
            config_instance = Config()
            _ = config_instance.get_api_key()
            _ = config_instance.get_base_url()
            _ = config_instance.get_model_name()
            _ = config_instance.get_temperature()
            _ = config_instance.get_max_tokens()
        except Exception as e:
            raise ValueError(f"Configuration validation failed for '{config.name}': {str(e)}")
        
        # Additional validation for base URL format
        if not config.base_url.startswith(('http://', 'https://')):
            raise ValueError(f"INVALID base URL for config '{config.name}': {config.base_url}")
    
    def add_api_config(self, config: APIConfig) -> None:
        """Add and validate API configuration"""
        self._validate_api_config(config)
        self.api_configs[config.name] = config
        self.logger.info(f"Added VALIDATED API config: {config.name} (model: {config.model})")
    
    def create_model_config(self, model_name: str, config_name: Optional[str] = None) -> APIConfig:
        """Create a new API configuration for a specific model"""
        if not config_name:
            config_name = f"model_{model_name.replace('-', '_').replace('.', '_')}"
        
        config = APIConfig(name=config_name, model=model_name)
        self.add_api_config(config)
        return config
    
    def add_model_configs(self, model_names: List[str]) -> None:
        """Add multiple model configurations at once"""
        for model_name in model_names:
            self.create_model_config(model_name)
    
    def add_prompt_variant(self, variant: PromptVariant) -> None:
        """Add a prompt variant with STRICT validation"""
        # Validate that the prompt key exists in the prompt library
        try:
            # Test loading the prompt to ensure it exists and is accessible
            prompt_loader.get_prompt(variant.prompt_key)
        except Exception as e:
            raise ValueError(f"Prompt key '{variant.prompt_key}' not found or inaccessible in prompt library: {str(e)}")
        
        self.prompt_variants[variant.id] = variant
        self.logger.info(f"Added VALIDATED prompt variant: {variant.name} ({variant.prompt_type.value}) -> {variant.prompt_key}")
    
    def add_test_case(self, case_id: str, test_data: Dict[str, Any]) -> None:
        """Add a test case"""
        self.test_cases[case_id] = test_data
        self.logger.info(f"Added test case: {case_id}")
    
    def _convert_entities_to_mcode(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert extracted entities to mCODE format using the mCODE mapper
        This allows for proper mCODE-based validation instead of exact text matching
        """
        if not entities:
            return []
        
        try:
            # Use the mCODE mapper to convert entities to mCODE format
            mapping_result = self.mcode_mapper.map_to_mcode(entities)
            return mapping_result.get('mapped_elements', [])
        except Exception as e:
            self.logger.warning(f"Failed to convert entities to mCODE format: {str(e)}")
            # Return empty list if conversion fails
            return []
    
    def load_test_cases_from_file(self, file_path: str) -> None:
        """Load test cases from JSON file"""
        try:
            with open(file_path, 'r') as f:
                test_cases = json.load(f)
            
            for case_id, case_data in test_cases.items():
                self.add_test_case(case_id, case_data)
            
            self.logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load test cases: {str(e)}")
            raise
    
    async def run_benchmark_async(self,
                                 prompt_variant_id: str,
                                 api_config_name: str,
                                 test_case_id: str,
                                 expected_entities: List[Dict[str, Any]] = None,
                                 expected_mappings: List[Dict[str, Any]] = None) -> BenchmarkResult:
        """
        Run a single benchmark asynchronously - simplified version for UI
        """
        result = BenchmarkResult(
            prompt_variant_id=prompt_variant_id,
            api_config_name=api_config_name,
            test_case_id=test_case_id
        )
        
        try:
            # Get the test case data
            test_case = self.test_cases.get(test_case_id)
            if not test_case:
                raise ValueError(f"Test case {test_case_id} not found")
            
            # Get the prompt variant
            prompt_variant = self.prompt_variants.get(prompt_variant_id)
            if not prompt_variant:
                raise ValueError(f"Prompt variant {prompt_variant_id} not found")
            
            # Get the API config
            api_config = self.api_configs.get(api_config_name)
            if not api_config:
                raise ValueError(f"API config {api_config_name} not found")
            
            # Get the actual prompt content from the prompt library
            prompt_content = prompt_variant.get_prompt_content()
            
            self.logger.info(f"🧪 Starting async benchmark:")
            self.logger.info(f"   📋 Prompt: {prompt_variant.name} ({prompt_variant.prompt_key})")
            self.logger.info(f"   🤖 Model: {api_config.model}")
            self.logger.info(f"   📊 Test Case: {test_case_id}")
            
            # Create pipeline callback
            def pipeline_callback(test_data):
                from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
                
                pipeline = StrictDynamicExtractionPipeline()
                
                # Set appropriate prompt based on type
                if prompt_variant.prompt_type == PromptType.NLP_EXTRACTION:
                    pipeline.nlp_engine.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
                elif prompt_variant.prompt_type == PromptType.MCODE_MAPPING:
                    # This would require changes to StrictMcodeMapper
                    pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE = prompt_content
                
                return pipeline.process_clinical_trial(test_data)
            
            # Run the pipeline processing
            start_time = time.time()
            pipeline_result = pipeline_callback(test_case)
            end_time = time.time()
            
            # Calculate duration
            result.duration_ms = (end_time - start_time) * 1000
            
            # Extract results
            if hasattr(pipeline_result, 'extracted_entities'):
                result.extracted_entities = pipeline_result.extracted_entities
                result.entities_extracted = len(pipeline_result.extracted_entities)
            
            if hasattr(pipeline_result, 'mcode_mappings'):
                result.mcode_mappings = pipeline_result.mcode_mappings
                result.entities_mapped = len(pipeline_result.mcode_mappings)
            
            if hasattr(pipeline_result, 'validation_results'):
                result.validation_results = pipeline_result.validation_results
                result.compliance_score = pipeline_result.validation_results.get('compliance_score', 0.0)
            
            # Mark the result as successful before calculating metrics
            result.success = True
            
            # Calculate metrics using provided gold standard data
            result.calculate_metrics(expected_entities, expected_mappings, self)
            
            self.logger.info(f"✅ Async benchmark completed in {result.duration_ms:.2f}ms")
            self.logger.info(f"   📊 Extraction: {result.entities_extracted} entities")
            self.logger.info(f"   🗺️  Mapping: {result.entities_mapped} mCODE elements")
            self.logger.info(f"   🎯 Metrics: F1={result.f1_score:.3f}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"❌ Async benchmark FAILED: {str(e)}")
            raise
        
        result.end_time = datetime.now()
        self.benchmark_results.append(result)
        
        # Save results
        self._save_benchmark_result(result)
        
        return result
    
    def run_benchmark(self,
                      prompt_variant_id: str,
                      api_config_name: str,
                      test_case_id: str,
                      pipeline_callback: Callable,
                      expected_entities: List[Dict[str, Any]] = None,
                      expected_mappings: List[Dict[str, Any]] = None,
                      current_index: int = None,
                      total_count: int = None,
                      benchmark_start_time: float = None) -> BenchmarkResult:
        """
        Run a single benchmark - STRICT mode, no fallbacks
        Now uses prompt library content instead of hardcoded templates
        
        Args:
            prompt_variant_id: ID of the prompt variant to test
            api_config_name: Name of the API configuration to use
            test_case_id: ID of the test case to run
            pipeline_callback: Callback function that executes the pipeline
            current_index: Current test index (for progress tracking)
            total_count: Total number of tests (for progress tracking)
            benchmark_start_time: Start time of the overall benchmark run (for ETR calculation)
        """
        result = BenchmarkResult(
            prompt_variant_id=prompt_variant_id,
            api_config_name=api_config_name,
            test_case_id=test_case_id
        )
        
        try:
            # Get the test case data
            test_case = self.test_cases.get(test_case_id)
            if not test_case:
                raise ValueError(f"Test case {test_case_id} not found")
            
            # Get the prompt variant
            prompt_variant = self.prompt_variants.get(prompt_variant_id)
            if not prompt_variant:
                raise ValueError(f"Prompt variant {prompt_variant_id} not found")
            
            # Get the API config
            api_config = self.api_configs.get(api_config_name)
            if not api_config:
                raise ValueError(f"API config {api_config_name} not found")
            
            # Get the actual prompt content from the prompt library
            prompt_content = prompt_variant.get_prompt_content()
            
            # Enhanced progress tracking with detailed information
            progress_info = ""
            if current_index is not None and total_count is not None:
                progress_info = f" [{current_index}/{total_count}]"
            
            # Get detailed model information
            model_info = f"model={api_config.model}"
            if api_config.temperature is not None:
                model_info += f", temp={api_config.temperature}"
            if api_config.max_tokens is not None:
                model_info += f", max_tokens={api_config.max_tokens}"
            
            self.logger.info(f"🧪{progress_info} Starting STRICT benchmark:")
            self.logger.info(f"   📋 Prompt: {prompt_variant.name} ({prompt_variant.prompt_key})")
            self.logger.info(f"   🤖 Model: {model_info}")
            self.logger.info(f"   📊 Test Case: {test_case_id}")
            self.logger.info(f"   🔄 Prompt Type: {prompt_variant.prompt_type.value}")
            
            # Note: The pipeline callback should use the Config class directly for API configuration
            # Environment variables are no longer needed since the pipeline components use Config class
            
            # Run the pipeline processing
            start_time = time.time()
            pipeline_result = pipeline_callback(test_case, prompt_content, prompt_variant_id, api_config_name)
            end_time = time.time()
            
            # Calculate duration
            result.duration_ms = (end_time - start_time) * 1000
            
            # Extract results
            if hasattr(pipeline_result, 'extracted_entities'):
                result.extracted_entities = pipeline_result.extracted_entities
                result.entities_extracted = len(pipeline_result.extracted_entities)
            
            if hasattr(pipeline_result, 'mcode_mappings'):
                result.mcode_mappings = pipeline_result.mcode_mappings
                result.entities_mapped = len(pipeline_result.mcode_mappings)
                self.logger.debug(f"Set mcode_mappings: {len(result.mcode_mappings)} mappings")
            
            if hasattr(pipeline_result, 'validation_results'):
                result.validation_results = pipeline_result.validation_results
                result.compliance_score = pipeline_result.validation_results.get('compliance_score', 0.0)

            if hasattr(pipeline_result, 'metadata') and 'token_usage' in pipeline_result.metadata:
                token_usage = pipeline_result.metadata['token_usage']
                if token_usage:
                    result.token_usage = token_usage.get('total_tokens', 0)
            
            # Also capture token usage from mcode_mappings
            if hasattr(result, 'mcode_mappings') and result.mcode_mappings:
                for mapping in result.mcode_mappings:
                    if 'metadata' in mapping and 'token_usage' in mapping['metadata']:
                        token_usage = mapping['metadata']['token_usage']
                        if token_usage:
                            result.token_usage += token_usage.get('total_tokens', 0)
            
            # Capture aggregate token usage from global tracker
            aggregate_token_usage = global_token_tracker.get_total_usage()
            if aggregate_token_usage.total_tokens > 0:
                result.token_usage = aggregate_token_usage.total_tokens
                self.logger.info(f"   📊 Aggregate token usage: {aggregate_token_usage.total_tokens} tokens")
                self.logger.info(f"      Prompt tokens: {aggregate_token_usage.prompt_tokens}")
                self.logger.info(f"      Completion tokens: {aggregate_token_usage.completion_tokens}")
            
            # Mark the result as successful before calculating metrics
            result.success = True
            
            # Calculate metrics using provided gold standard data
            result.calculate_metrics(expected_entities, expected_mappings, self)
            
            # Enhanced completion logging with detailed metrics
            completion_status = "✅"
            if current_index is not None and total_count is not None:
                completion_status = f"✅ [{current_index}/{total_count}]"
            
            # Calculate estimated time remaining if we have progress info
            time_remaining = ""
            if current_index is not None and total_count is not None and current_index > 1:
                avg_time_per_test = (time.time() - benchmark_start_time) / current_index
                remaining_tests = total_count - current_index
                remaining_seconds = avg_time_per_test * remaining_tests
                time_remaining = f" ⏰ ETR: {self._format_time_remaining(remaining_seconds)}"
            
            self.logger.info(f"{completion_status} STRICT benchmark completed in {result.duration_ms:.2f}ms{time_remaining}")
            self.logger.info(f"   📊 Extraction: {result.entities_extracted} entities")
            self.logger.info(f"   🗺️  Mapping: {result.entities_mapped} mCODE elements")
            self.logger.info(f"   🎯 Metrics: F1={result.f1_score:.3f}, Precision={result.precision:.3f}, Recall={result.recall:.3f}")
            self.logger.info(f"   ✅ Compliance: {result.compliance_score:.2%}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            
            # Enhanced error logging with progress info and time remaining
            error_status = "❌"
            time_remaining = ""
            if current_index is not None and total_count is not None:
                error_status = f"❌ [{current_index}/{total_count}]"
                
                # Calculate estimated time remaining for error cases too
                if current_index > 1:
                    avg_time_per_test = (time.time() - benchmark_start_time) / current_index
                    remaining_tests = total_count - current_index
                    remaining_seconds = avg_time_per_test * remaining_tests
                    time_remaining = f" ⏰ ETR: {self._format_time_remaining(remaining_seconds)}"
            
            self.logger.error(f"{error_status} STRICT benchmark FAILED in {result.duration_ms:.2f}ms{time_remaining}: {str(e)}")
            raise  # Re-raise to fail hard
        
        result.end_time = datetime.now()
        self.benchmark_results.append(result)
        
        # Save results
        self._save_benchmark_result(result)
        
        return result
    
    def _save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to disk"""
        result_file = self.results_dir / f"benchmark_{result.run_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_benchmark_results(self) -> None:
        """Load benchmark results from disk"""
        self.benchmark_results.clear()
        
        for result_file in self.results_dir.glob("benchmark_*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Convert back to BenchmarkResult object
                result = BenchmarkResult()
                result.run_id = result_data.get('run_id', '')
                result.prompt_variant_id = result_data.get('prompt_variant_id', '')
                result.api_config_name = result_data.get('api_config_name', '')
                result.test_case_id = result_data.get('test_case_id', '')
                result.start_time = datetime.fromisoformat(result_data.get('start_time', datetime.now().isoformat()))
                
                if result_data.get('end_time'):
                    result.end_time = datetime.fromisoformat(result_data['end_time'])
                
                result.duration_ms = result_data.get('duration_ms', 0.0)
                result.success = result_data.get('success', False)
                result.error_message = result_data.get('error_message', '')
                result.entities_extracted = result_data.get('entities_extracted', 0)
                result.entities_mapped = result_data.get('entities_mapped', 0)
                result.extraction_completeness = result_data.get('extraction_completeness', 0.0)
                result.mapping_accuracy = result_data.get('mapping_accuracy', 0.0)
                result.precision = result_data.get('precision', 0.0)
                result.recall = result_data.get('recall', 0.0)
                result.f1_score = result_data.get('f1_score', 0.0)
                result.compliance_score = result_data.get('compliance_score', 0.0)
                result.token_usage = result_data.get('token_usage', 0)
                
                # Load raw results for validation recalculation
                result.extracted_entities = result_data.get('extracted_entities', [])
                result.mcode_mappings = result_data.get('mcode_mappings', [])
                result.validation_results = result_data.get('validation_results', {})
                
                self.benchmark_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to load benchmark result {result_file}: {str(e)}")
                raise
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert benchmark results to pandas DataFrame"""
        if not self.benchmark_results:
            self.load_benchmark_results()
        
        data = []
        for result in self.benchmark_results:
            row = {
                'run_id': result.run_id,
                'prompt_variant_id': result.prompt_variant_id,
                'api_config_name': result.api_config_name,
                'test_case_id': result.test_case_id,
                'duration_ms': result.duration_ms,
                'success': result.success,
                'entities_extracted': result.entities_extracted,
                'entities_mapped': result.entities_mapped,
                'extraction_completeness': result.extraction_completeness,
                'mapping_accuracy': result.mapping_accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'compliance_score': result.compliance_score,
                'token_usage': result.token_usage,
                'aggregate_token_usage': result.token_usage  # For backward compatibility
            }
            
            # Add prompt variant details
            variant = self.prompt_variants.get(result.prompt_variant_id)
            if variant:
                row.update({
                    'prompt_name': variant.name,
                    'prompt_type': variant.prompt_type.value,
                    'prompt_version': variant.version,
                    'prompt_key': variant.prompt_key
                })
            
            # Add model information
            config = self.api_configs.get(result.api_config_name)
            if config:
                row.update({
                    'model': config.model,
                    'model_config_name': config.name
                })
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def run_all_combinations(self, test_case_ids: List[str], pipeline_callback: Callable,
                           expected_entities: List[Dict[str, Any]] = None,
                           expected_mappings: List[Dict[str, Any]] = None) -> None:
        """
        Run benchmarks across all combinations of prompt variants and API configurations
        
        Args:
            test_case_ids: List of test case IDs to run
            pipeline_callback: Callback function that executes the pipeline
            expected_entities: Gold standard entities for validation
            expected_mappings: Gold standard mCODE mappings for validation
        """
        if not self.prompt_variants:
            raise ValueError("No prompt variants configured")
        if not self.api_configs:
            raise ValueError("No API configurations configured")
        if not test_case_ids:
            raise ValueError("No test case IDs provided")
        
        total_combinations = len(self.prompt_variants) * len(self.api_configs) * len(test_case_ids)
        current_index = 0
        benchmark_start_time = time.time()
        
        self.logger.info(f"🧪 Starting optimization across {len(self.prompt_variants)} prompts × {len(self.api_configs)} models × {len(test_case_ids)} test cases = {total_combinations} combinations")
        self.logger.info(f"   Models: {[config.model for config in self.api_configs.values()]}")
        self.logger.info(f"   Prompts: {[variant.name for variant in self.prompt_variants.values()]}")
        self.logger.info(f"   Test Cases: {test_case_ids}")
        
        for prompt_id, prompt_variant in self.prompt_variants.items():
            for config_name, api_config in self.api_configs.items():
                for test_case_id in test_case_ids:
                    current_index += 1
                    try:
                        self.run_benchmark(
                            prompt_variant_id=prompt_id,
                            api_config_name=config_name,
                            test_case_id=test_case_id,
                            pipeline_callback=pipeline_callback,
                            expected_entities=expected_entities,
                            expected_mappings=expected_mappings,
                            current_index=current_index,
                            total_count=total_combinations,
                            benchmark_start_time=benchmark_start_time
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Failed combination {current_index}/{total_combinations}: {prompt_variant.name} + {config_name} + {test_case_id}: {str(e)}")
                        # Continue with other combinations despite failures
    
    def get_best_combinations(self, metric: str = 'f1_score', top_n: int = 5) -> pd.DataFrame:
        """
        Get the best prompt-model combinations based on the specified metric
        
        Args:
            metric: Metric to optimize for ('f1_score', 'precision', 'recall', 'compliance_score')
            top_n: Number of top combinations to return
        
        Returns:
            DataFrame with top combinations and their metrics
        """
        if not self.benchmark_results:
            self.load_benchmark_results()
        
        df = self.get_results_dataframe()
        
        if df.empty:
            return df
        
        # Group by prompt and model, calculate average metrics
        grouped = df.groupby(['prompt_variant_id', 'api_config_name']).agg({
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'compliance_score': 'mean',
            'duration_ms': 'mean',
            'success': lambda x: x.sum() / len(x)  # success rate
        }).reset_index()
        
        # Add prompt and model details
        for idx, row in grouped.iterrows():
            variant = self.prompt_variants.get(row['prompt_variant_id'])
            config = self.api_configs.get(row['api_config_name'])
            if variant:
                grouped.at[idx, 'prompt_name'] = variant.name
                grouped.at[idx, 'prompt_key'] = variant.prompt_key
                grouped.at[idx, 'prompt_type'] = variant.prompt_type.value
            if config:
                grouped.at[idx, 'model'] = config.model
        
        # Sort by the specified metric and return top N
        return grouped.sort_values(metric, ascending=False).head(top_n)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison statistics across different models"""
        if not self.benchmark_results:
            self.load_benchmark_results()
        
        df = self.get_results_dataframe()
        
        if df.empty:
            return df
        
        # Add model information to results
        for idx, row in df.iterrows():
            config = self.api_configs.get(row['api_config_name'])
            if config:
                df.at[idx, 'model'] = config.model
        
        # Group by model and calculate statistics
        model_stats = df.groupby('model').agg({
            'f1_score': ['mean', 'std', 'count'],
            'precision': 'mean',
            'recall': 'mean',
            'compliance_score': 'mean',
            'duration_ms': 'mean',
            'success': lambda x: x.sum() / len(x)  # success rate
        }).round(3)
        
        # Flatten column names
        model_stats.columns = ['_'.join(col).strip('_') for col in model_stats.columns.values]
        return model_stats
    
    def get_prompt_comparison(self) -> pd.DataFrame:
        """Get comparison statistics across different prompts"""
        if not self.benchmark_results:
            self.load_benchmark_results()
        
        df = self.get_results_dataframe()
        
        if df.empty:
            return df
        
        # Add prompt information to results
        for idx, row in df.iterrows():
            variant = self.prompt_variants.get(row['prompt_variant_id'])
            if variant:
                df.at[idx, 'prompt_name'] = variant.name
                df.at[idx, 'prompt_key'] = variant.prompt_key
        
        # Group by prompt and calculate statistics
        prompt_stats = df.groupby(['prompt_name', 'prompt_key']).agg({
            'f1_score': ['mean', 'std', 'count'],
            'precision': 'mean',
            'recall': 'mean',
            'compliance_score': 'mean',
            'duration_ms': 'mean',
            'success': lambda x: x.sum() / len(x)  # success rate
        }).round(3)
        
        # Flatten column names
        prompt_stats.columns = ['_'.join(col).strip('_') for col in prompt_stats.columns.values]
        return prompt_stats

    
    def set_default_prompt(self, prompt_type: str, prompt_name: str) -> None:
        """
        Set a prompt as default for its type.
        
        Args:
            prompt_type: The type of prompt (NLP_EXTRACTION or MCODE_MAPPING)
            prompt_name: The name of the prompt to set as default
        """
        # Find the prompt variant
        prompt_variant = None
        for variant in self.prompt_variants.values():
            if variant.name == prompt_name and variant.prompt_type.value == prompt_type.lower():
                prompt_variant = variant
                break
        
        if not prompt_variant:
            raise ValueError(f"Prompt '{prompt_name}' of type '{prompt_type}' not found")
        
        # Update the prompt configuration file to mark this prompt as default
        self._update_prompt_config_default(prompt_type, prompt_name)
        self.logger.info(f"Set '{prompt_name}' as default prompt for type '{prompt_type}'")
    
    def _update_prompt_config_default(self, prompt_type: str, prompt_name: str) -> None:
        """
        Update the prompt configuration file to mark a prompt as default.
        
        Args:
            prompt_type: The type of prompt (NLP_EXTRACTION or MCODE_MAPPING)
            prompt_name: The name of the prompt to set as default
        """
        config_path = Path("prompts/prompts_config.json")
        
        # Load the current configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Update the configuration to set the default prompt
        prompts = config_data["prompt_library"]["prompts"]
        for category in prompts.values():
            if prompt_type.lower() in category:
                for prompt in category[prompt_type.lower()]:
                    # Set default=True for the specified prompt, False for others
                    prompt["default"] = prompt["name"] == prompt_name
        
        # Save the updated configuration
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Reload the prompt loader to reflect the changes
        prompt_loader.reload_config()
    
    def get_default_prompt(self, prompt_type: str) -> Optional[str]:
        """
        Get the default prompt for a given type.
        
        Args:
            prompt_type: The type of prompt (NLP_EXTRACTION or MCODE_MAPPING)
            
        Returns:
            The name of the default prompt, or None if no default is set
        """
        config_path = Path("prompts/prompts_config.json")
        
        # Load the current configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Find the default prompt for the given type
        prompts = config_data["prompt_library"]["prompts"]
        for category in prompts.values():
            if prompt_type.lower() in category:
                for prompt in category[prompt_type.lower()]:
                    if prompt.get("default", False):
                        return prompt["name"]
        
        return None
    
    def set_default_model(self, model_name: str) -> None:
        """
        Set a model as default.
        
        Args:
            model_name: The name of the model to set as default
        """
        # Check if the model exists in our configurations
        model_found = False
        for config in self.api_configs.values():
            if config.model == model_name:
                model_found = True
                break
        
        if not model_found:
            # Check if it's a valid model in our model library
            try:
                from src.utils.model_loader import model_loader
                all_models = model_loader.get_all_models()
                if model_name not in all_models:
                    raise ValueError(f"Model '{model_name}' not found in model library")
            except Exception:
                raise ValueError(f"Model '{model_name}' not found")
        
        # Update the model configuration file to mark this model as default
        self._update_model_config_default(model_name)
        self.logger.info(f"Set '{model_name}' as default model")
    
    def _update_model_config_default(self, model_name: str) -> None:
        """
        Update the model configuration file to mark a model as default.
        
        Args:
            model_name: The name of the model to set as default
        """
        config_path = Path("models/models_config.json")
        
        # Load the current configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Update the configuration to set the default model
        models = config_data["model_library"]["models"]
        for category in models.values():
            for subcategory, model_list in category.items():
                for model in model_list:
                    # Set default=True for the specified model, False for others
                    model["default"] = model["name"] == model_name
        
        # Save the updated configuration
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Reload the model loader to reflect the changes
        try:
            from src.utils.model_loader import reload_models_config
            reload_models_config()
        except Exception as e:
            self.logger.warning(f"Failed to reload model configuration: {str(e)}")
    
    def get_default_model(self) -> Optional[str]:
        """
        Get the default model.
        
        Returns:
            The name of the default model, or None if no default is set
        """
        config_path = Path("models/models_config.json")
        
        # Load the current configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Find the default model
        models = config_data["model_library"]["models"]
        for category in models.values():
            for subcategory, model_list in category.items():
                for model in model_list:
                    if model.get("default", False):
                        return model["name"]
        
        return None
    
    def _format_time_remaining(self, seconds: float) -> str:
        """Format time remaining in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def export_results_to_csv(self, filename: str = None) -> str:
        """Export benchmark results to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)
        return filename
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        try:
            df = self.get_results_dataframe()
            
            if df.empty:
                return {'error': 'No benchmark data available'}
            
            # Get summary statistics
            summary_stats = {
                'total_experiments': len(df),
                'success_rate': df['success'].mean(),
                'avg_duration_ms': df['duration_ms'].mean(),
                'avg_entities': df['entities_extracted'].mean(),
                'avg_compliance': df['compliance_score'].mean(),
                'avg_f1_score': df['f1_score'].mean(),
                'models_tested': df['model'].nunique(),
                'prompts_tested': df['prompt_name'].nunique()
            }
            
            # Get best performers
            best_f1 = self.get_best_combinations('f1_score', 1)
            best_compliance = self.get_best_combinations('compliance_score', 1)
            
            return {
                'generated_at': datetime.now().isoformat(),
                'total_experiments': summary_stats['total_experiments'],
                'success_rate': summary_stats['success_rate'],
                'avg_duration_ms': summary_stats['avg_duration_ms'],
                'avg_entities': summary_stats['avg_entities'],
                'avg_compliance': summary_stats['avg_compliance'],
                'avg_f1_score': summary_stats['avg_f1_score'],
                'models_tested': summary_stats['models_tested'],
                'prompts_tested': summary_stats['prompts_tested'],
                'best_configs': {
                    'f1_score': {
                        'name': best_f1.iloc[0]['prompt_name'] if not best_f1.empty else 'N/A',
                        'model': best_f1.iloc[0]['model'] if not best_f1.empty else 'N/A',
                        'score': best_f1.iloc[0]['f1_score'] if not best_f1.empty else 0
                    },
                    'compliance_score': {
                        'name': best_compliance.iloc[0]['prompt_name'] if not best_compliance.empty else 'N/A',
                        'model': best_compliance.iloc[0]['model'] if not best_compliance.empty else 'N/A',
                        'score': best_compliance.iloc[0]['compliance_score'] if not best_compliance.empty else 0
                    }
                }
            }
            
        except Exception as e:
            return {'error': f'Failed to generate report: {str(e)}'}
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Generate data for visualizations"""
        df = self.get_results_dataframe()
        
        if df.empty:
            return {}
        
        # Success rates by prompt
        success_rates = df.groupby('prompt_name')['success'].mean().to_dict()
        
        # Compliance scores by prompt
        compliance_scores = df.groupby('prompt_name')['compliance_score'].mean().to_dict()
        
        # Performance comparison data
        performance_data = df.groupby(['prompt_name', 'model']).agg({
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'compliance_score': 'mean',
            'duration_ms': 'mean'
        }).reset_index()
        
        return {
            'success_rates': {
                'labels': list(success_rates.keys()),
                'values': list(success_rates.values())
            },
            'compliance_scores': {
                'labels': list(compliance_scores.keys()),
                'values': list(compliance_scores.values())
            },
            'performance_comparison': performance_data.to_dict('records')
        }