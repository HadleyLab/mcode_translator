#!/usr/bin/env python3
"""
Mcode Optimize - Unified Command Line Interface for Prompt and Model Optimization

This CLI provides comprehensive optimization capabilities across all prompts and models
with the ability to set the best performing combinations as defaults in their respective libraries.
"""

import json
import sys
import asyncio
import logging
import click
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
load_dotenv()

# Setup logging
from src.utils.logging_config import setup_logging, get_logger
setup_logging(logging.INFO)
logger = get_logger(__name__)

# Import pipeline components
from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from pipeline.nlp_extractor import NlpLlm
from src.pipeline.mcode_mapper import McodeMapper
from src.utils.prompt_loader import PromptLoader, load_prompt
from src.utils import ModelLoader, load_model

# Import optimization framework
from src.optimization.prompt_optimization_framework import (
    PromptOptimizationFramework, 
    PromptType, 
    APIConfig, 
    PromptVariant
)


class StrictValidator:
    """STRICT validation utilities for file and data validation"""
    
    @staticmethod
    def validate_file_exists(file_path: Path, description: str) -> None:
        """Validate that a file exists and is accessible"""
        if not file_path.exists():
            raise FileNotFoundError(
                f"{description} not found: {file_path}\n"
                f"Please ensure the file exists and is accessible."
            )
        
        if not file_path.is_file():
            raise ValueError(
                f"{description} is not a valid file: {file_path}\n"
                f"Expected a file, found a directory or invalid path."
            )
        
        # Check if file is readable
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1)  # Try to read first byte
        except (IOError, OSError, UnicodeDecodeError) as e:
            raise IOError(
                f"Cannot read {description}: {file_path}\n"
                f"Error: {e}\n"
                f"Please check file permissions and encoding."
            )
    
    @staticmethod
    def load_json_file(file_path: Path, description: str) -> Any:
        """Load JSON file with basic validation"""
        StrictValidator.validate_file_exists(file_path, description)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValueError(f"{description}: Expected dictionary, got {type(data).__name__}")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in {description}: {e}\n"
                f"Please validate the JSON structure of the file."
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading {description}: {e}")


class GoldStandardTester:
    """STRICT Gold standard testing functionality"""
    
    def __init__(self):
        self.pipeline = NlpMcodePipeline()
        self.token_usage = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'extraction_tokens': 0,
            'mapping_tokens': 0
        }
    
    def update_token_usage(self, usage_data: Dict[str, int]) -> None:
        """Update token usage statistics"""
        if usage_data:
            self.token_usage['total_tokens'] += usage_data.get('total_tokens', 0)
            self.token_usage['prompt_tokens'] += usage_data.get('prompt_tokens', 0)
            self.token_usage['completion_tokens'] += usage_data.get('completion_tokens', 0)
            self.token_usage['extraction_tokens'] += usage_data.get('extraction_tokens', 0)
            self.token_usage['mapping_tokens'] += usage_data.get('mapping_tokens', 0)
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        return self.token_usage.copy()
    
    def reset_token_usage(self) -> None:
        """Reset token usage statistics"""
        self.token_usage = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'extraction_tokens': 0,
            'mapping_tokens': 0
        }
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
            
        normalized1 = text1.lower().strip()
        normalized2 = text2.lower().strip()
        
        # Exact match
        if normalized1 == normalized2:
            return 1.0
        
        # Contains match
        if normalized1 in normalized2 or normalized2 in normalized1:
            return 0.9
        
        # Word overlap similarity
        words1 = set(normalized1.split())
        words2 = set(normalized2.split())
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        if total > 0:
            return overlap / total
        
        return 0.0
    
    def calculate_metrics(self, actual: List[Dict], expected: List[Dict], entity_type: str = "entity") -> Dict[str, float]:
        """Calculate precision, recall, and F1 score with semantic similarity matching"""
        # Debug: Check what types we're getting
        logger.info(f"DEBUG: calculate_metrics called with actual type: {type(actual)}, expected type: {type(expected)}")
        logger.info(f"DEBUG: Actual value: {actual}")
        logger.info(f"DEBUG: Expected value: {expected}")
        
        try:
            # Handle None values and ensure we have lists
            if actual is None:
                actual = []
            if expected is None:
                expected = []
                
            # Debug: Check what types we have after handling None
            logger.info(f"DEBUG: After None handling - actual type: {type(actual)}, expected type: {type(expected)}")
            
            # Debug: Check what we're passing to isinstance
            logger.info(f"DEBUG: About to check isinstance for actual: {type(actual)}, expected: {type(expected)}")
            
            try:
                if not isinstance(actual, list) or not isinstance(expected, list):
                    error_msg = f"Both actual and expected must be lists. Got actual: {type(actual)}, expected: {type(expected)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            except TypeError as e:
                logger.error(f"TypeError in isinstance check: {e}")
                logger.error(f"actual value: {actual}, type: {type(actual)}")
                logger.error(f"expected value: {expected}, type: {type(expected)}")
                raise
            
            if not actual and not expected:
                return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
            
            # For mapping validation, compare element_name fields with semantic similarity
            if entity_type == "mapping":
                return self._calculate_mapping_metrics(actual, expected)
            
            # For extraction validation, use original text-based comparison
            return self._calculate_extraction_metrics(actual, expected)
            
        except Exception as e:
            logger.error(f"Error in calculate_metrics: {e}")
            logger.error(f"Actual: {actual}")
            logger.error(f"Expected: {expected}")
            logger.error(f"Entity type: {entity_type}")
            raise
    
    def _calculate_extraction_metrics(self, actual: List[Dict], expected: List[Dict]) -> Dict[str, float]:
        """STRICT: Calculate metrics for entity extraction using exact LLM format matching"""
        # Debug: Check what types we're getting in helper method
        logger.info(f"DEBUG: _calculate_extraction_metrics called with actual type: {type(actual)}, expected type: {type(expected)}")
        
        # Extract texts from expected (gold standard) - STRICT LLM format
        expected_texts = set()
        for item in expected:
            logger.info(f"DEBUG: Expected item type: {type(item)}, value: {item}")
            if not isinstance(item, dict):
                raise ValueError(f"Expected item must be dictionary, got {type(item).__name__}")
            if 'text' not in item:
                raise ValueError("Gold standard items must contain 'text' field in LLM format")
            expected_texts.add(item['text'].lower().strip())
        
        # Extract texts from actual results - STRICT LLM format
        actual_texts = set()
        for item in actual:
            logger.info(f"DEBUG: Actual item type: {type(item)}, value: {item}")
            if not isinstance(item, dict):
                raise ValueError(f"Actual item must be dictionary, got {type(item).__name__}")
            if 'text' not in item:
                raise ValueError("Actual results must contain 'text' field in LLM format")
            actual_texts.add(item['text'].lower().strip())
        
        true_positives = len(actual_texts.intersection(expected_texts))
        false_positives = len(actual_texts - expected_texts)
        false_negatives = len(expected_texts - actual_texts)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def _calculate_mapping_metrics(self, actual: List[Dict], expected: List[Dict]) -> Dict[str, float]:
        """Calculate metrics for Mcode mapping using semantic similarity"""
        # Debug: Check what types we're getting in helper method
        logger.info(f"DEBUG: _calculate_mapping_metrics called with actual type: {type(actual)}, expected type: {type(expected)}")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Convert to dictionaries for easier lookup by element_name
        gold_dict = {}
        for mapping in expected:
            logger.info(f"DEBUG: Expected mapping type: {type(mapping)}, value: {mapping}")
            element_name = mapping.get('element_name', '')
            if element_name:
                gold_dict[element_name] = mapping
        
        pred_dict = {}
        for mapping in actual:
            logger.info(f"DEBUG: Actual mapping type: {type(mapping)}, value: {mapping}")
            element_name = mapping.get('element_name', '')
            if element_name:
                pred_dict[element_name] = mapping
        
        # Find matches using semantic similarity
        matched_gold = set()
        matched_pred = set()
        
        for gold_key, gold_mapping in gold_dict.items():
            for pred_key, pred_mapping in pred_dict.items():
                similarity = self.semantic_similarity(gold_key, pred_key)
                if similarity >= 0.7:  # 70% similarity threshold
                    true_positives += 1
                    matched_gold.add(gold_key)
                    matched_pred.add(pred_key)
                    break
        
        # Calculate remaining unmatched
        false_positives = len(pred_dict) - len(matched_pred)
        false_negatives = len(gold_dict) - len(matched_gold)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def test_single_case(self, case_name: str, test_data: Dict, gold_standard: Dict) -> Dict[str, Any]:
        """STRICT Test a single test case against the pipeline with no fallbacks"""
        logger.info(f"Testing: {case_name}")
        logger.info(f"DEBUG: test_data type: {type(test_data)}, gold_standard type: {type(gold_standard)}")
        logger.info(f"DEBUG: gold_standard keys: {list(gold_standard.keys()) if hasattr(gold_standard, 'keys') else 'N/A'}")
        logger.info(f"DEBUG: gold_standard value: {gold_standard}")
        
        try:
            # Validate input data structures
            if not isinstance(test_data, dict):
                raise ValueError(f"Test data for {case_name} must be a dictionary")
            if not isinstance(gold_standard, dict):
                raise ValueError(f"Gold standard for {case_name} must be a dictionary")
            
            # Run the STRICT pipeline with complete clinical trial data
            logger.info(f"Running STRICT pipeline for: {case_name}")
            result = self.pipeline.process_clinical_trial(test_data)
            
            # Validate pipeline result structure
            if not hasattr(result, 'extracted_entities'):
                raise ValueError("Pipeline result missing extracted_entities attribute")
            if not hasattr(result, 'mcode_mappings'):
                raise ValueError("Pipeline result missing mcode_mappings attribute")
            
            # Extract results from PipelineResult object
            extraction_result = {
                'entities': result.extracted_entities
            }
            mapping_result = {
                'mapped_elements': result.mcode_mappings
            }

            # Capture token usage if available
            if hasattr(result, 'token_usage') and result.token_usage:
                self.update_token_usage(result.token_usage)
            
            logger.info(f"Extraction result: {len(extraction_result['entities'])} entities")
            logger.info(f"Mapping result: {len(mapping_result['mapped_elements'])} mappings")
            
            # Compare with gold standard
            gold_extraction = gold_standard.get('expected_extraction', {})
            gold_mapping = gold_standard.get('expected_mcode_mappings', {})
            
            # Validate gold standard structure
            if not isinstance(gold_extraction, dict):
                raise ValueError("Gold standard extraction must be a dictionary")
            if not isinstance(gold_mapping, dict):
                raise ValueError("Gold standard mapping must be a dictionary")
            
            # Calculate metrics - gold_extraction and gold_mapping are dictionaries containing 'entities' and 'mapped_elements' lists
            # Ensure we have lists for calculate_metrics
            entities_list = extraction_result.get('entities', [])
            if not isinstance(entities_list, list):
                entities_list = []
            
            # Debug: Check what type gold_extraction is before isinstance check
            logger.info(f"DEBUG: gold_extraction type: {type(gold_extraction)}, value: {gold_extraction}")
            try:
                gold_entities_list = gold_extraction.get('entities', []) if isinstance(gold_extraction, dict) else []
            except TypeError as e:
                logger.error(f"TypeError in gold_extraction isinstance check: {e}")
                logger.error(f"gold_extraction value: {gold_extraction}")
                logger.error(f"gold_extraction type: {type(gold_extraction)}")
                raise
            if not isinstance(gold_entities_list, list):
                gold_entities_list = []
            
            # Debug: Check what we're passing to calculate_metrics for extraction
            logger.info(f"DEBUG: Extraction - entities_list type: {type(entities_list)}, value: {entities_list}")
            logger.info(f"DEBUG: Extraction - gold_entities_list type: {type(gold_entities_list)}, value: {gold_entities_list}")
            
            extraction_metrics = self.calculate_metrics(entities_list, gold_entities_list)
            
            mapped_elements_list = mapping_result.get('mapped_elements', [])
            if not isinstance(mapped_elements_list, list):
                mapped_elements_list = []
            
            # Debug: Check what type gold_mapping is before isinstance check
            logger.info(f"DEBUG: gold_mapping type: {type(gold_mapping)}, value: {gold_mapping}")
            try:
                gold_mapped_list = gold_mapping.get('mapped_elements', []) if isinstance(gold_mapping, dict) else []
            except TypeError as e:
                logger.error(f"TypeError in gold_mapping isinstance check: {e}")
                logger.error(f"gold_mapping value: {gold_mapping}")
                logger.error(f"gold_mapping type: {type(gold_mapping)}")
                raise
            if not isinstance(gold_mapped_list, list):
                gold_mapped_list = []
            
            # Debug: Check what we're passing to calculate_metrics for mapping
            logger.info(f"DEBUG: Mapping - mapped_elements_list type: {type(mapped_elements_list)}, value: {mapped_elements_list}")
            logger.info(f"DEBUG: Mapping - gold_mapped_list type: {type(gold_mapped_list)}, value: {gold_mapped_list}")
            
            mapping_metrics = self.calculate_metrics(
                mapped_elements_list,
                gold_mapped_list,
                entity_type="mapping"
            )
            
            return {
                "case_name": case_name,
                "extraction_metrics": extraction_metrics,
                "mapping_metrics": mapping_metrics,
                "extraction_result": extraction_result,
                "mapping_result": mapping_result,
                "success": True
            }
            
        except Exception as e:
            # Debug: Check what type of exception we're getting
            logger.error(f"Exception type: {type(e)}, exception args: {e.args}")
            error_msg = f"Error testing {case_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "case_name": case_name,
                "error": error_msg,
                "success": False
            }


class PromptManager:
    """Prompt library management functionality"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / 'prompts' / 'prompts_config.json'
        self.prompt_loader = PromptLoader(str(self.config_path))
    
    def get_prompt_requirements(self) -> Dict[str, List[str]]:
        """Get prompt template requirements by prompt type"""
        return {
            "NLP_EXTRACTION": ["{clinical_text}"],
            "MCODE_MAPPING": ["{entities_json}", "{trial_context}"]
        }
    
    def list_prompts(self, prompt_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """List available prompts with optional filtering"""
        prompts = []
        all_prompts = self.prompt_loader.list_available_prompts()
        
        for prompt_name, prompt_config in all_prompts.items():
            # Filter by type if specified
            if prompt_type and prompt_config.get('prompt_type') != prompt_type:
                continue
            
            # Filter by status if specified
            if status and prompt_config.get('status') != status:
                continue
            
            prompts.append({
                'name': prompt_name,
                'type': prompt_config.get('prompt_type'),
                'status': prompt_config.get('status'),
                'description': prompt_config.get('description'),
                'version': prompt_config.get('version'),
                'prompt_file': prompt_config.get('prompt_file')
            })
        
        return prompts
    
    def get_prompt_content(self, prompt_name: str) -> str:
        """Get the content of a specific prompt"""
        try:
            return self.prompt_loader.get_prompt(prompt_name)
        except Exception as e:
            raise ValueError(f"Failed to load prompt '{prompt_name}': {e}")


class ModelManager:
    """Model library management functionality"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / 'models' / 'models_config.json'
        self.model_loader = ModelLoader(str(self.config_path))
    
    def list_models(self, model_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
        """List available models with optional filtering"""
        models = []
        all_models = self.model_loader.list_available_models()
        
        for model_name, model_config in all_models.items():
            # Filter by type if specified
            if model_type and model_config.get('model_type') != model_type:
                continue
            
            # Filter by status if specified
            if status and model_config.get('status') != status:
                continue
            
            models.append({
                'name': model_name,
                'type': model_config.get('model_type'),
                'status': model_config.get('status'),
                'description': model_config.get('description'),
                'version': model_config.get('version'),
                'model_identifier': model_config.get('model_identifier')
            })
        
        return models
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get the configuration of a specific model"""
        try:
            model_config = self.model_loader.get_model(model_name)
            return model_config.to_dict()
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {e}")


# CLI Command Groups
@click.group()
def cli():
    """Mcode Optimize - Unified Command Line Interface for Prompt and Model Optimization"""
    pass


@cli.command()
@click.option('--test-cases', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to clinical trial test cases JSON file')
@click.option('--gold-standard', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to gold standard validation JSON file')
@click.option('--prompt-config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration JSON file (default: prompts/prompts_config.json)')
@click.option('--model-config', type=click.Path(exists=True, dir_okay=False),
              default='models/models_config.json',
              help='Path to model configuration JSON file (default: models/models_config.json)')
@click.option('--output', required=True, type=click.Path(),
              help='Output directory for results')
@click.option('--metric', type=click.Choice(['f1_score', 'precision', 'recall', 'compliance_score']),
              default='f1_score', help='Metric to optimize for')
@click.option('--top-n', type=int, default=5, help='Number of top combinations to consider')
@click.option('--prompt-filter', multiple=True, help='Filter prompts by name (can be used multiple times)')
@click.option('--prompt-type-filter', multiple=True, help='Filter prompts by type (can be used multiple times)')
@click.option('--model-filter', multiple=True, help='Filter models by name (can be used multiple times)')
@click.option('--model-type-filter', multiple=True, help='Filter models by type (can be used multiple times)')
def run(test_cases, gold_standard, prompt_config, model_config, output, metric, top_n,
        prompt_filter, prompt_type_filter, model_filter, model_type_filter):
    """Run full optimization across all prompts and models"""
    logger.info("🚀 Starting Full Optimization")
    logger.info("=" * 60)
    
    try:
        # Validate and load files
        test_cases_path = Path(test_cases)
        gold_standard_path = Path(gold_standard)
        prompt_config_path = Path(prompt_config)
        model_config_path = Path(model_config)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        test_cases_data = StrictValidator.load_json_file(
            test_cases_path, "Test cases file"
        )
        gold_standard_data = StrictValidator.load_json_file(
            gold_standard_path, "Gold standard file"
        )
        
        test_cases_dict = test_cases_data['test_cases']
        gold_standard_dict = gold_standard_data['gold_standard']
        
        # The gold standard file has structure: {"gold_standard": {"case_name": {"expected_extraction": {...}, "expected_mcode_mappings": {...}}}}
        # gold_standard_dict already contains the inner case data, so we can use it directly
        extracted_gold_standard = gold_standard_dict
        
        # Create optimization framework
        framework = PromptOptimizationFramework(results_dir=str(output_path))
        
        # Load prompt configuration
        prompt_manager = PromptManager(prompt_config_path)
        all_prompts = prompt_manager.list_prompts()
        
        # Apply prompt filters if specified
        if prompt_filter or prompt_type_filter:
            filtered_prompts = []
            for prompt_info in all_prompts:
                # Filter by prompt name
                if prompt_filter and prompt_info['name'] not in prompt_filter:
                    # Check if any of the filter names match (support for partial matches)
                    if not any(pf in prompt_info['name'] for pf in prompt_filter):
                        continue
                
                # Filter by prompt type
                if prompt_type_filter and prompt_info['type'] not in prompt_type_filter:
                    continue
                
                filtered_prompts.append(prompt_info)
            all_prompts = filtered_prompts
            logger.info(f"Filtered to {len(all_prompts)} prompts based on provided filters")
        
        # Add prompt variants from the standard prompt library using PromptLoader
        for prompt_info in all_prompts:
            # Convert string prompt type to PromptType enum
            prompt_type_str = prompt_info['type']
            if prompt_type_str == 'NLP_EXTRACTION':
                prompt_type = PromptType.NLP_EXTRACTION
            elif prompt_type_str == 'MCODE_MAPPING':
                prompt_type = PromptType.MCODE_MAPPING
            else:
                # Default to NLP_EXTRACTION for unknown types
                prompt_type = PromptType.NLP_EXTRACTION
                logger.warning(f"Unknown prompt type '{prompt_type_str}' for prompt '{prompt_info['name']}', defaulting to NLP_EXTRACTION")
            
            variant = PromptVariant(
                name=prompt_info['name'],
                prompt_type=prompt_type,
                prompt_key=prompt_info['name'],  # Use prompt name as key in prompt library
                description=prompt_info['description'],
                version=prompt_info['version'],
                tags=prompt_info.get('tags', []),
                parameters={}
            )
            framework.add_prompt_variant(variant)
        
        # Add test cases and merge with gold standard data if available
        for case_id, case_data in test_cases_dict.items():
            # Merge gold standard data if available for this case
            if case_id in extracted_gold_standard:
                gold_case_data = extracted_gold_standard[case_id]
                # Properly merge gold standard data into test case structure
                # The optimization framework expects expected_extraction and expected_mcode_mappings
                # to be at the top level of the test case data
                case_data = {
                    **case_data,  # Keep original test case data
                    **gold_case_data  # Add gold standard data (expected_extraction, expected_mcode_mappings)
                }
            framework.add_test_case(case_id, case_data)
        
        # Apply model filters if specified
        if model_filter or model_type_filter:
            # Get all available models
            model_manager = ModelManager(model_config_path)
            all_models = model_manager.list_models()
            
            # Filter models
            filtered_models = []
            for model_info in all_models:
                # Filter by model name
                if model_filter and model_info['name'] not in model_filter:
                    # Check if any of the filter names match (support for partial matches)
                    if not any(mf in model_info['name'] for mf in model_filter):
                        continue
                
                # Filter by model type
                if model_type_filter and model_info['type'] not in model_type_filter:
                    continue
                
                filtered_models.append(model_info)
            
            # Add filtered models to framework
            for model_info in filtered_models:
                try:
                    model_config = model_manager.get_model_config(model_info['name'])
                    api_config = APIConfig(
                        name=f"model_{model_info['name'].replace('-', '_')}",
                        model=model_config['model_identifier'],
                        temperature=model_config.get('temperature', 0.1),
                        max_tokens=model_config.get('max_tokens', 4000),
                        top_p=model_config.get('top_p', 0.9),
                        frequency_penalty=model_config.get('frequency_penalty', 0.0),
                        presence_penalty=model_config.get('presence_penalty', 0.0)
                    )
                    framework.add_api_config(api_config)
                except Exception as e:
                    logger.warning(f"Failed to load model config for {model_info['name']}: {e}")
            
            logger.info(f"Filtered to {len(filtered_models)} models based on provided filters")
        else:
            # If no model filters, the framework will automatically load all models from config
            pass
        
        # Run all test cases
        cases_to_run = list(test_cases_dict.keys())
        
        logger.info(f"Running {len(cases_to_run)} test case(s) with {len(framework.prompt_variants)} prompt variant(s)")
        logger.info("📊 Gold standard validation ENABLED")
        
        # Pipeline callback function
        def pipeline_callback(test_data, prompt_content, prompt_variant_id):
            pipeline = NlpMcodePipeline()
            return pipeline.process_clinical_trial(test_data)
        
        # Run benchmarks for all combinations
        framework.run_all_combinations(
            test_case_ids=cases_to_run,
            pipeline_callback=pipeline_callback,
            expected_entities=None,  # Will be handled by framework
            expected_mappings=None     # Will be handled by framework
        )
        
        # Get best combinations
        best_combinations = framework.get_best_combinations(metric=metric, top_n=top_n)
        
        # Save results
        results_file = output_path / "optimization_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(best_combinations.to_dict('records'), f, indent=2, ensure_ascii=False)
        
        # Display top results
        logger.info(f"\n🏆 Top {top_n} Combinations:")
        logger.info("=" * 60)
        
        for idx, (_, row) in enumerate(best_combinations.iterrows()):
            rank = idx + 1
            logger.info(f"{rank}. {row['prompt_name']} + {row['model']}")
            logger.info(f"   Score: {row[metric]:.3f}")
            logger.info(f"   F1: {row['f1_score']:.3f} | Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f}")
            logger.info("")
        
        logger.info(f"💾 Optimization results saved to: {results_file}")
        logger.info("✅ Full optimization completed!")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option('--prompt-type', type=click.Choice(['NLP_EXTRACTION', 'MCODE_MAPPING']),
              help='Prompt type to set as default')
@click.option('--prompt-name', help='Specific prompt name to set as default')
@click.option('--model-name', help='Specific model name to set as default')
@click.option('--from-results', is_flag=True, help='Use best results from latest optimization')
@click.option('--metric', type=click.Choice(['f1_score', 'precision', 'recall', 'compliance_score']),
              default='f1_score', help='Metric to use for selection')
@click.option('--results-dir', type=click.Path(exists=True, file_okay=False),
              help='Directory containing benchmark results')
def set_default(prompt_type, prompt_name, model_name, from_results, metric, results_dir):
    """Set the best performing prompt or model as default"""
    logger.info("🔧 Setting Default Prompt or Model")
    logger.info("=" * 60)
    
    try:
        framework = PromptOptimizationFramework()
        
        if prompt_name:
            # Set specific prompt as default
            framework.set_default_prompt(prompt_type, prompt_name)
            logger.info(f"✅ Set '{prompt_name}' as default prompt for type '{prompt_type}'")
        elif model_name:
            # Set specific model as default
            framework.set_default_model(model_name)
            logger.info(f"✅ Set '{model_name}' as default model")
        elif from_results:
            # Set defaults based on best results
            if not results_dir:
                logger.error("❌ --results-dir is required when using --from-results")
                sys.exit(1)
            
            results_dir_path = Path(results_dir)
            
            # Load benchmark results
            framework.load_benchmark_results()
            df = framework.get_results_dataframe()
            
            if df.empty:
                logger.error("❌ No benchmark results found")
                sys.exit(1)
            
            # Get best combination based on metric
            best_row = df.loc[df[metric].idxmax()]
            
            # Set both prompt and model as defaults
            prompt_name = best_row['prompt_name']
            model_name = best_row['model']
            
            framework.set_default_prompt(best_row['prompt_type'].upper(), prompt_name)
            framework.set_default_model(model_name)
            
            logger.info(f"✅ Set '{prompt_name}' as default prompt and '{model_name}' as default model")
            logger.info(f"   Based on {metric}: {best_row[metric]:.3f}")
        else:
            logger.error("❌ Please specify either --prompt-name, --model-name, or --from-results")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Failed to set default: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option('--results-dir', required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory containing benchmark results JSON files')
@click.option('--metric', type=click.Choice(['f1_score', 'precision', 'recall', 'compliance_score']),
              default='f1_score', help='Metric to sort by')
@click.option('--top-n', type=int, default=10, help='Number of top combinations to show')
@click.option('--format', type=click.Choice(['table', 'json', 'csv']), default='table',
              help='Output format')
@click.option('--export', type=click.Path(), help='Export results to file')
def view_results(results_dir, metric, top_n, format, export):
    """View optimization results and best combinations"""
    logger.info("📊 Viewing Optimization Results")
    logger.info("=" * 60)
    
    try:
        results_dir_path = Path(results_dir)
        
        # Create optimization framework and load results
        framework = PromptOptimizationFramework(results_dir=str(results_dir_path))
        framework.load_benchmark_results()
        df = framework.get_results_dataframe()
        
        if df.empty:
            logger.error("❌ No benchmark results found")
            sys.exit(1)
        
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
            variant = framework.prompt_variants.get(row['prompt_variant_id'])
            config = framework.api_configs.get(row['api_config_name'])
            if variant:
                grouped.at[idx, 'prompt_name'] = variant.name
                grouped.at[idx, 'prompt_key'] = variant.prompt_key
                grouped.at[idx, 'prompt_type'] = variant.prompt_type.value
            if config:
                grouped.at[idx, 'model'] = config.model
        
        # Sort by the specified metric and get top N
        sorted_results = grouped.sort_values(metric, ascending=False).head(top_n)
        
        # Display or export results
        if format == 'table':
            logger.info(f"\n🏆 Top {top_n} Combinations by {metric}:")
            logger.info("=" * 80)
            for idx, (_, row) in enumerate(sorted_results.iterrows()):
                rank = idx + 1
                logger.info(f"{rank}. {row['prompt_name']} + {row['model']}")
                logger.info(f"   {metric}: {row[metric]:.3f}")
                logger.info(f"   F1: {row['f1_score']:.3f} | Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f}")
                logger.info(f"   Compliance: {row['compliance_score']:.3f} | Duration: {row['duration_ms']:.0f}ms")
                logger.info("")
        elif format == 'json':
            results_json = sorted_results.to_dict('records')
            if export:
                with open(export, 'w', encoding='utf-8') as f:
                    json.dump(results_json, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Results exported to: {export}")
            else:
                logger.info(json.dumps(results_json, indent=2))
        elif format == 'csv':
            if export:
                sorted_results.to_csv(export, index=False)
                logger.info(f"💾 Results exported to: {export}")
            else:
                logger.info(sorted_results.to_csv(index=False))
        
        if export and format != 'json' and format != 'csv':
            # Also save as JSON by default
            results_json = sorted_results.to_dict('records')
            default_export = Path(export).with_suffix('.json')
            with open(default_export, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Results also saved to: {default_export}")
        
    except Exception as e:
        logger.error(f"❌ Failed to view results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option('--type', help='Filter by prompt type (extraction/mapping)')
@click.option('--status', help='Filter by status (production/experimental)')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration file')
@click.option('--default-only', is_flag=True, help='Show only default prompts')
def list_prompts(type, status, config, default_only):
    """List available prompts in the library"""
    try:
        config_path = Path(config)
        prompt_manager = PromptManager(config_path)
        prompts = prompt_manager.list_prompts(type, status)
        
        if default_only:
            # Filter to only show default prompts
            default_prompts = []
            for prompt in prompts:
                # Check if this is marked as default in the config
                prompt_config = prompt_manager.prompt_loader.get_prompt_metadata(prompt['name'])
                if prompt_config and prompt_config.get('default', False):
                    default_prompts.append(prompt)
            prompts = default_prompts
        
        if not prompts:
            logger.info("No prompts found matching the criteria")
            return
        
        # Group by type and status
        grouped = {}
        for prompt in prompts:
            key = f"{prompt['type']}_{prompt['status']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(prompt)
        
        # Get prompt requirements
        prompt_manager = PromptManager(config_path)
        requirements = prompt_manager.get_prompt_requirements()
        
        logger.info("📚 Available Prompts")
        logger.info("=" * 60)
        
        # Show prompt template requirements
        logger.info("\n📋 Prompt Template Requirements:")
        for prompt_type, placeholders in requirements.items():
            logger.info(f"  {prompt_type}: {', '.join(placeholders)}")
        
        for key, prompt_list in grouped.items():
            type_name, status_name = key.split('_', 1)
            logger.info(f"\n{type_name.upper()} - {status_name.title()}:")
            for prompt in prompt_list:
                default_marker = " [DEFAULT]" if prompt.get('default', False) else ""
                logger.info(f"  {prompt['name']} (v{prompt['version']}){default_marker} - {prompt['description']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to list prompts: {e}")
        sys.exit(1)


@cli.command()
@click.argument('prompt_name')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration file')
@click.option('--format', is_flag=True, help='Format the prompt with example placeholders')
@click.option('--requirements', is_flag=True, help='Show prompt template requirements')
def show_prompt(prompt_name, config, format, requirements):
    """Show the content of a specific prompt"""
    try:
        config_path = Path(config)
        prompt_manager = PromptManager(config_path)
        
        # Show prompt requirements if requested
        if requirements:
            reqs = prompt_manager.get_prompt_requirements()
            logger.info("📋 Prompt Template Requirements:")
            logger.info("=" * 40)
            for prompt_type, placeholders in reqs.items():
                logger.info(f"  {prompt_type}: {', '.join(placeholders)}")
            logger.info("")
        
        if format:
            # Format with example placeholders
            example_clinical_text = """
            The patient is a 65-year-old male with a history of hypertension,
            type 2 diabetes, and stage III colon cancer. Current medications
            include lisinopril 10mg daily, metformin 500mg twice daily, and
            undergoing FOLFOX chemotherapy regimen.
            """
            
            example_json_schema = {
                "entities": [
                    {
                        "type": "string",
                        "value": "string",
                        "context": "string",
                        "confidence": "float"
                    }
                ]
            }
            
            content = prompt_manager.get_prompt_content(
                prompt_name,
                text=example_clinical_text,
                json_schema=json.dumps(example_json_schema, indent=2)
            )
            logger.info(f"📝 Formatted Prompt: {prompt_name}")
        else:
            content = prompt_manager.get_prompt_content(prompt_name)
            logger.info(f"📝 Raw Prompt: {prompt_name}")
        
        logger.info("=" * 60)
        logger.info(f"\n{content}")
        
    except Exception as e:
        logger.error(f"❌ Failed to show prompt: {e}")
        sys.exit(1)


@cli.command()
@click.option('--type', help='Filter by model type')
@click.option('--status', help='Filter by status (production/experimental)')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='models/models_config.json',
              help='Path to model configuration file')
@click.option('--default-only', is_flag=True, help='Show only default models')
def list_models(type, status, config, default_only):
    """List available models in the library"""
    try:
        config_path = Path(config)
        model_manager = ModelManager(config_path)
        models = model_manager.list_models(type, status)
        
        if default_only:
            # Filter to only show default models
            default_models = []
            for model in models:
                # Check if this is marked as default in the config
                try:
                    model_config = model_manager.model_loader.get_model_metadata(model['name'])
                    if model_config and model_config.get('default', False):
                        default_models.append(model)
                except Exception:
                    pass
            models = default_models
        
        if not models:
            logger.info("No models found matching the criteria")
            return
        
        # Group by type and status
        grouped = {}
        for model in models:
            key = f"{model['type']}_{model['status']}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(model)
        
        logger.info("🤖 Available Models")
        logger.info("=" * 60)
        
        for key, model_list in grouped.items():
            type_name, status_name = key.split('_', 1)
            logger.info(f"\n{type_name.upper()} - {status_name.title()}:")
            for model in model_list:
                default_marker = " [DEFAULT]" if model.get('default', False) else ""
                logger.info(f"  {model['name']} (v{model['version']}){default_marker} - {model['description']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to list models: {e}")
        sys.exit(1)


@cli.command()
@click.argument('model_name')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='models/models_config.json',
              help='Path to model configuration file')
def show_model(model_name, config):
    """Show the configuration of a specific model"""
    try:
        config_path = Path(config)
        model_manager = ModelManager(config_path)
        model_config = model_manager.get_model_config(model_name)
        
        logger.info(f"🤖 Model Configuration: {model_name}")
        logger.info("=" * 60)
        
        for key, value in model_config.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"❌ Failed to show model: {e}")
        sys.exit(1)


# Main execution
if __name__ == "__main__":
    cli()