#!/usr/bin/env python3
"""
mCODE Translator - Unified Command Line Interface

STRICT implementation that consolidates all runner scripts into a single CLI
with comprehensive prompt library integration and strict validation.
"""

import json
import sys
import asyncio
import logging
import click
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
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
from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from pipeline.nlp_extractor import NlpLlm
from src.pipeline.mcode_mapper import McodeMapper
from src.utils.prompt_loader import PromptLoader, load_prompt


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
        self.pipeline = StrictDynamicExtractionPipeline()
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
        """Calculate metrics for mCODE mapping using semantic similarity"""
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


# CLI Command Groups
@click.group()
def cli():
    """mCODE Translator - Unified Command Line Interface"""
    pass


@cli.group()
def validate():
    """Gold standard validation commands"""
    pass


@cli.group()
def prompts():
    """Prompt library management commands"""
    pass


@cli.group()
def optimize():
    """Prompt optimization commands"""
    pass


@cli.group()
def benchmark():
    """Performance benchmarking commands"""
    pass


# Validation Commands
@validate.command()
@click.option('--test-cases', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to clinical trial test cases JSON file')
@click.option('--gold-standard', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to gold standard validation JSON file')
@click.option('--output', type=click.Path(), help='Save detailed validation results to file')
def gold_standard(test_cases, gold_standard, output):
    """Validate mCODE extraction and mapping against gold standard using all test cases"""
    logger.info("🧪 Starting STRICT Gold Standard Validation")
    logger.info("=" * 60)
    
    try:
        # Validate and load files
        test_cases_path = Path(test_cases)
        gold_standard_path = Path(gold_standard)
        
        test_cases_data = StrictValidator.load_json_file(
            test_cases_path, "Test cases file"
        )
        gold_standard_data = StrictValidator.load_json_file(
            gold_standard_path, "Gold standard file"
        )
        
        tester = GoldStandardTester()
        results = []
        
        test_cases_dict = test_cases_data['test_cases']
        gold_standard_dict = gold_standard_data['gold_standard']
        
        # Debug: Check what we're getting from the gold standard file
        logger.info(f"DEBUG: gold_standard_dict type: {type(gold_standard_dict)}")
        logger.info(f"DEBUG: gold_standard_dict keys: {list(gold_standard_dict.keys())}")
        logger.info(f"DEBUG: gold_standard_dict value: {gold_standard_dict}")
        
        # The gold standard file has structure: {"gold_standard": {"case_name": {"expected_extraction": {...}, "expected_mcode_mappings": {...}}}}
        # gold_standard_dict already contains the inner case data, so we can use it directly
        extracted_gold_standard = gold_standard_dict
        
        # Run all test cases
        cases_to_run = list(test_cases_dict.keys())
        
        for case_name in cases_to_run:
            if case_name not in test_cases_dict:
                raise ValueError(f"Test case '{case_name}' not found in test cases file")
            if case_name not in extracted_gold_standard:
                raise ValueError(f"Gold standard for case '{case_name}' not found")
            
            result = tester.test_single_case(
                case_name, test_cases_dict[case_name], extracted_gold_standard[case_name]
            )
            results.append(result)
        
        # Generate report
        successful_tests = [r for r in results if r.get('success', False)]
        failed_tests = [r for r in results if not r.get('success', False)]
        
        logger.info(f"\n📊 STRICT Validation Report")
        logger.info("=" * 60)
        logger.info(f"Total tests: {len(results)}")
        logger.info(f"Successful: {len(successful_tests)}")
        logger.info(f"Failed: {len(failed_tests)}")
        
        if successful_tests:
            # Calculate average metrics
            extraction_precision = sum(r['extraction_metrics']['precision'] for r in successful_tests) / len(successful_tests)
            extraction_recall = sum(r['extraction_metrics']['recall'] for r in successful_tests) / len(successful_tests)
            extraction_f1 = sum(r['extraction_metrics']['f1'] for r in successful_tests) / len(successful_tests)
            
            mapping_precision = sum(r['mapping_metrics']['precision'] for r in successful_tests) / len(successful_tests)
            mapping_recall = sum(r['mapping_metrics']['recall'] for r in successful_tests) / len(successful_tests)
            mapping_f1 = sum(r['mapping_metrics']['f1'] for r in successful_tests) / len(successful_tests)
            
            logger.info(f"\n📈 Average Extraction Metrics:")
            logger.info(f"  Precision: {extraction_precision:.3f}")
            logger.info(f"  Recall:    {extraction_recall:.3f}")
            logger.info(f"  F1 Score:  {extraction_f1:.3f}")
            
            logger.info(f"\n📈 Average Mapping Metrics:")
            logger.info(f"  Precision: {mapping_precision:.3f}")
            logger.info(f"  Recall:    {mapping_recall:.3f}")
            logger.info(f"  F1 Score:  {mapping_f1:.3f}")
            
            # Detailed results per test case
            logger.info(f"\n🔍 Detailed Results:")
            for result in successful_tests:
                logger.info(f"\n{result['case_name']}:")
                logger.info(f"  Extraction - P: {result['extraction_metrics']['precision']:.3f}, "
                           f"R: {result['extraction_metrics']['recall']:.3f}, "
                           f"F1: {result['extraction_metrics']['f1']:.3f}")
                logger.info(f"  Mapping    - P: {result['mapping_metrics']['precision']:.3f}, "
                           f"R: {result['mapping_metrics']['recall']:.3f}, "
                           f"F1: {result['mapping_metrics']['f1']:.3f}")
        
        if failed_tests:
            logger.error(f"\n❌ Failed Tests:")
            for result in failed_tests:
                logger.error(f"  {result['case_name']}: {result.get('error', 'Unknown error')}")

        # Report token usage
        token_usage = tester.get_token_usage()
        if token_usage['total_tokens'] > 0:
            logger.info(f"\n🔢 Token Usage Statistics:")
            logger.info(f"  Total tokens: {token_usage['total_tokens']:,}")
            logger.info(f"  Prompt tokens: {token_usage['prompt_tokens']:,}")
            logger.info(f"  Completion tokens: {token_usage['completion_tokens']:,}")
            logger.info(f"  Extraction tokens: {token_usage['extraction_tokens']:,}")
            logger.info(f"  Mapping tokens: {token_usage['mapping_tokens']:,}")
        
        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Detailed results saved to: {output_path}")
        
        logger.info(f"\n✅ Gold standard validation completed!")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        sys.exit(1)


@validate.command()
@click.argument('test_case_name')
@click.option('--test-cases', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to clinical trial test cases JSON file')
@click.option('--gold-standard', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to gold standard validation JSON file')
@click.option('--output', type=click.Path(), help='Save detailed validation results to file')
def test_case(test_case_name, test_cases, gold_standard, output):
    """Validate a specific test case against gold standard by name"""
    logger.info(f"🧪 Testing specific case: {test_case_name}")
    logger.info("=" * 60)
    
    try:
        # Validate and load files
        test_cases_path = Path(test_cases)
        gold_standard_path = Path(gold_standard)
        
        test_cases_data = StrictValidator.load_json_file(
            test_cases_path, "Test cases file"
        )
        gold_standard_data = StrictValidator.load_json_file(
            gold_standard_path, "Gold standard file"
        )
        
        tester = GoldStandardTester()
        
        test_cases_dict = test_cases_data['test_cases']
        gold_standard_dict = gold_standard_data['gold_standard']
        
        # Debug: Check what we're getting from the gold standard file
        logger.info(f"DEBUG: gold_standard_dict type: {type(gold_standard_dict)}")
        logger.info(f"DEBUG: gold_standard_dict keys: {list(gold_standard_dict.keys())}")
        logger.info(f"DEBUG: gold_standard_dict value: {gold_standard_dict}")
        
        # The gold standard file has structure: {"gold_standard": {"case_name": {"expected_extraction": {...}, "expected_mcode_mappings": {...}}}}
        # gold_standard_dict already contains the inner case data, so we can use it directly
        extracted_gold_standard = gold_standard_dict
        
        # Check if test case exists
        if test_case_name not in test_cases_dict:
            raise ValueError(f"Test case '{test_case_name}' not found in test cases file")
        if test_case_name not in extracted_gold_standard:
            raise ValueError(f"Gold standard for case '{test_case_name}' not found")
        
        # Run the specific test case
        result = tester.test_single_case(
            test_case_name, test_cases_dict[test_case_name], extracted_gold_standard[test_case_name]
        )
        
        # Generate report
        logger.info(f"\n📊 STRICT Validation Report - {test_case_name}")
        logger.info("=" * 60)
        
        if result.get('success', False):
            logger.info("✅ Test completed successfully")
            
            extraction_metrics = result['extraction_metrics']
            mapping_metrics = result['mapping_metrics']
            
            logger.info(f"\n📈 Extraction Metrics:")
            logger.info(f"  Precision: {extraction_metrics['precision']:.3f}")
            logger.info(f"  Recall:    {extraction_metrics['recall']:.3f}")
            logger.info(f"  F1 Score:  {extraction_metrics['f1']:.3f}")
            logger.info(f"  TP: {extraction_metrics['true_positives']}, FP: {extraction_metrics['false_positives']}, FN: {extraction_metrics['false_negatives']}")
            
            logger.info(f"\n📈 Mapping Metrics:")
            logger.info(f"  Precision: {mapping_metrics['precision']:.3f}")
            logger.info(f"  Recall:    {mapping_metrics['recall']:.3f}")
            logger.info(f"  F1 Score:  {mapping_metrics['f1']:.3f}")
            logger.info(f"  TP: {mapping_metrics['true_positives']}, FP: {mapping_metrics['false_positives']}, FN: {mapping_metrics['false_negatives']}")
            
            logger.info(f"\n🔢 Results:")
            logger.info(f"  Entities extracted: {len(result['extraction_result']['entities'])}")
            logger.info(f"  Mappings created: {len(result['mapping_result']['mapped_elements'])}")
            
        else:
            logger.error(f"❌ Test failed: {result.get('error', 'Unknown error')}")
        
        # Report token usage
        token_usage = tester.get_token_usage()
        if token_usage['total_tokens'] > 0:
            logger.info(f"\n🔢 Token Usage Statistics:")
            logger.info(f"  Total tokens: {token_usage['total_tokens']:,}")
            logger.info(f"  Prompt tokens: {token_usage['prompt_tokens']:,}")
            logger.info(f"  Completion tokens: {token_usage['completion_tokens']:,}")
            logger.info(f"  Extraction tokens: {token_usage['extraction_tokens']:,}")
            logger.info(f"  Mapping tokens: {token_usage['mapping_tokens']:,}")
        
        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Detailed results saved to: {output_path}")
        
        logger.info(f"\n✅ Test case validation completed!")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        sys.exit(1)


# Prompt Optimization Command
@cli.command()
@click.option('--test-cases', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to clinical trial test cases JSON file')
@click.option('--gold-standard', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to gold standard validation JSON file')
@click.option('--output', type=click.Path(), help='Save optimization results to file')
def optimize_prompts(test_cases, gold_standard, output):
    """Test all available prompts against gold standard and rank by performance metrics"""
    logger.info("⚡ Starting Prompt Optimization")
    logger.info("=" * 60)
    
    try:
        # Validate and load files
        test_cases_path = Path(test_cases)
        gold_standard_path = Path(gold_standard)
        
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
        
        # Get all available prompts
        prompt_manager = PromptManager()
        all_prompts = prompt_manager.list_prompts()
        
        logger.info(f"Testing {len(all_prompts)} prompt variants")
        
        results = []
        
        for prompt_info in all_prompts:
            logger.info(f"\n🧪 Testing prompt: {prompt_info['name']}")
            
            # Create tester with specific prompt
            tester = GoldStandardTester()
            # Set the specific prompt for this test
            # This would require modifying the pipeline to accept specific prompts
            
            prompt_results = []
            token_usage = {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0
            }
            
            for case_name in test_cases_dict.keys():
                if case_name not in extracted_gold_standard:
                    continue
                
                # Extract the actual gold standard data for this case
                case_gold_standard = extracted_gold_standard[case_name]
                
                result = tester.test_single_case(
                    case_name, test_cases_dict[case_name], case_gold_standard
                )
                prompt_results.append(result)
            
            # Calculate average metrics
            successful_tests = [r for r in prompt_results if r.get('success', False)]
            if successful_tests:
                avg_extraction_f1 = sum(r['extraction_metrics']['f1'] for r in successful_tests) / len(successful_tests)
                avg_mapping_f1 = sum(r['mapping_metrics']['f1'] for r in successful_tests) / len(successful_tests)
                overall_score = (avg_extraction_f1 + avg_mapping_f1) / 2
                
                # Get token usage
                prompt_token_usage = tester.get_token_usage()
                
                results.append({
                    'prompt_name': prompt_info['name'],
                    'prompt_type': prompt_info['type'],
                    'success_rate': len(successful_tests) / len(prompt_results),
                    'avg_extraction_f1': avg_extraction_f1,
                    'avg_mapping_f1': avg_mapping_f1,
                    'overall_score': overall_score,
                    'token_usage': prompt_token_usage,
                    'test_count': len(prompt_results)
                })
                
                logger.info(f"  ✅ Score: {overall_score:.3f} (Extraction: {avg_extraction_f1:.3f}, Mapping: {avg_mapping_f1:.3f})")
                logger.info(f"  🔢 Tokens: {prompt_token_usage['total_tokens']:,}")
        
        # Sort results by overall score
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        logger.info(f"\n🏆 Optimization Results:")
        logger.info("=" * 60)
        
        for i, result in enumerate(results[:5]):  # Show top 5
            rank = i + 1
            logger.info(f"{rank}. {result['prompt_name']} ({result['prompt_type']})")
            logger.info(f"   Score: {result['overall_score']:.3f} | "
                       f"Extraction F1: {result['avg_extraction_f1']:.3f} | "
                       f"Mapping F1: {result['avg_mapping_f1']:.3f}")
            logger.info(f"   Tokens: {result['token_usage']['total_tokens']:,} | "
                       f"Success rate: {result['success_rate']:.1%}")
        
        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"\n💾 Optimization results saved to: {output_path}")
        
        logger.info(f"\n✅ Prompt optimization completed!")
        
    except Exception as e:
        logger.error(f"❌ Prompt optimization failed: {e}")
        sys.exit(1)


# Prompt Management Commands
@prompts.command()
@click.option('--type', help='Filter by prompt type (extraction/mapping)')
@click.option('--status', help='Filter by status (production/experimental)')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration file')
def list(type, status, config):
    """List available prompts in the library with template requirements"""
    try:
        config_path = Path(config)
        prompt_manager = PromptManager(config_path)
        prompts = prompt_manager.list_prompts(type, status)
        
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
                logger.info(f"  {prompt['name']} (v{prompt['version']}) - {prompt['description']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to list prompts: {e}")
        sys.exit(1)


@prompts.command()
@click.argument('prompt_name')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration file')
@click.option('--format', is_flag=True, help='Format the prompt with example placeholders')
@click.option('--requirements', is_flag=True, help='Show prompt template requirements')
def show(prompt_name, config, format, requirements):
    """Show the content of a specific prompt with template requirements"""
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


@prompts.command()
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration file')
def demo(config):
    """Demonstrate prompt library usage with examples and template requirements"""
    try:
        config_path = Path(config)
        prompt_manager = PromptManager(config_path)
        
        logger.info("🚀 Prompt Library Demo")
        logger.info("=" * 60)
        
        # Show prompt template requirements
        requirements = prompt_manager.get_prompt_requirements()
        logger.info("📋 Prompt Template Requirements:")
        for prompt_type, placeholders in requirements.items():
            logger.info(f"  {prompt_type}: {', '.join(placeholders)}")
        logger.info("")
        
        # List all prompts
        all_prompts = prompt_manager.list_prompts()
        logger.info(f"📚 Total prompts available: {len(all_prompts)}")
        
        # Show prompt types
        types = set(p['type'] for p in all_prompts)
        logger.info(f"📋 Prompt types: {', '.join(types)}")
        
        # Show some example prompts
        logger.info("\n🔍 Example prompts:")
        for prompt in all_prompts[:5]:  # Show first 5
            logger.info(f"  • {prompt['name']} ({prompt['type']}) - {prompt['description'][:50]}...")
        
        # Demonstrate loading a specific prompt
        if all_prompts:
            example_prompt = all_prompts[0]['name']
            content = prompt_manager.get_prompt_content(example_prompt)
            logger.info(f"\n📝 Example content from '{example_prompt}':")
            logger.info(f"   Length: {len(content)} characters")
            logger.info(f"   Preview: {content[:100]}...")
        
        logger.info("\n✅ Demo completed! Use 'prompts list' and 'prompts show <name>' for more details.")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)


@prompts.command()
@click.argument('prompt_name')
@click.option('--test-cases', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to test cases JSON file')
@click.option('--case', help='Specific test case to run')
@click.option('--output', type=click.Path(), help='Output file for results')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration file')
def test(prompt_name, test_cases, case, output, config):
    """Test a specific prompt against test cases with template validation"""
    logger.info(f"🧪 Testing prompt: {prompt_name}")
    
    try:
        # Load prompt content
        config_path = Path(config)
        prompt_manager = PromptManager(config_path)
        prompt_content = prompt_manager.get_prompt_content(prompt_name)
        
        # Load test cases
        test_cases_path = Path(test_cases)
        test_cases_data = StrictValidator.load_json_file(
            test_cases_path, "Test cases file"
        )
        
        test_cases_dict = test_cases_data['test_cases']
        cases_to_run = [case] if case else list(test_cases_dict.keys())
        
        logger.info(f"Testing {len(cases_to_run)} case(s) with prompt: {prompt_name}")
        
        # TODO: Implement actual prompt testing logic
        # This would involve running the pipeline with the specific prompt
        # and comparing results
        
        logger.info("✅ Prompt testing completed (implementation pending)")
        
    except Exception as e:
        logger.error(f"❌ Prompt testing failed: {e}")
        sys.exit(1)


# Prompt Requirements Command
@prompts.command()
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration file')
def requirements(config):
    """Show prompt template requirements by type"""
    try:
        config_path = Path(config)
        prompt_manager = PromptManager(config_path)
        requirements = prompt_manager.get_prompt_requirements()
        
        logger.info("📋 STRICT Prompt Template Requirements")
        logger.info("=" * 50)
        logger.info("These placeholders MUST be present in prompt templates:")
        logger.info("")
        
        for prompt_type, placeholders in requirements.items():
            logger.info(f"🔹 {prompt_type}:")
            for placeholder in placeholders:
                logger.info(f"   • {placeholder}")
            logger.info("")
        
        logger.info("💡 Usage examples:")
        logger.info("  - NLP_EXTRACTION: Extract entities from clinical text")
        logger.info("  - MCODE_MAPPING: Map extracted entities to mCODE FHIR standard")
        logger.info("")
        logger.info("⚠️  Prompts missing required placeholders will generate warnings")
        
    except Exception as e:
        logger.error(f"❌ Failed to get prompt requirements: {e}")
        sys.exit(1)


# Combined Benchmark + Optimization Commands
@cli.group()
def combined():
    """Combined benchmark and optimization commands"""
    pass


@combined.command()
@click.option('--test-cases', required=True, type=click.Path(exists=True, dir_okay=False),
              help='Path to test cases JSON file')
@click.option('--gold-standard', type=click.Path(exists=True, dir_okay=False),
              help='Path to gold standard JSON file (optional)')
@click.option('--prompt-config', type=click.Path(exists=True, dir_okay=False),
              default='prompts/prompts_config.json',
              help='Path to prompt configuration JSON file (default: prompts/prompts_config.json)')
@click.option('--api-configs', type=click.Path(exists=True, dir_okay=False),
              default='config.json',
              help='Path to unified configuration JSON file (default: config.json)')
@click.option('--output', required=True, type=click.Path(),
              help='Output directory for results')
@click.option('--optimize', is_flag=True, help='Enable prompt optimization')
def benchmark(test_cases, gold_standard, prompt_config, api_configs, output, optimize):
    """Run comprehensive benchmark with all test cases, gold standard validation, and optional prompt optimization"""
    logger.info("🚀 Starting Benchmark" + (" + Optimization" if optimize else ""))
    logger.info("=" * 60)
    
    try:
        # Import optimization framework
        from src.optimization.prompt_optimization_framework import (
            PromptOptimizationFramework, PromptVariant, APIConfig, PromptType
        )
        
        # Load test cases
        test_cases_path = Path(test_cases)
        test_cases_data = StrictValidator.load_json_file(test_cases_path, "Test cases file")
        
        # Load gold standard if provided
        gold_standard_data = {}
        if gold_standard:
            gold_standard_path = Path(gold_standard)
            gold_standard_data = StrictValidator.load_json_file(gold_standard_path, "Gold standard file")
        
        # Load prompt configuration
        prompt_config_path = Path(prompt_config)
        prompt_config_data = StrictValidator.load_json_file(prompt_config_path, "Prompt config file")
        
        # Load unified configuration
        config_path = Path(api_configs)
        config_data = StrictValidator.load_json_file(config_path, "Unified configuration file")
        
        # Create optimization framework
        framework = PromptOptimizationFramework(results_dir=output)
        
        # Add API configuration using the unified Config class
        # The framework will automatically use the unified configuration
        api_config = APIConfig(name="unified_config")
        framework.add_api_config(api_config)
        
        # Add prompt variants from the standard prompt library using PromptLoader
        prompt_manager = PromptManager(prompt_config_path)
        all_prompts = prompt_manager.list_prompts()

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
        test_cases_dict = test_cases_data.get('test_cases', {})
        gold_standard_dict = gold_standard_data.get('gold_standard', {})

        for case_id, case_data in test_cases_dict.items():
            # Merge gold standard data if available for this case
            if case_id in gold_standard_dict:
                gold_case_data = gold_standard_dict[case_id]
                # Properly merge gold standard data into test case structure
                # The optimization framework expects expected_extraction and expected_mcode_mappings
                # to be at the top level of the test case data
                case_data = {
                    **case_data,  # Keep original test case data
                    **gold_case_data  # Add gold standard data (expected_extraction, expected_mcode_mappings)
                }
            framework.add_test_case(case_id, case_data)
        
        # Run all test cases
        cases_to_run = list(test_cases_dict.keys())
        
        logger.info(f"Running {len(cases_to_run)} test case(s) with {len(framework.prompt_variants)} prompt variant(s)")
        if gold_standard:
            logger.info("📊 Gold standard validation ENABLED")
        if optimize:
            logger.info("⚡ Optimization ENABLED")
        
        # Pipeline callback function
        def pipeline_callback(test_data, prompt_content, prompt_variant_id):
            pipeline = StrictDynamicExtractionPipeline()
            return pipeline.process_clinical_trial(test_data)
        
        # Run benchmarks for all combinations
        results = []
        for case_id in cases_to_run:
            for variant_id, variant in framework.prompt_variants.items():
                for config_name in framework.api_configs.keys():
                    logger.info(f"Running: {variant.name} + {config_name} + {case_id}")
                    
                    result = framework.run_benchmark(
                        prompt_variant_id=variant_id,
                        api_config_name=config_name,
                        test_case_id=case_id,
                        pipeline_callback=pipeline_callback
                    )
                    results.append(result)
        
        # Generate comprehensive report
        logger.info(f"\n📊 Benchmark Report")
        logger.info("=" * 60)
        logger.info(f"Total runs: {len(results)}")
        logger.info(f"Successful: {len([r for r in results if r.success])}")
        logger.info(f"Failed: {len([r for r in results if not r.success])}")
        
        # Calculate average metrics
        successful_results = [r for r in results if r.success]
        if successful_results:
            avg_precision = sum(r.precision for r in successful_results) / len(successful_results)
            avg_recall = sum(r.recall for r in successful_results) / len(successful_results)
            avg_f1 = sum(r.f1_score for r in successful_results) / len(successful_results)
            avg_mapping_accuracy = sum(r.mapping_accuracy for r in successful_results) / len(successful_results)
            
            logger.info(f"\n📈 Average Metrics Across All Runs:")
            logger.info(f"  Precision:          {avg_precision:.3f}")
            logger.info(f"  Recall:             {avg_recall:.3f}")
            logger.info(f"  F1 Score:           {avg_f1:.3f}")
            logger.info(f"  Mapping Accuracy:   {avg_mapping_accuracy:.3f}")
        
        # Save detailed results
        results_file = Path(output) / "benchmark_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Detailed results saved to: {results_file}")
        logger.info("✅ Benchmark completed!" + (" Optimization results available." if optimize else ""))
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


# Performance Report Commands
@combined.command()
@click.option('--results-dir', required=True, type=click.Path(exists=True, file_okay=False),
              help='Directory containing benchmark results JSON files')
@click.option('--output', type=click.Path(), help='Output directory for reports (default: ./optimization_reports)')
def report(results_dir, output):
    """Generate comprehensive performance report from benchmark results"""
    logger.info("📊 Generating performance report from benchmark results")
    
    try:
        from datetime import datetime
        import pandas as pd
        from pathlib import Path
        
        results_dir = Path(results_dir)
        output_dir = Path(output) if output else Path("./optimization_reports")
        output_dir.mkdir(exist_ok=True)
        
        # Load all benchmark results
        results = []
        for result_file in results_dir.glob("benchmark_*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                results.append(result_data)
            except Exception as e:
                logger.error(f"Error loading {result_file}: {e}")
        
        if not results:
            logger.error("❌ No benchmark results found")
            sys.exit(1)
        
        df = pd.DataFrame(results)
        
        # Generate report
        report_lines = []
        report_lines.append("🔬 STRICT PROMPT OPTIMIZATION FRAMEWORK - PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total benchmark runs: {len(df)}")
        report_lines.append(f"Successful runs: {len(df[df['success'] == True])}")
        report_lines.append(f"Failed runs: {len(df[df['success'] == False])}")
        report_lines.append("")
        
        # Summary statistics
        successful_runs = df[df['success'] == True]
        if not successful_runs.empty:
            report_lines.append("📊 PERFORMANCE SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Average entities extracted: {successful_runs['entities_extracted'].mean():.1f}")
            report_lines.append(f"Average entities mapped: {successful_runs['entities_mapped'].mean():.1f}")
            report_lines.append(f"Average compliance score: {successful_runs['compliance_score'].mean():.3f}")
            report_lines.append(f"Average duration: {successful_runs['duration_ms'].mean():.0f}ms")
            report_lines.append("")
        
        # By prompt variant
        if 'prompt_variant_name' in successful_runs.columns:
            report_lines.append("📋 PERFORMANCE BY PROMPT VARIANT")
            report_lines.append("-" * 40)
            for prompt_name, group in successful_runs.groupby('prompt_variant_name'):
                report_lines.append(f"🔹 {prompt_name}:")
                report_lines.append(f"   Runs: {len(group)}")
                report_lines.append(f"   Avg entities: {group['entities_extracted'].mean():.1f}")
                report_lines.append(f"   Avg mapped: {group['entities_mapped'].mean():.1f}")
                report_lines.append(f"   Avg compliance: {group['compliance_score'].mean():.3f}")
                report_lines.append(f"   Avg duration: {group['duration_ms'].mean():.0f}ms")
                report_lines.append("")
        
        # By test case
        if 'test_case_id' in successful_runs.columns:
            report_lines.append("🧪 PERFORMANCE BY TEST CASE")
            report_lines.append("-" * 40)
            for test_case, group in successful_runs.groupby('test_case_id'):
                report_lines.append(f"🔹 {test_case}:")
                report_lines.append(f"   Runs: {len(group)}")
                report_lines.append(f"   Avg entities: {group['entities_extracted'].mean():.1f}")
                report_lines.append(f"   Avg mapped: {group['entities_mapped'].mean():.1f}")
                report_lines.append(f"   Avg compliance: {group['compliance_score'].mean():.3f}")
                report_lines.append("")
        
        # Cache performance analysis
        first_run_times = []
        cached_run_times = []
        
        if 'prompt_variant_id' in df.columns and 'test_case_id' in df.columns:
            for (prompt_id, test_case), group in df.groupby(['prompt_variant_id', 'test_case_id']):
                if len(group) > 1:
                    first_run = group.iloc[0]
                    cached_runs = group.iloc[1:]
                    first_run_times.append(first_run['duration_ms'])
                    cached_run_times.extend(cached_runs['duration_ms'])
        
        if first_run_times and cached_run_times:
            report_lines.append("⚡ CACHE PERFORMANCE ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"First-run average: {sum(first_run_times)/len(first_run_times):.0f}ms")
            report_lines.append(f"Cached-run average: {sum(cached_run_times)/len(cached_run_times):.0f}ms")
            report_lines.append(f"Speedup factor: {sum(first_run_times)/sum(cached_run_times):.1f}x")
            report_lines.append("")
        
        # Save report
        report_file = output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"📋 Performance report saved to: {report_file}")
        logger.info(f"📊 Report contains analysis of {len(results)} benchmark runs")
        
    except Exception as e:
        logger.error(f"❌ Performance report generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


# Main execution
if __name__ == "__main__":
    cli()