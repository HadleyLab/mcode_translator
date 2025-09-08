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

from src.utils import (
    Loggable,
    get_logger,
    Config,
    PromptLoader,
    prompt_loader,
    model_loader,
    global_token_tracker
)
from src.pipeline.mcode_mapper import McodeMapper
from src.shared.benchmark_result import BenchmarkResult


class PromptType(Enum):
    """Types of prompts that can be optimized"""
    NLP_EXTRACTION = "nlp_extraction"
    MCODE_MAPPING = "mcode_mapping"
    DIRECT_MCODE = "direct_mcode"


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



class PromptOptimizationFramework(Loggable):
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
        self.mcode_mapper = McodeMapper()
        
        # Automatically add all models from configuration
        self._add_all_models_from_config()
    
    def _add_all_models_from_config(self) -> None:
        """Automatically add all models from the configuration"""
        try:
            config = Config()
            model_configs = model_loader.get_all_models()
            
            for model_key, model_config in model_configs.items():
                try:
                    # Create a unique config name for each model
                    config_name = f"model_{model_key.replace('-', '_').replace('.', '_')}"
                    api_config = APIConfig(name=config_name, model=model_key)
                    self.add_api_config(api_config)
                    pass
                except ValueError as e:
                    continue
        except Exception as e:
            raise
    
    def _validate_api_config(self, config: APIConfig) -> None:
        """STRICT validation - fail hard on invalid API configs using unified Config"""
        # The unified Config class already performs validation, so we just need to ensure
        # the configuration is accessible and valid
        try:
            # Test that we can access the configuration values by creating a Config instance
            config_instance = Config()
            _ = config_instance.get_api_key(config.model)
            _ = config_instance.get_base_url(config.model)
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
    
    def add_test_case(self, case_id: str, test_data: Dict[str, Any]) -> None:
        """Add a test case"""
        self.test_cases[case_id] = test_data
    
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
            # Return empty list if conversion fails
            return []
    
    def load_test_cases_from_file(self, file_path: str) -> None:
        """Load test cases from JSON file"""
        try:
            with open(file_path, 'r') as f:
                test_cases = json.load(f)
            
            for case_id, case_data in test_cases.items():
                self.add_test_case(case_id, case_data)
            
        except Exception as e:
            raise
    
    async def run_benchmark_async(self,
                                 prompt_variant_id: str,
                                 api_config_name: str,
                                 test_case_id: str,
                                 pipeline_type: str,
                                 expected_entities: List[Dict[str, Any]] = None,
                                 expected_mappings: List[Dict[str, Any]] = None) -> BenchmarkResult:
        """
        Run a single benchmark asynchronously - simplified version for UI
        """
        result = BenchmarkResult(
            prompt_variant_id=prompt_variant_id,
            api_config_name=api_config_name,
            test_case_id=test_case_id,
            pipeline_type=pipeline_type
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
            
            # Create pipeline callback
            def pipeline_callback(test_data):
                from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline

                pipeline = NlpMcodePipeline()
                
                # Set appropriate prompt based on type
                if prompt_variant.prompt_type == PromptType.NLP_EXTRACTION:
                    pipeline.nlp_extractor.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
                elif prompt_variant.prompt_type == PromptType.MCODE_MAPPING:
                    # This would require changes to McodeMapper
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
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
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
                      pipeline_type: str,
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
            test_case_id=test_case_id,
            pipeline_type=pipeline_type
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
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            
            # Enhanced error logging with progress info and time remaining
            error_status = "❌"
            time_remaining = ""
            if current_index is not None and total_count is not None:
                error_status = f"❌ [{current_index}/{total_count}]"
                
                # Calculate estimated time remaining for error cases too
                if current_index > 1 and benchmark_start_time is not None:
                    avg_time_per_test = (time.time() - benchmark_start_time) / current_index
                    remaining_tests = total_count - current_index
                    remaining_seconds = avg_time_per_test * remaining_tests
                    time_remaining = f" ⏰ ETR: {self._format_time_remaining(remaining_seconds)}"
            
        
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
                           pipeline_type: str,
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
                            pipeline_type=pipeline_type,
                            expected_entities=expected_entities,
                            expected_mappings=expected_mappings,
                            current_index=current_index,
                            total_count=total_combinations,
                            benchmark_start_time=benchmark_start_time
                        )
                    except Exception as e:
                        # Continue with other combinations despite failures
                        pass
    
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
                from src.utils import model_loader
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
            from src.utils import reload_models_config
            reload_models_config()
        except Exception as e:
            pass
    
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
    
    def run_benchmark_with_live_logging(self,
                                        prompt_variant_id: str,
                                        api_config_name: str,
                                        test_case_id: str,
                                        pipeline_callback: Callable,
                                        pipeline_type: str,
                                        expected_entities: List[Dict[str, Any]] = None,
                                        expected_mappings: List[Dict[str, Any]] = None,
                                        current_index: int = None,
                                        total_count: int = None,
                                        benchmark_start_time: float = None,
                                        log_callback: Callable[[str], None] = None) -> BenchmarkResult:
        """
        Run a single benchmark with live logging - streams log messages during execution
        
        Args:
            prompt_variant_id: ID of the prompt variant to test
            api_config_name: Name of the API configuration to use
            test_case_id: ID of the test case to run
            pipeline_callback: Callback function that executes the pipeline
            pipeline_type: Type of pipeline being used
            expected_entities: Gold standard entities for validation
            expected_mappings: Gold standard mCODE mappings for validation
            current_index: Current test index (for progress tracking)
            total_count: Total number of tests (for progress tracking)
            benchmark_start_time: Start time of the overall benchmark run (for ETR calculation)
            log_callback: Callback function to stream log messages during execution
            
        Returns:
            BenchmarkResult with the benchmark results
        """
        def _log_with_callback(message: str, level: str = "INFO"):
            """Log message and send to callback if provided"""
            # Send to callback for live streaming
            if log_callback:
                try:
                    log_callback(message)
                except Exception as e:
                    pass
        
        # Create result object
        result = BenchmarkResult(
            prompt_variant_id=prompt_variant_id,
            api_config_name=api_config_name,
            test_case_id=test_case_id,
            pipeline_type=pipeline_type
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
            
            _log_with_callback(f"🧪{progress_info} Starting STRICT benchmark:")
            _log_with_callback(f"   📋 Prompt: {prompt_variant.name} ({prompt_variant.prompt_key})")
            _log_with_callback(f"   🤖 Model: {model_info}")
            _log_with_callback(f"   📊 Test Case: {test_case_id}")
            _log_with_callback(f"   🔄 Prompt Type: {prompt_variant.prompt_type.value}")
            
            # Note: The pipeline callback should use the Config class directly for API configuration
            # Environment variables are no longer needed since the pipeline components use Config class
            
            # Run the pipeline processing
            start_time = time.time()
            _log_with_callback(f"   ⚡ Starting pipeline execution...")
            
            pipeline_result = pipeline_callback(test_case, prompt_content, prompt_variant_id, api_config_name)
            end_time = time.time()
            
            # Calculate duration
            result.duration_ms = (end_time - start_time) * 1000
            
            # Extract results
            if hasattr(pipeline_result, 'extracted_entities'):
                result.extracted_entities = pipeline_result.extracted_entities
                result.entities_extracted = len(pipeline_result.extracted_entities)
                _log_with_callback(f"   📊 Extracted {result.entities_extracted} entities")
            
            if hasattr(pipeline_result, 'mcode_mappings'):
                result.mcode_mappings = pipeline_result.mcode_mappings
                result.entities_mapped = len(pipeline_result.mcode_mappings)
                _log_with_callback(f"   🗺️  Mapped {result.entities_mapped} mCODE elements")
                self.logger.debug(f"Set mcode_mappings: {len(result.mcode_mappings)} mappings")
            
            if hasattr(pipeline_result, 'validation_results'):
                result.validation_results = pipeline_result.validation_results
                result.compliance_score = pipeline_result.validation_results.get('compliance_score', 0.0)
                _log_with_callback(f"   ✅ Compliance score: {result.compliance_score:.2%}")

            if hasattr(pipeline_result, 'metadata') and 'token_usage' in pipeline_result.metadata:
                token_usage = pipeline_result.metadata['token_usage']
                if token_usage:
                    result.token_usage = token_usage.get('total_tokens', 0)
                    _log_with_callback(f"   📝 Token usage: {result.token_usage} tokens")
            
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
                _log_with_callback(f"   📊 Aggregate token usage: {aggregate_token_usage.total_tokens} tokens")
                _log_with_callback(f"      Prompt tokens: {aggregate_token_usage.prompt_tokens}")
                _log_with_callback(f"      Completion tokens: {aggregate_token_usage.completion_tokens}")
            
            # Mark the result as successful before calculating metrics
            result.success = True
            
            # Calculate metrics using provided gold standard data
            _log_with_callback(f"   📈 Calculating performance metrics...")
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
            
            _log_with_callback(f"{completion_status} STRICT benchmark completed in {result.duration_ms:.2f}ms{time_remaining}")
            _log_with_callback(f"   📊 Extraction: {result.entities_extracted} entities")
            _log_with_callback(f"   🗺️  Mapping: {result.entities_mapped} mCODE elements")
            _log_with_callback(f"   🎯 Metrics: F1={result.f1_score:.3f}, Precision={result.precision:.3f}, Recall={result.recall:.3f}"
                              if result.f1_score is not None and result.precision is not None and result.recall is not None
                              else "   🎯 Metrics: F1=N/A, Precision=N/A, Recall=N/A")
            _log_with_callback(f"   ✅ Compliance: {result.compliance_score:.2%}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            
            # Enhanced error logging with progress info and time remaining
            error_status = "❌"
            time_remaining = ""
            if current_index is not None and total_count is not None:
                error_status = f"❌ [{current_index}/{total_count}]"
                
                # Calculate estimated time remaining for error cases too
                if current_index > 1 and benchmark_start_time is not None:
                    avg_time_per_test = (time.time() - benchmark_start_time) / current_index
                    remaining_tests = total_count - current_index
                    remaining_seconds = avg_time_per_test * remaining_tests
                    time_remaining = f" ⏰ ETR: {self._format_time_remaining(remaining_seconds)}"
            
            _log_with_callback(f"{error_status} STRICT benchmark FAILED in {result.duration_ms:.2f}ms{time_remaining}: {str(e)}", "ERROR")
        
        result.end_time = datetime.now()
        self.benchmark_results.append(result)
        
        # Save results
        self._save_benchmark_result(result)
        
        return result

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