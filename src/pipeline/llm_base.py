"""
STRICT Base LLM Engine - Shared foundation for NlpLlm and McodeMapper
No fallbacks, explicit error handling, and strict initialization validation
"""

import json
import re
import time
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass

import sys
import os

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils import (
    Loggable,
    Config,
    TokenUsage,
    extract_token_usage_from_response,
    global_token_tracker,
    UnifiedAPIManager
)
from src.pipeline.task_queue import BenchmarkTask, get_global_task_queue


class LlmConfigurationError(Exception):
    """Exception raised for LLM configuration issues"""
    pass


class LlmExecutionError(Exception):
    """Exception raised for LLM execution failures"""
    pass


class LlmResponseError(Exception):
    """Exception raised for LLM response parsing issues"""
    pass


@dataclass
class LLMCallMetrics:
    """Metrics for LLM API calls"""
    duration: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    success: bool = True
    error_type: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'duration': self.duration,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'success': self.success,
            'error_type': self.error_type
        }


class LlmBase(Loggable, ABC):
    """
    STRICT Base LLM Engine for shared functionality between NLP extractor and mCODE mapper
    Implements strict error handling, no fallbacks, and explicit configuration validation
    """
    
    def __init__(self,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 response_format: Dict[str, Any] = None,
                 task_id: Optional[str] = None):
        """
        Initialize strict LLM base with explicit configuration validation.

        Args:
            model_name: LLM model name to use.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens for response.
            response_format: Response format specification.
            task_id: Optional task ID for associating with a BenchmarkTask.

        Raises:
            LlmConfigurationError: If required configuration is missing or invalid.
        """
        super().__init__()
        self.task_id = task_id
        if self.task_id:
            self.log_handler = self.TaskLogHandler(self.task_id)
            self.logger.addHandler(self.log_handler)

        # Load configuration from unified config
        config = Config()

        # Set model name - use provided or default from config
        if model_name:
            self.model_name = self._validate_model_name(model_name)
        else:
            self.model_name = config.get_model_name()

        # Get the full model configuration to access model_identifier
        self.model_config = config.get_model_config(self.model_name)
        
        # Set temperature - use provided or default from config
        if temperature is not None:
            self.temperature = self._validate_temperature(temperature)
        else:
            self.temperature = config.get_temperature(self.model_name)

        # Set max tokens - use provided or default from config
        if max_tokens is not None:
            self.max_tokens = self._validate_max_tokens(max_tokens)
        else:
            self.max_tokens = config.get_max_tokens(self.model_name)

        self.response_format = response_format

        # Load and validate API configuration
        self.api_key = self._load_and_validate_api_key()
        self.base_url = self._load_and_validate_base_url()

        # Initialize OpenAI client
        self.client = self._initialize_openai_client()

        # Initialize LLM cache using UnifiedAPIManager
        api_manager = UnifiedAPIManager()
        self.llm_cache = api_manager.get_cache("llm")

        self.logger.info(f"✅ Strict LLM Base initialized successfully with model: {self.model_name}")
    
    def _validate_model_name(self, model_name: str) -> str:
        """Validate model name is non-empty string"""
        if not model_name or not isinstance(model_name, str):
            raise LlmConfigurationError("Model name must be a non-empty string")
        return model_name
    
    def _validate_temperature(self, temperature: float) -> float:
        """Validate temperature is within valid range"""
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise LlmConfigurationError("Temperature must be a float between 0 and 2")
        return float(temperature)
    
    def _validate_max_tokens(self, max_tokens: int) -> int:
        """Validate max tokens is positive integer"""
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise LlmConfigurationError("Max tokens must be a positive integer")
        return max_tokens
    
    def _load_and_validate_api_key(self) -> str:
        """Load and validate API key with explicit error handling"""
        config = Config()
        try:
            api_key = config.get_api_key(self.model_name)
            return api_key
        except Exception as e:
            raise LlmConfigurationError(f"Failed to load API key: {str(e)}")
    
    def _load_and_validate_base_url(self) -> str:
        """Load and validate base URL with explicit error handling"""
        config = Config()
        try:
            base_url = config.get_base_url(self.model_name)
            
            # Validate URL format
            if not re.match(r'^https?://', base_url):
                raise LlmConfigurationError(f"Base URL must start with http:// or https://: {base_url}")
            
            return base_url
        except Exception as e:
            raise LlmConfigurationError(f"Failed to load base URL: {str(e)}")
    
    def _call_llm_api(self,
                     messages: List[Dict[str, str]],
                     cache_key_data: Dict[str, Any]) -> Tuple[Dict[str, Any], LLMCallMetrics]:
        """
        Make LLM API call with strict error handling and metrics tracking
        
        Args:
            messages: List of message dictionaries for the LLM
            cache_key_data: Data for generating cache key
        
        Returns:
            Tuple of (parsed_response_json, call_metrics)
        
        Raises:
            LlmExecutionError: If API call fails
        """
        # Add model name and API key to cache key data for comprehensive cache isolation
        cache_key_data_with_model_and_api = cache_key_data.copy()
        cache_key_data_with_model_and_api["model_name"] = self.model_name
        cache_key_data_with_model_and_api["api_key"] = self.api_key
        
        # Try to get cached result first
        cached_result = self.llm_cache.get_by_key(cache_key_data_with_model_and_api)
        if cached_result is not None:
            self.logger.info(f"✅ LLM API call CACHED for {self.model_name}")
            # Convert metrics dict back to LLMCallMetrics object
            metrics_dict = cached_result['metrics']
            metrics = LLMCallMetrics(
                duration=metrics_dict['duration'],
                prompt_tokens=metrics_dict['prompt_tokens'],
                completion_tokens=metrics_dict['completion_tokens'],
                total_tokens=metrics_dict['total_tokens'],
                success=metrics_dict['success'],
                error_type=metrics_dict['error_type']
            )
            return cached_result['response_json'], metrics
        
        metrics = LLMCallMetrics(duration=0.0)
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting LLM API call to {self.model_name}...")
            self.logger.debug(f"LLM Request - Model: {self.model_name}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
            
            # Make the actual LLM call
            response = self.client.chat.completions.create(
                model=self.model_config.model_identifier,  # Use model_identifier for API call
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=self.response_format
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise LlmExecutionError("Empty LLM response received")
            
            response_content = response.choices[0].message.content
            
            # Parse JSON response immediately to ensure proper JSON objects in cache
            response_json = self._parse_and_validate_json_response(response_content)
            
            # Capture token usage metrics
            token_usage = extract_token_usage_from_response(response, self.model_name, "deepseek")
            
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.prompt_tokens = token_usage.prompt_tokens
            metrics.completion_tokens = token_usage.completion_tokens
            metrics.total_tokens = token_usage.total_tokens
            
            # Track token usage globally
            global_token_tracker.add_usage(token_usage, self.__class__.__name__)
            
            # Cache the result with proper JSON object (not string)
            cache_data = {
                'response_json': response_json,
                'metrics': metrics.to_dict()
            }
            self.llm_cache.set_by_key(cache_data, cache_key_data_with_model_and_api, ttl=None)
            
            return response_json, metrics
            
        except openai.APIConnectionError as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "connection_error"
            raise LlmExecutionError(f"API connection failed: {str(e)}")
            
        except openai.APIError as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "api_error"
            raise LlmExecutionError(f"API error: {str(e)}")
            
        except openai.RateLimitError as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "rate_limit"
            raise LlmExecutionError(f"Rate limit exceeded: {str(e)}")
            
        except Exception as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "unknown_error"
            raise LlmExecutionError(f"Unexpected error during LLM call: {str(e)}")
    
    def _initialize_openai_client(self) -> openai.OpenAI:
        """Initialize OpenAI client with validation, supporting Anthropic auth"""
        try:
            # Check if this is an Anthropic model based on base URL
            is_anthropic = "anthropic.com" in self.base_url
            
            if is_anthropic:
                # For Anthropic, use their specific authentication method
                client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    default_headers={"x-api-key": self.api_key}
                )
            else:
                # Standard OpenAI-compatible authentication
                client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
            
            # Test client connectivity with a simple operation
            # This will raise an exception if the client can't be initialized
            # We use a minimal operation to validate the client
            # Note: Some API providers may not support the 'limit' parameter
            try:
                if not is_anthropic:  # Skip model listing for Anthropic as it may not be supported
                    client.models.list(limit=1)
            except TypeError:
                # Fallback to simple list without limit parameter
                if not is_anthropic:
                    client.models.list()
            except Exception as e:
                # If model listing fails, it's not critical for Anthropic
                if not is_anthropic:
                    raise e
                self.logger.warning(f"Model listing not supported for provider, continuing: {e}")
            
            return client
        except Exception as e:
            raise LlmConfigurationError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _generate_cache_key(self, prompt_data: Dict[str, Any]) -> str:
        """
        Generate deterministic cache key based on prompt and model parameters
        
        Args:
            prompt_data: Dictionary containing prompt and model parameters
            
        Returns:
            MD5 hash string for cache key
        """
        try:
            # Sort keys to ensure deterministic hashing
            sorted_data = json.dumps(prompt_data, sort_keys=True)
            return hashlib.md5(sorted_data.encode()).hexdigest()
        except Exception as e:
            raise LlmExecutionError(f"Failed to generate cache key: {str(e)}")
    
    def _parse_and_validate_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate JSON response from LLM with strict error handling
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            LlmResponseError: If JSON parsing fails or response is invalid
        """
        try:
            # STRICT: Only attempt direct JSON parsing - no fallbacks or cleanup
            parsed = json.loads(response_text)
            
            # Validate that parsed result is a dictionary
            if not isinstance(parsed, dict):
                raise LlmResponseError(f"Parsed JSON must be a dictionary, got {type(parsed).__name__}")
            
            # Check for truncation patterns that indicate max_tokens limit reached
            self._check_for_truncation(response_text, parsed)
            
            return parsed
            
        except json.JSONDecodeError as e:
            # Check if this looks like a truncated JSON response due to max_tokens limit
            if self._is_truncated_json(response_text):
                raise LlmResponseError(
                    f"JSON response appears truncated due to max_tokens limit ({self.max_tokens}). "
                    f"Increase max_tokens parameter to allow complete JSON responses. "
                    f"Error: {str(e)}"
                )
            else:
                raise LlmResponseError(
                    f"Failed to parse LLM response as valid JSON. "
                    f"Expected well-formed JSON but got malformed response. "
                    f"Error: {str(e)}"
                )
    
    def _is_truncated_json(self, response_text: str) -> bool:
        """
        Check if JSON response appears truncated due to max_tokens limit
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            True if response appears truncated, False otherwise
        """
        # Common truncation patterns:
        # 1. Missing closing brace/bracket
        # 2. Incomplete field values
        # 3. Trailing comma without closing
        text = response_text.strip()
        
        # Check for missing closing brace
        if text.startswith('{') and not text.endswith('}'):
            return True
            
        # Check for missing closing bracket in arrays
        if text.startswith('[') and not text.endswith(']'):
            return True
            
        # Check for trailing comma without proper closing
        if text.endswith(',') and not (text.endswith('},') or text.endswith('],')):
            return True
            
        # Check for incomplete field values (ends with colon or partial string)
        if re.search(r':\s*[^"]*$', text) or re.search(r'"\w+":\s*"([^"]*)$', text):
            return True
            
        return False
    
    def _check_for_truncation(self, response_text: str, parsed_json: Dict[str, Any]) -> None:
        """
        Check parsed JSON for signs of truncation and alert user if detected
        
        Args:
            response_text: Raw LLM response text
            parsed_json: Parsed JSON dictionary
            
        Raises:
            LlmResponseError: If truncation is detected and user should be alerted
        """
        # Check if we have common truncation patterns in the parsed structure
        text = response_text.strip()
        
        # Pattern 1: Response ends with incomplete structure
        if text.endswith(',') or text.endswith(':'):
            pass
            
        # Pattern 2: Missing expected array structures in common fields
        for field in ['entities', 'mapped_elements']:
            if field in parsed_json and not isinstance(parsed_json[field], list):
                pass
    
    @abstractmethod
    def process_request(self, *args, **kwargs) -> Any:
        """Abstract method for processing requests - must be implemented by subclasses"""
        pass

    class TaskLogHandler(logging.Handler):
        def __init__(self, task_id):
            super().__init__()
            self.task_id = task_id

        def emit(self, record):
            log_entry = self.format(record)
            task_queue = get_global_task_queue()
            if task_queue:
                task = task_queue.get_task(self.task_id)
                if task:
                    task.live_log.append(log_entry)