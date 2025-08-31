"""
STRICT Base LLM Engine - Shared foundation for StrictNlpExtractor and StrictMcodeMapper
No fallbacks, explicit error handling, and strict initialization validation
"""

import json
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
import openai
from abc import ABC, abstractmethod
from dataclasses import dataclass

import sys
import os

# Add parent directory to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logging_config import Loggable
from utils.config import Config
from utils.token_tracker import TokenUsage, extract_token_usage_from_response, global_token_tracker
from utils.cache_manager import cache_manager


class LLMConfigurationError(Exception):
    """Exception raised for LLM configuration issues"""
    pass


class LLMExecutionError(Exception):
    """Exception raised for LLM execution failures"""
    pass


class LLMResponseError(Exception):
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


class StrictLLMBase(Loggable, ABC):
    """
    STRICT Base LLM Engine for shared functionality between NLP extractor and MCode mapper
    Implements strict error handling, no fallbacks, and explicit configuration validation
    """
    
    def __init__(self,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 response_format: Dict[str, Any] = None):
        """
        Initialize strict LLM base with explicit configuration validation
        
        Args:
            model_name: LLM model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            response_format: Response format specification
            
        Raises:
            LLMConfigurationError: If required configuration is missing or invalid
        """
        super().__init__()
        
        # Load configuration from unified config
        config = Config()
        
        # Set model name - use provided or default from config
        if model_name:
            self.model_name = self._validate_model_name(model_name)
        else:
            self.model_name = config.get_model_name()
        
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
        
        self.logger.info(f"✅ Strict LLM Base initialized successfully with model: {self.model_name}")
    
    def _validate_model_name(self, model_name: str) -> str:
        """Validate model name is non-empty string"""
        if not model_name or not isinstance(model_name, str):
            raise LLMConfigurationError("Model name must be a non-empty string")
        return model_name
    
    def _validate_temperature(self, temperature: float) -> float:
        """Validate temperature is within valid range"""
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise LLMConfigurationError("Temperature must be a float between 0 and 2")
        return float(temperature)
    
    def _validate_max_tokens(self, max_tokens: int) -> int:
        """Validate max tokens is positive integer"""
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise LLMConfigurationError("Max tokens must be a positive integer")
        return max_tokens
    
    def _load_and_validate_api_key(self) -> str:
        """Load and validate API key with explicit error handling"""
        config = Config()
        try:
            api_key = config.get_api_key(self.model_name)
            return api_key
        except Exception as e:
            raise LLMConfigurationError(f"Failed to load API key: {str(e)}")
    
    def _load_and_validate_base_url(self) -> str:
        """Load and validate base URL with explicit error handling"""
        config = Config()
        try:
            base_url = config.get_base_url(self.model_name)
            
            # Validate URL format
            if not re.match(r'^https?://', base_url):
                raise LLMConfigurationError(f"Base URL must start with http:// or https://: {base_url}")
            
            return base_url
        except Exception as e:
            raise LLMConfigurationError(f"Failed to load base URL: {str(e)}")
    
    def _call_llm_api(self,
                     messages: List[Dict[str, str]],
                     cache_key_data: Dict[str, Any]) -> Tuple[str, LLMCallMetrics]:
        """
        Make LLM API call with strict error handling and metrics tracking
        
        Args:
            messages: List of message dictionaries for the LLM
            cache_key_data: Data for generating cache key
            
        Returns:
            Tuple of (response_content, call_metrics)
            
        Raises:
            LLMExecutionError: If API call fails
        """
        metrics = LLMCallMetrics(duration=0.0)
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting LLM API call to {self.model_name}...")
            self.logger.debug(f"LLM Request - Model: {self.model_name}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
            
            # Create a comprehensive cache data structure that includes all parameters
            # We include instance-specific parameters to ensure cache isolation
            complete_cache_data = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "response_format": self.response_format,
                "messages": messages,
                **cache_key_data  # Include the original cache key data
            }
            
            # Generate a complete cache key that includes all parameters
            complete_cache_key = json.dumps(complete_cache_data, sort_keys=True)
            
            # Call the cached LLM method with just the complete cache key
            response_content, token_usage_dict = self._cached_llm_call(complete_cache_key)
            
            end_time = time.time()
            metrics.duration = end_time - start_time
            
            # Convert token usage dict back to TokenUsage object
            token_usage = TokenUsage(
                prompt_tokens=token_usage_dict["prompt_tokens"],
                completion_tokens=token_usage_dict["completion_tokens"],
                total_tokens=token_usage_dict["total_tokens"],
                model_name=self.model_name,
                provider_name="deepseek"
            )
            
            metrics.prompt_tokens = token_usage.prompt_tokens
            metrics.completion_tokens = token_usage.completion_tokens
            metrics.total_tokens = token_usage.total_tokens
            
            # Track token usage globally
            global_token_tracker.add_usage(token_usage, self.__class__.__name__)
            
            self.logger.info(f"✅ LLM API call completed in {metrics.duration:.2f}s")
            self.logger.info(f"   📊 Token usage - Prompt: {token_usage.prompt_tokens}, Completion: {token_usage.completion_tokens}, Total: {token_usage.total_tokens}")
            self.logger.debug(f"Raw LLM response length: {len(response_content)} characters")
            
            return response_content, metrics
            
        except openai.APIConnectionError as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "connection_error"
            raise LLMExecutionError(f"API connection failed: {str(e)}")
            
        except openai.APIError as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "api_error"
            raise LLMExecutionError(f"API error: {str(e)}")
            
        except openai.RateLimitError as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "rate_limit"
            raise LLMExecutionError(f"Rate limit exceeded: {str(e)}")
            
        except Exception as e:
            end_time = time.time()
            metrics.duration = end_time - start_time
            metrics.success = False
            metrics.error_type = "unknown_error"
            raise LLMExecutionError(f"Unexpected error during LLM call: {str(e)}")
    
    def _cached_llm_call(self, cache_key: str) -> Tuple[str, Dict[str, Any]]:
        """
        Disk-based cached LLM call with token usage tracking
        
        Args:
            cache_key: Cache key for the request (contains hash of all parameters)
        
        Returns:
            Tuple of (response_content, token_usage_dict)
        
        Raises:
            LLMExecutionError: If API call fails
        """
        # Check disk-based cache first
        cached_result = cache_manager.llm_cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"📦 Cache hit for key: {cache_key[:50]}...")
            return cached_result
        
        # The cache_key contains all the necessary information for the LLM call
        # We need to parse it to extract all the parameters
        try:
            import json
            cache_data = json.loads(cache_key)
            
            # Extract ALL parameters from cache data to ensure cache isolation
            messages = cache_data["messages"]
            model_name = cache_data["model"]
            temperature = cache_data["temperature"]
            max_tokens = cache_data["max_tokens"]
            response_format = cache_data["response_format"]
            
            # Make the actual LLM call using the parameters from the cache data
            # This ensures that cached responses match the exact configuration
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise LLMExecutionError("Empty LLM response received")
            
            response_content = response.choices[0].message.content
            
            # Capture token usage metrics
            token_usage = extract_token_usage_from_response(response, model_name, "deepseek")
            
            # Store result in disk-based cache
            result = (response_content, {
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "total_tokens": token_usage.total_tokens
            })
            
            cache_manager.llm_cache.set(cache_key, result)
            self.logger.debug(f"📦 Cache miss - stored new entry for key: {cache_key[:50]}...")
            
            return result
            
        except Exception as e:
            raise LLMExecutionError(f"Failed to process cached LLM call: {str(e)}")
    
    def _initialize_openai_client(self) -> openai.OpenAI:
        """Initialize OpenAI client with validation"""
        try:
            client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
            
            # Test client connectivity with a simple operation
            # This will raise an exception if the client can't be initialized
            # We use a minimal operation to validate the client
            # Note: Some API providers may not support the 'limit' parameter
            try:
                client.models.list(limit=1)
            except TypeError:
                # Fallback to simple list without limit parameter
                client.models.list()
            
            return client
        except Exception as e:
            raise LLMConfigurationError(f"Failed to initialize OpenAI client: {str(e)}")
    
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
            raise LLMExecutionError(f"Failed to generate cache key: {str(e)}")
    
    def _parse_and_validate_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate JSON response from LLM with strict error handling
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            LLMResponseError: If JSON parsing fails or response is invalid
        """
        # Log the raw response for debugging
        self.logger.debug(f"Raw LLM response: {repr(response_text[:500])}")
        
        try:
            # STRICT: Only attempt direct JSON parsing - no fallbacks or cleanup
            parsed = json.loads(response_text)
            
            # Validate that parsed result is a dictionary
            if not isinstance(parsed, dict):
                raise LLMResponseError(f"Parsed JSON must be a dictionary, got {type(parsed).__name__}")
            
            # Check for truncation patterns that indicate max_tokens limit reached
            self._check_for_truncation(response_text, parsed)
            
            self.logger.debug("✅ Direct JSON parsing successful")
            return parsed
            
        except json.JSONDecodeError as e:
            # Check if this looks like a truncated JSON response due to max_tokens limit
            if self._is_truncated_json(response_text):
                raise LLMResponseError(
                    f"JSON response appears truncated due to max_tokens limit ({self.max_tokens}). "
                    f"Increase max_tokens parameter to allow complete JSON responses. "
                    f"Error: {str(e)}"
                )
            else:
                raise LLMResponseError(
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
            LLMResponseError: If truncation is detected and user should be alerted
        """
        # Check if we have common truncation patterns in the parsed structure
        text = response_text.strip()
        
        # Pattern 1: Response ends with incomplete structure
        if text.endswith(',') or text.endswith(':'):
            self.logger.warning(
                f"⚠️  JSON response appears truncated (ends with '{text[-10:]}'). "
                f"Consider increasing max_tokens from {self.max_tokens} for complete responses."
            )
            
        # Pattern 2: Missing expected array structures in common fields
        for field in ['entities', 'mapped_elements']:
            if field in parsed_json and not isinstance(parsed_json[field], list):
                self.logger.warning(
                    f"⚠️  Field '{field}' is not an array as expected. "
                    f"This may indicate truncation. Current max_tokens: {self.max_tokens}"
                )
    
    @abstractmethod
    def process_request(self, *args, **kwargs) -> Any:
        """Abstract method for processing requests - must be implemented by subclasses"""
        pass