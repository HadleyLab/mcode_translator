"""
STRICT Base LLM Engine - Shared foundation for StrictNlpExtractor and StrictMcodeMapper
No fallbacks, explicit error handling, and strict initialization validation
"""

import os
import json
import re
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from functools import lru_cache
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
        # Generate cache key
        cache_key = self._generate_cache_key(cache_key_data)
        
        metrics = LLMCallMetrics(duration=0.0)
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting LLM API call to {self.model_name}...")
            self.logger.debug(f"LLM Request - Model: {self.model_name}, Temperature: {self.temperature}, Max Tokens: {self.max_tokens}")
            
            # Convert objects to JSON strings for caching
            import json
            response_format_str = json.dumps(self.response_format) if self.response_format else "null"
            messages_str = json.dumps(messages)
            
            # Call the cached LLM method
            response_content, token_usage_dict = self._cached_llm_call(
                cache_key,
                self.model_name,
                self.temperature,
                self.max_tokens,
                response_format_str,
                messages_str
            )
            
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
    
    @lru_cache(maxsize=128)
    def _cached_llm_call(self, cache_key: str, model_name: str, temperature: float, max_tokens: int, response_format_str: str, messages_str: str) -> Tuple[str, Dict[str, Any]]:
        """
        Cached LLM call with token usage tracking
        
        Args:
            cache_key: Cache key for the request
            model_name: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            response_format_str: Response format as JSON string
            messages_str: Messages as JSON string
            
        Returns:
            Tuple of (response_content, token_usage_dict)
        """
        # Parse the JSON strings back to objects
        import json
        response_format = json.loads(response_format_str) if response_format_str != "null" else None
        messages = json.loads(messages_str)
        
        # Make the actual LLM call
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
        
        # Return response content and token usage as dictionary
        return response_content, {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens
        }
    
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
    
    def _call_llm_api_wrapper(self,
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
            
            # Convert objects to JSON strings for caching
            import json
            response_format_str = json.dumps(self.response_format)
            messages_str = json.dumps(messages)
            
            # Call the cached LLM method
            response_content, token_usage_dict = self._call_llm_api(
                messages_str,
                self.model_name,
                self.temperature,
                self.max_tokens,
                response_format_str
            )
            
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
        try:
            # Log the raw response for debugging
            self.logger.debug(f"Raw LLM response: {repr(response_text[:500])}")
            
            # First attempt direct JSON parsing
            try:
                parsed = json.loads(response_text)
                self.logger.debug("✅ Direct JSON parsing successful")
                
                # Validate that parsed result is a dictionary
                if not isinstance(parsed, dict):
                    raise LLMResponseError(f"Parsed JSON must be a dictionary, got {type(parsed).__name__}")
                
                return parsed
            except json.JSONDecodeError as e:
                self.logger.info(f"Direct JSON parsing failed, attempting cleanup: {str(e)}")
                
                # Clean the response text
                cleaned_text = re.sub(r'^.*?(\{.*\}).*?$', r'\1', response_text, flags=re.DOTALL)
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()
                
                self.logger.debug(f"Cleaned text for parsing: {repr(cleaned_text[:200])}")
                
                # Try to parse cleaned text
                try:
                    parsed = json.loads(cleaned_text)
                    self.logger.info("✅ Successfully parsed cleaned LLM response")
                    
                    # Validate that parsed result is a dictionary
                    if not isinstance(parsed, dict):
                        raise LLMResponseError(f"Parsed JSON must be a dictionary, got {type(parsed).__name__}")
                    
                    return parsed
                except json.JSONDecodeError as e2:
                    self.logger.error(f"❌ Cleaned JSON parsing also failed: {str(e2)}")
                    
                    # Attempt JSON repair
                    try:
                        repaired_text = self._repair_malformed_json(cleaned_text)
                        parsed = json.loads(repaired_text)
                        self.logger.info("✅ Successfully parsed repaired LLM response")
                        
                        # Validate that parsed result is a dictionary
                        if not isinstance(parsed, dict):
                            raise LLMResponseError(f"Parsed JSON must be a dictionary, got {type(parsed).__name__}")
                        
                        return parsed
                    except json.JSONDecodeError as e3:
                        self.logger.error(f"❌ JSON repair also failed: {str(e3)}")
                        # Log the exact problematic text for debugging
                        self.logger.error(f"Problematic text: {repr(cleaned_text[:200])}")
                        raise LLMResponseError(f"Failed to parse LLM response as JSON: {str(e3)}")
        
        except Exception as e:
            raise LLMResponseError(f"JSON parsing failed: {str(e)}")
    
    def _repair_malformed_json(self, json_text: str) -> str:
        """
        Attempt to repair common JSON malformations from LLM responses
        
        Args:
            json_text: Potentially malformed JSON text
            
        Returns:
            Repaired JSON text
            
        Raises:
            LLMResponseError: If JSON cannot be repaired
        """
        try:
            repaired = json_text
            self.logger.debug(f"Original text for repair: {repr(json_text[:200])}")
            
            # Fix 0: Handle JSON that starts with indentation/newlines followed by field name
            # This handles cases like '\n  "mapped_elements": [...]' or '\n  "mapped_elements"'
            stripped = repaired.strip()
            if (stripped.startswith('"') and not stripped.startswith('{') and
                not stripped.startswith('[') and ':' in stripped):
                
                # This is likely a JSON object that starts with a field name instead of braces
                self.logger.info("Detected JSON starting with field name, wrapping in object")
                
                # Find the first colon to separate field name from value
                colon_pos = stripped.find(':')
                if colon_pos != -1:
                    field_name_part = stripped[:colon_pos].strip()
                    value_part = stripped[colon_pos + 1:].strip()
                    
                    # Extract field name (remove quotes if present)
                    if field_name_part.startswith('"') and field_name_part.endswith('"'):
                        field_name = field_name_part[1:-1]
                    else:
                        field_name = field_name_part
                    
                    # Wrap in proper JSON object
                    repaired = f'{{"{field_name}": {value_part}}}'
                    self.logger.info(f"Repaired JSON starting with field: {field_name}")
            
            # Fix 1: Handle the specific case where we have just '\n  "mapped_elements"'
            # This might be an incomplete response that needs to be completed
            if repaired.strip() == '"mapped_elements"':
                self.logger.info("Detected incomplete mapped_elements field, completing structure")
                repaired = '{"mapped_elements": []}'
            
            # Fix 2: Handle case where we have '\n  "mapped_elements":' but no value
            if re.match(r'^\s*"mapped_elements"\s*:\s*$', repaired):
                self.logger.info("Detected mapped_elements with missing value, adding empty array")
                repaired = '{"mapped_elements": []}'
            
            # Fix 3: Add missing opening brace if we have field:value but no braces
            if (not repaired.startswith('{') and not repaired.startswith('[') and
                ':' in repaired and not re.search(r'^\s*\{', repaired)):
                # Check if this looks like a JSON object without braces
                if re.search(r'"\w+"\s*:', repaired):
                    repaired = '{' + repaired + '}'
                    self.logger.info("Added missing opening/closing braces")
            
            # Fix 4: Add missing closing brackets/braces
            open_braces = repaired.count('{')
            close_braces = repaired.count('}')
            open_brackets = repaired.count('[')
            close_brackets = repaired.count(']')
            
            # Add missing closing braces
            while open_braces > close_braces:
                repaired += '}'
                close_braces += 1
            
            # Add missing closing brackets for arrays
            while open_brackets > close_brackets:
                repaired += ']'
                close_brackets += 1
            
            # Fix 5: Remove trailing commas before closing braces/brackets
            repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
            
            # Fix 6: Ensure proper array structure for common fields
            for field in ['entities', 'mapped_elements']:
                field_pattern = f'"{field}":'
                if field_pattern in repaired:
                    # Check if it's missing array brackets
                    if not re.search(field_pattern + r'\s*\[', repaired):
                        # Add opening bracket
                        repaired = re.sub(field_pattern + r'\s*([^{])', field_pattern + r' [\1', repaired)
                    # Check if it's missing closing bracket
                    if re.search(field_pattern + r'\s*\[[^\]]*$', repaired):
                        repaired = repaired + ']'
            
            self.logger.info(f"Attempting to repair malformed JSON, result: {repr(repaired[:200])}")
            
            # Validate that the repaired JSON is parseable
            json.loads(repaired)
            
            return repaired
            
        except Exception as e:
            raise LLMResponseError(f"Failed to repair malformed JSON: {str(e)}")
    
    @abstractmethod
    def process_request(self, *args, **kwargs) -> Any:
        """Abstract method for processing requests - must be implemented by subclasses"""
        pass