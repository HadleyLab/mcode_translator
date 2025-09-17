"""
Ultra-Lean LLM Service for mCODE

Leverages existing utils infrastructure for maximum performance and minimal redundancy.
"""

import json
from typing import Any, Dict, List, Optional

from src.shared.models import McodeElement
from src.utils.api_manager import APIManager
from src.utils.config import Config
from src.utils.llm_loader import llm_loader
from src.utils.logging_config import get_logger
from src.utils.prompt_loader import prompt_loader
from src.utils.token_tracker import global_token_tracker


class LLMService:
    """
    Ultra-lean LLM service leveraging all existing utils.

    No new models, no redundancy - just coordinates existing excellent infrastructure.
    """

    def __init__(self, config: Config, model_name: str, prompt_name: str):
        """
        Initialize with existing infrastructure.

        Args:
            config: Existing Config instance
            model_name: Model name (uses existing llm_loader)
            prompt_name: Prompt name (uses existing prompt_loader)
        """
        self.config = config
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.logger = get_logger(__name__)

        # Leverage existing utils
        self.llm_loader = llm_loader
        self.prompt_loader = prompt_loader
        self.api_manager = APIManager()
        self.token_tracker = global_token_tracker

    def map_to_mcode(self, clinical_text: str) -> List[McodeElement]:
        """
        Map clinical text to mCODE elements using existing infrastructure.

        Args:
            clinical_text: Clinical trial text to process

        Returns:
            List of McodeElement instances
        """
        try:
            # Check if API key is available (graceful degradation for testing)
            try:
                api_key = self.config.get_api_key(self.model_name)
                if not api_key:
                    self.logger.warning(f"No API key available for {self.model_name} - returning empty results")
                    return []
            except Exception as e:
                self.logger.warning(f"API key check failed for {self.model_name}: {e} - returning empty results")
                return []

            # Get LLM config from existing file-based system
            llm_config = self.llm_loader.get_llm(self.model_name)

            # Get prompt from existing file-based system
            prompt = self.prompt_loader.get_prompt(
                self.prompt_name,
                clinical_text=clinical_text
            )

            # Use existing API manager for caching
            cache_key = {
                "model": self.model_name,
                "prompt": self.prompt_name,
                "text_hash": hash(clinical_text)
            }

            llm_cache = self.api_manager.get_cache("llm")
            cached_result = llm_cache.get_by_key(cache_key)

            if cached_result is not None:
                self.logger.debug(f"LLM cache hit for {self.model_name}")
                return cached_result.get("mcode_elements", [])

            # Make LLM call using existing infrastructure
            response_json = self._call_llm_api(prompt, llm_config)

            # Parse response using existing models
            mcode_elements = self._parse_llm_response(response_json)

            # Cache result using existing API manager
            cache_data = {"mcode_elements": [elem.model_dump() for elem in mcode_elements]}
            llm_cache.set_by_key(cache_data, cache_key)

            return mcode_elements

        except Exception as e:
            self.logger.error(f"LLM mapping failed: {str(e)}")
            return []

    def _call_llm_api(self, prompt: str, llm_config) -> Dict[str, Any]:
        """
        Make LLM API call using existing infrastructure.

        Args:
            prompt: Formatted prompt
            llm_config: LLM configuration from existing loader

        Returns:
            Parsed JSON response
        """
        import openai

        try:
            # Use existing config for API details
            api_key = self.config.get_api_key(self.model_name)
            base_url = self.config.get_base_url(self.model_name)
            temperature = self.config.get_temperature(self.model_name)
            max_tokens = self.config.get_max_tokens(self.model_name)

            # Initialize client (reuse existing pattern)
            client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key
            )

            # Make API call - conditionally use response_format for supported models
            call_params = {
                "model": llm_config.model_identifier,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Use response_format for models that support it
            if ("gpt-4o" in llm_config.model_identifier.lower() or
                "gpt-4-turbo" in llm_config.model_identifier.lower() or
                "deepseek" in llm_config.model_identifier.lower()):
                call_params["response_format"] = {"type": "json_object"}
                self.logger.debug(f"Using response_format for model: {llm_config.model_identifier}")

            self.logger.debug(f"Making LLM API call to {llm_config.model_identifier}")
            response = client.chat.completions.create(**call_params)

            # Extract token usage using existing utility
            from src.utils.token_tracker import extract_token_usage_from_response
            token_usage = extract_token_usage_from_response(
                response, self.model_name, "provider"
            )

            # Track tokens using existing global tracker
            self.token_tracker.add_usage(token_usage, "llm_service")

            # Parse response
            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Empty LLM response")

            self.logger.debug(f"Raw LLM response: {response_content}")

            try:
                # Handle markdown-wrapped JSON responses (```json ... ```)
                cleaned_content = response_content.strip()

                # Handle deepseek-specific response formats
                if self.model_name == "deepseek-coder":
                    # DeepSeek might return JSON with extra formatting or prefixes
                    if cleaned_content.startswith('```json'):
                        # Extract JSON from markdown code block
                        json_start = cleaned_content.find('{')
                        json_end = cleaned_content.rfind('}')
                        if json_start != -1 and json_end != -1:
                            cleaned_content = cleaned_content[json_start:json_end+1]
                        else:
                            raise ValueError(f"DeepSeek response contains malformed markdown JSON block: {cleaned_content[:200]}...")

                    # STRICT: Fail fast on truncated or incomplete JSON
                    if cleaned_content.startswith('{') and not cleaned_content.endswith('}'):
                        raise ValueError(f"DeepSeek response contains truncated JSON (missing closing brace): {cleaned_content[:200]}...")

                    if cleaned_content.startswith('[') and not cleaned_content.endswith(']'):
                        raise ValueError(f"DeepSeek response contains truncated JSON (missing closing bracket): {cleaned_content[:200]}...")

                    # Check for obvious JSON structure issues
                    if cleaned_content.count('{') != cleaned_content.count('}'):
                        raise ValueError(f"DeepSeek response has mismatched braces: {cleaned_content.count('{')} opening vs {cleaned_content.count('}')} closing")

                    if cleaned_content.count('[') != cleaned_content.count(']'):
                        raise ValueError(f"DeepSeek response has mismatched brackets: {cleaned_content.count('[')} opening vs {cleaned_content.count(']')} closing")

                    # Remove trailing commas only if they appear to be formatting errors
                    # But fail if the JSON structure looks fundamentally broken
                    if ',}' in cleaned_content or ',]' in cleaned_content:
                        self.logger.warning(f"DeepSeek response contains trailing commas, attempting cleanup: {cleaned_content[:100]}...")
                        cleaned_content = cleaned_content.replace(',}', '}').replace(',]', ']')

                # Handle general markdown-wrapped JSON responses
                elif cleaned_content.startswith('```json') and cleaned_content.endswith('```'):
                    # Extract JSON from markdown code block
                    json_start = cleaned_content.find('{')
                    json_end = cleaned_content.rfind('}')
                    if json_start != -1 and json_end != -1:
                        cleaned_content = cleaned_content[json_start:json_end+1]

                # Clean up common JSON formatting issues
                cleaned_content = cleaned_content.strip()
                if cleaned_content.startswith('```') and cleaned_content.endswith('```'):
                    # Remove markdown code blocks if still present
                    lines = cleaned_content.split('\n')
                    if len(lines) > 2:
                        cleaned_content = '\n'.join(lines[1:-1])

                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode failed for model {self.model_name}, response: {response_content[:500]}...")
                # Try to provide more specific error information
                if "Expecting ',' delimiter" in str(e):
                    self.logger.error("JSON parsing error: Missing comma or malformed structure")
                elif "Expecting ':' delimiter" in str(e):
                    self.logger.error("JSON parsing error: Missing colon in key-value pair")
                elif "Expecting value" in str(e):
                    self.logger.error("JSON parsing error: Unexpected token or missing value")
                raise ValueError(f"Invalid JSON response from {self.model_name}: {str(e)}") from e

        except Exception as e:
            self.logger.error(f"LLM API call failed: {str(e)}")
            raise

    def _parse_llm_response(self, response_json: Dict[str, Any]) -> List[McodeElement]:
        """
        Parse LLM response into McodeElement instances.

        Args:
            response_json: Raw LLM response

        Returns:
            List of validated McodeElement instances
        """
        elements = []

        # Handle different response formats from LLM
        mcode_data = response_json.get("mcode_mappings", response_json.get("mappings", []))

        for item in mcode_data:
            try:
                # Use existing McodeElement model
                element = McodeElement(**item)
                elements.append(element)
            except Exception as e:
                self.logger.warning(f"Failed to parse mCODE element: {e}")
                continue

        return elements