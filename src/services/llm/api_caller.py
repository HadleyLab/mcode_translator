"""
LLM API Caller Component - Handles LLM API calls with rate limiting and retries.

This module provides specialized functionality for making LLM API calls
with comprehensive error handling, rate limiting, and retry logic.
"""

import time
from typing import Any

from src.shared.models import ParsedLLMResponse
from src.utils.config import Config
from src.utils.logging_config import get_logger
from src.utils.token_tracker import extract_token_usage_from_response, global_token_tracker


class LLMAPICaller:
    """
    Specialized component for making LLM API calls with rate limiting and retries.

    Handles async API calls, rate limiting, exponential backoff, and token tracking.
    """

    def __init__(self, config: Config, model_name: str):
        """
        Initialize the API caller.

        Args:
            config: Configuration instance
            model_name: Name of the LLM model
        """
        self.config = config
        self.model_name = model_name
        self.logger = get_logger(__name__)

    async def call_llm_api_async(self, prompt: str, llm_config: Any) -> ParsedLLMResponse:
        """
        Make optimized async LLM API call with connection reuse and performance monitoring.

        Args:
            prompt: Formatted prompt
            llm_config: LLM configuration from existing loader

        Returns:
            ParsedLLMResponse with validated and cleaned response data
        """
        import openai

        # Check API key availability before making any calls
        api_key = self.config.get_api_key(self.model_name)
        if not api_key:
            raise ValueError(f"API key not configured for model {self.model_name}")

        # Use existing config for API details
        temperature = self.config.get_temperature(self.model_name)
        max_tokens = self.config.get_max_tokens(self.model_name)

        # Create fresh async client for each request (simple pattern)
        base_url = self.config.get_base_url(self.model_name)

        client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=self.config.get_timeout(self.model_name),
        )

        # Make async API call
        response = await self._make_async_api_call(
            client, llm_config, prompt, temperature, max_tokens
        )

        # Extract token usage using existing utility
        token_usage = extract_token_usage_from_response(response, self.model_name, "provider")

        # Track tokens using existing global tracker
        global_token_tracker.add_usage(token_usage, "llm_service")

        # Parse response
        response_content = response.choices[0].message.content
        if not response_content:
            raise ValueError("Empty LLM response")

        self.logger.debug(f"Raw LLM response: {response_content}")

        # Parse and clean the response content
        return self._parse_and_clean_response(response_content)

    def _parse_and_clean_response(self, response_content: str) -> ParsedLLMResponse:
        """
        Parse and clean LLM response content with comprehensive validation.

        Args:
            response_content: Raw response content from LLM

        Returns:
            ParsedLLMResponse with validated and cleaned response data
        """
        import json

        # Create ParsedLLMResponse to track validation process
        parsed_response = ParsedLLMResponse(
            raw_content=response_content,
            parsed_json=None,
            is_valid_json=False,
            validation_errors=[]
        )

        # Handle markdown-wrapped JSON responses (```json ... ```)
        cleaned_content = response_content.strip()

        # Handle deepseek-specific response formats
        if self.model_name in ["deepseek-coder", "deepseek-reasoner"]:
            # DeepSeek models might return JSON with extra formatting or prefixes
            if cleaned_content.startswith("```json"):
                # Extract JSON from markdown code block
                json_start = cleaned_content.find("{")
                json_end = cleaned_content.rfind("}")
                if json_start != -1 and json_end != -1:
                    cleaned_content = cleaned_content[json_start : json_end + 1]
                else:
                    error_msg = f"DeepSeek response contains malformed markdown JSON block: {cleaned_content[:200]}..."
                    parsed_response.validation_errors.append(error_msg)
                    raise ValueError(error_msg)

            # STRICT: Fail fast on truncated or incomplete JSON - no fallback processing
            if cleaned_content.startswith("{") and not cleaned_content.endswith("}"):
                error_msg = f"DeepSeek response contains truncated JSON (missing closing brace): {cleaned_content[:200]}..."
                parsed_response.validation_errors.append(error_msg)
                raise ValueError(error_msg)

            if cleaned_content.startswith("[") and not cleaned_content.endswith("]"):
                error_msg = f"DeepSeek response contains truncated JSON (missing closing bracket): {cleaned_content[:200]}..."
                parsed_response.validation_errors.append(error_msg)
                raise ValueError(error_msg)

            # Check for obvious JSON structure issues
            if cleaned_content.count("{") != cleaned_content.count("}"):
                opening_braces = cleaned_content.count("{")
                closing_braces = cleaned_content.count("}")
                if opening_braces > closing_braces:
                    # Add missing closing braces
                    missing_braces = opening_braces - closing_braces
                    cleaned_content += "}" * missing_braces
                    self.logger.warning(
                        f"DeepSeek response had {missing_braces} missing closing braces, added them"
                    )
                elif closing_braces > opening_braces:
                    # Remove extra closing braces
                    extra_braces = closing_braces - opening_braces
                    cleaned_content = cleaned_content.rstrip("}")[:-extra_braces] + cleaned_content.rstrip("}")[extra_braces:]
                    self.logger.warning(
                        f"DeepSeek response had {extra_braces} extra closing braces, removed them"
                    )

            if cleaned_content.count("[") != cleaned_content.count("]"):
                opening_brackets = cleaned_content.count("[")
                closing_brackets = cleaned_content.count("]")
                if opening_brackets > closing_brackets:
                    # Add missing closing brackets
                    missing_brackets = opening_brackets - closing_brackets
                    cleaned_content += "]" * missing_brackets
                    self.logger.warning(
                        f"DeepSeek response had {missing_brackets} missing closing brackets, added them"
                    )
                elif closing_brackets > opening_brackets:
                    # Remove extra closing brackets
                    extra_brackets = closing_brackets - opening_brackets
                    cleaned_content = cleaned_content.rstrip("]")[:-extra_brackets] + cleaned_content.rstrip("]")[extra_brackets:]
                    self.logger.warning(
                        f"DeepSeek response had {extra_brackets} extra closing brackets, removed them"
                    )

            # Remove trailing commas only if they appear to be formatting errors
            # But fail if the JSON structure looks fundamentally broken
            if ",}" in cleaned_content or ",]" in cleaned_content:
                self.logger.warning(
                    f"DeepSeek response contains trailing commas, attempting cleanup: {cleaned_content[:100]}..."
                )
                cleaned_content = cleaned_content.replace(",}", "}").replace(",]", "]")

            # Additional cleanup for deepseek-reasoner which may produce more verbose output
            if self.model_name == "deepseek-reasoner":
                # Remove any reasoning/thinking content that might precede JSON
                json_start = cleaned_content.find("{")
                if json_start > 0:
                    # Look for common reasoning markers
                    reasoning_markers = [
                        "Let me think",
                        "First,",
                        "The task is",
                        "I need to",
                        "Looking at",
                    ]
                    for marker in reasoning_markers:
                        marker_pos = cleaned_content.find(marker)
                        if marker_pos != -1 and marker_pos < json_start:
                            # Remove reasoning content before JSON
                            cleaned_content = cleaned_content[json_start:]
                            self.logger.info(
                                "Removed reasoning content from deepseek-reasoner response"
                            )
                            break

        # Handle general markdown-wrapped JSON responses
        elif cleaned_content.startswith("```json") and cleaned_content.endswith("```"):
            # Extract JSON from markdown code block
            json_start = cleaned_content.find("{")
            json_end = cleaned_content.rfind("}")
            if json_start != -1 and json_end != -1:
                cleaned_content = cleaned_content[json_start : json_end + 1]

        # Clean up common JSON formatting issues
        cleaned_content = cleaned_content.strip()
        if cleaned_content.startswith("```") and cleaned_content.endswith("```"):
            # Remove markdown code blocks if still present
            lines = cleaned_content.split("\n")
            if len(lines) > 2:
                cleaned_content = "\n".join(lines[1:-1])

            # Store cleaned content
            parsed_response.cleaned_content = cleaned_content

            # Parse JSON and validate
            parsed_json = json.loads(cleaned_content)
            parsed_response.parsed_json = parsed_json
            parsed_response.is_valid_json = True
            self.logger.debug(f"✅ Successfully parsed JSON response from {self.model_name}")
            return parsed_response

    async def _make_async_api_call(
        self,
        client: Any,
        llm_config: Any,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """
        Make async API call - fail fast on any error.

        Args:
            client: AsyncOpenAI client instance
            llm_config: LLM configuration
            prompt: Formatted prompt
            temperature: Temperature parameter
            max_tokens: Max tokens parameter

        Returns:
            OpenAI API response
        """
        start_time = time.time()

        # Make API call - conditionally use response_format for supported models
        call_params = {
            "model": llm_config.model_identifier,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Use response_format for models that support it
        if (
            "gpt-4o" in llm_config.model_identifier.lower()
            or "gpt-4-turbo" in llm_config.model_identifier.lower()
            or "deepseek" in llm_config.model_identifier.lower()
        ):
            call_params["response_format"] = {"type": "json_object"}
            self.logger.debug(f"Using response_format for model: {llm_config.model_identifier}")

        # Make single API call - fail fast on any error
        response = await client.chat.completions.create(**call_params)
        api_time = time.time() - start_time
        self.logger.info(
            f"✅ API RESPONSE: {llm_config.model_identifier} ({api_time:.2f}s)"
        )
        return response
