"""
LLM API Caller Component - Handles LLM API calls with rate limiting and retries.

This module provides specialized functionality for making LLM API calls
with comprehensive error handling, rate limiting, and retry logic.
"""

import asyncio
import random
import time
from typing import Any, Dict

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

    async def call_llm_api_async(self, prompt: str, llm_config) -> Dict[str, Any]:
        """
        Make optimized async LLM API call with connection reuse and performance monitoring.

        Args:
            prompt: Formatted prompt
            llm_config: LLM configuration from existing loader

        Returns:
            Parsed JSON response
        """
        import openai

        try:
            # Check API key availability before making any calls
            try:
                api_key = self.config.get_api_key(self.model_name)
                if not api_key:
                    self.logger.warning(
                        f"No API key available for {self.model_name} - skipping API call"
                    )
                    raise ValueError(
                        f"API key not configured for model {self.model_name}"
                    )
            except Exception as e:
                if "not found in config" in str(e) or "API key not configured" in str(
                    e
                ):
                    self.logger.warning(
                        f"API key configuration issue for {self.model_name}: {e}"
                    )
                    raise ValueError(
                        f"API key not configured for model {self.model_name}"
                    )
                else:
                    raise

            # Use existing config for API details
            temperature = self.config.get_temperature(self.model_name)
            max_tokens = self.config.get_max_tokens(self.model_name)

            # Create fresh async client for each request (simple pattern)
            api_key = self.config.get_api_key(self.model_name)
            base_url = self.config.get_base_url(self.model_name)

            client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                max_retries=0,
                timeout=self.config.get_timeout(self.model_name),
            )

            # Make async API call with aggressive rate limiting handling
            response = await self._make_async_api_call_with_rate_limiting(
                client, llm_config, prompt, temperature, max_tokens
            )

            # Extract token usage using existing utility
            token_usage = extract_token_usage_from_response(
                response, self.model_name, "provider"
            )

            # Track tokens using existing global tracker
            global_token_tracker.add_usage(token_usage, "llm_service")

            # Parse response
            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Empty LLM response")

            self.logger.debug(f"Raw LLM response: {response_content}")

            # Parse and clean the response content
            return self._parse_and_clean_response(response_content)

        except Exception as e:
            self.logger.error(f"💥 LLM API call failed for {self.model_name}: {str(e)}")
            raise

    def _parse_and_clean_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse and clean LLM response content.

        Args:
            response_content: Raw response content from LLM

        Returns:
            Parsed JSON response
        """
        import json

        try:
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
                        raise ValueError(
                            f"DeepSeek response contains malformed markdown JSON block: {cleaned_content[:200]}..."
                        )

                # STRICT: Fail fast on truncated or incomplete JSON - no fallback processing
                if cleaned_content.startswith("{") and not cleaned_content.endswith(
                    "}"
                ):
                    raise ValueError(
                        f"DeepSeek response contains truncated JSON (missing closing brace): {cleaned_content[:200]}..."
                    )

                if cleaned_content.startswith("[") and not cleaned_content.endswith(
                    "]"
                ):
                    raise ValueError(
                        f"DeepSeek response contains truncated JSON (missing closing bracket): {cleaned_content[:200]}..."
                    )

                # Check for obvious JSON structure issues
                if cleaned_content.count("{") != cleaned_content.count("}"):
                    raise ValueError(
                        f"DeepSeek response has mismatched braces: {cleaned_content.count('{')} opening vs {cleaned_content.count('}')} closing"
                    )

                if cleaned_content.count("[") != cleaned_content.count("]"):
                    raise ValueError(
                        f"DeepSeek response has mismatched brackets: {cleaned_content.count('[')} opening vs {cleaned_content.count(']')} closing"
                    )

                # Remove trailing commas only if they appear to be formatting errors
                # But fail if the JSON structure looks fundamentally broken
                if ",}" in cleaned_content or ",]" in cleaned_content:
                    self.logger.warning(
                        f"DeepSeek response contains trailing commas, attempting cleanup: {cleaned_content[:100]}..."
                    )
                    cleaned_content = cleaned_content.replace(",}", "}").replace(
                        ",]", "]"
                    )

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
            elif cleaned_content.startswith("```json") and cleaned_content.endswith(
                "```"
            ):
                # Extract JSON from markdown code block
                json_start = cleaned_content.find("{")
                json_end = cleaned_content.rfind("}")
                if json_start != -1 and json_end != -1:
                    cleaned_content = cleaned_content[json_start : json_end + 1]

            # Clean up common JSON formatting issues
            cleaned_content = cleaned_content.strip()
            if cleaned_content.startswith("```") and cleaned_content.endswith(
                "```"
            ):
                # Remove markdown code blocks if still present
                lines = cleaned_content.split("\n")
                if len(lines) > 2:
                    cleaned_content = "\n".join(lines[1:-1])

            return json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            # Provide detailed error information for debugging
            self.logger.error(
                f"❌ JSON parsing error for {self.model_name}: {str(e)}"
            )
            self.logger.error(f"  Response preview: {response_content[:500]}...")

            # Check for common issues
            if "Expecting ',' delimiter" in str(e):
                self.logger.error(
                    "  Issue: Missing comma or malformed JSON structure"
                )
            elif "Expecting ':' delimiter" in str(e):
                self.logger.error("  Issue: Missing colon in key-value pair")
            elif "Expecting value" in str(e):
                self.logger.error("  Issue: Unexpected token or missing value")
            elif "Unterminated string" in str(e):
                self.logger.error("  Issue: Unterminated string literal")
            else:
                self.logger.error("  Issue: General JSON syntax error")

            # Check if response looks like plain text instead of JSON
            if not cleaned_content.strip().startswith(("{", "[")):
                self.logger.error(
                    f"  Model {self.model_name} returned plain text instead of JSON"
                )
                self.logger.error(
                    "  This model may not support structured JSON output properly"
                )

            raise ValueError(
                f"Model {self.model_name} returned invalid JSON: {str(e)} | Response: {response_content[:300]}..."
            ) from e

    async def _make_async_api_call_with_rate_limiting(
        self, client, llm_config, prompt: str, temperature: float, max_tokens: int
    ):
        """
        Make async API call with aggressive rate limiting and exponential backoff retry logic.

        Args:
            client: AsyncOpenAI client instance
            llm_config: LLM configuration
            prompt: Formatted prompt
            temperature: Temperature parameter
            max_tokens: Max tokens parameter

        Returns:
            OpenAI API response

        Raises:
            Exception: If all retry attempts fail
        """
        start_time = time.time()
        # Aggressive retry configuration for rate limiting
        max_retries = 10  # Much more aggressive than the default 3
        base_delay = 1.0  # Base delay in seconds
        max_delay = 60.0  # Maximum delay between retries
        backoff_factor = 2.0  # Exponential backoff multiplier

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
            self.logger.debug(
                f"Using response_format for model: {llm_config.model_identifier}"
            )

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    self.logger.info(f"🚀 API CALL: {llm_config.model_identifier}")
                    # DEBUG: Direct print for immediate visibility
                    print(
                        f"DEBUG - API CALL STARTED: {llm_config.model_identifier}",
                        flush=True,
                    )
                else:
                    self.logger.warning(
                        f"🔄 RETRY {attempt}/{max_retries}: {llm_config.model_identifier}"
                    )
                    print(
                        f"DEBUG - API RETRY {attempt}: {llm_config.model_identifier}",
                        flush=True,
                    )

                response = await client.chat.completions.create(**call_params)
                api_time = time.time() - start_time
                self.logger.info(
                    f"✅ API RESPONSE: {llm_config.model_identifier} ({api_time:.2f}s)"
                )
                print(
                    f"DEBUG - API RESPONSE: {llm_config.model_identifier} ({api_time:.2f}s)",
                    flush=True,
                )
                return response

            except Exception as e:
                # Check error type by examining the error message/code
                error_str = str(e).lower()

                # Check for quota errors first (these should be flagged and model skipped)
                is_quota_error = (
                    "insufficient_quota" in error_str
                    or "quota exceeded" in error_str
                    or "billing" in error_str
                    or "check your plan" in error_str
                )

                if is_quota_error:
                    # Quota errors should be flagged and the model should be skipped
                    self.logger.error(
                        f"💰 QUOTA ERROR for {llm_config.model_identifier}: {str(e)}"
                    )
                    self.logger.error(
                        "  This model has exceeded its quota and will be skipped"
                    )
                    self.logger.error(
                        "  Consider upgrading your plan or using a different model"
                    )
                    raise ValueError(
                        f"Model {llm_config.model_identifier} has exceeded its quota"
                    ) from e

                # Check for rate limiting errors (these should be retried)
                is_rate_limit = (
                    "rate limit" in error_str
                    or "429" in error_str
                    or "too many requests" in error_str
                )

                if is_rate_limit:
                    # Extract rate limit information from the error
                    error_data = getattr(e, "body", {})
                    if isinstance(error_data, dict) and "error" in error_data:
                        error_info = error_data["error"]
                        error_type = error_info.get("type", "unknown")
                        error_message = error_info.get("message", str(e))

                        # Parse retry time from message if available
                        retry_after = None
                        if "Please try again in" in error_message:
                            try:
                                # Extract milliseconds from message like "Please try again in 524ms"
                                retry_match = error_message.split(
                                    "Please try again in"
                                )[1].strip()
                                if retry_match.endswith("ms"):
                                    retry_after = (
                                        float(retry_match[:-2]) / 1000.0
                                    )  # Convert ms to seconds
                                elif retry_match.endswith("s"):
                                    retry_after = float(retry_match[:-1])
                            except (ValueError, IndexError):
                                pass

                        self.logger.warning(
                            f"🚦 RATE LIMIT hit for {llm_config.model_identifier}"
                        )
                        self.logger.warning(f"  Type: {error_type}")
                        self.logger.warning(f"  Message: {error_message}")
                        if retry_after:
                            self.logger.warning(
                                f"  Suggested retry after: {retry_after:.2f}s"
                            )
                    else:
                        # Fallback for rate limit detection without structured error data
                        self.logger.warning(
                            f"🚦 RATE LIMIT detected for {llm_config.model_identifier}"
                        )
                        self.logger.warning(f"  Error: {str(e)}")
                        retry_after = None

                    # If this is the last attempt, don't retry
                    if attempt == max_retries:
                        self.logger.error(
                            f"💥 RATE LIMIT retry exhausted after {max_retries} attempts for {llm_config.model_identifier}"
                        )
                        raise

                    # Calculate delay with exponential backoff and jitter
                    if retry_after:
                        # Use the API's suggested retry time if available
                        delay = retry_after
                    else:
                        # Calculate exponential backoff delay
                        delay = min(base_delay * (backoff_factor**attempt), max_delay)

                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0.1, 1.0) * delay * 0.1
                    total_delay = delay + jitter

                    self.logger.info(
                        f"⏳ Waiting {total_delay:.2f}s before retry {attempt + 1}/{max_retries}"
                    )
                    await asyncio.sleep(total_delay)
                    continue  # Continue to next retry attempt

                # If not a rate limit error, handle as regular API error
                if attempt == max_retries:
                    self.logger.error(
                        f"💥 API error retry exhausted after {max_retries} attempts: {str(e)}"
                    )
                    raise
                else:
                    # For non-rate-limit errors, use shorter delay
                    delay = min(
                        base_delay * (1.5**attempt), 10.0
                    )  # Shorter backoff for API errors
                    jitter = random.uniform(0.1, 1.0) * delay * 0.1
                    total_delay = delay + jitter

                    self.logger.warning(
                        f"⚠️ API error on attempt {attempt + 1}: {str(e)}"
                    )
                    self.logger.info(f"⏳ Waiting {total_delay:.2f}s before retry")
                    await asyncio.sleep(total_delay)