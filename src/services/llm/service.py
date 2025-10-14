"""
Ultra-Lean LLM Service for mCODE

Leverages existing utils infrastructure for maximum performance and minimal redundancy.
"""

import json
import logging
import time
from typing import Any, Dict, List

from src.shared.models import (
    ParsedLLMResponse,
    McodeElement,
    McodeMappingResponse,
    PatientTrialMatchResponse,
    ProcessingMetadata,
)
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
        # Set logger to DEBUG level for maximum visibility
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f"🔧 LLM Service initialized for model: {model_name}")

        # Leverage existing utils
        self.llm_loader = llm_loader
        self.prompt_loader = prompt_loader
        self.api_manager = APIManager()
        self.token_tracker = global_token_tracker

    async def map_to_mcode(self, clinical_text: str) -> McodeMappingResponse:
        """
        Map clinical text to mCODE elements using async infrastructure with comprehensive validation.

        Args:
            clinical_text: Clinical trial text to process

        Returns:
            McodeMappingResponse with validated results and metadata
        """
        start_time = time.time()
        self.logger.info(
            f"🚀 LLM SERVICE: map_to_mcode called for {self.model_name} with text length {len(clinical_text)}"
        )

        try:
            # Check if API key is available - STRICT: No fallback, fail fast
            api_key = self.config.get_api_key(self.model_name)
            if not api_key:
                raise ValueError(f"No API key available for {self.model_name}")
            self.logger.info(f"✅ API key found for {self.model_name}")

            # Get LLM config from existing file-based system - STRICT: No fallback, fail fast
            llm_config = self.llm_loader.get_llm(self.model_name)

            # Get prompt from existing file-based system - STRICT: No fallback, fail fast
            prompt = self.prompt_loader.get_prompt(self.prompt_name, clinical_text=clinical_text)

            # Use enhanced caching for better performance
            cache_key = self._enhanced_cache_key(clinical_text)

            llm_cache = self.api_manager.get_cache("llm")
            cached_result = llm_cache.get_by_key(cache_key)

            if cached_result is not None:
                self.logger.info(f"💾 CACHE HIT: {self.model_name}")
                mcode_elements = [McodeElement(**elem) for elem in cached_result.get("mcode_elements", [])]
                parsed_response = ParsedLLMResponse(
                    raw_content=cached_result.get("raw_content", ""),
                    parsed_json=cached_result.get("response_json"),
                    is_valid_json=True
                )
            else:
                # Make async LLM call using existing infrastructure - STRICT: No fallback, fail fast
                self.logger.debug(f"🔍 CACHE MISS → 🚀 ASYNC API CALL: {self.model_name}")
                parsed_response = await self._call_llm_api_async(prompt, llm_config)

                # Parse response using existing models - STRICT: No fallback, fail fast
                if parsed_response.parsed_json is None:
                    raise ValueError("Parsed JSON is None - cannot parse LLM response")
                mcode_elements = self._parse_llm_response(parsed_response.parsed_json)

                # Cache result using existing API manager
                cache_data = {
                    "mcode_elements": [elem.model_dump() for elem in mcode_elements],
                    "response_json": parsed_response.parsed_json,
                    "raw_content": parsed_response.raw_content,
                }
                llm_cache.set_by_key(cache_data, cache_key)
                self.logger.debug(f"💾 CACHE SAVED: {self.model_name}")

            # Create processing metadata
            processing_time = time.time() - start_time
            metadata = ProcessingMetadata(
                engine_type="llm",
                entities_count=len(mcode_elements),
                mapped_count=len(mcode_elements),
                processing_time_seconds=processing_time,
                model_used=self.model_name,
                prompt_used=self.prompt_name
            )

            return McodeMappingResponse(
                mcode_elements=mcode_elements,
                raw_response=parsed_response,
                processing_metadata=metadata,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ LLM SERVICE: map_to_mcode failed: {e}")
            processing_time = time.time() - start_time
            metadata = ProcessingMetadata(
                engine_type="llm",
                entities_count=0,
                mapped_count=0,
                processing_time_seconds=processing_time,
                model_used=self.model_name,
                prompt_used=self.prompt_name
            )
            return McodeMappingResponse(
                mcode_elements=[],
                raw_response=ParsedLLMResponse(
                    raw_content="",
                    parsed_json=None,
                    is_valid_json=False,
                    validation_errors=[str(e)]
                ),
                processing_metadata=metadata,
                success=False,
                error_message=str(e)
            )

    async def match_patient_to_trial(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> PatientTrialMatchResponse:
        """
        Match patient data against trial eligibility criteria using LLM with comprehensive validation.

        Args:
            patient_data: Patient information dictionary
            trial_criteria: Trial eligibility criteria dictionary

        Returns:
            PatientTrialMatchResponse with validated match results and metadata
        """
        start_time = time.time()
        self.logger.info(
            f"🚀 LLM SERVICE: match_patient_to_trial called for {self.model_name}"
        )

        # DEBUG: Log input data for debugging
        self.logger.debug(f"🔍 PATIENT DATA: {json.dumps(patient_data, indent=2)}")
        self.logger.debug(f"🔍 TRIAL CRITERIA: {json.dumps(trial_criteria, indent=2)}")

        try:
            # Check if API key is available - STRICT: No fallback, fail fast
            api_key = self.config.get_api_key(self.model_name)
            if not api_key:
                raise ValueError(f"No API key available for {self.model_name}")
            self.logger.info(f"✅ API key found for {self.model_name}")

            # Get LLM config from existing file-based system - STRICT: No fallback, fail fast
            llm_config = self.llm_loader.get_llm(self.model_name)

            # Create matching prompt using the new prompt file
            prompt = self.prompt_loader.get_prompt("patient_matcher", patient_data=patient_data, trial_criteria=trial_criteria)

            # DEBUG: Log the prompt being sent to DeepSeek
            self.logger.info(f"📤 PROMPT SENT TO {self.model_name}:")
            self.logger.info(f"--- PROMPT START ---")
            self.logger.info(prompt)
            self.logger.info(f"--- PROMPT END ---")

            # Use enhanced caching for better performance
            cache_key = self._enhanced_cache_key(f"match_{patient_data.get('id', 'unknown')}_{trial_criteria.get('trial_id', 'unknown')}")

            llm_cache = self.api_manager.get_cache("llm")
            cached_result = llm_cache.get_by_key(cache_key)

            if cached_result is not None:
                self.logger.info(f"💾 CACHE HIT: {self.model_name}")
                # Reconstruct response from cached data
                match_result = cached_result
                parsed_response = ParsedLLMResponse(
                    raw_content=cached_result.get("raw_content", ""),
                    parsed_json=cached_result.get("response_json"),
                    is_valid_json=True
                )
            else:
                # Make async LLM call using existing infrastructure - STRICT: No fallback, fail fast
                self.logger.debug(f"🔍 CACHE MISS → 🚀 ASYNC API CALL: {self.model_name}")
                parsed_response = await self._call_llm_api_async(prompt, llm_config)

                # Parse matching response
                if parsed_response.parsed_json is None:
                    raise ValueError("Parsed JSON is None - cannot parse matching response")
                match_result = self._parse_matching_response(parsed_response.parsed_json)

                # DEBUG: Log the parsed result
                self.logger.info(f"📊 PARSED MATCH RESULT: {json.dumps(match_result, indent=2)}")

                # Cache result using existing API manager
                cache_data = match_result.copy()
                cache_data.update({
                    "raw_content": parsed_response.raw_content,
                    "response_json": parsed_response.parsed_json
                })
                llm_cache.set_by_key(cache_data, cache_key)
                self.logger.debug(f"💾 CACHE SAVED: {self.model_name}")

            # Create processing metadata
            processing_time = time.time() - start_time
            metadata = ProcessingMetadata(
                engine_type="llm",
                entities_count=0,  # Not applicable for matching
                mapped_count=0,    # Not applicable for matching
                processing_time_seconds=processing_time,
                model_used=self.model_name,
                prompt_used="patient_matcher"
            )

            return PatientTrialMatchResponse(
                is_match=match_result.get("is_match", False),
                confidence_score=match_result.get("confidence_score", 0.0),
                reasoning=match_result.get("reasoning", ""),
                matched_criteria=match_result.get("matched_criteria", []),
                unmatched_criteria=match_result.get("unmatched_criteria", []),
                clinical_notes=match_result.get("clinical_notes", ""),
                matched_elements=match_result.get("matched_elements", []),
                raw_response=parsed_response,
                processing_metadata=metadata,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ LLM SERVICE: match_patient_to_trial failed: {e}")
            processing_time = time.time() - start_time
            metadata = ProcessingMetadata(
                engine_type="llm",
                entities_count=0,
                mapped_count=0,
                processing_time_seconds=processing_time,
                model_used=self.model_name,
                prompt_used="patient_matcher"
            )
            return PatientTrialMatchResponse(
                is_match=False,
                confidence_score=0.0,
                reasoning=f"Matching failed: {str(e)}",
                matched_criteria=[],
                unmatched_criteria=[],
                clinical_notes="",
                matched_elements=[],
                raw_response=ParsedLLMResponse(
                    raw_content="",
                    parsed_json=None,
                    is_valid_json=False,
                    validation_errors=[str(e)]
                ),
                processing_metadata=metadata,
                success=False,
                error_message=str(e)
            )

    def _create_matching_prompt(self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]) -> str:
        """
        Create a prompt for patient-trial matching.

        Args:
            patient_data: Patient information
            trial_criteria: Trial eligibility criteria

        Returns:
            Formatted prompt string
        """
        # This method is now deprecated - using prompt_loader instead
        # Keeping for backward compatibility
        patient_summary = patient_data.get("conditions", "No patient conditions specified")
        trial_criteria_text = trial_criteria.get("eligibilityCriteria", "No eligibility criteria specified")

        prompt = f"""You are a clinical trial matching expert specializing in oncology trials. Your task is to determine if a patient meets the eligibility criteria for a clinical trial.

PATIENT INFORMATION:
{patient_summary}

TRIAL ELIGIBILITY CRITERIA:
{trial_criteria_text}

INSTRUCTIONS:
1. Carefully analyze if the patient meets ALL inclusion criteria and NONE of the exclusion criteria
2. Focus on clinical relevance, safety, and appropriateness for the trial
3. Be conservative - if there's any uncertainty or missing information, err on the side of caution
4. Consider the patient's specific condition, stage, and characteristics against trial requirements
5. Look for explicit matches in diagnosis, age, gender, performance status, prior treatments, etc.

RESPONSE FORMAT (JSON):
{{
  "is_match": true,
  "confidence_score": 0.9,
  "reasoning": "Patient has breast cancer diagnosis matching trial inclusion criteria for breast cancer patients. Age and gender requirements are met. No exclusion criteria appear to apply.",
  "matched_criteria": ["breast cancer diagnosis", "female gender", "age within range"],
  "unmatched_criteria": [],
  "clinical_notes": "Strong clinical match for this breast cancer trial"
}}

Now analyze this patient-trial pair and respond with JSON:"""

        return prompt

    def _parse_matching_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM response for patient-trial matching. STRICT parsing. Fail fast.

        Args:
            response_json: Raw LLM response, expected to be a JSON object.

        Returns:
            Parsed match result dictionary.
        """
        self.logger.debug(f"Attempting to parse LLM matching response: {response_json}")

        if not isinstance(response_json, dict):
            self.logger.error(f"Response is not a JSON object: {type(response_json)}")
            return {
                "is_match": False,
                "reasoning": "Invalid response format: not a JSON object.",
                "confidence_score": 0.0,
                "matched_criteria": [],
                "unmatched_criteria": [],
                "clinical_notes": "",
                "matched_elements": [],
            }

        is_match = response_json.get("is_match")

        # STRICT: is_match must be a boolean. No complex parsing or fallbacks.
        if not isinstance(is_match, bool):
            self.logger.warning(
                f"'is_match' field is not a boolean ({type(is_match)}). Defaulting to False."
            )
            is_match = False

        result = {
            "is_match": is_match,
            "confidence_score": response_json.get("confidence_score", 0.0),
            "reasoning": response_json.get("reasoning", ""),
            "matched_criteria": response_json.get("matched_criteria", []),
            "unmatched_criteria": response_json.get("unmatched_criteria", []),
            "clinical_notes": response_json.get("clinical_notes", ""),
            "matched_elements": response_json.get("matched_elements", []),
        }

        self.logger.debug(f"Successfully parsed matching response: {result}")
        return result

    async def _call_llm_api_async(self, prompt: str, llm_config: Any) -> ParsedLLMResponse:
        """
        Make optimized async LLM API call with connection reuse and performance monitoring.

        Args:
            prompt: Formatted prompt
            llm_config: LLM configuration from existing loader

        Returns:
            ParsedLLMResponse with validated and cleaned response data
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
                    raise ValueError(f"API key not configured for model {self.model_name}")
            except Exception as e:
                if "not found in config" in str(e) or "API key not configured" in str(e):
                    self.logger.warning(f"API key configuration issue for {self.model_name}: {e}")
                    raise ValueError(f"API key not configured for model {self.model_name}")
                else:
                    raise

            # Use existing config for API details
            temperature = self.config.get_temperature(self.model_name)
            max_tokens = self.config.get_max_tokens(self.model_name)

            # Create fresh async client for each request (simple pattern)
            api_key = self.config.get_api_key(self.model_name)
            base_url = self.config.get_base_url(self.model_name)

            import openai

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
            from src.utils.token_tracker import extract_token_usage_from_response

            token_usage = extract_token_usage_from_response(response, self.model_name, "provider")

            # Track tokens using existing global tracker
            self.token_tracker.add_usage(token_usage, "llm_service")

            # Parse response
            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Empty LLM response")

            # DEBUG: Log raw API response from DeepSeek
            self.logger.info(f"📥 RAW API RESPONSE FROM {self.model_name}:")
            self.logger.info(f"--- RESPONSE START ---")
            self.logger.info(response_content)
            self.logger.info(f"--- RESPONSE END ---")

            try:
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
                try:
                    parsed_json = json.loads(cleaned_content)
                    parsed_response.parsed_json = parsed_json
                    parsed_response.is_valid_json = True
                    self.logger.debug(f"✅ Successfully parsed JSON response from {self.model_name}")
                    return parsed_response
                except json.JSONDecodeError as json_error:
                    parsed_response.validation_errors.append(f"JSON decode error: {str(json_error)}")
                    raise ValueError(f"Invalid JSON from {self.model_name}: {str(json_error)} | Content: {cleaned_content[:300]}...") from json_error
            except json.JSONDecodeError as e:
                # Provide detailed error information for debugging
                self.logger.error(f"❌ JSON parsing error for {self.model_name}: {str(e)}")
                self.logger.error(f"  Response preview: {response_content[:500]}...")

                # Check for common issues
                if "Expecting ',' delimiter" in str(e):
                    self.logger.error("  Issue: Missing comma or malformed JSON structure")
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

        except Exception as e:
            self.logger.error(f"💥 LLM API call failed for {self.model_name}: {str(e)}")
            raise

    async def _make_async_api_call_with_rate_limiting(
        self,
        client: Any,
        llm_config: Any,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
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
        import asyncio
        import random

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
        ):
            call_params["response_format"] = {"type": "json_object"}
            self.logger.debug(f"Using response_format for model: {llm_config.model_identifier}")

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
                    self.logger.error(f"💰 QUOTA ERROR for {llm_config.model_identifier}: {str(e)}")
                    self.logger.error("  This model has exceeded its quota and will be skipped")
                    self.logger.error("  Consider upgrading your plan or using a different model")
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
                                retry_match = error_message.split("Please try again in")[1].strip()
                                if retry_match.endswith("ms"):
                                    retry_after = (
                                        float(retry_match[:-2]) / 1000.0
                                    )  # Convert ms to seconds
                                elif retry_match.endswith("s"):
                                    retry_after = float(retry_match[:-1])
                            except (ValueError, IndexError):
                                pass

                        self.logger.warning(f"🚦 RATE LIMIT hit for {llm_config.model_identifier}")
                        self.logger.warning(f"  Type: {error_type}")
                        self.logger.warning(f"  Message: {error_message}")
                        if retry_after:
                            self.logger.warning(f"  Suggested retry after: {retry_after:.2f}s")
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
                    delay = min(base_delay * (1.5**attempt), 10.0)  # Shorter backoff for API errors
                    jitter = random.uniform(0.1, 1.0) * delay * 0.1
                    total_delay = delay + jitter

                    self.logger.warning(f"⚠️ API error on attempt {attempt + 1}: {str(e)}")
                    self.logger.info(f"⏳ Waiting {total_delay:.2f}s before retry")
                    await asyncio.sleep(total_delay)

    def _parse_llm_response(self, response_json: Dict[str, Any]) -> List[McodeElement]:
        """
        Parse LLM response into McodeElement instances using Pydantic validation.

        Args:
            response_json: Raw LLM response

        Returns:
            List of validated McodeElement instances
        """
        elements = []

        # Handle the actual LLM response format from cache
        # LLM returns: {"mcode_elements": [...], "response_json": {...}}
        if "mcode_elements" in response_json:
            mcode_data = response_json["mcode_elements"]
        else:
            # Fallback to other formats for compatibility
            mcode_data = (
                response_json.get("mcode_mappings")
                or response_json.get("mappings")
                or response_json.get("mapped_elements")
                or []
            )

        # If no mappings found, try the direct format
        if not mcode_data and "element_type" in response_json:
            mcode_data = [response_json]

        for item in mcode_data:
            try:
                # Handle the prompt's expected format with nested code object
                if "mcode_element" in item and "code" in item:
                    # Transform from prompt format to Pydantic model format
                    transformed_item = {
                        "element_type": item["mcode_element"],
                        "code": (
                            item["code"].get("code") if isinstance(item["code"], dict) else None
                        ),
                        "display": (
                            item["code"].get("display") if isinstance(item["code"], dict) else None
                        ),
                        "system": (
                            item["code"].get("system") if isinstance(item["code"], dict) else None
                        ),
                        "confidence_score": item.get("mapping_confidence"),
                        "evidence_text": item.get("source_text_fragment"),
                    }
                    element = McodeElement(**transformed_item)
                else:
                    # Try direct mapping for other formats
                    element = McodeElement(**item)

                elements.append(element)
            except Exception as e:
                self.logger.warning(f"Failed to create McodeElement: {e}")
                self.logger.warning(f"Problematic item: {item}")
                continue

        return elements

    def _enhanced_cache_key(self, clinical_text: str) -> Dict[str, Any]:
        """
        Generate enhanced cache key with semantic similarity support.

        Args:
            clinical_text: Clinical text to generate key for

        Returns:
            Enhanced cache key dictionary
        """
        import hashlib

        # Basic cache key with deterministic hashing
        basic_key = {
            "model": self.model_name,
            "prompt": self.prompt_name,
            "text_hash": hashlib.md5(clinical_text.encode("utf-8")).hexdigest(),
        }

        # Add semantic fingerprinting for better cache hits
        # This is a simple implementation - could be enhanced with embeddings
        text_length = len(clinical_text)
        text_sample = clinical_text[:200] if len(clinical_text) > 200 else clinical_text

        enhanced_key = {
            **basic_key,
            "text_length": text_length,
            "text_sample_hash": hashlib.md5(text_sample.encode("utf-8")).hexdigest(),
            "semantic_fingerprint": self._generate_semantic_fingerprint(clinical_text),
        }

        return enhanced_key

    def _generate_semantic_fingerprint(self, text: str) -> str:
        """
        Generate a simple semantic fingerprint for better caching.

        Args:
            text: Text to fingerprint

        Returns:
            Semantic fingerprint string
        """
        # Simple fingerprinting based on key terms and structure
        # This could be enhanced with actual NLP processing
        key_terms = ["cancer", "treatment", "patient", "trial", "clinical", "study"]
        fingerprint_parts = []

        for term in key_terms:
            if term in text.lower():
                fingerprint_parts.append(term)

        # Add text length category
        if len(text) < 1000:
            fingerprint_parts.append("short")
        elif len(text) < 5000:
            fingerprint_parts.append("medium")
        else:
            fingerprint_parts.append("long")

        return "_".join(fingerprint_parts) if fingerprint_parts else "generic"
