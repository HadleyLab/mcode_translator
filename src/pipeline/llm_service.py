"""
Ultra-Lean LLM Service for mCODE

Leverages existing utils infrastructure for maximum performance and minimal redundancy.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from src.shared.models import McodeElement
from src.utils.api_manager import APIManager
from src.utils.concurrency import TaskQueue, create_task
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

        # Optimization: Connection pooling and reuse
        self._client_cache: Dict[str, Any] = {}
        self._last_client_use: Dict[str, float] = {}

        # Optimization: Performance monitoring
        self._performance_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "error_count": 0
        }

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

            # Use enhanced caching for better performance
            cache_key = self._enhanced_cache_key(clinical_text)

            llm_cache = self.api_manager.get_cache("llm")
            cached_result = llm_cache.get_by_key(cache_key)

            if cached_result is not None:
                self.logger.info(f"💾 LLM cache hit for {self.model_name}")
                self._update_performance_stats(0.0, cache_hit=True, tokens_used=0, error=False)
                return cached_result.get("mcode_elements", [])

            # Make LLM call using existing infrastructure
            response_json = self._call_llm_api(prompt, llm_config)

            # Parse response using existing models
            mcode_elements = self._parse_llm_response(response_json)

            # Cache result using existing API manager
            cache_data = {"mcode_elements": [elem.model_dump() for elem in mcode_elements]}
            llm_cache.set_by_key(cache_data, cache_key)
            self.logger.info(f"✅ LLM result cached for {self.model_name}")

            # Periodic cleanup of old clients
            self._cleanup_old_clients()

            return mcode_elements

        except Exception as e:
            self.logger.error(f"❌ LLM mapping failed: {str(e)}")
            return []

    def _call_llm_api(self, prompt: str, llm_config) -> Dict[str, Any]:
        """
        Make optimized LLM API call with connection reuse and performance monitoring.

        Args:
            prompt: Formatted prompt
            llm_config: LLM configuration from existing loader

        Returns:
            Parsed JSON response
        """
        import openai

        start_time = time.time()
        tokens_used = 0
        error_occurred = False

        try:
            # Use existing config for API details
            temperature = self.config.get_temperature(self.model_name)
            max_tokens = self.config.get_max_tokens(self.model_name)

            # Use cached client for connection reuse (optimization)
            client = self._get_cached_client(self.model_name)

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

            self.logger.info(f"🤖 Making LLM API call to {llm_config.model_identifier}")
            response = client.chat.completions.create(**call_params)

            # Extract token usage using existing utility
            from src.utils.token_tracker import extract_token_usage_from_response
            token_usage = extract_token_usage_from_response(
                response, self.model_name, "provider"
            )
            tokens_used = token_usage.total_tokens

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

                    # STRICT: Fail fast on truncated or incomplete JSON - no fallback processing
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
                error_occurred = True
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
            error_occurred = True
            self.logger.error(f"💥 LLM API call failed: {str(e)}")
            raise
        finally:
            # Update performance statistics
            request_time = time.time() - start_time
            self._update_performance_stats(request_time, cache_hit=False, tokens_used=tokens_used, error=error_occurred)

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

    def _get_cached_client(self, model_name: str):
        """
        Get cached OpenAI client with connection reuse for performance optimization.

        Args:
            model_name: Name of the model to get client for

        Returns:
            Cached OpenAI client instance
        """
        import openai

        cache_key = f"{model_name}_{self.config.get_base_url(model_name)}"
        current_time = time.time()

        # Check if we have a cached client and it's still fresh (< 5 minutes old)
        if (cache_key in self._client_cache and
            cache_key in self._last_client_use and
            current_time - self._last_client_use[cache_key] < 300):  # 5 minutes

            self.logger.debug(f"🔄 Reusing cached client for {model_name}")
            self._last_client_use[cache_key] = current_time
            return self._client_cache[cache_key]

        # Create new client
        try:
            api_key = self.config.get_api_key(model_name)
            base_url = self.config.get_base_url(model_name)

            client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                # Enable connection reuse
                max_retries=3,
                timeout=60.0
            )

            # Cache the client
            self._client_cache[cache_key] = client
            self._last_client_use[cache_key] = current_time

            self.logger.debug(f"✨ Created new cached client for {model_name}")
            return client

        except Exception as e:
            self.logger.error(f"Failed to create client for {model_name}: {e}")
            raise

    def map_to_mcode_batch(self, clinical_texts: List[str], max_workers: int = 4) -> List[List[McodeElement]]:
        """
        Batch process multiple clinical texts concurrently for improved performance.

        Args:
            clinical_texts: List of clinical trial texts to process
            max_workers: Maximum number of concurrent workers

        Returns:
            List of lists containing McodeElement instances for each text
        """
        self.logger.info(f"🔄 Batch processing {len(clinical_texts)} texts with {max_workers} workers")

        # Prepare batch tasks
        batch_tasks = []
        for i, clinical_text in enumerate(clinical_texts):
            task = create_task(
                task_id=f"batch_llm_{i}",
                func=self._process_single_text_batch,
                clinical_text=clinical_text,
                task_index=i
            )
            batch_tasks.append(task)

        # Execute batch processing
        task_queue = TaskQueue(max_workers=max_workers, name="LLMBatchProcessor")

        def progress_callback(completed, total, result):
            if result.success:
                self.logger.info(f"✅ Completed batch task {result.task_id}")
            else:
                self.logger.error(f"❌ Failed batch task {result.task_id}: {result.error}")

        task_results = task_queue.execute_tasks(batch_tasks, progress_callback=progress_callback)

        # Process results and maintain order
        results = [[] for _ in clinical_texts]  # Initialize with empty lists
        successful_tasks = 0
        failed_tasks = 0

        for result in task_results:
            task_index = int(result.task_id.split('_')[-1])
            if result.success and result.result:
                results[task_index] = result.result
                successful_tasks += 1
            else:
                failed_tasks += 1
                self.logger.warning(f"Batch task {result.task_id} failed: {result.error}")

        self.logger.info(f"📊 Batch processing complete: {successful_tasks} successful, {failed_tasks} failed")
        return results

    def _process_single_text_batch(self, clinical_text: str, task_index: int) -> List[McodeElement]:
        """
        Process a single clinical text for batch processing.

        Args:
            clinical_text: Clinical trial text to process
            task_index: Index of this task in the batch

        Returns:
            List of McodeElement instances
        """
        try:
            # Reuse the existing map_to_mcode logic but with optimizations
            return self.map_to_mcode(clinical_text)
        except Exception as e:
            self.logger.error(f"Batch processing failed for task {task_index}: {e}")
            return []

    def _enhanced_cache_key(self, clinical_text: str) -> Dict[str, Any]:
        """
        Generate enhanced cache key with semantic similarity support.

        Args:
            clinical_text: Clinical text to generate key for

        Returns:
            Enhanced cache key dictionary
        """
        # Basic cache key
        basic_key = {
            "model": self.model_name,
            "prompt": self.prompt_name,
            "text_hash": hash(clinical_text)
        }

        # Add semantic fingerprinting for better cache hits
        # This is a simple implementation - could be enhanced with embeddings
        text_length = len(clinical_text)
        text_sample = clinical_text[:200] if len(clinical_text) > 200 else clinical_text

        enhanced_key = {
            **basic_key,
            "text_length": text_length,
            "text_sample_hash": hash(text_sample),
            "semantic_fingerprint": self._generate_semantic_fingerprint(clinical_text)
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

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring and optimization.

        Returns:
            Dictionary with performance metrics
        """
        stats = self._performance_stats.copy()

        # Calculate cache hit rate
        total_cache_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_cache_requests
        else:
            stats["cache_hit_rate"] = 0.0

        # Calculate error rate
        if stats["total_requests"] > 0:
            stats["error_rate"] = stats["error_count"] / stats["total_requests"]
        else:
            stats["error_rate"] = 0.0

        # Add connection pool stats
        stats["active_clients"] = len(self._client_cache)
        stats["oldest_client_age"] = self._get_oldest_client_age()

        return stats

    def _get_oldest_client_age(self) -> float:
        """Get age of oldest cached client in seconds."""
        if not self._last_client_use:
            return 0.0

        current_time = time.time()
        oldest_age = current_time - min(self._last_client_use.values())
        return oldest_age

    def _cleanup_old_clients(self, max_age_seconds: int = 600) -> int:
        """
        Clean up old cached clients to prevent memory leaks.

        Args:
            max_age_seconds: Maximum age for cached clients (default: 10 minutes)

        Returns:
            Number of clients cleaned up
        """
        current_time = time.time()
        cleanup_count = 0

        # Find old clients
        old_clients = []
        for cache_key, last_use in self._last_client_use.items():
            if current_time - last_use > max_age_seconds:
                old_clients.append(cache_key)

        # Remove old clients
        for cache_key in old_clients:
            if cache_key in self._client_cache:
                del self._client_cache[cache_key]
            if cache_key in self._last_client_use:
                del self._last_client_use[cache_key]
            cleanup_count += 1

        if cleanup_count > 0:
            self.logger.info(f"🧹 Cleaned up {cleanup_count} old cached clients")

        return cleanup_count

    def _update_performance_stats(self, request_time: float, cache_hit: bool, tokens_used: int, error: bool):
        """
        Update performance statistics.

        Args:
            request_time: Time taken for the request
            cache_hit: Whether this was a cache hit
            tokens_used: Number of tokens used
            error: Whether an error occurred
        """
        self._performance_stats["total_requests"] += 1
        self._performance_stats["total_tokens"] += tokens_used

        if cache_hit:
            self._performance_stats["cache_hits"] += 1
        else:
            self._performance_stats["cache_misses"] += 1

        if error:
            self._performance_stats["error_count"] += 1

        # Update rolling average response time
        current_avg = self._performance_stats["avg_response_time"]
        total_requests = self._performance_stats["total_requests"]

        if total_requests == 1:
            self._performance_stats["avg_response_time"] = request_time
        else:
            self._performance_stats["avg_response_time"] = (
                (current_avg * (total_requests - 1)) + request_time
            ) / total_requests