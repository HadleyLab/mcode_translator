"""
LLMMatchingEngine - An AI-driven matching engine using Large Language Models.
"""

from typing import Any, Dict

from src.matching.base import MatchingEngineBase
from src.services.llm.service import LLMService
from src.utils.config import Config
from src.utils.logging_config import get_logger


class LLMMatchingEngine(MatchingEngineBase):
    """
    A matching engine that uses an LLM to find matches between patient and trial data.
    """

    def __init__(self, model_name: str, prompt_name: str, cache_enabled: bool = True, max_retries: int = 3):
        """
        Initializes the engine with a specific LLM model and prompt.

        Args:
            model_name: The name of the LLM model to use.
            prompt_name: The name of the prompt to use for matching.
            cache_enabled: Whether to enable caching for API calls.
            max_retries: Maximum number of retries on failure.
        """
        super().__init__(cache_enabled=cache_enabled, max_retries=max_retries)
        self.logger = get_logger(__name__)
        config = Config()
        self.llm_service = LLMService(config, model_name, prompt_name)
        self.logger.info(
            f"✅ LLMMatchingEngine initialized with model: {model_name}, prompt: {prompt_name}, cache: {cache_enabled}, retries: {max_retries}"
        )

    async def match(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> bool:
        """
        Matches patient data against trial criteria using an LLM.

        Args:
            patient_data: The patient's data, to be included in the prompt.
            trial_criteria: The trial's eligibility criteria.

        Returns:
            A boolean indicating if a match was found.
        """
        try:
            # Use LLM service for patient-trial matching
            match_result = await self.llm_service.match_patient_to_trial(
                patient_data, trial_criteria
            )
            self.logger.debug(f"LLMMatchingEngine received match_result with is_match: {match_result.is_match}")
            return match_result.is_match

        except Exception as e:
            self.logger.error(f"❌ LLMMatchingEngine failed: {e}")
            return False