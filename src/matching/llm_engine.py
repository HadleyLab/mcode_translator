"""
LLMMatchingEngine - An AI-driven matching engine using Large Language Models.

Enhanced with expert panel support for ensemble decision making and
comprehensive clinical trial matching analysis.
"""

from typing import Any, Dict, Optional

from matching.base import MatchingEngineBase
from matching.expert_panel_manager import ExpertPanelManager
from services.llm.service import LLMService
from utils.config import Config
from utils.logging_config import get_logger


class LLMMatchingEngine(MatchingEngineBase):
    """
    A matching engine that uses an LLM to find matches between patient and trial data.

    Enhanced with expert panel support for ensemble decision making and
    comprehensive clinical trial matching analysis.
    """

    def __init__(
        self,
        model_name: str,
        prompt_name: str,
        cache_enabled: bool = True,
        max_retries: int = 3,
        enable_expert_panel: bool = False,
        expert_panel_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the engine with a specific LLM model and prompt.

        Args:
            model_name: The name of the LLM model to use.
            prompt_name: The name of the prompt to use for matching.
            cache_enabled: Whether to enable caching for API calls.
            max_retries: Maximum number of retries on failure.
            enable_expert_panel: Whether to use expert panel for ensemble decisions.
            expert_panel_config: Configuration for expert panel (if enabled).
        """
        super().__init__(cache_enabled=cache_enabled, max_retries=max_retries)
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.enable_expert_panel = enable_expert_panel

        config = Config()
        self.llm_service = LLMService(config, model_name, prompt_name)

        # Initialize expert panel if enabled
        self.expert_panel = None
        if enable_expert_panel:
            panel_config = expert_panel_config or {}
            self.expert_panel = ExpertPanelManager(
                model_name=model_name,
                config=config,
                max_concurrent_experts=panel_config.get("max_concurrent_experts", 3),
                enable_diversity_selection=panel_config.get("enable_diversity_selection", True)
            )

        self.logger.info(
            f"✅ LLMMatchingEngine initialized with model: {model_name}, prompt: {prompt_name}, "
            f"cache: {cache_enabled}, retries: {max_retries}, expert_panel: {enable_expert_panel}"
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
            if self.enable_expert_panel and self.expert_panel:
                # Use expert panel for comprehensive assessment
                panel_result = await self.expert_panel.assess_with_expert_panel(
                    patient_data, trial_criteria
                )
                return panel_result.get("is_match", False)
            else:
                # Use standard LLM service
                return await self._match_with_llm_service(patient_data, trial_criteria)

        except Exception as e:
            self.logger.error(f"❌ LLMMatchingEngine failed: {e}")
            return False

    async def _match_with_llm_service(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> bool:
        """Match using standard LLM service.

        Args:
            patient_data: Patient data dictionary
            trial_criteria: Trial criteria dictionary

        Returns:
            Boolean match result
        """
        # Use LLM service for patient-trial matching
        match_result = await self.llm_service.match_patient_to_trial(
            patient_data, trial_criteria
        )
        self.logger.debug(f"LLMMatchingEngine received match_result with is_match: {match_result.is_match}")
        return match_result.is_match

    async def _match_with_expert_panel(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> bool:
        """Match using expert panel for ensemble decision."""
        # Get expert panel assessment
        panel_result = await self.expert_panel.assess_with_expert_panel(
            patient_data, trial_criteria
        )
        return panel_result.get("is_match", False)

    async def get_detailed_assessment(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get detailed assessment with reasoning and confidence scores.

        Args:
            patient_data: Patient data dictionary
            trial_criteria: Trial criteria dictionary

        Returns:
            Detailed assessment dictionary
        """
        try:
            if self.enable_expert_panel and self.expert_panel:
                # Get comprehensive expert panel assessment
                return await self.expert_panel.assess_with_expert_panel(
                    patient_data, trial_criteria
                )
            else:
                # Get standard LLM assessment
                match_result = await self.llm_service.match_patient_to_trial(
                    patient_data, trial_criteria
                )

                return {
                    "is_match": match_result.is_match,
                    "confidence_score": match_result.confidence_score,
                    "reasoning": match_result.reasoning,
                    "matched_criteria": match_result.matched_criteria,
                    "unmatched_criteria": match_result.unmatched_criteria,
                    "clinical_notes": match_result.clinical_notes,
                    "model_used": self.model_name,
                    "prompt_used": self.prompt_name,
                    "assessment_method": "single_llm",
                    "processing_metadata": match_result.processing_metadata
                }

        except Exception as e:
            self.logger.error(f"❌ Detailed assessment failed: {e}")
            return {
                "is_match": False,
                "confidence_score": 0.0,
                "reasoning": f"Assessment failed: {str(e)}",
                "matched_criteria": [],
                "unmatched_criteria": [],
                "clinical_notes": "",
                "error": str(e),
                "assessment_method": "failed"
            }

    async def get_expert_panel_status(self) -> Dict[str, Any]:
        """Get status of the expert panel if enabled.

        Returns:
            Expert panel status information
        """
        if self.expert_panel:
            return await self.expert_panel.get_expert_panel_status()
        else:
            return {
                "expert_panel_enabled": False,
                "reason": "Expert panel not initialized"
            }

    def shutdown(self):
        """Shutdown the LLM engine and cleanup resources."""
        self.logger.info("🔄 Shutting down LLMMatchingEngine")

        # Shutdown expert panel if initialized
        if self.expert_panel:
            self.expert_panel.shutdown()

        self.logger.info("✅ LLMMatchingEngine shutdown complete")
