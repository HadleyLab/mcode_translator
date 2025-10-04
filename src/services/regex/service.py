"""
Regex Service for mCODE - Deterministic pattern-based extraction.

Reuses the same pipeline architecture as LLM service but uses regex patterns
instead of API calls for mCODE element extraction.
"""

from typing import Any, Dict, List

from src.shared.models import McodeElement
from src.utils.logging_config import get_logger
from src.workflows.trial_extractor import TrialExtractor


class RegexService:
    """
    Regex-based service that mimics LLMService interface for pipeline compatibility.

    Uses deterministic regex patterns instead of LLM API calls for mCODE extraction.
    """

    def __init__(self, config: Any, model_name: str = "regex", prompt_name: str = "patterns"):
        """
        Initialize with same interface as LLMService.

        Args:
            config: Config instance (not used for regex)
            model_name: Ignored for regex (always "regex")
            prompt_name: Ignored for regex (always "patterns")
        """
        self.config = config
        self.model_name = "regex"  # Always regex
        self.prompt_name = "patterns"  # Always patterns
        self.logger = get_logger(__name__)
        self.extractor = TrialExtractor()

    async def map_to_mcode(self, clinical_text: str) -> List[McodeElement]:
        """
        Map clinical text to mCODE elements using regex patterns.

        Args:
            clinical_text: Clinical trial text to process

        Returns:
            List of McodeElement instances
        """
        self.logger.info(
            f"🔍 REGEX SERVICE: map_to_mcode called for text length {len(clinical_text)}"
        )

        # For regex processing, we need the full trial data, not just text
        # This is a limitation - regex service needs full trial structure
        # For now, return empty list since we can't process text fragments
        self.logger.warning(
            "Regex service cannot process text fragments - needs full trial data structure"
        )
        return []

    def map_trial_to_mcode(self, trial_data: Dict[str, Any]) -> List[McodeElement]:
        """
        Map full trial data to mCODE elements using regex patterns.

        Args:
            trial_data: Full clinical trial data dictionary

        Returns:
            List of McodeElement instances
        """
        self.logger.info(
            f"🔍 REGEX SERVICE: map_trial_to_mcode called for trial data"
        )

        # Extract mCODE elements using the existing TrialExtractor
        mcode_elements = self.extractor.extract_trial_mcode_elements(trial_data)

        # Convert to McodeElement format expected by pipeline
        elements = []
        for element_name, element_data in mcode_elements.items():
            if isinstance(element_data, list):
                for item in element_data:
                    if isinstance(item, dict):
                        element = McodeElement(
                            element_type=element_name,
                            code=item.get("code"),
                            display=item.get("display"),
                            system=item.get("system"),
                            confidence_score=1.0,  # Regex is deterministic
                            evidence_text=f"Extracted from trial data using regex patterns"
                        )
                        elements.append(element)
            elif isinstance(element_data, dict):
                element = McodeElement(
                    element_type=element_name,
                    code=element_data.get("code"),
                    display=element_data.get("display"),
                    system=element_data.get("system"),
                    confidence_score=1.0,  # Regex is deterministic
                    evidence_text=f"Extracted from trial data using regex patterns"
                )
                elements.append(element)

        self.logger.info(
            f"✅ REGEX SERVICE: extracted {len(elements)} mCODE elements"
        )
        return elements