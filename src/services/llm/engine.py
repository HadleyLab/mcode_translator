"""
LLMEngine - Clean LLM-Based Processing Engine

Implements the ProcessingEngine protocol for LLM-based content processing.
Contains only LLM-specific logic - no pipeline orchestration.
"""

from typing import List

from src.services.llm.service import LLMService
from src.shared.models import McodeElement
from src.utils.config import Config
from src.utils.logging_config import get_logger


class LLMEngine:
    """
    Clean LLM-based processing engine.

    Only contains LLM API calling and response processing logic.
    No pipeline orchestration - that's handled by ProcessingService.
    """

    def __init__(
        self,
        model_name: str = "deepseek-coder",
        prompt_name: str = "direct_mcode_evidence_based_concise",
    ):
        """Initialize the LLM engine with model and prompt configuration."""
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.prompt_name = prompt_name

        # Initialize the LLM service (sophisticated API handling)
        self.config = Config()
        self.llm_service = LLMService(self.config, model_name, prompt_name)

        self.logger.info(
            f"✅ LLMEngine initialized with model: {model_name}, prompt: {prompt_name}"
        )

    async def process_section(self, section_content: str) -> List[McodeElement]:
        """
        Process a single document section using LLM analysis.

        Args:
            section_content: Text content of the section to process

        Returns:
            List of extracted mCODE elements from LLM analysis
        """
        self.logger.debug(
            f"🤖 LLMEngine processing section content (length: {len(section_content)})"
        )

        try:
            # Use the sophisticated LLM service for processing
            elements = await self.llm_service.map_to_mcode(section_content)
            self.logger.debug(f"✅ LLMEngine extracted {len(elements)} elements from section")
            return elements

        except Exception as e:
            self.logger.error(f"❌ LLMEngine failed to process section: {e}")
            return []
