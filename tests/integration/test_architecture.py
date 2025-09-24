"""
Test the refactored mCODE translator architecture.

This module tests the new modular architecture to ensure all components
work together correctly.
"""

import pytest

from src.utils.config import Config
from src.pipeline import McodePipeline
from src.pipeline.document_ingestor import DocumentIngestor
from src.pipeline.llm_service import LLMService


class TestArchitecture:
    """Test the new ultra-lean architecture components."""

    def test_pipeline_initialization(self):
        """Test that the main pipeline initializes correctly."""
        pipeline = McodePipeline()
        assert isinstance(pipeline, McodePipeline)
        assert hasattr(pipeline, "process")

    def test_llm_service_initialization(self):
        """Test that the LLM service initializes correctly."""
        config = Config()
        llm_service = LLMService(config, "deepseek-coder", "direct_mcode")
        assert isinstance(llm_service, LLMService)
        assert hasattr(llm_service, "map_to_mcode")

    def test_document_ingestor_initialization(self):
        """Test that the document ingestor initializes correctly."""
        ingestor = DocumentIngestor()
        assert isinstance(ingestor, DocumentIngestor)
        assert hasattr(ingestor, "ingest_clinical_trial_document")


if __name__ == "__main__":
    pytest.main([__file__])
