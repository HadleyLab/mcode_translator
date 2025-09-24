"""
Unit tests for the new ultra-lean pipeline architecture.
"""

import json
from unittest.mock import patch

import pytest

from src.pipeline import McodePipeline
from src.shared.models import PipelineResult, McodeElement, ValidationResult
from src.utils.config import Config


@pytest.fixture
def sample_trial_data():
    """Fixture for sample clinical trial data."""
    with open("tests/data/sample_trial.json", "r") as f:
        return json.load(f)


class TestMcodePipeline:
    """Test the new McodePipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = McodePipeline()
        assert pipeline is not None
        assert isinstance(pipeline.config, Config)
        assert pipeline.model_name == "deepseek-coder"
        assert pipeline.prompt_name == "direct_mcode_evidence_based_concise"

    def test_initialization_with_custom_params(self):
        """Test pipeline initialization with custom parameters."""
        pipeline = McodePipeline(model_name="gpt-4", prompt_name="custom_prompt")
        assert pipeline.model_name == "gpt-4"
        assert pipeline.prompt_name == "custom_prompt"

    @patch("src.pipeline.llm_service.LLMService.map_to_mcode")
    @pytest.mark.asyncio
    async def test_process_successful(self, mock_map_to_mcode, sample_trial_data):
        """Test successful processing of a trial."""
        # Mock the LLM service to return a sample mCODE element
        mock_map_to_mcode.return_value = [
            McodeElement(
                element_type="CancerCondition", code="C123", display="Test Cancer"
            )
        ]

        pipeline = McodePipeline()
        result = await pipeline.process(sample_trial_data)

        assert isinstance(result, PipelineResult)
        assert result.error is None
        assert len(result.mcode_mappings) > 0
        assert result.mcode_mappings[0].element_type == "CancerCondition"
        assert result.metadata.mapped_count > 0

    @pytest.mark.asyncio
    async def test_process_with_invalid_data(self):
        """Test processing with invalid trial data."""
        pipeline = McodePipeline()
        invalid_data = {"foo": "bar"}  # Missing required fields

        # This should raise a Pydantic validation error, not return a result
        with pytest.raises(Exception) as exc_info:
            await pipeline.process(invalid_data)
        assert "protocolSection" in str(exc_info.value)

    @patch("src.pipeline.llm_service.LLMService.map_to_mcode")
    @pytest.mark.asyncio
    async def test_process_with_llm_error(self, mock_map_to_mcode, sample_trial_data):
        """Test processing when the LLM service raises an exception."""
        mock_map_to_mcode.side_effect = Exception("LLM API is down")

        pipeline = McodePipeline()

        # The current implementation raises exceptions rather than handling them gracefully
        with pytest.raises(Exception) as exc_info:
            await pipeline.process(sample_trial_data)
        assert "LLM API is down" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_processing(self, sample_trial_data):
        """Test batch processing of trials."""
        # For this test, we can just check if it runs without error
        # A more thorough test would mock the process method
        with patch.object(McodePipeline, "process") as mock_process:
            mock_process.return_value = PipelineResult(
                mcode_mappings=[],
                validation_results=ValidationResult(compliance_score=1.0),
                metadata={
                    "engine_type": "test",
                    "total_count": 1,
                    "successful": 1,
                    "failed": 0,
                    "success_rate": 1.0,
                },
                original_data={},
            )

            pipeline = McodePipeline()
            batch_results = await pipeline.process_batch(
                [sample_trial_data, sample_trial_data]
            )

            assert len(batch_results) == 2
            assert mock_process.call_count == 2

    def test_compliance_score_calculation(self):
        """Test the compliance score calculation."""
        pipeline = McodePipeline()

        # No elements
        assert pipeline._calculate_compliance_score([]) == 0.0

        # One required element
        elements1 = [McodeElement(element_type="CancerCondition")]
        assert pipeline._calculate_compliance_score(elements1) > 0.3

        # All required elements
        elements3 = [
            McodeElement(element_type="CancerCondition"),
            McodeElement(element_type="CancerTreatment"),
            McodeElement(element_type="TumorMarker"),
        ]
        assert pipeline._calculate_compliance_score(elements3) == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
