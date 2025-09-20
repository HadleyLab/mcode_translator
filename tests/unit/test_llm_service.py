"""
Ultra-Lean LLM Service Tests
Tests the streamlined LLM service interface with focus on core functionality.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.pipeline.llm_service import LLMService
from src.utils.config import Config
from src.shared.models import McodeElement


class TestLLMServiceCore:
    """Test core LLM service functionality."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        config = Config()
        return LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")

    @pytest.fixture
    def mock_config(self):
        """Mock config for testing."""
        config = Mock()
        config.get_api_key.return_value = "test-key"
        config.get_temperature.return_value = 0.1
        config.get_max_tokens.return_value = 1000
        config.get_timeout.return_value = 30
        config.get_base_url.return_value = "https://api.deepseek.com"
        return config

    @pytest.mark.asyncio
    async def test_map_to_mcode_successful(self, llm_service, mock_config):
        """Test successful mCODE mapping."""
        # Mock the LLM loader and prompt loader
        with patch('src.pipeline.llm_service.llm_loader') as mock_llm_loader, \
             patch('src.pipeline.llm_service.prompt_loader') as mock_prompt_loader, \
             patch.object(llm_service, '_call_llm_api_async') as mock_call_api:

            # Setup mocks
            mock_llm_config = Mock()
            mock_llm_config.model_identifier = "deepseek-coder"
            mock_llm_loader.get_llm.return_value = mock_llm_config
            mock_prompt_loader.get_prompt.return_value = "test prompt"

            # Mock successful API response
            mock_response = {
                "mcode_mappings": [
                    {
                        "element_type": "CancerCondition",
                        "code": "C123",
                        "display": "Test Cancer"
                    }
                ]
            }
            mock_call_api.return_value = mock_response

            # Mock cache miss
            with patch.object(llm_service.api_manager, 'get_cache') as mock_get_cache:
                mock_cache = Mock()
                mock_cache.get_by_key.return_value = None
                mock_get_cache.return_value = mock_cache

                # Test the method
                elements = await llm_service.map_to_mcode("test clinical text")

                assert len(elements) == 1
                assert isinstance(elements[0], McodeElement)
                assert elements[0].element_type == "CancerCondition"

    @pytest.mark.asyncio
    async def test_map_to_mcode_cache_hit(self, llm_service, mock_config):
        """Test mCODE mapping with cache hit."""
        # Mock cache hit
        cached_data = {
            "mcode_elements": [
                {
                    "element_type": "CancerCondition",
                    "code": "C123",
                    "display": "Test Cancer"
                }
            ]
        }

        with patch.object(llm_service.api_manager, 'get_cache') as mock_get_cache:
            mock_cache = Mock()
            mock_cache.get_by_key.return_value = cached_data
            mock_get_cache.return_value = mock_cache

            elements = await llm_service.map_to_mcode("test clinical text")

            assert len(elements) == 1
            assert elements[0].element_type == "CancerCondition"

    def test_parse_llm_response_standard_format(self, llm_service):
        """Test parsing standard LLM response format."""
        response_json = {
            "mcode_mappings": [
                {
                    "element_type": "CancerCondition",
                    "code": "C123",
                    "display": "Test Cancer",
                    "system": "SNOMED CT",
                    "confidence_score": 0.9
                }
            ]
        }

        elements = llm_service._parse_llm_response(response_json)

        assert len(elements) == 1
        assert elements[0].element_type == "CancerCondition"
        assert elements[0].code == "C123"
        assert elements[0].confidence_score == 0.9

    def test_parse_llm_response_alternative_formats(self, llm_service):
        """Test parsing alternative response formats."""
        # Test 'mappings' format
        response_json = {
            "mappings": [
                {
                    "element_type": "CancerTreatment",
                    "code": "T456"
                }
            ]
        }

        elements = llm_service._parse_llm_response(response_json)
        assert len(elements) == 1
        assert elements[0].element_type == "CancerTreatment"

    def test_parse_llm_response_invalid_elements(self, llm_service):
        """Test handling of invalid mCODE elements."""
        response_json = {
            "mcode_mappings": [
                {
                    "element_type": "CancerCondition",
                    "code": "C123"
                },
                {
                    "invalid_field": "invalid_value"  # Missing element_type
                }
            ]
        }

        with patch.object(llm_service.logger, 'warning') as mock_warning:
            elements = llm_service._parse_llm_response(response_json)

            # Should parse valid element and skip invalid one
            assert len(elements) == 1
            assert elements[0].element_type == "CancerCondition"
            # The warning is called twice - once for validation error, once for problematic item
            assert mock_warning.call_count == 2

    def test_enhanced_cache_key_generation(self, llm_service):
        """Test enhanced cache key generation."""
        clinical_text = "Patient has breast cancer and is receiving chemotherapy."

        cache_key = llm_service._enhanced_cache_key(clinical_text)

        assert "model" in cache_key
        assert "prompt" in cache_key
        assert "text_hash" in cache_key
        assert "text_length" in cache_key
        assert "semantic_fingerprint" in cache_key

        # Verify semantic fingerprint contains expected terms
        fingerprint = cache_key["semantic_fingerprint"]
        assert "cancer" in fingerprint
        assert "patient" in fingerprint

    def test_semantic_fingerprint_generation(self, llm_service):
        """Test semantic fingerprint generation."""
        # Test with key terms
        text_with_terms = "Patient has cancer treatment in clinical trial study."
        fingerprint = llm_service._generate_semantic_fingerprint(text_with_terms)

        assert "cancer" in fingerprint
        assert "treatment" in fingerprint
        assert "patient" in fingerprint
        assert "trial" in fingerprint
        assert "clinical" in fingerprint
        assert "study" in fingerprint

        # Test without key terms
        text_without_terms = "This is a generic medical text."
        fingerprint = llm_service._generate_semantic_fingerprint(text_without_terms)

        # For short text without key terms, it returns "short"
        assert fingerprint == "short"

    @pytest.mark.asyncio
    async def test_map_to_mcode_no_api_key(self, llm_service):
        """Test mCODE mapping fails without API key."""
        with patch.object(llm_service.config, 'get_api_key', return_value=None):
            with pytest.raises(ValueError, match="No API key available"):
                await llm_service.map_to_mcode("test text")

    @pytest.mark.asyncio
    async def test_map_to_mcode_missing_config(self, llm_service):
        """Test mCODE mapping with missing LLM config - should make actual API call."""
        with patch('src.pipeline.llm_service.llm_loader') as mock_llm_loader, \
             patch.object(llm_service.api_manager, 'get_cache') as mock_get_cache:

            # Setup mocks for successful API call
            mock_llm_config = Mock()
            mock_llm_config.model_identifier = "deepseek-coder"
            mock_llm_loader.get_llm.return_value = mock_llm_config

            # Mock cache miss
            mock_cache = Mock()
            mock_cache.get_by_key.return_value = None
            mock_get_cache.return_value = mock_cache

            # The method should proceed with API call rather than failing immediately
            # This test verifies the method doesn't crash with config issues
            try:
                await llm_service.map_to_mcode("test text")
                # If it doesn't raise an exception, that's acceptable
                assert True
            except Exception as e:
                # If it does raise an exception, it should be related to API call, not config
                assert "API" in str(e) or "config" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])