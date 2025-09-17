"""
Unit tests for LLM Service JSON parsing and response handling.
Tests all the deepseek-specific fixes and fail-fast behavior.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.pipeline.llm_service import LLMService
from src.utils.config import Config
from src.shared.models import McodeElement


class TestLLMServiceJSONParsing:
    """Test JSON parsing fixes for LLM service."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance for testing."""
        config = Config()
        return LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        with patch('openai.OpenAI') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Mock successful response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = '{"mcode_mappings": []}'
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150

            mock_instance.chat.completions.create.return_value = mock_response
            yield mock_instance

    def test_successful_json_parsing(self, llm_service, mock_openai_client):
        """Test successful JSON parsing with valid response."""
        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        response_json = llm_service._call_llm_api("test prompt", mock_config)
        assert isinstance(response_json, dict)
        assert "mcode_mappings" in response_json

    def test_deepseek_response_format_enabled(self, llm_service, mock_openai_client):
        """Test that response_format is enabled for deepseek models."""
        llm_config = Mock()
        llm_config.model_identifier = "deepseek-coder"

        llm_service._call_llm_api("test prompt", llm_config)

        # Check that response_format was passed
        call_args = mock_openai_client.chat.completions.create.call_args
        assert "response_format" in call_args[1]
        assert call_args[1]["response_format"]["type"] == "json_object"

    def test_deepseek_markdown_json_parsing(self, llm_service, mock_openai_client):
        """Test parsing JSON wrapped in markdown code blocks."""
        # Mock response with markdown-wrapped JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''```json
{
  "mcode_mappings": [
    {
      "element_type": "CancerCondition",
      "code": "C123",
      "display": "Test Cancer"
    }
  ]
}
```'''
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        response_json = llm_service._call_llm_api("test prompt", mock_config)
        assert isinstance(response_json, dict)
        assert "mcode_mappings" in response_json
        assert len(response_json["mcode_mappings"]) == 1

    def test_deepseek_truncated_json_fails(self, llm_service, mock_openai_client):
        """Test that truncated JSON responses raise errors."""
        # Mock response with truncated JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"mcode_mappings": [{"element_type": "CancerCondition"'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        with pytest.raises(ValueError, match="truncated JSON"):
            llm_service._call_llm_api("test prompt", mock_config)

    def test_deepseek_mismatched_braces_fails(self, llm_service, mock_openai_client):
        """Test that JSON with mismatched braces raises errors."""
        # Mock response with mismatched braces
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"mcode_mappings": [{"element_type": "CancerCondition"}]}}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        with pytest.raises(ValueError, match="mismatched braces"):
            llm_service._call_llm_api("test prompt", mock_config)

    def test_deepseek_mismatched_brackets_fails(self, llm_service, mock_openai_client):
        """Test that JSON with mismatched brackets raises errors."""
        # Mock response with mismatched brackets
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"mcode_mappings": ["item1", "item2"}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        with pytest.raises(ValueError, match="mismatched brackets"):
            llm_service._call_llm_api("test prompt", mock_config)

    def test_deepseek_malformed_markdown_fails(self, llm_service, mock_openai_client):
        """Test that malformed markdown JSON blocks raise errors."""
        # Mock response with malformed markdown
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''```json
{
  "mcode_mappings": [
'''
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        with pytest.raises(ValueError, match="malformed markdown JSON block"):
            llm_service._call_llm_api("test prompt", mock_config)

    def test_deepseek_trailing_comma_cleanup_warning(self, llm_service, mock_openai_client):
        """Test that trailing comma cleanup generates warnings."""
        # Mock response with trailing comma
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"mcode_mappings": [],}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        with patch.object(llm_service.logger, 'warning') as mock_warning:
            response_json = llm_service._call_llm_api("test prompt", mock_config)
            assert isinstance(response_json, dict)
            mock_warning.assert_called_once()
            assert "trailing commas" in mock_warning.call_args[0][0]

    def test_deepseek_empty_response_fails(self, llm_service, mock_openai_client):
        """Test that empty responses raise errors."""
        # Mock empty response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = ""
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        with pytest.raises(ValueError, match="Empty LLM response"):
            llm_service._call_llm_api("test prompt", mock_config)

    def test_deepseek_invalid_json_fails(self, llm_service, mock_openai_client):
        """Test that invalid JSON raises appropriate errors."""
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"invalid": json}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        mock_openai_client.chat.completions.create.return_value = mock_response

        mock_config = Mock()
        mock_config.model_identifier = "deepseek-coder"
        with pytest.raises(ValueError, match="Invalid JSON response"):
            llm_service._call_llm_api("test prompt", mock_config)

    def test_deepseek_response_element_parsing(self, llm_service):
        """Test parsing of mCODE elements from deepseek response."""
        response_json = {
            "mcode_mappings": [
                {
                    "element_type": "CancerCondition",
                    "code": "C123",
                    "display": "Test Cancer",
                    "system": "SNOMED CT",
                    "confidence_score": 0.9,
                    "evidence_text": "Test evidence"
                }
            ]
        }

        elements = llm_service._parse_llm_response(response_json)

        assert len(elements) == 1
        assert isinstance(elements[0], McodeElement)
        assert elements[0].element_type == "CancerCondition"
        assert elements[0].code == "C123"

    def test_deepseek_response_alternative_format(self, llm_service):
        """Test parsing alternative response format (mappings vs mcode_mappings)."""
        response_json = {
            "mappings": [
                {
                    "element_type": "CancerTreatment",
                    "code": "T456",
                    "display": "Test Treatment"
                }
            ]
        }

        elements = llm_service._parse_llm_response(response_json)

        assert len(elements) == 1
        assert elements[0].element_type == "CancerTreatment"

    def test_deepseek_invalid_element_handling(self, llm_service):
        """Test handling of invalid mCODE elements."""
        response_json = {
            "mcode_mappings": [
                {
                    "element_type": "CancerCondition",
                    "code": "C123",
                    "display": "Test Cancer"
                },
                {
                    "invalid_field": "invalid_value"
                }
            ]
        }

        with patch.object(llm_service.logger, 'warning') as mock_warning:
            elements = llm_service._parse_llm_response(response_json)

            # Should have parsed the valid element and skipped the invalid one
            assert len(elements) == 1
            assert elements[0].element_type == "CancerCondition"
            mock_warning.assert_called_once()

    def test_non_deepseek_model_no_special_handling(self):
        """Test that non-deepseek models don't get special JSON handling."""
        config = Config()
        llm_service = LLMService(config, "gpt-4", "direct_mcode_evidence_based_concise")

        # This should work without deepseek-specific parsing
        response_json = {
            "mcode_mappings": [
                {
                    "element_type": "CancerCondition",
                    "code": "C123",
                    "display": "Test Cancer"
                }
            ]
        }

        elements = llm_service._parse_llm_response(response_json)
        assert len(elements) == 1


if __name__ == "__main__":
    pytest.main([__file__])