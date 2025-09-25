"""
Unit tests for LLMAPICaller class.
"""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.pipeline.llm_api_caller import LLMAPICaller


class TestLLMAPICaller:
    """Test cases for LLMAPICaller class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.get_api_key.return_value = "test-api-key"
        config.get_base_url.return_value = "https://api.test.com"
        config.get_temperature.return_value = 0.7
        config.get_max_tokens.return_value = 1000
        config.get_timeout.return_value = 30.0
        return config

    @pytest.fixture
    def mock_llm_config(self):
        """Create mock LLM config."""
        config = MagicMock()
        config.model_identifier = "test-model"
        return config

    @pytest.fixture
    def api_caller(self, mock_config):
        """Create LLMAPICaller instance."""
        return LLMAPICaller(mock_config, "test-model")

    def test_init(self, mock_config):
        """Test initialization."""
        caller = LLMAPICaller(mock_config, "test-model")
        assert caller.config == mock_config
        assert caller.model_name == "test-model"
        assert hasattr(caller, 'logger')

    @pytest.mark.asyncio
    @patch('src.pipeline.llm_api_caller.extract_token_usage_from_response')
    @patch('src.pipeline.llm_api_caller.global_token_tracker')
    @patch('openai.AsyncOpenAI')
    async def test_call_llm_api_async_success(self, mock_openai_class, mock_token_tracker, mock_extract_tokens, api_caller, mock_llm_config):
        """Test successful async LLM API call."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"test": "response"}'
        mock_client.chat.completions.create.return_value = mock_response

        mock_extract_tokens.return_value = {"tokens": 100}

        # Test
        result = await api_caller.call_llm_api_async("test prompt", mock_llm_config)

        # Assertions
        assert result == {"test": "response"}
        mock_openai_class.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()
        mock_extract_tokens.assert_called_once_with(mock_response, "test-model", "provider")
        mock_token_tracker.add_usage.assert_called_once_with({"tokens": 100}, "llm_service")

    @pytest.mark.asyncio
    async def test_call_llm_api_async_no_api_key(self, api_caller, mock_llm_config):
        """Test API call with no API key."""
        api_caller.config.get_api_key.return_value = None

        with pytest.raises(ValueError, match="API key not configured"):
            await api_caller.call_llm_api_async("test prompt", mock_llm_config)

    @pytest.mark.asyncio
    async def test_call_llm_api_async_config_error(self, api_caller, mock_llm_config):
        """Test API call with config error."""
        api_caller.config.get_api_key.side_effect = Exception("not found in config")

        with pytest.raises(ValueError, match="API key not configured"):
            await api_caller.call_llm_api_async("test prompt", mock_llm_config)

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_call_llm_api_async_empty_response(self, mock_openai_class, api_caller, mock_llm_config):
        """Test API call with empty response."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="Empty LLM response"):
            await api_caller.call_llm_api_async("test prompt", mock_llm_config)

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_call_llm_api_async_general_exception(self, mock_openai_class, api_caller, mock_llm_config):
        """Test API call with general exception."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            await api_caller.call_llm_api_async("test prompt", mock_llm_config)

    def test_parse_and_clean_response_valid_json(self, api_caller):
        """Test parsing valid JSON response."""
        response = '{"key": "value", "number": 42}'
        result = api_caller._parse_and_clean_response(response)
        assert result == {"key": "value", "number": 42}

    def test_parse_and_clean_response_markdown_json(self, api_caller):
        """Test parsing markdown-wrapped JSON."""
        response = '```json\n{"key": "value"}\n```'
        result = api_caller._parse_and_clean_response(response)
        assert result == {"key": "value"}

    def test_parse_and_clean_response_deepseek_markdown(self, api_caller):
        """Test parsing DeepSeek markdown JSON."""
        api_caller.model_name = "deepseek-coder"
        response = '```json\n{"key": "value"}\n```'
        result = api_caller._parse_and_clean_response(response)
        assert result == {"key": "value"}

    def test_parse_and_clean_response_deepseek_truncated_json(self, api_caller):
        """Test parsing DeepSeek truncated JSON."""
        api_caller.model_name = "deepseek-coder"
        response = '{"key": "value"'  # Missing closing brace

        with pytest.raises(ValueError, match="truncated JSON"):
            api_caller._parse_and_clean_response(response)

    def test_parse_and_clean_response_deepseek_mismatched_braces(self, api_caller):
        """Test parsing DeepSeek JSON with mismatched braces."""
        api_caller.model_name = "deepseek-coder"
        response = '{"key": "value", "nested": {"incomplete": "json"}'  # Missing closing brace

        with pytest.raises(ValueError, match="mismatched braces"):
            api_caller._parse_and_clean_response(response)

    def test_parse_and_clean_response_deepseek_trailing_comma(self, api_caller):
        """Test parsing DeepSeek JSON with trailing comma."""
        api_caller.model_name = "deepseek-coder"
        response = '{"key": "value",}'
        result = api_caller._parse_and_clean_response(response)
        assert result == {"key": "value"}

    def test_parse_and_clean_response_deepseek_reasoner_with_reasoning(self, api_caller):
        """Test parsing DeepSeek reasoner response with reasoning content."""
        api_caller.model_name = "deepseek-reasoner"
        response = 'Let me think about this. The task requires analyzing the data. {"key": "value"}'
        result = api_caller._parse_and_clean_response(response)
        assert result == {"key": "value"}

    def test_parse_and_clean_response_invalid_json(self, api_caller):
        """Test parsing invalid JSON."""
        response = '{"key": "value", invalid}'

        with pytest.raises(ValueError, match="invalid JSON"):
            api_caller._parse_and_clean_response(response)

    def test_parse_and_clean_response_plain_text(self, api_caller):
        """Test parsing plain text response."""
        response = "This is plain text, not JSON"

        with pytest.raises(ValueError, match="returned invalid JSON"):
            api_caller._parse_and_clean_response(response)

    def test_parse_and_clean_response_deepseek_malformed_markdown(self, api_caller):
        """Test parsing DeepSeek response with malformed markdown JSON block."""
        api_caller.model_name = "deepseek-coder"
        response = "```json\nincomplete json block"

        with pytest.raises(ValueError, match="malformed markdown JSON block"):
            api_caller._parse_and_clean_response(response)

    def test_parse_and_clean_response_deepseek_truncated_bracket(self, api_caller):
        """Test parsing DeepSeek JSON with truncated bracket."""
        api_caller.model_name = "deepseek-coder"
        response = '["item1", "item2"'  # Missing closing bracket

        with pytest.raises(ValueError, match="truncated JSON.*missing closing bracket"):
            api_caller._parse_and_clean_response(response)

    def test_parse_and_clean_response_deepseek_mismatched_brackets(self, api_caller):
        """Test parsing DeepSeek JSON with mismatched brackets."""
        api_caller.model_name = "deepseek-coder"
        response = '{"key": "value", "array": ["item1", "item2"}'  # Missing closing bracket for array

        with pytest.raises(ValueError, match="mismatched brackets"):
            api_caller._parse_and_clean_response(response)

    def test_parse_and_clean_response_markdown_cleanup(self, api_caller):
        """Test parsing response with markdown code blocks that need cleanup."""
        response = "```\n{\"key\": \"value\"}\n```"

        result = api_caller._parse_and_clean_response(response)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    @patch('src.pipeline.llm_api_caller.LLMAPICaller._parse_and_clean_response')
    async def test_call_llm_api_async_unexpected_config_exception(self, mock_parse, api_caller, mock_llm_config):
        """Test API call with unexpected config exception."""
        # Mock the config to raise an unexpected exception
        api_caller.config.get_api_key.side_effect = RuntimeError("unexpected error")

        with pytest.raises(RuntimeError, match="unexpected error"):
            await api_caller.call_llm_api_async("test prompt", mock_llm_config)

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_with_rate_limiting_success(self, mock_openai_class, api_caller, mock_llm_config):
        """Test successful async API call with rate limiting."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        assert result == mock_response
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_with_rate_limiting_quota_error(self, mock_openai_class, api_caller, mock_llm_config):
        """Test API call with quota error."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.side_effect = Exception("insufficient_quota")

        with pytest.raises(ValueError, match="exceeded its quota"):
            await api_caller._make_async_api_call_with_rate_limiting(
                mock_client, mock_llm_config, "test prompt", 0.7, 1000
            )

    @pytest.mark.asyncio
    @patch('asyncio.sleep')
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_with_rate_limiting_retry_success(self, mock_openai_class, mock_sleep, api_caller, mock_llm_config):
        """Test API call with rate limiting that succeeds on retry."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # First call fails with rate limit, second succeeds
        mock_response = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            Exception("rate limit exceeded"),
            mock_response
        ]

        result = await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    @patch('asyncio.sleep')
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_with_rate_limiting_max_retries_exceeded(self, mock_openai_class, mock_sleep, api_caller, mock_llm_config):
        """Test API call with rate limiting that exhausts all retries."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # All calls fail with rate limit
        mock_client.chat.completions.create.side_effect = Exception("rate limit exceeded")

        with pytest.raises(Exception, match="rate limit exceeded"):
            await api_caller._make_async_api_call_with_rate_limiting(
                mock_client, mock_llm_config, "test prompt", 0.7, 1000
            )

        # Should have tried max_retries + 1 times (11 times)
        assert mock_client.chat.completions.create.call_count == 11
        assert mock_sleep.call_count == 10  # 10 retries

    @pytest.mark.asyncio
    @patch('asyncio.sleep')
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_with_rate_limiting_api_error_retry(self, mock_openai_class, mock_sleep, api_caller, mock_llm_config):
        """Test API call with general API error that retries."""
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            Exception("connection timeout"),
            mock_response
        ]

        result = await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_response_format_gpt4(self, mock_openai_class, api_caller):
        """Test that response_format is used for GPT-4 models."""
        mock_llm_config = MagicMock()
        mock_llm_config.model_identifier = "gpt-4o"

        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        call_args = mock_client.chat.completions.create.call_args
        assert "response_format" in call_args[1]
        assert call_args[1]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_response_format_deepseek(self, mock_openai_class, api_caller):
        """Test that response_format is used for DeepSeek models."""
        mock_llm_config = MagicMock()
        mock_llm_config.model_identifier = "deepseek-coder"

        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        call_args = mock_client.chat.completions.create.call_args
        assert "response_format" in call_args[1]
        assert call_args[1]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_no_response_format(self, mock_openai_class, api_caller):
        """Test that response_format is not used for unsupported models."""
        mock_llm_config = MagicMock()
        mock_llm_config.model_identifier = "unsupported-model"

        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock()

        await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        call_args = mock_client.chat.completions.create.call_args
        assert "response_format" not in call_args[1]

    @pytest.mark.asyncio
    @patch('random.uniform')
    @patch('asyncio.sleep')
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_retry_after_parsing(self, mock_openai_class, mock_sleep, mock_uniform, api_caller, mock_llm_config):
        """Test parsing of retry-after time from rate limit error."""
        mock_uniform.return_value = 0.5  # Fixed jitter value
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Create an exception with retry-after information
        error = Exception("rate limit exceeded")
        error.body = {"error": {"type": "rate_limit", "message": "Please try again in 5s"}}

        mock_response = MagicMock()
        mock_client.chat.completions.create.side_effect = [error, mock_response]

        await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        # Should have used the parsed retry time plus jitter
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert sleep_time == 5.25  # 5s + (0.5 * 5 * 0.1) jitter

    @pytest.mark.asyncio
    @patch('random.uniform')
    @patch('asyncio.sleep')
    @patch('openai.AsyncOpenAI')
    async def test_make_async_api_call_retry_after_ms_parsing(self, mock_openai_class, mock_sleep, mock_uniform, api_caller, mock_llm_config):
        """Test parsing of retry-after time in milliseconds."""
        mock_uniform.return_value = 0.5  # Fixed jitter value
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        # Create an exception with retry-after in milliseconds
        error = Exception("rate limit exceeded")
        error.body = {"error": {"type": "rate_limit", "message": "Please try again in 2000ms"}}

        mock_response = MagicMock()
        mock_client.chat.completions.create.side_effect = [error, mock_response]

        await api_caller._make_async_api_call_with_rate_limiting(
            mock_client, mock_llm_config, "test prompt", 0.7, 1000
        )

        # Should have converted ms to seconds plus jitter
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert sleep_time == 2.1  # 2s + (0.5 * 2 * 0.1) jitter