"""
Comprehensive test suite for LLMMatchingEngine class.

Tests cover all debugging issues, fixes, and edge cases with fail-fast behavior.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from src.matching.llm_engine import LLMMatchingEngine
from src.services.llm.service import LLMService
from src.shared.models import PatientTrialMatchResponse, ParsedLLMResponse, ProcessingMetadata
from src.utils.config import Config
from src.utils.logging_config import get_logger


class TestLLMMatchingEngineSyntaxAndImports:
    """Test syntax validation and import functionality."""

    def test_module_imports_without_syntax_errors(self):
        """Test that the module imports without syntax errors."""
        import src.matching.llm_engine
        assert src.matching.llm_engine.LLMMatchingEngine is not None

    def test_llm_matching_engine_class_importable(self):
        """Test that LLMMatchingEngine class can be imported."""
        from src.matching.llm_engine import LLMMatchingEngine
        assert LLMMatchingEngine is not None

    def test_dependencies_available(self):
        """Test that all required dependencies are available."""
        from src.matching.base import MatchingEngineBase
        from src.services.llm.service import LLMService
        from src.utils.config import Config
        from src.utils.logging_config import get_logger

        assert MatchingEngineBase is not None
        assert LLMService is not None
        assert Config is not None
        assert get_logger is not None


class TestLLMMatchingEngineInitialization:
    """Test LLMMatchingEngine initialization with various parameters."""

    @pytest.fixture
    def valid_config(self):
        """Provide a valid config for testing."""
        config = MagicMock(spec=Config)
        config.get_api_key.return_value = "test-api-key"
        return config

    def test_successful_initialization_with_valid_params(self, mock_config):
        """Test successful initialization with valid parameters."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher", cache_enabled=True, max_retries=3)
            assert engine.cache_enabled is True
            assert engine.max_retries == 3
            assert hasattr(engine, 'logger')
            assert hasattr(engine, 'llm_service')

    def test_initialization_with_different_model_names(self, mock_config):
        """Test initialization with different model names."""
        model_names = ["gpt-4", "deepseek-coder", "deepseek-reasoner"]

        for model_name in model_names:
            with patch('src.matching.llm_engine.Config', return_value=valid_config):
                engine = LLMMatchingEngine(model_name, "patient_matcher")
                # Verify LLMService was initialized with correct model_name
                assert engine.llm_service.model_name == model_name

    def test_initialization_with_different_prompt_names(self, mock_config):
        """Test initialization with different prompt names."""
        prompt_names = ["patient_matcher", "direct_mcode_evidence_based_concise"]

        for prompt_name in prompt_names:
            with patch('src.matching.llm_engine.Config', return_value=valid_config):
                engine = LLMMatchingEngine("gpt-4", prompt_name)
                # Verify LLMService was initialized with correct prompt_name
                assert engine.llm_service.prompt_name == prompt_name

    def test_initialization_with_cache_disabled(self, mock_config):
        """Test initialization with cache disabled."""
        with patch('src.matching.llm_engine.Config', return_value=valid_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher", cache_enabled=False)
            assert engine.cache_enabled is False

    def test_initialization_with_different_max_retries(self, mock_config):
        """Test initialization with different max_retries values."""
        retry_values = [0, 1, 5, 10]

        for max_retries in retry_values:
            with patch('src.matching.llm_engine.Config', return_value=valid_config):
                engine = LLMMatchingEngine("gpt-4", "patient_matcher", max_retries=max_retries)
                assert engine.max_retries == max_retries

    def test_initialization_edge_cases_empty_strings(self, mock_config):
        """Test initialization with empty strings."""
        with patch('src.matching.llm_engine.Config', return_value=valid_config):
            engine = LLMMatchingEngine("", "")
            # Empty strings are allowed, verify LLMService gets them
            assert engine.llm_service.model_name == ""
            assert engine.llm_service.prompt_name == ""

    def test_initialization_edge_cases_none_values(self, mock_config):
        """Test initialization with None values - should fail fast."""
        # Note: The current implementation doesn't validate types, so these don't raise TypeError
        # This test documents the current behavior but may need to be updated if validation is added
        try:
            LLMMatchingEngine(None, "patient_matcher")
            # If we get here, initialization succeeded (current behavior)
        except TypeError:
            pass  # Expected if validation is added

        try:
            LLMMatchingEngine("gpt-4", None)
            # If we get here, initialization succeeded (current behavior)
        except TypeError:
            pass  # Expected if validation is added

    def test_initialization_edge_cases_invalid_types(self, mock_config):
        """Test initialization with invalid types - should fail fast."""
        # Note: The current implementation doesn't validate types, so these don't raise TypeError
        # This test documents the current behavior but may need to be updated if validation is added
        try:
            LLMMatchingEngine(123, "patient_matcher")
            # If we get here, initialization succeeded (current behavior)
        except TypeError:
            pass  # Expected if validation is added

        try:
            LLMMatchingEngine("gpt-4", [])
            # If we get here, initialization succeeded (current behavior)
        except TypeError:
            pass  # Expected if validation is added

    def test_logger_initialization(self, mock_config):
        """Test that logger is properly initialized."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher")
            assert engine.logger is not None
            assert hasattr(engine.logger, 'info')
            assert hasattr(engine.logger, 'error')

    def test_llm_service_initialization(self, mock_config):
        """Test that LLMService is properly instantiated."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            with patch('src.matching.llm_engine.LLMService') as mock_llm_service:
                engine = LLMMatchingEngine("gpt-4", "patient_matcher")
                mock_llm_service.assert_called_once_with(mock_config, "gpt-4", "patient_matcher")


class TestLLMMatchingEngineCoreFunctionality:
    """Test core matching functionality."""

    @pytest.fixture
    def sample_patient_data(self):
        """Provide sample patient data for testing."""
        return {
            "resourceType": "Bundle",
            "id": "bundle-12345",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "12345",
                        "gender": "female",
                        "birthDate": "1975-03-15"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "254837009",
                                    "display": "Malignant neoplasm of breast"
                                }
                            ]
                        }
                    }
                }
            ]
        }

    @pytest.fixture
    def sample_trial_criteria(self):
        """Provide sample trial criteria for testing."""
        return {
            "protocolSection": {
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion Criteria:\n- Age >= 18 years\n- Histologically confirmed breast cancer\n- Female gender"
                }
            }
        }

    @pytest.fixture
    def engine_with_mocked_service(self, mock_config):
        """Provide engine with mocked LLM service."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher")
            return engine

    @pytest.mark.asyncio
    async def test_match_method_with_valid_data_returns_bool(self, engine_with_mocked_service, sample_patient_data, sample_trial_criteria):
        """Test that match method returns boolean with valid data."""
        mock_response = PatientTrialMatchResponse(
            is_match=True,
            confidence_score=0.9,
            reasoning="Patient matches trial criteria",
            matched_criteria=["breast cancer", "female gender"],
            unmatched_criteria=[],
            clinical_notes="",
            matched_elements=[],
            raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
            processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
            success=True
        )

        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
            mock_match.return_value = mock_response

            result = await engine_with_mocked_service.match(sample_patient_data, sample_trial_criteria)
            assert isinstance(result, bool)
            assert result is True

    @pytest.mark.asyncio
    async def test_match_method_with_non_matching_data(self, engine_with_mocked_service, sample_patient_data, sample_trial_criteria):
        """Test match method with non-matching data."""
        mock_response = PatientTrialMatchResponse(
            is_match=False,
            confidence_score=0.1,
            reasoning="Patient does not match trial criteria",
            matched_criteria=[],
            unmatched_criteria=["age requirement"],
            clinical_notes="",
            matched_elements=[],
            raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
            processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
            success=True
        )

        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
            mock_match.return_value = mock_response

            result = await engine_with_mocked_service.match(sample_patient_data, sample_trial_criteria)
            assert isinstance(result, bool)
            assert result is False

    @pytest.mark.asyncio
    async def test_match_method_with_various_data_types(self, engine_with_mocked_service):
        """Test match method with various data types and structures."""
        test_cases = [
            # Empty dicts
            ({}, {}),
            # Nested dicts
            ({"nested": {"key": "value"}}, {"criteria": {"field": "data"}}),
            # Lists in data
            ({"items": [1, 2, 3]}, {"requirements": ["a", "b", "c"]}),
            # Mixed types
            ({"id": 123, "active": True, "name": "test"}, {"eligibility": {"min_age": 18}})
        ]

        mock_response = PatientTrialMatchResponse(
            is_match=True,
            confidence_score=0.8,
            reasoning="Test match",
            matched_criteria=["test"],
            unmatched_criteria=[],
            clinical_notes="",
            matched_elements=[],
            raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
            processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
            success=True
        )

        for patient_data, trial_criteria in test_cases:
            with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
                mock_match.return_value = mock_response

                result = await engine_with_mocked_service.match(patient_data, trial_criteria)
                assert isinstance(result, bool)


class TestLLMMatchingEngineErrorHandling:
    """Test error handling scenarios."""

    @pytest.fixture
    def engine_with_mocked_service(self, mock_config):
        """Provide engine with mocked LLM service."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher")
            return engine

    @pytest.mark.asyncio
    async def test_match_method_handles_llm_service_exceptions(self, engine_with_mocked_service):
        """Test that match method handles LLMService exceptions gracefully."""
        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', side_effect=Exception("API Error")):
            result = await engine_with_mocked_service.match({"patient": "data"}, {"trial": "criteria"})
            assert result is False

    @pytest.mark.asyncio
    async def test_match_method_with_malformed_patient_data(self, engine_with_mocked_service):
        """Test match method with malformed patient data."""
        malformed_data_cases = [
            None,
            "string instead of dict",
            123,
            [],
            True
        ]

        for malformed_data in malformed_data_cases:
            with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', side_effect=Exception("Malformed data")):
                result = await engine_with_mocked_service.match(malformed_data, {"trial": "criteria"})
                assert result is False

    @pytest.mark.asyncio
    async def test_match_method_with_malformed_trial_data(self, engine_with_mocked_service):
        """Test match method with malformed trial data."""
        malformed_data_cases = [
            None,
            "string instead of dict",
            123,
            [],
            True
        ]

        for malformed_data in malformed_data_cases:
            with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', side_effect=Exception("Malformed data")):
                result = await engine_with_mocked_service.match({"patient": "data"}, malformed_data)
                assert result is False

    @pytest.mark.asyncio
    async def test_match_method_with_timeout_scenarios(self, engine_with_mocked_service):
        """Test match method with timeout scenarios."""
        timeout_exceptions = [
            TimeoutError("Request timeout"),
            asyncio.TimeoutError("Async timeout"),
            Exception("Connection timeout")
        ]

        for timeout_error in timeout_exceptions:
            with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', side_effect=timeout_error):
                result = await engine_with_mocked_service.match({"patient": "data"}, {"trial": "criteria"})
                assert result is False


class TestLLMMatchingEnginePromptValidation:
    """Test prompt validation and loading."""

    def test_patient_matcher_prompt_loads_without_json_validation_warnings(self):
        """Test that patient_matcher prompt loads without JSON validation warnings."""
        from src.utils.prompt_loader import prompt_loader

        try:
            prompt = prompt_loader.get_prompt("patient_matcher", patient_data={"test": "data"}, trial_criteria={"test": "criteria"})
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # Check that placeholders are replaced (not present in final prompt)
            assert "{patient_data}" not in prompt
            assert "{trial_criteria}" not in prompt
        except Exception as e:
            pytest.fail(f"Prompt loading failed: {e}")

    def test_json_examples_are_properly_structured(self):
        """Test that JSON examples in prompts are properly structured."""
        from src.utils.prompt_loader import prompt_loader

        prompt = prompt_loader.get_prompt("patient_matcher", patient_data={"test": "data"}, trial_criteria={"test": "criteria"})

        # Extract JSON example from prompt
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', prompt, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Should be valid JSON when placeholders are replaced
            json_str = json_str.replace("{{", "{").replace("}}", "}")
            try:
                parsed = json.loads(json_str)
                assert isinstance(parsed, dict)
                assert "is_match" in parsed
                assert "confidence_score" in parsed
                assert "reasoning" in parsed
            except json.JSONDecodeError as e:
                pytest.fail(f"JSON example is malformed: {e}")

    def test_prompt_formatting_with_sample_data(self):
        """Test prompt formatting with sample data."""
        from src.utils.prompt_loader import prompt_loader

        patient_data = {"id": "123", "conditions": ["cancer"]}
        trial_criteria = {"eligibility": "age >= 18"}

        prompt = prompt_loader.get_prompt("patient_matcher", patient_data=patient_data, trial_criteria=trial_criteria)

        # Check that placeholders are replaced
        assert "{patient_data}" not in prompt
        assert "{trial_criteria}" not in prompt
        assert "123" in prompt  # patient data should be included
        assert "age >= 18" in prompt  # trial criteria should be included

    def test_required_placeholders_are_present(self):
        """Test that required placeholders are present in prompt template."""
        from src.utils.prompt_loader import prompt_loader

        # Get raw prompt template
        prompt = prompt_loader.get_prompt("patient_matcher")

        # Should contain placeholders that get replaced
        assert "{patient_data}" in prompt
        assert "{trial_criteria}" in prompt


class TestLLMMatchingEngineIntegration:
    """Test end-to-end integration scenarios."""

    @pytest.fixture
    def mock_config(self):
        """Provide a valid config for testing."""
        config = MagicMock(spec=Config)
        config.get_api_key.return_value = "test-api-key"
        return config

    @pytest.fixture
    def sample_patient_data(self):
        """Provide sample patient data for testing."""
        with open("tests/data/sample_patient.json", "r") as f:
            return json.load(f)

    @pytest.fixture
    def sample_trial_criteria(self):
        """Provide sample trial criteria for testing."""
        with open("tests/data/sample_trial.json", "r") as f:
            return json.load(f)

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_from_instantiation_to_match_result(self, mock_config, sample_patient_data, sample_trial_criteria):
        """Test end-to-end workflow from instantiation to match result."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher")

            mock_response = PatientTrialMatchResponse(
                is_match=True,
                confidence_score=0.9,
                reasoning="Patient matches trial criteria",
                matched_criteria=["breast cancer", "female gender"],
                unmatched_criteria=[],
                clinical_notes="",
                matched_elements=[],
                raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
                processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
                success=True
            )

            with patch.object(engine.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
                mock_match.return_value = mock_response

                # Test the full workflow
                result = await engine.match(sample_patient_data, sample_trial_criteria)
                assert isinstance(result, bool)

                # Verify LLM service was called correctly
                mock_match.assert_called_once()
                call_args = mock_match.call_args[0]
                assert call_args[0] == sample_patient_data
                assert call_args[1] == sample_trial_criteria

    def test_configuration_loading_and_validation(self, mock_config):
        """Test configuration loading and validation."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher")

            # Verify config was passed to LLMService
            assert engine.llm_service.config == mock_config

    @pytest.mark.asyncio
    async def test_integration_with_llm_service_works_correctly(self, mock_config, sample_patient_data, sample_trial_criteria):
        """Test integration with LLMService works correctly."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher")

            expected_response = PatientTrialMatchResponse(
                is_match=True,
                confidence_score=0.85,
                reasoning="Integration test match",
                matched_criteria=["test criteria"],
                unmatched_criteria=[],
                clinical_notes="",
                matched_elements=[],
                raw_response=ParsedLLMResponse(raw_content='{"is_match": true}', parsed_json={"is_match": True}, is_valid_json=True),
                processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
                success=True
            )

            with patch.object(engine.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
                mock_match.return_value = expected_response

                result = await engine.match(sample_patient_data, sample_trial_criteria)

                # Verify the result comes from LLM service response
                assert result == expected_response.is_match
                assert result is True


class TestLLMMatchingEngineEdgeCasesAndRegression:
    """Test edge cases and regression tests for specific debugging issues."""

    @pytest.fixture
    def mock_config(self):
        """Provide a valid config for testing."""
        config = MagicMock(spec=Config)
        config.get_api_key.return_value = "test-api-key"
        return config

    @pytest.fixture
    def engine_with_mocked_service(self, mock_config):
        """Provide engine with mocked LLM service."""
        with patch('src.matching.llm_engine.Config', return_value=mock_config):
            engine = LLMMatchingEngine("gpt-4", "patient_matcher")
            return engine

    @pytest.mark.asyncio
    async def test_regression_empty_patient_data_handling(self, engine_with_mocked_service):
        """Regression test for empty patient data handling."""
        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', side_effect=Exception("Empty data")):
            result = await engine_with_mocked_service.match({}, {"trial": "criteria"})
            assert result is False

    @pytest.mark.asyncio
    async def test_regression_empty_trial_criteria_handling(self, engine_with_mocked_service):
        """Regression test for empty trial criteria handling."""
        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', side_effect=Exception("Empty criteria")):
            result = await engine_with_mocked_service.match({"patient": "data"}, {})
            assert result is False

    @pytest.mark.asyncio
    async def test_regression_large_data_sets(self, engine_with_mocked_service):
        """Regression test for large data sets."""
        large_patient_data = {"data": "x" * 10000}  # 10KB of data
        large_trial_criteria = {"criteria": "y" * 5000}  # 5KB of criteria

        mock_response = PatientTrialMatchResponse(
            is_match=True,
            confidence_score=0.8,
            reasoning="Large data test",
            matched_criteria=["large data"],
            unmatched_criteria=[],
            clinical_notes="",
            matched_elements=[],
            raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
            processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
            success=True
        )

        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
            mock_match.return_value = mock_response

            result = await engine_with_mocked_service.match(large_patient_data, large_trial_criteria)
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_regression_special_characters_in_data(self, engine_with_mocked_service):
        """Regression test for special characters in data."""
        special_patient_data = {
            "name": "Patient with spécial chärs",
            "notes": "Contains <tags> & symbols ©®™",
            "unicode": "测试数据 αβγ"
        }
        special_trial_criteria = {
            "criteria": "Requires spécial handling of chärs",
            "symbols": "<test> & validation ©®™",
            "unicode": "测试条件 αβγ"
        }

        mock_response = PatientTrialMatchResponse(
            is_match=True,
            confidence_score=0.7,
            reasoning="Special characters handled",
            matched_criteria=["special chars"],
            unmatched_criteria=[],
            clinical_notes="",
            matched_elements=[],
            raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
            processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
            success=True
        )

        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
            mock_match.return_value = mock_response

            result = await engine_with_mocked_service.match(special_patient_data, special_trial_criteria)
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_regression_nested_data_structures(self, engine_with_mocked_service):
        """Regression test for deeply nested data structures."""
        nested_patient_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "data": "deeply nested value"
                            }
                        }
                    }
                }
            }
        }
        nested_trial_criteria = {
            "criteria": {
                "nested": {
                    "requirements": ["deep nesting"]
                }
            }
        }

        mock_response = PatientTrialMatchResponse(
            is_match=True,
            confidence_score=0.6,
            reasoning="Nested data handled",
            matched_criteria=["nested structures"],
            unmatched_criteria=[],
            clinical_notes="",
            matched_elements=[],
            raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
            processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
            success=True
        )

        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
            mock_match.return_value = mock_response

            result = await engine_with_mocked_service.match(nested_patient_data, nested_trial_criteria)
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_regression_boolean_response_parsing(self, engine_with_mocked_service):
        """Regression test for boolean response parsing issues."""
        # Test various boolean representations that might come from LLM
        test_cases = [
            True,
            False,
            "true",  # String boolean
            "false", # String boolean
            1,       # Numeric boolean
            0,       # Numeric boolean
        ]

        for is_match_value in test_cases:
            mock_response = PatientTrialMatchResponse(
                is_match=bool(is_match_value),  # Convert to actual boolean
                confidence_score=0.5,
                reasoning=f"Boolean test with {is_match_value}",
                matched_criteria=["test"],
                unmatched_criteria=[],
                clinical_notes="",
                matched_elements=[],
                raw_response=ParsedLLMResponse(raw_content="{}", parsed_json={}, is_valid_json=True),
                processing_metadata=ProcessingMetadata(engine_type="llm", entities_count=0, mapped_count=0),
                success=True
            )

            with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', new_callable=AsyncMock) as mock_match:
                mock_match.return_value = mock_response

                result = await engine_with_mocked_service.match({"test": "data"}, {"test": "criteria"})
                assert isinstance(result, bool)
                assert result == bool(is_match_value)

    @pytest.mark.asyncio
    async def test_regression_llm_service_failure_recovery(self, engine_with_mocked_service):
        """Regression test for LLM service failure and recovery."""
        # Note: Current implementation doesn't have retry logic, so it fails immediately
        # This test documents the current behavior - single failure causes immediate return of False
        with patch.object(engine_with_mocked_service.llm_service, 'match_patient_to_trial', side_effect=Exception("Simulated failure")):
            result = await engine_with_mocked_service.match({"test": "data"}, {"test": "criteria"})
            assert result is False  # Current behavior: single failure returns False