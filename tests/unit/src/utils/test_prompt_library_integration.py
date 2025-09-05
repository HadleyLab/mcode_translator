"""
Pytest tests for prompt library integration functionality.
Replaces examples/test_prompt_library_integration.py
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.optimization.prompt_optimization_framework import (
    PromptOptimizationFramework, PromptVariant, PromptType, APIConfig
)
from src.utils.prompt_loader import prompt_loader


class TestPromptLibraryIntegration:
    """Test prompt library integration with strict optimization framework"""

    def test_prompt_library_loading(self):
        """Test that prompt library keys can be loaded successfully"""
        # Test loading some known prompt keys
        test_keys = [
            "generic_extraction",
            "generic_mapping",
        ]
        
        for key in test_keys:
            try:
                prompt_content = prompt_loader.get_prompt(key)
                assert len(prompt_content) > 0
                assert isinstance(prompt_content, str)
            except Exception as e:
                pytest.fail(f"Failed to load prompt '{key}': {e}")

    def test_prompt_variant_validation(self):
        """Test that prompt variants validate against prompt library"""
        framework = PromptOptimizationFramework()
        
        # Test valid prompt variant
        valid_variant = PromptVariant(
            name="Test Extraction Variant",
            prompt_type=PromptType.NLP_EXTRACTION,
            prompt_key="generic_extraction",
            description="Test variant using generic extraction prompt"
        )
        
        # Should not raise an exception
        framework.add_prompt_variant(valid_variant)
        
        # Test invalid prompt variant
        invalid_variant = PromptVariant(
            name="Invalid Variant",
            prompt_type=PromptType.NLP_EXTRACTION,
            prompt_key="nonexistent_prompt_key",
            description="This should fail validation"
        )
        
        # Should raise an exception
        with pytest.raises(Exception):
            framework.add_prompt_variant(invalid_variant)

    def test_api_config_validation(self):
        """Test API configuration validation"""
        framework = PromptOptimizationFramework()
        
        # Test valid API config
        valid_config = APIConfig(
            name="test_config",
            base_url="https://api.example.com",
            api_key="valid-api-key-123",  # Realistic API key format
            model="deepseek-coder"
        )
        
        # Should not raise an exception
        framework.add_api_config(valid_config)
        
        # Test invalid API config (placeholder key)
        invalid_config = APIConfig(
            name="invalid_config",
            base_url="https://api.example.com",
            api_key="your-api-key-here",  # Placeholder should be rejected
            model="deepseek-coder"
        )
        
        # Should raise an exception
        with pytest.raises(Exception):
            framework.add_api_config(invalid_config)

    def test_prompt_content_generation(self):
        """Test that prompt variants can generate content with formatting"""
        # Create a test variant
        variant = PromptVariant(
            name="Formatted Extraction",
            prompt_type=PromptType.NLP_EXTRACTION,
            prompt_key="generic_extraction",
            description="Test formatted prompt generation"
        )
        
        # Test formatting with clinical text
        test_clinical_text = "Patient presents with fever, cough, and shortness of breath."
        
        prompt_content = variant.get_prompt_content(clinical_text=test_clinical_text)
        
        assert len(prompt_content) > 0
        assert isinstance(prompt_content, str)
        assert test_clinical_text in prompt_content

    def test_framework_integration(self):
        """Test full framework integration with prompt library"""
        framework = PromptOptimizationFramework()
        
        # Create a minimal test case
        framework.add_test_case("test_case_1", {
            "clinical_text": "45-year-old male with hypertension and diabetes",
            "expected_entities": ["hypertension", "diabetes"]
        })
        
        # Add prompt variants using prompt library keys
        prompt_variants = [
            {
                "name": "Generic Extraction",
                "prompt_type": "nlp_extraction",
                "prompt_key": "generic_extraction",
                "description": "Standard extraction prompt"
            }
        ]
        
        for variant_data in prompt_variants:
            variant = PromptVariant.from_dict(variant_data)
            framework.add_prompt_variant(variant)
            assert variant.name in [v.name for v in framework.prompt_variants.values()]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])