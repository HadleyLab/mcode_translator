"""
Pytest tests for combined benchmark + optimization functionality.
Replaces examples/test_combined_functionality.py
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.optimization.strict_prompt_optimization_framework import (
    StrictPromptOptimizationFramework, PromptVariant, APIConfig, PromptType
)


class TestCombinedFunctionality:
    """Test combined benchmark + optimization functionality"""

    def test_combined_benchmark_optimization(self):
        """Test that the combined benchmark + optimization works correctly"""
        # Create a simple test case with gold standard data
        test_case = {
            "clinical_data": {
                "text": "Patient has breast cancer and received chemotherapy treatment."
            },
            "expected_extraction": {
                "entities": [
                    {"text": "breast cancer", "type": "condition"},
                    {"text": "chemotherapy", "type": "treatment"}
                ]
            },
            "expected_mcode_mappings": {
                "mapped_elements": [
                    {"element_name": "CancerCondition", "value": "breast cancer"},
                    {"element_name": "CancerRelatedMedicationAdministered", "value": "chemotherapy"}
                ]
            }
        }
        
        # Create optimization framework
        framework = StrictPromptOptimizationFramework(results_dir="./test_results")
        
        # Add a test API config
        api_config = APIConfig(
            name="test_api",
            base_url="https://api.example.com",
            api_key="sk-real-looking-key-abcdef1234567890",
            model="test-model",
            temperature=0.2,
            max_tokens=1000,
            timeout=10
        )
        framework.add_api_config(api_config)
        
        # Add a prompt variant using an actual prompt that exists
        prompt_variant = PromptVariant(
            name="Test Extraction Prompt",
            prompt_type=PromptType.NLP_EXTRACTION,
            prompt_key="basic_extraction",
            description="Test prompt for combined functionality",
            version="1.0.0"
        )
        framework.add_prompt_variant(prompt_variant)
        
        # Add the test case
        framework.add_test_case("test_case_1", test_case)
        
        # Create a mock pipeline callback for testing
        def mock_pipeline_callback(test_data, prompt_content, prompt_variant_id):
            """Mock pipeline callback that returns test results"""
            class MockResult:
                def __init__(self):
                    self.extracted_entities = [
                        {"text": "breast cancer", "type": "condition"},
                        {"text": "chemotherapy", "type": "treatment"}
                    ]
                    self.mcode_mappings = [
                        {"element_name": "CancerCondition", "value": "breast cancer"},
                        {"element_name": "CancerRelatedMedicationAdministered", "value": "chemotherapy"}
                    ]
                    self.validation_results = {"compliance_score": 1.0}
            
            return MockResult()
        
        # Run the benchmark
        result = framework.run_benchmark(
            prompt_variant_id=prompt_variant.id,
            api_config_name="test_api",
            test_case_id="test_case_1",
            pipeline_callback=mock_pipeline_callback
        )
        
        # Verify the metrics are calculated correctly
        assert result.success is True
        assert result.precision > 0
        assert result.recall > 0
        assert result.f1_score > 0
        assert result.mapping_accuracy > 0
        
        # Verify metrics are not placeholder values
        assert result.precision != 0.8, "Should not use placeholder precision"
        assert result.recall != 0.7, "Should not use placeholder recall"
        assert result.f1_score != 0.75, "Should not use placeholder F1"
        
        # Clean up
        import shutil
        if Path("./test_results").exists():
            shutil.rmtree("./test_results")

    def test_benchmark_with_mock_llm(self):
        """Test benchmark functionality with mocked LLM responses"""
        framework = StrictPromptOptimizationFramework(results_dir="./test_results_mock")
        
        # Add test API config
        api_config = APIConfig(
            name="mock_api",
            base_url="https://api.example.com",
            api_key="mock-key",
            model="mock-model"
        )
        framework.add_api_config(api_config)
        
        # Add test prompt variant
        prompt_variant = PromptVariant(
            name="Mock Test Prompt",
            prompt_type=PromptType.NLP_EXTRACTION,
            prompt_key="generic_extraction"
        )
        framework.add_prompt_variant(prompt_variant)
        
        # Add test case
        test_case = {
            "clinical_text": "Patient with diabetes and hypertension",
            "expected_entities": ["diabetes", "hypertension"]
        }
        framework.add_test_case("mock_case", test_case)
        
        # Mock pipeline callback
        def mock_callback(test_data, prompt_content, prompt_variant_id):
            class MockResult:
                def __init__(self):
                    self.extracted_entities = [
                        {"text": "diabetes", "type": "condition"},
                        {"text": "hypertension", "type": "condition"}
                    ]
                    self.mcode_mappings = []
                    self.validation_results = {"compliance_score": 1.0}
            return MockResult()
        
        # Run benchmark
        result = framework.run_benchmark(
            prompt_variant_id=prompt_variant.id,
            api_config_name="mock_api",
            test_case_id="mock_case",
            pipeline_callback=mock_callback
        )
        
        assert result.success is True
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        
        # Clean up
        import shutil
        if Path("./test_results_mock").exists():
            shutil.rmtree("./test_results_mock")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])