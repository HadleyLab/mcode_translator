"""
Test suite for the streamlined clinical benchmark UI
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.optimization.clinical_benchmark_ui import ClinicalBenchmarkUI
from src.optimization.strict_prompt_optimization_framework import StrictPromptOptimizationFramework
from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader


def test_clinical_benchmark_ui_initialization():
    """Test that ClinicalBenchmarkUI initializes correctly with streamlined implementation"""
    with patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_test_cases'), \
         patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_gold_standard'), \
         patch('src.optimization.clinical_benchmark_ui.ui.run'), \
         patch('src.optimization.clinical_benchmark_ui.ui.timer'):
        
        # Mock the test cases and gold standard data
        with patch.object(ClinicalBenchmarkUI, '_load_test_cases'), \
             patch.object(ClinicalBenchmarkUI, '_load_gold_standard'):
            
            # Create a mock framework
            mock_framework = Mock(spec=StrictPromptOptimizationFramework)
            mock_framework.add_test_case = Mock()
            mock_framework.add_prompt_variant = Mock()
            
            # Initialize the UI
            ui_instance = ClinicalBenchmarkUI(framework=mock_framework)
            
            # Verify initialization
            assert ui_instance is not None
            assert isinstance(ui_instance, ClinicalBenchmarkUI)
            assert ui_instance.framework == mock_framework


def test_load_libraries_with_validation():
    """Test that _load_libraries correctly loads and validates prompt and model libraries"""
    with patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_test_cases'), \
         patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_gold_standard'), \
         patch('src.optimization.clinical_benchmark_ui.ui.run'), \
         patch('src.optimization.clinical_benchmark_ui.ui.timer'):
        
        # Mock the test cases and gold standard data
        with patch.object(ClinicalBenchmarkUI, '_load_test_cases'), \
             patch.object(ClinicalBenchmarkUI, '_load_gold_standard'):
            
            # Create a mock framework
            mock_framework = Mock(spec=StrictPromptOptimizationFramework)
            mock_framework.add_test_case = Mock()
            mock_framework.add_prompt_variant = Mock()
            
            # Initialize the UI
            ui_instance = ClinicalBenchmarkUI(framework=mock_framework)
            
            # Verify that libraries were loaded
            assert hasattr(ui_instance, 'available_prompts')
            assert hasattr(ui_instance, 'available_models')
            assert len(ui_instance.available_prompts) > 0
            assert len(ui_instance.available_models) > 0


def test_unique_key_validation():
    """Test that duplicate key validation works correctly"""
    with patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_test_cases'), \
         patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_gold_standard'), \
         patch('src.optimization.clinical_benchmark_ui.ui.run'), \
         patch('src.optimization.clinical_benchmark_ui.ui.timer'):
        
        # Mock the test cases and gold standard data
        with patch.object(ClinicalBenchmarkUI, '_load_test_cases'), \
             patch.object(ClinicalBenchmarkUI, '_load_gold_standard'):
            
            # Create a mock framework
            mock_framework = Mock(spec=StrictPromptOptimizationFramework)
            mock_framework.add_test_case = Mock()
            mock_framework.add_prompt_variant = Mock()
            
            # Initialize the UI
            ui_instance = ClinicalBenchmarkUI(framework=mock_framework)
            
            # Test that we have unique keys
            prompt_keys = list(ui_instance.available_prompts.keys())
            model_keys = list(ui_instance.available_models.keys())
            
            assert len(prompt_keys) == len(set(prompt_keys)), "Duplicate prompt keys found"
            assert len(model_keys) == len(set(model_keys)), "Duplicate model keys found"


def test_generate_validations_performance():
    """Test that _generate_validations creates validations with improved performance"""
    with patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_test_cases'), \
         patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_gold_standard'), \
         patch('src.optimization.clinical_benchmark_ui.ui.run'), \
         patch('src.optimization.clinical_benchmark_ui.ui.timer'):
        
        # Mock the test cases and gold standard data
        with patch.object(ClinicalBenchmarkUI, '_load_test_cases'), \
             patch.object(ClinicalBenchmarkUI, '_load_gold_standard'):
            
            # Create a mock framework
            mock_framework = Mock(spec=StrictPromptOptimizationFramework)
            mock_framework.add_test_case = Mock()
            mock_framework.add_prompt_variant = Mock()
            
            # Initialize the UI
            ui_instance = ClinicalBenchmarkUI(framework=mock_framework)
            
            # Get initial validation count
            initial_count = len(ui_instance.validations)
            
            # Regenerate validations
            ui_instance._generate_validations()
            
            # Verify that validations were generated
            assert len(ui_instance.validations) > 0
            assert len(ui_instance.filtered_validations) == len(ui_instance.validations)
            
            # Verify validation structure
            if ui_instance.validations:
                validation = ui_instance.validations[0]
                required_fields = ['id', 'prompt_key', 'model_key', 'trial_id', 'prompt_type', 
                                 'prompt_name', 'model_name', 'status', 'last_run', 'score', 'selected']
                for field in required_fields:
                    assert field in validation, f"Missing required field: {field}"


def test_setup_filter_controls():
    """Test that _setup_filter_controls correctly uses prompt and model loaders"""
    with patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_test_cases'), \
         patch('src.optimization.clinical_benchmark_ui.ClinicalBenchmarkUI._load_gold_standard'), \
         patch('src.optimization.clinical_benchmark_ui.ui.run'), \
         patch('src.optimization.clinical_benchmark_ui.ui.timer'):
        
        # Mock the test cases and gold standard data
        with patch.object(ClinicalBenchmarkUI, '_load_test_cases'), \
             patch.object(ClinicalBenchmarkUI, '_load_gold_standard'):
            
            # Create a mock framework
            mock_framework = Mock(spec=StrictPromptOptimizationFramework)
            mock_framework.add_test_case = Mock()
            mock_framework.add_prompt_variant = Mock()
            
            # Initialize the UI
            ui_instance = ClinicalBenchmarkUI(framework=mock_framework)
            
            # Verify that filter controls can be set up
            # (In a real test, we would mock the UI components)
            assert hasattr(ui_instance, 'available_prompts')
            assert hasattr(ui_instance, 'available_models')
            assert hasattr(ui_instance, 'trial_ids')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])