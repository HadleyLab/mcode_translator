#!/usr/bin/env python3
"""
Test script for Clinical Benchmark UI functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.clinical_benchmark_ui import ClinicalBenchmarkUI
from src.optimization.strict_prompt_optimization_framework import StrictPromptOptimizationFramework


def test_ui_initialization():
    """Test that the UI initializes correctly"""
    print("🧪 Testing UI initialization...")
    
    try:
        framework = StrictPromptOptimizationFramework()
        ui = ClinicalBenchmarkUI(framework)
        
        # Test library loading
        assert hasattr(ui, 'available_prompts'), "UI should have available_prompts"
        assert hasattr(ui, 'available_models'), "UI should have available_models"
        assert len(ui.available_prompts) > 0, "Should load prompts from library"
        assert len(ui.available_models) > 0, "Should load models from library"
        
        # Test validation generation
        ui._generate_validations()
        assert len(ui.validations) > 0, "Should generate validation combinations"
        
        print("✅ UI initialization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ UI initialization test failed: {e}")
        return False


def test_filter_functionality():
    """Test filter functionality"""
    print("🧪 Testing filter functionality...")
    
    try:
        ui = ClinicalBenchmarkUI()
        ui._generate_validations()
        
        # Test initial state
        initial_count = len(ui.filtered_validations)
        assert initial_count > 0, "Should have initial validations"
        
        # Test prompt type filter
        ui.prompt_type_filter = type('MockFilter', (), {'value': 'NLP_EXTRACTION'})()
        ui._apply_filters()
        
        nlp_count = len(ui.filtered_validations)
        assert nlp_count > 0, "Should have NLP validations"
        assert nlp_count <= initial_count, "NLP filter should reduce count"
        
        # Verify all filtered validations are NLP
        for validation in ui.filtered_validations:
            assert validation['prompt_type'] == 'NLP_EXTRACTION', "All should be NLP"
        
        print("✅ Filter functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Filter functionality test failed: {e}")
        return False


def test_validation_structure():
    """Test validation structure"""
    print("🧪 Testing validation structure...")
    
    try:
        ui = ClinicalBenchmarkUI()
        ui._generate_validations()
        
        # Check validation structure
        for validation in ui.validations[:5]:  # Check first 5
            assert 'id' in validation, "Validation should have ID"
            assert 'prompt_key' in validation, "Validation should have prompt key"
            assert 'model_key' in validation, "Validation should have model key"
            assert 'trial_id' in validation, "Validation should have trial ID"
            assert 'prompt_type' in validation, "Validation should have prompt type"
            assert validation['prompt_type'] in ['NLP_EXTRACTION', 'MCODE_MAPPING'], "Valid prompt type"
        
        print("✅ Validation structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation structure test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Clinical Benchmark UI tests...\n")
    
    tests = [
        test_ui_initialization,
        test_filter_functionality,
        test_validation_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Clinical Benchmark UI is ready.")
        print("\n🌐 Access the UI at: http://localhost:8085")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        sys.exit(1)