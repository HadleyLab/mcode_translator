#!/usr/bin/env python3
"""
Breast cancer specific test runner for the mcode_translator project.
Runs all breast cancer related tests including gold standard validation.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_breast_cancer_tests():
    """Run all breast cancer related tests."""
    print("🧪 Running Breast Cancer Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Run pipeline tests with prompts (breast cancer focused)
    e2e_test_path = "tests/e2e/test_pipeline_with_prompts.py"
    if Path(e2e_test_path).exists():
        print(f"\n🚀 Running pipeline tests with prompts...")
        result = subprocess.run([sys.executable, "-m", "pytest", e2e_test_path, "-v"],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Pipeline tests with prompts failed")
            all_passed = False
        else:
            print("✅ Pipeline tests with prompts passed")
    
    # Run simple pipeline tests
    simple_test_path = "tests/e2e/test_pipeline_simple.py"
    if Path(simple_test_path).exists():
        print(f"\n🧪 Running simple pipeline tests...")
        result = subprocess.run([sys.executable, "-m", "pytest", simple_test_path, "-v"],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("❌ Simple pipeline tests failed")
            all_passed = False
        else:
            print("✅ Simple pipeline tests passed")
    
    return all_passed

def main():
    """Main function to run breast cancer test suite."""
    print("Breast Cancer Test Runner")
    print("=" * 50)
    print("This runner executes the pipeline testing suite:")
    print("1. Pipeline tests with prompt integration")
    print("2. Simple pipeline functionality tests")
    print()
    
    success = run_breast_cancer_tests()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 ALL BREAST CANCER TESTS PASSED!")
        print("The breast cancer pipeline is functioning correctly.")
        sys.exit(0)
    else:
        print("💥 SOME BREAST CANCER TESTS FAILED!")
        print("Please check the test output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()