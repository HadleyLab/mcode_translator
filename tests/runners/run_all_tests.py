#!/usr/bin/env python3
"""
Main test runner for the mcode_translator project.
Runs all test suites in the proper order.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_path, test_type="unit"):
    """Run tests in the specified directory and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_type} tests: {test_path}")
    print(f"{'='*60}")
    
    # Use python -m pytest for better module resolution
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            print(f"❌ {test_type.upper()} TESTS FAILED")
            return False
        else:
            print(f"✅ {test_type.upper()} TESTS PASSED")
            return True
    except Exception as e:
        print(f"Error running {test_type} tests: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run mcode_translator test suites")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run only end-to-end tests")
    parser.add_argument("--breast-cancer", action="store_true", help="Run only breast cancer tests")
    parser.add_argument("--multi-cancer", action="store_true", help="Run only multi-cancer tests")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    
    args = parser.parse_args()
    
    # Base test directories
    unit_tests = "tests/unit"
    integration_tests = "tests/integration"
    e2e_tests = "tests/e2e"
    
    all_passed = True
    
    # Run specific test types if requested
    if args.unit:
        all_passed = run_tests(unit_tests, "unit") and all_passed
    elif args.integration:
        all_passed = run_tests(integration_tests, "integration") and all_passed
    elif args.e2e:
        all_passed = run_tests(e2e_tests, "end-to-end") and all_passed
    elif args.breast_cancer:
        all_passed = run_tests(f"{e2e_tests}/breast_cancer", "breast cancer") and all_passed
    elif args.multi_cancer:
        all_passed = run_tests(f"{e2e_tests}/multi_cancer", "multi-cancer") and all_passed
    else:
        # Run all tests in recommended order
        print("🧪 Running complete mcode_translator test suite")
        print("Recommended order: Unit → Integration → E2E")
        
        # Unit tests
        all_passed = run_tests(unit_tests, "unit") and all_passed
        
        # Integration tests (if they exist)
        if Path(integration_tests).exists() and any(Path(integration_tests).iterdir()):
            all_passed = run_tests(integration_tests, "integration") and all_passed
        
        # E2E tests
        all_passed = run_tests(e2e_tests, "end-to-end") and all_passed
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()