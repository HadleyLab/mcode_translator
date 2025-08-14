#!/usr/bin/env python3
"""
Main test runner for the mCODE Translator project
Runs all unit tests and provides a comprehensive test report
"""

import sys
import os
import unittest
import subprocess

def run_tests():
    """Run all tests and provide a comprehensive report"""
    print("Running Comprehensive Test Suite for mCODE Translator")
    print("=" * 60)
    
    # Test directories
    test_dirs = [
        "tests",
    ]
    
    # Test files to run
    test_files = [
        "tests/test_code_extraction.py",
        "tests/test_mcode_mapping_engine.py",
        "tests/test_structured_data_generator.py",
        "tests/test_output_formatter.py"
    ]
    
    # Run each test file individually and capture results
    results = []
    
    for test_file in test_files:
        print(f"\nRunning tests in {test_file}...")
        print("-" * 40)
        
        try:
            # Run the test file
            result = subprocess.run([
                sys.executable, "-m", "unittest", test_file
            ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            # Parse the output to extract test results
            output_lines = result.stdout.strip().split('\n')
            test_summary = None
            for line in output_lines:
                if "Ran" in line and ("tests" in line or "test" in line):
                    test_summary = line
                    break
            
            if test_summary is None:
                test_summary = "No test summary found"
            
            results.append({
                "file": test_file,
                "passed": result.returncode == 0,
                "summary": test_summary,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            if result.returncode == 0:
                print(f"  Status: PASSED")
                print(f"  Summary: {test_summary}")
            else:
                print(f"  Status: FAILED")
                print(f"  Summary: {test_summary}")
                if result.stderr:
                    print(f"  Errors: {result.stderr}")
        except Exception as e:
            print(f"  Status: ERROR - {str(e)}")
            results.append({
                "file": test_file,
                "passed": False,
                "summary": f"Error running tests: {str(e)}",
                "stdout": "",
                "stderr": str(e)
            })
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    failed_tests = total_tests - passed_tests
    
    print(f"Total test files: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "Success rate: 0%")
    
    if failed_tests > 0:
        print("\nFailed test files:")
        for result in results:
            if not result["passed"]:
                print(f"  - {result['file']}: {result['summary']}")
        return False
    else:
        print("\nAll tests passed!")
        return True

def main():
    """Main function"""
    success = run_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()