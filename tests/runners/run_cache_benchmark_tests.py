#!/usr/bin/env python3
"""
Test runner for cache and benchmark functionality tests
"""
import sys
import os
import unittest
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all cache and benchmark tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add unit tests
    suite.addTests(loader.discover(
        start_dir=str(project_root / 'tests' / 'unit'),
        pattern='test_*benchmark*.py'
    ))
    
    suite.addTests(loader.discover(
        start_dir=str(project_root / 'tests' / 'unit'),
        pattern='test_*fetcher*.py'
    ))
    
    # Add integration tests
    suite.addTests(loader.discover(
        start_dir=str(project_root / 'tests' / 'integration'),
        pattern='test_*benchmark*.py'
    ))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)