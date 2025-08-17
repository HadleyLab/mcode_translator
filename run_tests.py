#!/usr/bin/env python3
"""
Simple script to run the mCODE Translator test suite
"""

import sys
import os

# Add the tests directory to the path so we can import the test runner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

def main():
    """Run the test suite using the new test runner"""
    try:
        from test_runner import TestRunner
        runner = TestRunner()
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"Error importing test runner: {e}")
        print("Make sure you're running this script from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()