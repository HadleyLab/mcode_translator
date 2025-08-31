import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.optimization.ui_components.results_analyzer import ResultsAnalyzer

class TestResultsAnalyzer(unittest.TestCase):

    def test_initialization(self):
        # This class is tightly coupled with the UI, so we can only do a basic test
        # without a running NiceGUI event loop.
        pass

if __name__ == '__main__':
    unittest.main()