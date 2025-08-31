import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils.pattern_config import BIOMARKER_PATTERNS, GENE_PATTERN

class TestPatternConfig(unittest.TestCase):

    def test_biomarker_patterns(self):
        self.assertIn("ER", BIOMARKER_PATTERNS)
        self.assertIsNotNone(BIOMARKER_PATTERNS["ER"])

    def test_gene_pattern(self):
        self.assertIsNotNone(GENE_PATTERN)

if __name__ == '__main__':
    unittest.main()