import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipeline.llm_base import LlmBase

class TestLlmBase(unittest.TestCase):

    def test_initialization(self):
        base = LlmBase()
        self.assertIsNotNone(base)
        self.assertIsNotNone(base.logger)
        self.assertIsNotNone(base.llm_client)

if __name__ == '__main__':
    unittest.main()