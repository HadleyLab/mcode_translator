import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipeline.strict_llm_base import StrictLLMBase

class TestStrictLLMBase(unittest.TestCase):

    def test_initialization(self):
        base = StrictLLMBase()
        self.assertIsNotNone(base)
        self.assertIsNotNone(base.logger)
        self.assertIsNotNone(base.llm_client)

if __name__ == '__main__':
    unittest.main()