import unittest
import sys
import os
from unittest.mock import MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipeline.mcode_mapper import StrictMcodeMapper

class TestStrictMcodeMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = StrictMcodeMapper()

    def test_initialization(self):
        self.assertIsNotNone(self.mapper)
        self.assertIsNotNone(self.mapper.logger)

    def test_map_to_mcode(self):
        # This is a complex function that would require significant mocking of the LLM
        # For a unit test, we'll just test the basic flow
        nlp_entities = [{"text": "ER Positive", "type": "Biomarker"}]
        
        # Mocking the LLM call
        self.mapper.llm_client.chat.completions.create = MagicMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"mcode": "some_mcode"}'))]))

        result = self.mapper.map_to_mcode(nlp_entities)
        self.assertIn("mcode", result)
        self.assertEqual(result["mcode"], "some_mcode")

if __name__ == '__main__':
    unittest.main()