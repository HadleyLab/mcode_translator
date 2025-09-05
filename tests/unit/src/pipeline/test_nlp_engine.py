import unittest
import sys
import os
from unittest.mock import MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from pipeline.nlp_extractor import NlpBase

class TestNlpBase(unittest.TestCase):

    def setUp(self):
        self.engine = NlpBase()

    def test_initialization(self):
        self.assertIsNotNone(self.engine)
        self.assertIsNotNone(self.engine.logger)

    def test_extract_entities(self):
        # This is a complex function that would require significant mocking of the LLM
        # For a unit test, we'll just test the basic flow
        text = "ER Positive"
        
        # Mocking the LLM call
        self.engine.llm_client.chat.completions.create = MagicMock(return_value=MagicMock(choices=[MagicMock(message=MagicMock(content='{"entities": [{"text": "ER Positive", "type": "Biomarker"}]}'))]))

        result = self.engine.extract_entities(text)
        self.assertIn("entities", result)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["text"], "ER Positive")

if __name__ == '__main__':
    unittest.main()