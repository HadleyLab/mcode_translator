import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipeline.prompt_model_interface import set_extraction_prompt, set_mapping_prompt, set_model, create_configured_pipeline

class TestPromptModelInterface(unittest.TestCase):

    def test_set_extraction_prompt(self):
        set_extraction_prompt("test_prompt")
        # We can't easily assert the effect, but we can check that it runs without error
        self.assertTrue(True)

    def test_set_mapping_prompt(self):
        set_mapping_prompt("test_prompt")
        self.assertTrue(True)

    def test_set_model(self):
        set_model("test_model")
        self.assertTrue(True)

    def test_create_configured_pipeline(self):
        pipeline = create_configured_pipeline()
        self.assertIsNotNone(pipeline)

if __name__ == '__main__':
    unittest.main()