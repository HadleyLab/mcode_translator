import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.pipeline.nlp_base import NlpBase

class TestNlpBase(unittest.TestCase):

    def test_initialization(self):
        # This is an abstract class, so we can't instantiate it directly.
        # We can create a concrete subclass for testing.
        class ConcreteNlpBase(NlpBase):
            def process_text(self, text: str):
                return super().process_text(text)
        
        engine = ConcreteNlpBase()
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.logger)

if __name__ == '__main__':
    unittest.main()