import unittest
import time
from src.nlp_engine.spacy_nlp_engine import SpacyNLPEngine
from src.nlp_engine.nlp_engine import ProcessingResult

class TestSpacyNLPEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SpacyNLPEngine()
        
    def test_process_text(self):
        """Test processing text with spaCy"""
        sample_text = "Male patient with breast cancer, ER+"
        
        start_time = time.time()
        result = self.engine.process_text(sample_text)
        processing_time = time.time() - start_time
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsNotNone(result.entities)
        self.assertIsNotNone(result.features)
        self.assertIsNotNone(result.metadata)
        
        # Validate entity extraction
        self.assertGreater(len(result.entities), 2)
        
        # Validate features
        self.assertIn('demographics', result.features)
        self.assertIn('cancer_characteristics', result.features)
        self.assertIn('biomarkers', result.features)
        
        # Validate demographics
        self.assertEqual(result.features['demographics']['gender'], 'Male')
        
        # Validate metadata
        self.assertGreater(result.metadata['processing_time'], 0)
        self.assertEqual(result.metadata['engine'], 'spacy')
        
        # Log performance metrics
        print(f"\nSpaCy NLP Engine Performance:")
        print(f"- Processing time: {processing_time:.4f} seconds")
        print(f"- Text length: {len(sample_text)} characters")
        print(f"- Entities found: {len(result.entities)}")