import unittest
import time
from src.regex_nlp_engine import RegexNLPEngine
from src.nlp_engine import ProcessingResult

class TestRegexNLPEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RegexNLPEngine()
        
    def test_process_text(self):
        """Test processing text with regex patterns"""
        sample_text = "Male patient with breast cancer, ER+"
        
        start_time = time.time()
        result = self.engine.process_text(sample_text)
        processing_time = time.time() - start_time
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertIsNotNone(result.features)
        self.assertIsNotNone(result.entities)
        self.assertIsNotNone(result.metadata)
        
        # Validate feature extraction
        self.assertIn('demographics', result.features)
        self.assertIn('cancer_characteristics', result.features)
        self.assertIn('biomarkers', result.features)
        
        # Validate demographics
        self.assertEqual(result.features['demographics']['gender'], 'Male')
        
        # Validate cancer type
        self.assertEqual(result.features['cancer_characteristics']['cancer_type'], 'Breast cancer')
        
        # Validate biomarkers
        self.assertGreater(len(result.features['biomarkers']), 0)
        self.assertEqual(result.features['biomarkers'][0]['name'], 'ER')
        self.assertEqual(result.features['biomarkers'][0]['status'], 'Positive')
        
        # Validate metadata
        self.assertGreater(result.metadata['processing_time'], 0)
        self.assertEqual(result.metadata['engine'], 'regex')
        
        # Log performance metrics
        print(f"\nRegex NLP Engine Performance:")
        print(f"- Processing time: {processing_time:.4f} seconds")
        print(f"- Text length: {len(sample_text)} characters")
        print(f"- Biomarkers found: {result.metadata['biomarkers_count']}")