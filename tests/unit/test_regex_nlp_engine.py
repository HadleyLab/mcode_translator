import unittest
import time
from src.regex_nlp_engine import RegexNLPEngine

class TestRegexNLPEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RegexNLPEngine()
        
    def test_process_criteria(self):
        """Test processing criteria text"""
        sample_text = """
        Inclusion Criteria:
        - Male or female patients aged 18 years or older
        - Histologically confirmed diagnosis of breast cancer
        - Must have received prior chemotherapy treatment
        - Currently receiving radiation therapy
        - Laboratory values within normal limits
        
        Exclusion Criteria:
        - Pregnant or nursing women
        - History of other malignancies within the past 5 years
        - Allergy to contrast agents
        - Unable to undergo MRI scanning
        """
        
        start_time = time.time()
        result = self.engine.process_criteria(sample_text)
        processing_time = time.time() - start_time
        
        self.assertIsInstance(result, dict)
        self.assertIn('entities', result)
        self.assertIn('demographics', result)
        self.assertIn('conditions', result)
        self.assertIn('procedures', result)
        self.assertIn('metadata', result)
        
        # Validate entity extraction
        self.assertGreater(len(result['conditions']), 1)
        
        # Validate demographics
        self.assertGreater(len(result['demographics']['age']), 0)
        self.assertGreater(len(result['demographics']['gender']), 0)
        
        # Validate metadata
        self.assertGreater(result['metadata']['text_length'], 100)
        self.assertGreaterEqual(result['metadata']['condition_count'], 4)
        
        # Log performance metrics
        print(f"\nRegex NLP Engine Performance:")
        print(f"- Processing time: {processing_time:.4f} seconds")
        print(f"- Text length: {result['metadata']['text_length']} characters")
        print(f"- Conditions found: {result['metadata']['condition_count']}")
        print(f"- Procedures found: {result['metadata']['procedure_count']}")