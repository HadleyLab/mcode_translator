import unittest
from src.llm_nlp_engine import LLMNLPEngine

class TestLLMNLPEngine(unittest.TestCase):
    def setUp(self):
        self.engine = LLMNLPEngine()
        
    def test_extract_mcode_features(self):
        criteria = "Inclusion: ER+ breast cancer, HER2-negative, BRCA1 mutation"
        result = self.engine.extract_mcode_features(criteria)
        self.assertIsInstance(result, dict)
        self.assertIn('genomic_variants', result)
        self.assertIn('biomarkers', result)
        self.assertIn('cancer_characteristics', result)
        
        # Validate breast cancer-specific structure
        self.assertTrue(any(v['gene'] == 'BRCA1' for v in result['genomic_variants']))
        self.assertTrue(any(b['name'] == 'ER' for b in result['biomarkers']))
        self.assertTrue(any(b['name'] == 'HER2' for b in result['biomarkers']))

if __name__ == '__main__':
    unittest.main()