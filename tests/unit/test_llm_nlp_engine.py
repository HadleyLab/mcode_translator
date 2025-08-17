import unittest
import json
from unittest.mock import patch, MagicMock
from src.nlp_engine.llm_nlp_engine import LLMNLPEngine
from src.nlp_engine.nlp_engine import ProcessingResult

class TestLLMNLPEngine(unittest.TestCase):
    def setUp(self):
        self.engine = LLMNLPEngine()
        self.mock_response = {
            "genomic_variants": [
                {"gene": "BRCA1", "variant": "", "significance": ""}
            ],
            "biomarkers": [
                {"name": "ER", "status": "positive", "value": ""},
                {"name": "HER2", "status": "negative", "value": ""}
            ],
            "cancer_characteristics": {},
            "treatment_history": {},
            "performance_status": {},
            "demographics": {}
        }
        
    @patch('src.nlp_engine.llm_nlp_engine.openai.OpenAI')
    def test_process_text(self, mock_openai):
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(self.mock_response)))]
        )
        
        criteria = "Inclusion: ER+ breast cancer, HER2-negative, BRCA1 mutation"
        result = self.engine.process_text(criteria)
        self.assertIsInstance(result, ProcessingResult)
        self.assertIn('genomic_variants', result.features)
        self.assertIn('biomarkers', result.features)
        self.assertIn('cancer_characteristics', result.features)
        
        # Validate breast cancer-specific structure
        self.assertTrue(any(v['gene'] == 'BRCA1' for v in result.features['genomic_variants']))
        self.assertTrue(any(b['name'] == 'ER' for b in result.features['biomarkers']))
        self.assertTrue(any(b['name'] == 'HER2' for b in result.features['biomarkers']))

if __name__ == '__main__':
    unittest.main()