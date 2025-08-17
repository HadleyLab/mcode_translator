"""Simple test to verify ClinicalTrials.gov API connectivity"""
import unittest
from src.fetcher import search_trials

class TestAPISanityCheck(unittest.TestCase):
    """Basic connectivity test for ClinicalTrials.gov API"""
    
    def test_api_connectivity(self):
        """Test that we can connect to the API and get results"""
        print("Testing API connectivity...")
        try:
            # Simple search for cancer trials
            result = search_trials("cancer", max_results=2)
            
            # Basic validation
            self.assertIsInstance(result, dict, "Response should be a dictionary")
            self.assertIn("studies", result, "Response should contain studies")
            self.assertGreaterEqual(len(result["studies"]), 1, "Should return at least 1 study")
            
            print(f"API connectivity test passed. Found {len(result['studies'])} studies.")
        except Exception as e:
            self.fail(f"API connectivity test failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()