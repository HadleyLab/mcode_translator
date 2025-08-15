import unittest
from src.breast_cancer_profile import BreastCancerProfile
from src.mcode_mapping_engine import MCODEMappingEngine

class TestBreastCancerMCODE(unittest.TestCase):
    """Test suite for breast cancer mCODE trials matching"""
    
    def setUp(self):
        self.mapping_engine = MCODEMappingEngine()
        self.profile = BreastCancerProfile()
        
    def test_er_pos_matching(self):
        """Test matching for ER+ breast cancer trials"""
        # Create mCODE bundle with ER positive observation
        mcode_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [{"code": "LP417347-6"}]
                        },
                        "valueCodeableConcept": {
                            "coding": [{"code": "LA6576-8"}]  # Positive
                        }
                    }
                }
            ]
        }
        
        matches = self.profile.matches_er_positive(mcode_bundle)
        self.assertTrue(matches, "ER+ patient should match ER+ trials")
        
    def test_her2_pos_matching(self):
        """Test matching for HER2+ breast cancer trials"""
        # Create mCODE bundle with HER2 positive observation
        mcode_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [{"code": "LP417351-8"}]  # HER2
                        },
                        "valueCodeableConcept": {
                            "coding": [{"code": "LA6576-8"}]  # Positive
                        }
                    }
                }
            ]
        }
        
        matches = self.profile.matches_her2_positive(mcode_bundle)
        self.assertTrue(matches, "HER2+ patient should match HER2+ trials")
        
    def test_tnbc_matching(self):
        """Test matching for triple-negative breast cancer trials"""
        # Create mCODE bundle with negative biomarkers
        mcode_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [{"code": "LP417347-6"}]  # ER
                        },
                        "valueCodeableConcept": {
                            "coding": [{"code": "LA6577-6"}]  # Negative
                        }
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [{"code": "LP417348-4"}]  # PR
                        },
                        "valueCodeableConcept": {
                            "coding": [{"code": "LA6577-6"}]  # Negative
                        }
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [{"code": "LP417351-8"}]  # HER2
                        },
                        "valueCodeableConcept": {
                            "coding": [{"code": "LA6577-6"}]  # Negative
                        }
                    }
                }
            ]
        }
        
        matches = self.profile.matches_triple_negative(mcode_bundle)
        self.assertTrue(matches, "TNBC patient should match TNBC trials")

if __name__ == '__main__':
    unittest.main()