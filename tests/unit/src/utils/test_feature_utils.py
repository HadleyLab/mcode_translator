import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils.feature_utils import standardize_features, standardize_biomarkers, standardize_variants

class TestFeatureUtils(unittest.TestCase):

    def test_standardize_features(self):
        features = {"demographics": {}, "cancer_characteristics": {}, "biomarkers": [], "genomic_variants": [], "treatment_history": {}, "performance_status": {}}
        standardized = standardize_features(features)
        self.assertEqual(features, standardized)

    def test_standardize_biomarkers(self):
        biomarkers = {"ER": "Positive"}
        standardized = standardize_biomarkers(biomarkers)
        self.assertEqual(standardized, [{'name': 'ER', 'status': 'Positive'}])

    def test_standardize_variants(self):
        variants = [{"gene": "BRCA1"}]
        standardized = standardize_variants(variants)
        self.assertEqual(standardized, [{'gene': 'BRCA1', 'variant': '', 'significance': ''}])

if __name__ == '__main__':
    unittest.main()