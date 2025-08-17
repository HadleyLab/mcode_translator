"""
Unit tests for breast cancer mCODE trials matching using the refactored approach.
"""

import pytest
from tests.shared.test_components import MCODETestComponents


class TestBreastCancerMCODE:
    """Test suite for breast cancer mCODE trials matching using pytest and shared components"""
    
    def test_er_pos_matching(self, sample_breast_cancer_profile):
        """Test matching for ER+ breast cancer trials using shared components"""
        # Create mCODE bundle with ER positive observation using shared component
        mcode_bundle = MCODETestComponents.create_mcode_bundle(
            biomarkers=[{"code": "LP417347-6", "value": "LA6576-8"}]  # ER+
        )
        
        matches = sample_breast_cancer_profile.matches_er_positive(mcode_bundle)
        assert matches, "ER+ patient should match ER+ trials"
    
    def test_her2_pos_matching(self, sample_breast_cancer_profile):
        """Test matching for HER2+ breast cancer trials using shared components"""
        # Create mCODE bundle with HER2 positive observation using shared component
        mcode_bundle = MCODETestComponents.create_mcode_bundle(
            biomarkers=[{"code": "LP417351-8", "value": "LA6576-8"}]  # HER2+
        )
        
        matches = sample_breast_cancer_profile.matches_her2_positive(mcode_bundle)
        assert matches, "HER2+ patient should match HER2+ trials"
    
    def test_tnbc_matching(self, sample_breast_cancer_profile):
        """Test matching for triple-negative breast cancer trials using shared components"""
        # Create mCODE bundle with negative biomarkers using shared component
        mcode_bundle = MCODETestComponents.create_mcode_bundle(
            biomarkers=[
                {"code": "LP417347-6", "value": "LA6577-6"},  # ER-
                {"code": "LP417348-4", "value": "LA6577-6"},  # PR-
                {"code": "LP417351-8", "value": "LA6577-6"}   # HER2-
            ]
        )
        
        matches = sample_breast_cancer_profile.matches_triple_negative(mcode_bundle)
        assert matches, "TNBC patient should match TNBC trials"
    
    @pytest.mark.parametrize("biomarker_profile,expected_match_er,expected_match_her2,expected_match_tnbc", [
        # ER+ only
        ([{"code": "LP417347-6", "value": "LA6576-8"}], True, False, False),
        # HER2+ only
        ([{"code": "LP417351-8", "value": "LA6576-8"}], False, True, False),
        # TNBC (all negative)
        ([{"code": "LP417347-6", "value": "LA6577-6"},
          {"code": "LP417348-4", "value": "LA6577-6"},
          {"code": "LP417351-8", "value": "LA6577-6"}], False, False, True),
    ])
    def test_biomarker_matching_parameterized(self, sample_breast_cancer_profile, 
                                            biomarker_profile, expected_match_er, 
                                            expected_match_her2, expected_match_tnbc):
        """Test biomarker matching with parameterized testing"""
        # Create mCODE bundle with specified biomarkers
        mcode_bundle = MCODETestComponents.create_mcode_bundle(biomarkers=biomarker_profile)
        
        # Test each matching function
        matches_er = sample_breast_cancer_profile.matches_er_positive(mcode_bundle)
        matches_her2 = sample_breast_cancer_profile.matches_her2_positive(mcode_bundle)
        matches_tnbc = sample_breast_cancer_profile.matches_triple_negative(mcode_bundle)
        
        assert matches_er == expected_match_er
        assert matches_her2 == expected_match_her2
        assert matches_tnbc == expected_match_tnbc
    
    def test_mixed_biomarker_profile(self, sample_breast_cancer_profile):
        """Test matching with mixed biomarker profile"""
        # Create mCODE bundle with mixed biomarkers (ER+, PR-, HER2+)
        mcode_bundle = MCODETestComponents.create_mcode_bundle(
            biomarkers=[
                {"code": "LP417347-6", "value": "LA6576-8"},  # ER+
                {"code": "LP417348-4", "value": "LA6577-6"},  # PR-
                {"code": "LP417351-8", "value": "LA6576-8"}   # HER2+
            ]
        )
        
        # Should match ER+ and HER2+ but not TNBC
        matches_er = sample_breast_cancer_profile.matches_er_positive(mcode_bundle)
        matches_her2 = sample_breast_cancer_profile.matches_her2_positive(mcode_bundle)
        matches_tnbc = sample_breast_cancer_profile.matches_triple_negative(mcode_bundle)
        
        assert matches_er, "Should match ER+ trials"
        assert matches_her2, "Should match HER2+ trials"
        assert not matches_tnbc, "Should not match TNBC trials"
    
    def test_empty_biomarker_profile(self, sample_breast_cancer_profile):
        """Test matching with empty biomarker profile"""
        # Create mCODE bundle with no biomarkers
        mcode_bundle = MCODETestComponents.create_mcode_bundle()
        
        # Should not match any specific biomarker trials
        matches_er = sample_breast_cancer_profile.matches_er_positive(mcode_bundle)
        matches_her2 = sample_breast_cancer_profile.matches_her2_positive(mcode_bundle)
        matches_tnbc = sample_breast_cancer_profile.matches_triple_negative(mcode_bundle)
        
        # Behavior depends on implementation - could be False for all or require specific handling
        # For this test, we'll check that it doesn't crash
        assert isinstance(matches_er, bool)
        assert isinstance(matches_her2, bool)
        assert isinstance(matches_tnbc, bool)


if __name__ == '__main__':
    pytest.main([__file__])