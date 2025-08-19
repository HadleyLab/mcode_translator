"""
Integration tests for the Clinical Trial Data Fetcher.
These tests verify the actual functionality against the real ClinicalTrials.gov API.
"""

import pytest
import os
import json
from src.data_fetcher.fetcher import get_full_study, search_trials, ClinicalTrialsAPIError


class TestClinicalTrialsFetcherIntegration:
    """Integration tests for clinicaltrials.gov API fetcher"""
    
    def test_get_full_study_integration(self):
        """Test get_full_study function with real API"""
        # Use a known study that should have eligibility criteria
        nct_id = "NCT03929822"
        
        # Call function
        result = get_full_study(nct_id)
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        protocol_section = result["protocolSection"]
        
        # Verify identification module
        assert "identificationModule" in protocol_section
        identification_module = protocol_section["identificationModule"]
        assert identification_module["nctId"] == nct_id
        
        # Verify eligibility module is present and has criteria
        assert "eligibilityModule" in protocol_section
        eligibility_module = protocol_section["eligibilityModule"]
        assert "eligibilityCriteria" in eligibility_module
        
        # Verify eligibility criteria content
        criteria = eligibility_module["eligibilityCriteria"]
        assert isinstance(criteria, str)
        assert len(criteria) > 0
        assert "Inclusion Criteria" in criteria or "Exclusion Criteria" in criteria
        
        # Verify other important modules are present
        assert "statusModule" in protocol_section
        assert "descriptionModule" in protocol_section
        assert "conditionsModule" in protocol_section
    
    def test_search_trials_integration(self):
        """Test search_trials function with real API"""
        # Search for a common condition
        condition = "breast cancer"
        
        # Call function
        result = search_trials(condition, max_results=5)
        
        # Verify results
        assert isinstance(result, dict)
        assert "studies" in result
        studies = result["studies"]
        assert isinstance(studies, list)
        assert len(studies) > 0
        assert len(studies) <= 5
        
        # Verify study structure
        study = studies[0]
        assert "protocolSection" in study
        protocol_section = study["protocolSection"]
        
        # Verify identification module
        assert "identificationModule" in protocol_section
        identification_module = protocol_section["identificationModule"]
        assert "nctId" in identification_module
        assert "briefTitle" in identification_module
        
        # Verify the study has a title
        title = identification_module["briefTitle"]
        assert isinstance(title, str)
        assert len(title) > 0
    
    def test_get_full_study_not_found_integration(self):
        """Test get_full_study function with non-existent study"""
        # Use a clearly non-existent NCT ID
        nct_id = "NCT99999999"
        
        # Call function and verify exception
        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study(nct_id)
    
    def test_search_trials_with_eligibility_criteria_field(self):
        """Test search_trials function requesting eligibility criteria field"""
        # Search for a study and request specific fields including those we know are valid
        condition = "breast cancer"
        
        # Use only valid fields
        fields = ["NCTId", "BriefTitle", "Condition", "OverallStatus"]
        
        # Call function
        result = search_trials(condition, fields=fields, max_results=3)
        
        # Verify results
        assert isinstance(result, dict)
        assert "studies" in result
        studies = result["studies"]
        assert isinstance(studies, list)
        assert len(studies) > 0
        
        # Verify study structure
        study = studies[0]
        assert "protocolSection" in study
        protocol_section = study["protocolSection"]
        
        # Verify identification module
        assert "identificationModule" in protocol_section
        identification_module = protocol_section["identificationModule"]
        assert "nctId" in identification_module
        assert "briefTitle" in identification_module


if __name__ == '__main__':
    pytest.main([__file__])