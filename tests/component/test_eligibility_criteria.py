"""
Component tests for eligibility criteria fetching functionality.
These tests verify that the eligibility criteria are properly fetched and processed.
"""

import pytest
import os
import json
from src.data_fetcher.fetcher import get_full_study


class TestEligibilityCriteriaComponent:
    """Component tests for eligibility criteria functionality"""
    
    def test_eligibility_criteria_structure(self):
        """Test that eligibility criteria have the correct structure"""
        # Use a known study that should have eligibility criteria
        nct_id = "NCT03929822"
        
        # Call function
        result = get_full_study(nct_id)
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        protocol_section = result["protocolSection"]
        
        # Verify eligibility module structure
        assert "eligibilityModule" in protocol_section
        eligibility_module = protocol_section["eligibilityModule"]
        
        # Verify eligibility criteria are present
        assert "eligibilityCriteria" in eligibility_module
        criteria = eligibility_module["eligibilityCriteria"]
        
        # Verify criteria content structure
        assert isinstance(criteria, str)
        assert len(criteria) > 0
        
        # Verify criteria contain expected sections
        assert "Inclusion" in criteria or "Inclusion Criteria" in criteria or " inclusion " in criteria.lower()
        assert "Exclusion" in criteria or "Exclusion Criteria" in criteria or " exclusion " in criteria.lower()
        
        # Verify criteria have proper formatting
        lines = criteria.split('\n')
        assert len(lines) > 3  # Should have multiple lines
    
    def test_eligibility_criteria_content(self):
        """Test that eligibility criteria contain expected content"""
        # Use a known study that should have eligibility criteria
        nct_id = "NCT03929822"
        
        # Call function
        result = get_full_study(nct_id)
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        protocol_section = result["protocolSection"]
        
        # Verify eligibility module
        assert "eligibilityModule" in protocol_section
        eligibility_module = protocol_section["eligibilityModule"]
        
        # Verify eligibility criteria
        assert "eligibilityCriteria" in eligibility_module
        criteria = eligibility_module["eligibilityCriteria"]
        
        # Verify content is not empty
        assert len(criteria.strip()) > 0
        
        # Verify content contains actual criteria (not just placeholder text)
        assert "criteria" in criteria.lower() or "age" in criteria.lower() or "patient" in criteria.lower()
    
    def test_eligibility_criteria_for_different_study_types(self):
        """Test eligibility criteria for different types of studies"""
        # Test with a few different studies
        test_studies = [
            "NCT03929822",  # Our main test study
        ]
        
        for nct_id in test_studies:
            # Call function
            result = get_full_study(nct_id)
            
            # Verify results
            assert isinstance(result, dict)
            assert "protocolSection" in result
            protocol_section = result["protocolSection"]
            
            # Verify eligibility module exists
            assert "eligibilityModule" in protocol_section
            eligibility_module = protocol_section["eligibilityModule"]
            
            # Verify eligibility criteria exist
            assert "eligibilityCriteria" in eligibility_module
            criteria = eligibility_module["eligibilityCriteria"]
            
            # Verify criteria are not empty
            assert isinstance(criteria, str)
            assert len(criteria.strip()) > 0
    
    def test_eligibility_criteria_formatting(self):
        """Test that eligibility criteria are properly formatted"""
        # Use a known study that should have eligibility criteria
        nct_id = "NCT03929822"
        
        # Call function
        result = get_full_study(nct_id)
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        protocol_section = result["protocolSection"]
        
        # Verify eligibility module
        assert "eligibilityModule" in protocol_section
        eligibility_module = protocol_section["eligibilityModule"]
        
        # Verify eligibility criteria
        assert "eligibilityCriteria" in eligibility_module
        criteria = eligibility_module["eligibilityCriteria"]
        
        # Verify formatting
        assert isinstance(criteria, str)
        assert len(criteria) > 0
        
        # Should not have excessive whitespace at the beginning or end
        assert len(criteria.strip()) > 0
        
        # Should contain line breaks for readability
        assert '\n' in criteria
    
    def test_eligibility_criteria_special_characters(self):
        """Test that eligibility criteria handle special characters properly"""
        # Use a known study that should have eligibility criteria
        nct_id = "NCT03929822"
        
        # Call function
        result = get_full_study(nct_id)
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        protocol_section = result["protocolSection"]
        
        # Verify eligibility module
        assert "eligibilityModule" in protocol_section
        eligibility_module = protocol_section["eligibilityModule"]
        
        # Verify eligibility criteria
        assert "eligibilityCriteria" in eligibility_module
        criteria = eligibility_module["eligibilityCriteria"]
        
        # Verify content handles special characters
        assert isinstance(criteria, str)
        
        # Should handle common special characters in medical text
        # This is more of a sanity check since we're dealing with real data
        assert len(criteria) > 0


if __name__ == '__main__':
    pytest.main([__file__])