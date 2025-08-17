"""
Unit tests for the Clinical Trial Data Fetcher using the refactored approach.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from tests.shared.test_components import MockClinicalTrialsAPI, MockCacheManager


class TestClinicalTrialsFetcher:
    """Unit tests for clinicaltrials.gov API fetcher using pytest"""
    
    def test_search_trials(self, mock_clinical_trials_api, mock_cache_manager):
        """Test search_trials function with mocks"""
        from src.data_fetcher.fetcher import search_trials
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "StudyFields": [
                {"NCTId": ["NCT12345678"], "BriefTitle": ["Test Study 1"]},
                {"NCTId": ["NCT87654321"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = search_trials("cancer", fields=["NCTId", "BriefTitle"], max_results=2)
        
        # Verify results
        assert isinstance(result, dict)
        assert "StudyFields" in result
        assert len(result["StudyFields"]) == 2
        assert result["StudyFields"][0]["NCTId"][0] == "NCT12345678"
        
        # Verify cache was set
        mock_cache_manager.return_value.set.assert_called()
    
    def test_search_trials_cached(self, mock_clinical_trials_api, mock_cache_manager):
        """Test search_trials with cached results"""
        from src.data_fetcher.fetcher import search_trials
        
        # Set up mocks
        mock_cache_manager.return_value.get.return_value = {
            "StudyFields": [
                {"NCTId": ["NCT12345678"], "BriefTitle": ["Test Study 1"]},
                {"NCTId": ["NCT87654321"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        
        # Call function
        result = search_trials("cancer", fields=["NCTId", "BriefTitle"], max_results=2)
        
        # Verify results
        assert isinstance(result, dict)
        assert "StudyFields" in result
        assert len(result["StudyFields"]) == 2
        
        # Verify API was not called
        mock_clinical_trials_api.return_value.get_study_fields.assert_not_called()
    
    def test_get_full_study(self, mock_clinical_trials_api, mock_cache_manager):
        """Test get_full_study function"""
        from src.data_fetcher.fetcher import get_full_study
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "StudyFields": [
                {"NCTId": ["NCT12345678"], "BriefTitle": ["Test Study 1"]}
            ]
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = get_full_study("NCT12345678")
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        assert result["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"
        
        # Verify cache was set
        mock_cache_manager.return_value.set.assert_called()
    
    def test_get_full_study_cached(self, mock_cache_manager):
        """Test get_full_study with cached results"""
        from src.data_fetcher.fetcher import get_full_study
        
        # Set up mocks
        mock_cache_manager.return_value.get.return_value = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study 1"
                }
            }
        }
        
        # Call function
        result = get_full_study("NCT12345678")
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        assert result["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"
    
    def test_search_trials_error(self, mock_clinical_trials_api, mock_cache_manager):
        """Test search_trials with API error"""
        from src.data_fetcher.fetcher import search_trials, ClinicalTrialsAPIError
        
        # Set up mocks to raise exception
        mock_clinical_trials_api.return_value.get_study_fields.side_effect = Exception("API error")
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function and verify exception
        with pytest.raises(ClinicalTrialsAPIError):
            search_trials("cancer")


if __name__ == '__main__':
    pytest.main([__file__])