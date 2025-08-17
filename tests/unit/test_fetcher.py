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
        mock_clinical_trials_api.return_value.get_full_studies.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Test Study 1"
                        }
                    }
                }
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

    def test_search_trials_pagination(self, mock_clinical_trials_api, mock_cache_manager):
        """Test search_trials function with pagination"""
        from src.data_fetcher.fetcher import search_trials
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "StudyFields": [
                {"NCTId": ["NCT00000001"], "BriefTitle": ["Test Study 1"]},
                {"NCTId": ["NCT00000002"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function with pagination parameters
        result = search_trials("cancer", fields=["NCTId", "BriefTitle"], max_results=2, min_rank=5)
        
        # Verify results
        assert isinstance(result, dict)
        assert "StudyFields" in result
        assert len(result["StudyFields"]) == 2
        assert result["StudyFields"][0]["NCTId"][0] == "NCT00000001"
        
        # Verify pagination metadata
        assert "pagination" in result
        assert result["pagination"]["max_results"] == 2
        assert result["pagination"]["min_rank"] == 5
        
        # Verify cache was set
        mock_cache_manager.return_value.set.assert_called()
    
    def test_calculate_total_studies(self, mock_clinical_trials_api, mock_cache_manager):
        """Test calculate_total_studies function"""
        from src.data_fetcher.fetcher import calculate_total_studies
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT00000001",
                            "briefTitle": "Test Study 1"
                        }
                    }
                }
            ],
            "totalCount": 1000
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = calculate_total_studies("cancer", fields=["NCTId", "BriefTitle"])
        
        # Verify results
        assert isinstance(result, dict)
        assert result["total_studies"] >= 1
        assert result["total_pages"] >= 1
        assert result["page_size"] == 100
    
    def test_calculate_total_studies_no_results(self, mock_clinical_trials_api, mock_cache_manager):
        """Test calculate_total_studies function with no results"""
        from src.data_fetcher.fetcher import calculate_total_studies
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "StudyFields": []
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = calculate_total_studies("nonexistentcondition", fields=["NCTId", "BriefTitle"])
        
        # Verify results
        assert isinstance(result, dict)
        assert result["total_studies"] == 0
        assert result["total_pages"] == 0
        assert result["page_size"] == 100
    
    def test_search_trials_empty_fields(self, mock_clinical_trials_api, mock_cache_manager):
        """Test search_trials function with empty fields list"""
        from src.data_fetcher.fetcher import search_trials
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "StudyFields": [
                {"NCTId": ["NCT12345678"], "BriefTitle": ["Test Study 1"]}
            ]
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function with None fields (should use default fields)
        result = search_trials("cancer", fields=None, max_results=1)
        
        # Verify results
        assert isinstance(result, dict)
        assert "StudyFields" in result
        assert len(result["StudyFields"]) == 1
    
    def test_get_full_study_not_found(self, mock_clinical_trials_api, mock_cache_manager):
        """Test get_full_study function when study is not found"""
        from src.data_fetcher.fetcher import get_full_study, ClinicalTrialsAPIError
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_full_studies.return_value = {
            "studies": []
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function and verify exception
        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT99999999")

    def test_search_trials_data_structure(self, mock_clinical_trials_api, mock_cache_manager):
        """Test that search_trials returns correct data structure"""
        from src.data_fetcher.fetcher import search_trials
        
        # Set up mocks with correct structure
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Test Study 1"
                        }
                    }
                }
            ],
            "nextPageToken": "token123"
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = search_trials("cancer", fields=["NCTId", "BriefTitle"], max_results=1)
        
        # Verify results structure
        assert isinstance(result, dict)
        assert "studies" in result
        assert isinstance(result["studies"], list)
        assert len(result["studies"]) == 1
        
        # Verify study structure
        study = result["studies"][0]
        assert "protocolSection" in study
        assert "identificationModule" in study["protocolSection"]
        assert study["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"
        
        # Verify pagination structure
        assert "pagination" in result
        assert "max_results" in result["pagination"]
        assert "min_rank" in result["pagination"]
        assert "next_min_rank" in result["pagination"]
    
    def test_get_full_study_data_structure(self, mock_clinical_trials_api, mock_cache_manager):
        """Test that get_full_study returns correct data structure"""
        from src.data_fetcher.fetcher import get_full_study
        
        # Set up mocks with correct structure
        mock_clinical_trials_api.return_value.get_full_studies.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Test Study 1"
                        }
                    },
                    "derivedSection": {},
                    "hasResults": False
                }
            ]
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = get_full_study("NCT12345678")
        
        # Verify results structure
        assert isinstance(result, dict)
        assert "protocolSection" in result
        assert "derivedSection" in result
        assert "hasResults" in result
        
        # Verify identification module structure
        assert "identificationModule" in result["protocolSection"]
        assert result["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"


if __name__ == '__main__':
    pytest.main([__file__])