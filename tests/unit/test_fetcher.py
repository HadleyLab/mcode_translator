"""
Unit tests for the Clinical Trial Data Fetcher using the refactored approach.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from tests.shared.test_components import MockClinicalTrialsAPI, MockCacheManager

class TestClinicalTrialsFetcher:
    def setup_method(self):
        # We'll call the functions directly since there's no ClinicalTrialsFetcher class
        pass
    
    """Unit tests for clinicaltrials.gov API fetcher using pytest"""
    
    def test_search_trials(self, mock_clinical_trials_api, mock_cache_manager):
        """Test search_trials function with mocks"""
        from src.data_fetcher.fetcher import search_trials
        
        # Set up mocks
        mock_clinical_trials_api.return_value.get_study_fields.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Test Study 1"
                        }
                    }
                },
                {"NCTId": ["NCT87654321"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = search_trials("cancer", fields=["NCTId", "BriefTitle"], max_results=2)
        
        # Verify results
        assert isinstance(result, dict)
        assert "studies" in result
        assert len(result["studies"]) == 2
        assert result["studies"][0]["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"
        
        # Verify cache was set
        mock_cache_manager.return_value.set.assert_called()
    
    def test_search_trials_cached(self, mock_clinical_trials_api, mock_cache_manager):
        """Test search_trials with cached results"""
        from src.data_fetcher.fetcher import search_trials
        
        # Set up mocks
        mock_cache_manager.return_value.get.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Test Study 1"
                        }
                    }
                },
                {"NCTId": ["NCT87654321"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        
        # Call function
        result = search_trials("cancer", fields=["NCTId", "BriefTitle"], max_results=2)
        
        # Verify results
        assert isinstance(result, dict)
        assert "studies" in result
        assert len(result["studies"]) == 2
        
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
    
    def test_get_full_study_with_eligibility_criteria(self, mock_clinical_trials_api, mock_cache_manager):
        """Test get_full_study function returns eligibility criteria"""
        from src.data_fetcher.fetcher import get_full_study
        
        # Set up mocks with eligibility criteria
        mock_clinical_trials_api.return_value.get_full_studies.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Test Study 1"
                        },
                        "eligibilityModule": {
                            "eligibilityCriteria": "Inclusion Criteria:\n- Age > 18 years\n\nExclusion Criteria:\n- Pregnant women"
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
        
        # Verify eligibility criteria are included
        assert "eligibilityModule" in result["protocolSection"]
        assert "eligibilityCriteria" in result["protocolSection"]["eligibilityModule"]
        criteria = result["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
        assert "Inclusion Criteria" in criteria
        assert "Exclusion Criteria" in criteria
        
        # Verify cache was set
        mock_cache_manager.return_value.set.assert_called()
    
    def test_get_full_study_with_quotes_fallback(self, mock_clinical_trials_api, mock_cache_manager):
        """Test get_full_study function with quotes fallback"""
        from src.data_fetcher.fetcher import get_full_study
        
        # Set up mocks to simulate first attempt failing and second attempt succeeding
        mock_clinical_trials_api.return_value.get_full_studies.side_effect = [
            Exception("First attempt failed"),  # First call fails
            {
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
            }  # Second call succeeds
        ]
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function
        result = get_full_study("NCT12345678")
        
        # Verify results
        assert isinstance(result, dict)
        assert "protocolSection" in result
        assert result["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"
        
        # Verify that get_full_studies was called twice (first attempt failed, second succeeded)
        assert mock_clinical_trials_api.return_value.get_full_studies.call_count == 2
        
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
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT00000001",
                            "briefTitle": "Test Study 1"
                        }
                    }
                },
                {"NCTId": ["NCT00000002"], "BriefTitle": ["Test Study 2"]}
            ],
            "nextPageToken": "token123"
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function with pagination parameters
        result = search_trials("cancer", fields=["NCTId", "BriefTitle"], max_results=2)
        
        # Verify results
        assert isinstance(result, dict)
        assert "studies" in result
        assert len(result["studies"]) == 2
        assert result["studies"][0]["protocolSection"]["identificationModule"]["nctId"] == "NCT00000001"
        
        # Verify pagination metadata
        assert "pagination" in result
        assert result["pagination"]["max_results"] == 2
        # Check if nextPageToken is in the result or pagination
        assert "nextPageToken" in result or "nextPageToken" in result.get("pagination", {})
        
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
            "studies": []
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
            "studies": [
                {"NCTId": ["NCT12345678"], "BriefTitle": ["Test Study 1"]}
            ]
        }
        mock_cache_manager.return_value.get.return_value = None
        
        # Call function with None fields (should use default fields)
        result = search_trials("cancer", fields=None, max_results=1)
        
        # Verify results
        assert isinstance(result, dict)
        assert "studies" in result
        assert len(result["studies"]) == 1
    
    def test_get_full_study_not_found(self, mock_clinical_trials_api, mock_cache_manager):
        """Test get_full_study function when study is not found"""
        from src.data_fetcher.fetcher import get_full_study, ClinicalTrialsAPIError
        
        # Set up mocks to raise exception
        mock_clinical_trials_api.return_value.get_study_fields.side_effect = ValueError("No study found for NCT ID NCT99999999 after multiple attempts")
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
        assert "max_results" in result["pagination"]
        # Check if nextPageToken is in the result or pagination
        assert "nextPageToken" in result or "nextPageToken" in result.get("pagination", {})
    
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
        # Check if protocolSection is in the result (actual structure)
        assert "protocolSection" in result
        
        # Verify identification module structure
        assert "identificationModule" in result["protocolSection"]
        assert result["protocolSection"]["identificationModule"]["nctId"] == "NCT12345678"
    
    def test_get_study_fields(self, mock_clinical_trials_api):
        """Test get_study_fields function"""
        from src.data_fetcher.fetcher import get_study_fields
        
        # Set up mocks
        mock_study_fields = {
            'csv': ['NCT Number', 'Study Title', 'Study Status'],
            'json': ['NCTId', 'BriefTitle', 'OverallStatus']
        }
        mock_clinical_trials_api.return_value.study_fields = mock_study_fields
        
        # Call function
        result = get_study_fields()
        
        # Verify results
        assert isinstance(result, dict)
        assert 'csv' in result
        assert 'json' in result
        assert result['json'] == ['NCTId', 'BriefTitle', 'OverallStatus']
    
    def test_get_study_fields_error(self, mock_clinical_trials_api):
        """Test get_study_fields function with API error"""
        from src.data_fetcher.fetcher import get_study_fields, ClinicalTrialsAPIError
        
        # Set up mocks to raise exception
        mock_clinical_trials_api.side_effect = Exception("API error")
        
        # Call function and verify exception
        with pytest.raises(ClinicalTrialsAPIError):
            get_study_fields()
    
    def test_valid_json_fields_constant(self):
        """Test that VALID_JSON_FIELDS contains only valid fields"""
        from src.data_fetcher.fetcher import VALID_JSON_FIELDS
        
        # These are the known valid JSON fields from the API
        valid_fields = [
            "NCTId",
            "BriefTitle",
            "Condition",
            "OverallStatus",
            "BriefSummary",
            "StartDate",
            "CompletionDate"
        ]
        
        # Verify all fields in VALID_JSON_FIELDS are valid
        for field in VALID_JSON_FIELDS:
            assert field in valid_fields, f"Field {field} is not a valid JSON field"
    
    def test_field_mapping_constant(self):
        """Test that FIELD_MAPPING maps to valid API fields"""
        from src.data_fetcher.fetcher import FIELD_MAPPING
        
        # These are the known valid JSON fields from the API
        valid_api_fields = [
            "NCTId",
            "BriefTitle",
            "Condition",
            "OverallStatus",
            "BriefSummary",
            "StartDate",
            "CompletionDate"
        ]
        
        # Verify all mapped fields are valid API fields
        for our_field, api_field in FIELD_MAPPING.items():
            assert api_field in valid_api_fields, f"API field {api_field} for our field {our_field} is not valid"
    
    def test_default_search_fields_constant(self):
        """Test that DEFAULT_SEARCH_FIELDS contains only valid fields"""
        from src.data_fetcher.fetcher import DEFAULT_SEARCH_FIELDS
        
        # These are the known valid JSON fields from the API
        valid_fields = [
            "NCTId",
            "BriefTitle",
            "Condition",
            "OverallStatus",
            "BriefSummary",
            "StartDate",
            "CompletionDate"
        ]
        
        # Verify all fields in DEFAULT_SEARCH_FIELDS are valid
        for field in DEFAULT_SEARCH_FIELDS:
            assert field in valid_fields, f"Field {field} is not a valid search field"
    
    def test_invalid_field_detection_in_constants(self):
        """Test that our field constants don't contain invalid fields like 'EligibilityCriteria'"""
        from src.data_fetcher.fetcher import VALID_JSON_FIELDS, FIELD_MAPPING, DEFAULT_SEARCH_FIELDS
        
        # These are the known valid JSON fields from the API
        valid_fields = [
            "NCTId",
            "BriefTitle",
            "Condition",
            "OverallStatus",
            "BriefSummary",
            "StartDate",
            "CompletionDate"
        ]
        
        # Verify VALID_JSON_FIELDS doesn't contain invalid fields
        for field in VALID_JSON_FIELDS:
            assert field in valid_fields, f"VALID_JSON_FIELDS contains invalid field: {field}"
        
        # Verify FIELD_MAPPING doesn't map to invalid fields
        for our_field, api_field in FIELD_MAPPING.items():
            assert api_field in valid_fields, f"FIELD_MAPPING maps to invalid field: {api_field}"
        
        # Verify DEFAULT_SEARCH_FIELDS doesn't contain invalid fields
        for field in DEFAULT_SEARCH_FIELDS:
            assert field in valid_fields, f"DEFAULT_SEARCH_FIELDS contains invalid field: {field}"
        
        # Specifically check that 'EligibilityCriteria' is not in any of our field constants
        # This was the source of the original bug
        assert 'EligibilityCriteria' not in VALID_JSON_FIELDS, \
            "'EligibilityCriteria' should not be in VALID_JSON_FIELDS"
        assert 'EligibilityCriteria' not in FIELD_MAPPING.values(), \
            "'EligibilityCriteria' should not be in FIELD_MAPPING values"
        assert 'EligibilityCriteria' not in DEFAULT_SEARCH_FIELDS, \
            "'EligibilityCriteria' should not be in DEFAULT_SEARCH_FIELDS"


if __name__ == '__main__':
    pytest.main([__file__])