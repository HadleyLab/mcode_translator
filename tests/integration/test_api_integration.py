"""
Integration tests for API interactions with proper mocking.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock
from src.utils.fetcher import ClinicalTrialsAPIError, get_full_study, search_trials


@pytest.mark.integration
class TestApiIntegration:
    """Integration tests for API interactions."""

    @patch('src.utils.fetcher.requests.get')
    def test_get_full_study_success(self, mock_get):
        """Test successful API call to get_full_study."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"protocolSection": {"identificationModule": {"nctId": "NCT123456"}}}
        mock_get.return_value = mock_response

        trial = get_full_study("NCT123456")
        assert trial is not None
        assert trial["protocolSection"]["identificationModule"]["nctId"] == "NCT123456"
        mock_get.assert_called_once_with("https://clinicaltrials.gov/api/v2/studies/NCT123456", params={"format": "json"}, timeout=30)

    @patch('src.utils.fetcher.requests.get')
    def test_get_full_study_404_error(self, mock_get):
        """Test 404 error when calling get_full_study."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT123456")

    @patch('src.utils.fetcher.requests.get')
    def test_search_trials_success(self, mock_get):
        """Test successful API call to search_trials."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT123456"}}}]}
        mock_get.return_value = mock_response

        results = search_trials("breast cancer")
        assert results is not None
        assert len(results["studies"]) == 1
        assert results["studies"][0]["protocolSection"]["identificationModule"]["nctId"] == "NCT123456"

    @patch('src.utils.fetcher.requests.get')
    def test_search_trials_api_error(self, mock_get):
        """Test API error when calling search_trials."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            search_trials("breast cancer")