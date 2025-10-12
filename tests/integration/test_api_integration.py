"""
Integration tests for API interactions with comprehensive mocking and edge cases.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils.fetcher import ClinicalTrialsAPIError, get_full_study, search_trials


@pytest.mark.integration
class TestApiIntegration:
    """Integration tests for API interactions."""

    @patch("src.utils.fetcher.requests.get")
    def test_get_full_study_success(self, mock_get):
        """Test successful API call to get_full_study."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "protocolSection": {"identificationModule": {"nctId": "NCT123456"}}
        }
        mock_get.return_value = mock_response

        trial = get_full_study("NCT123456")
        assert trial is not None
        assert trial["protocolSection"]["identificationModule"]["nctId"] == "NCT123456"
        mock_get.assert_called_once_with(
            "https://clinicaltrials.gov/api/v2/studies/NCT123456",
            params={"format": "json"},
            timeout=30,
        )

    @patch("src.utils.fetcher.requests.get")
    def test_get_full_study_404_error(self, mock_get):
        """Test 404 error when calling get_full_study."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT123456")

    @patch("src.utils.fetcher.requests.get")
    def test_get_full_study_rate_limit_error(self, mock_get):
        """Test rate limit error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT123456")

    @patch("src.utils.fetcher.requests.get")
    def test_get_full_study_timeout_error(self, mock_get):
        """Test timeout error handling."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT123456")

    @patch("src.utils.fetcher.requests.get")
    def test_get_full_study_network_error(self, mock_get):
        """Test network connection error handling."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network is unreachable")

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT123456")

    @patch("src.utils.fetcher.requests.get")
    def test_get_full_study_malformed_json_response(self, mock_get):
        """Test handling of malformed JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT123456")

    @patch("src.utils.fetcher.requests.get")
    def test_get_full_study_empty_response(self, mock_get):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        trial = get_full_study("NCT123456")
        assert trial == {}  # Should return empty dict, not raise error

    @patch("src.utils.fetcher.requests.get")
    def test_search_trials_success(self, mock_get):
        """Test successful API call to search_trials."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT123456"}}}]
        }
        mock_get.return_value = mock_response

        results = search_trials("breast cancer")
        assert results is not None
        assert len(results["studies"]) == 1
        assert (
            results["studies"][0]["protocolSection"]["identificationModule"]["nctId"] == "NCT123456"
        )

    @patch("src.utils.fetcher.requests.get")
    def test_search_trials_with_pagination(self, mock_get):
        """Test search_trials with pagination parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT123456"}}}],
            "nextPageToken": "token123",
        }
        mock_get.return_value = mock_response

        results = search_trials("breast cancer", page_token="token123")
        assert results is not None
        assert "nextPageToken" in results
        mock_get.assert_called_once_with(
            "https://clinicaltrials.gov/api/v2/studies",
            params={
                "format": "json",
                "query.term": "breast cancer",
                "pageSize": 100,
                "pageToken": "token123",
            },
            timeout=30,
        )

    @patch("src.utils.fetcher.requests.get")
    def test_search_trials_api_error(self, mock_get):
        """Test API error when calling search_trials."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            search_trials("breast cancer")

    @patch("src.utils.fetcher.requests.get")
    def test_search_trials_empty_results(self, mock_get):
        """Test search_trials with no results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"studies": []}
        mock_get.return_value = mock_response

        results = search_trials("nonexistent condition")
        assert results is not None
        assert len(results["studies"]) == 0

    @patch("src.utils.fetcher.requests.get")
    def test_search_trials_empty_query(self, mock_get):
        """Test search_trials with empty query (should still make API call)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"studies": []}
        mock_get.return_value = mock_response

        results = search_trials("")  # Empty query
        assert results is not None
        assert len(results["studies"]) == 0
        mock_get.assert_called_once_with(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"format": "json", "query.term": "", "pageSize": 100},
            timeout=30,
        )

    @patch("src.utils.fetcher.requests.get")
    def test_concurrent_api_calls(self, mock_get):
        """Test handling of concurrent API calls."""
        # Mock multiple responses for concurrent calls
        responses = []
        for i in range(3):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "protocolSection": {"identificationModule": {"nctId": f"NCT{i}"}}
            }
            responses.append(mock_resp)

        mock_get.side_effect = responses

        # Simulate concurrent calls
        results = []
        for i in range(3):
            trial = get_full_study(f"NCT{i}")
            results.append(trial)

        assert len(results) == 3
        assert all(trial is not None for trial in results)
        assert mock_get.call_count == 3
