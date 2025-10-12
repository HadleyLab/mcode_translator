"""
Unit tests for fetcher module.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from src.utils.fetcher import (
    ClinicalTrialsAPIError,
    _fetch_single_study_with_error_handling,
    _search_single_page,
    calculate_total_studies,
    get_full_studies_batch,
    get_full_study,
    search_multiple_queries,
    search_trials,
    search_trials_parallel,
)


class TestSearchTrials:
    """Test search_trials function."""

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_search_trials_success(self, mock_config, mock_sleep, mock_get):
        """Test successful trial search."""
        # Mock config
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "studies": [{"nctId": "NCT001"}],
            "nextPageToken": "token123",
        }
        mock_get.return_value = mock_response

        result = search_trials("breast cancer", ["nctId"], 50, "page1")

        assert "studies" in result
        assert result["pagination"]["max_results"] == 50
        assert result["nextPageToken"] == "token123"

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["params"]["query.term"] == "breast cancer"
        assert call_args[1]["params"]["pageSize"] == 50
        assert call_args[1]["params"]["fields"] == "nctId"
        assert call_args[1]["params"]["pageToken"] == "page1"

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_search_trials_no_fields(self, mock_config, mock_sleep, mock_get):
        """Test search without specifying fields."""
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"studies": []}
        mock_get.return_value = mock_response

        search_trials("cancer")

        # Should not have fields parameter
        call_args = mock_get.call_args
        assert "fields" not in call_args[1]["params"]

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_search_trials_api_error(self, mock_config, mock_sleep, mock_get):
        """Test API error handling."""
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Network error")
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            search_trials("cancer")


class TestGetFullStudiesBatch:
    """Test get_full_studies_batch function."""

    @patch("src.utils.fetcher.TaskQueue")
    @patch("src.utils.fetcher.create_task")
    def test_batch_fetch_success(self, mock_create_task, mock_task_queue):
        """Test successful batch fetch."""
        # Mock task creation
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        # Mock task queue
        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance
        mock_queue_instance.execute_tasks.return_value = [
            Mock(task_id="fetch_NCT001", success=True, result={"study": "data1"}),
            Mock(task_id="fetch_NCT002", success=True, result={"study": "data2"}),
        ]

        result = get_full_studies_batch(["NCT001", "NCT002"])

        assert "NCT001" in result
        assert "NCT002" in result
        assert result["NCT001"] == {"study": "data1"}
        assert result["NCT002"] == {"study": "data2"}

    @patch("src.utils.fetcher.TaskQueue")
    @patch("src.utils.fetcher.create_task")
    def test_batch_fetch_partial_failure(self, mock_create_task, mock_task_queue):
        """Test batch fetch with some failures."""
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance
        mock_queue_instance.execute_tasks.return_value = [
            Mock(task_id="fetch_NCT001", success=True, result={"study": "data1"}),
            Mock(task_id="fetch_NCT002", success=False, error="Not found"),
        ]

        result = get_full_studies_batch(["NCT001", "NCT002"])

        assert result["NCT001"] == {"study": "data1"}
        assert result["NCT002"] == {"error": "Not found"}

    def test_batch_fetch_empty_list(self):
        """Test batch fetch with empty list."""
        result = get_full_studies_batch([])
        assert result == {}


class TestSearchTrialsParallel:
    """Test search_trials_parallel function."""

    @patch("src.utils.fetcher.calculate_total_studies")
    @patch("src.utils.fetcher.TaskQueue")
    @patch("src.utils.fetcher.create_task")
    @patch("src.utils.fetcher.Config")
    def test_parallel_search_success(
        self, mock_config, mock_create_task, mock_task_queue, mock_calc_total
    ):
        """Test successful parallel search."""
        # Mock config
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        # Mock total calculation
        mock_calc_total.return_value = {"total_studies": 250, "total_pages": 3}

        # Mock task creation
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        # Mock task queue
        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance
        mock_queue_instance.execute_tasks.return_value = [
            Mock(task_id="page_0", success=True, result={"studies": [{"id": "1"}]}),
            Mock(task_id="page_1", success=True, result={"studies": [{"id": "2"}]}),
            Mock(task_id="page_2", success=True, result={"studies": [{"id": "3"}]}),
        ]

        result = search_trials_parallel("cancer", max_results=250, page_size=100)

        assert len(result["studies"]) == 3
        assert result["totalCount"] == 3
        assert result["pagination"]["total_pages"] == 3
        assert result["pagination"]["successful_pages"] == 3
        assert result["pagination"]["failed_pages"] == 0

    @patch("src.utils.fetcher.calculate_total_studies")
    def test_parallel_search_no_results(self, mock_calc_total):
        """Test parallel search with no results."""
        mock_calc_total.return_value = {"total_studies": 0, "total_pages": 0}

        result = search_trials_parallel("nonexistent")

        assert result["studies"] == []
        assert result["totalCount"] == 0
        assert result["pagination"]["total_pages"] == 0


class TestSearchMultipleQueries:
    """Test search_multiple_queries function."""

    @patch("src.utils.fetcher.TaskQueue")
    @patch("src.utils.fetcher.create_task")
    def test_multiple_queries_success(self, mock_create_task, mock_task_queue):
        """Test successful multiple queries."""
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance
        mock_queue_instance.execute_tasks.return_value = [
            Mock(
                task_id="query_abc12345",
                success=True,
                result={"studies": [{"id": "1"}]},
            ),
            Mock(
                task_id="query_def67890",
                success=True,
                result={"studies": [{"id": "2"}]},
            ),
        ]

        queries = ["cancer", "diabetes"]
        result = search_multiple_queries(queries)

        assert len(result) == 2
        assert "cancer" in result
        assert "diabetes" in result

    def test_multiple_queries_empty_list(self):
        """Test multiple queries with empty list."""
        result = search_multiple_queries([])
        assert result == {}


class TestGetFullStudy:
    """Test get_full_study function."""

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_get_full_study_success(self, mock_config, mock_sleep, mock_get):
        """Test successful full study fetch."""
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"study": {"nctId": "NCT001"}}
        mock_get.return_value = mock_response

        result = get_full_study("NCT001")

        assert result["study"]["nctId"] == "NCT001"

        # Verify URL construction
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://api.example.com/NCT001"

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_get_full_study_invalid_response(self, mock_config, mock_sleep, mock_get):
        """Test invalid response handling."""
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = None  # Invalid response
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT001")

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_get_full_study_non_dict_response(self, mock_config, mock_sleep, mock_get):
        """Test non-dict response handling."""
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = "invalid"  # Non-dict response
        mock_get.return_value = mock_response

        with pytest.raises(ClinicalTrialsAPIError):
            get_full_study("NCT001")


class TestCalculateTotalStudies:
    """Test calculate_total_studies function."""

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_calculate_total_success(self, mock_config, mock_sleep, mock_get):
        """Test successful total calculation."""
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"totalCount": 1500}
        mock_get.return_value = mock_response

        result = calculate_total_studies("cancer", page_size=100)

        assert result["total_studies"] == 1500
        assert result["total_pages"] == 15  # 1500 / 100
        assert result["page_size"] == 100

    @patch("src.utils.fetcher.requests.get")
    @patch("src.utils.fetcher.time.sleep")
    @patch("src.utils.fetcher.Config")
    def test_calculate_total_zero_results(self, mock_config, mock_sleep, mock_get):
        """Test calculation with zero results."""
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0.1
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://api.example.com"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"totalCount": 0}
        mock_get.return_value = mock_response

        result = calculate_total_studies("nonexistent")

        assert result["total_studies"] == 0
        assert result["total_pages"] == 0


class TestHelperFunctions:
    """Test helper functions."""

    @patch("src.utils.fetcher.get_full_study")
    def test_fetch_single_study_success(self, mock_get_full_study):
        """Test successful single study fetch."""
        mock_get_full_study.return_value = {"study": "data"}

        result = _fetch_single_study_with_error_handling("NCT001")

        assert result == {"study": "data"}

    @patch("src.utils.fetcher.get_full_study")
    def test_fetch_single_study_failure(self, mock_get_full_study):
        """Test single study fetch failure."""
        mock_get_full_study.side_effect = Exception("API error")

        with pytest.raises(Exception):
            _fetch_single_study_with_error_handling("NCT001")

    @patch("src.utils.fetcher.search_trials")
    def test_search_single_page(self, mock_search_trials):
        """Test single page search."""
        mock_search_trials.return_value = {"studies": [{"id": "1"}]}

        result = _search_single_page("cancer", ["nctId"], 50, "token")

        assert result == {"studies": [{"id": "1"}]}
        mock_search_trials.assert_called_once_with("cancer", ["nctId"], 50, "token")


class TestClinicalTrialsAPIError:
    """Test ClinicalTrialsAPIError exception."""

    def test_exception_creation(self):
        """Test exception creation."""
        error = ClinicalTrialsAPIError("Test error")
        assert str(error) == "Test error"


if __name__ == "__main__":
    pytest.main([__file__])
