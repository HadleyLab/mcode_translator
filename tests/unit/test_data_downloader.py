"""
Unit tests for data_downloader module.
"""

from unittest.mock import Mock, mock_open, patch

import pytest
import requests

from src.utils.data_downloader import (
    _download_single_archive,
    _download_single_file,
    download_multiple_files,
    download_synthetic_patient_archives,
    download_synthetic_patient_archives_concurrent,
    get_archive_paths,
)


class TestDownloadSyntheticPatientArchives:
    """Test download_synthetic_patient_archives function."""

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.makedirs")
    @patch("src.utils.data_downloader.os.path.exists")
    def test_download_archives_success(self, mock_exists, mock_makedirs, mock_get):
        """Test successful archive download."""
        mock_exists.return_value = False
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        with patch("builtins.open", mock_open()) as mock_file:
            result = download_synthetic_patient_archives()

        assert len(result) == 4  # 2 cancer types * 2 durations
        assert "mixed_cancer_10_years.zip" in result
        assert "breast_cancer_lifetime.zip" in result

        # Verify directories were created
        assert mock_makedirs.call_count == 4

        # Verify files were opened for writing
        assert mock_file.call_count == 4

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.path.exists")
    def test_download_archives_already_exists(self, mock_exists, mock_get):
        """Test skipping download when archive already exists."""
        mock_exists.return_value = True

        result = download_synthetic_patient_archives()

        assert len(result) == 4
        # Should not call requests.get
        mock_get.assert_not_called()

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.makedirs")
    @patch("src.utils.data_downloader.os.path.exists")
    @patch("src.utils.data_downloader.os.remove")
    def test_download_archives_request_failure(
        self, mock_remove, mock_exists, mock_makedirs, mock_get
    ):
        """Test handling of download request failure."""
        # Files don't exist initially
        mock_exists.return_value = False
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("Download failed")
        mock_get.return_value = mock_response

        with patch("builtins.open", mock_open()):
            result = download_synthetic_patient_archives()

        # Should return empty dict on failure (no successful downloads)
        assert result == {}

        # Should attempt to remove partial downloads (but only if file was created)
        # In this case, since raise_for_status fails before writing, no cleanup occurs
        mock_remove.assert_not_called()

    @patch("src.utils.data_downloader.os.walk")
    def test_get_archive_paths(self, mock_walk):
        """Test getting archive paths."""
        mock_walk.return_value = [
            ("data/synthetic_patients", ["mixed_cancer"], []),
            ("data/synthetic_patients/mixed_cancer", ["10_years"], []),
            (
                "data/synthetic_patients/mixed_cancer/10_years",
                [],
                ["mixed_cancer_10_years.zip"],
            ),
            ("data/synthetic_patients/breast_cancer", ["lifetime"], []),
            (
                "data/synthetic_patients/breast_cancer/lifetime",
                [],
                ["breast_cancer_lifetime.zip"],
            ),
        ]

        result = get_archive_paths()

        assert len(result) == 2
        assert "mixed_cancer_10_years.zip" in result
        assert "breast_cancer_lifetime.zip" in result


class TestDownloadSyntheticPatientArchivesConcurrent:
    """Test download_synthetic_patient_archives_concurrent function."""

    @patch("src.utils.data_downloader.TaskQueue")
    @patch("src.utils.data_downloader.create_task")
    @patch("src.utils.data_downloader.os.makedirs")
    @patch("src.utils.data_downloader.os.path.exists")
    def test_concurrent_download_success(
        self, mock_exists, mock_makedirs, mock_create_task, mock_task_queue
    ):
        """Test successful concurrent archive download."""
        mock_exists.return_value = False

        # Mock task creation
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        # Mock task queue
        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance
        mock_queue_instance.execute_tasks.return_value = [
            Mock(task_id="download_test.zip", success=True)
        ]

        result = download_synthetic_patient_archives_concurrent()

        assert len(result) == 4
        mock_create_task.assert_called()
        mock_queue_instance.execute_tasks.assert_called()

    @patch("src.utils.data_downloader.os.path.exists")
    def test_concurrent_download_already_exists(self, mock_exists):
        """Test skipping concurrent download when archives exist."""
        mock_exists.return_value = True

        result = download_synthetic_patient_archives_concurrent()

        assert len(result) == 4
        # Should not create any tasks


class TestDownloadSingleArchive:
    """Test _download_single_archive function."""

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.makedirs")
    @patch("src.utils.data_downloader.os.path.exists")
    @patch("src.utils.data_downloader.os.remove")
    def test_download_single_archive_success(
        self, mock_remove, mock_exists, mock_makedirs, mock_get
    ):
        """Test successful single archive download."""
        mock_exists.return_value = True  # Parent directory exists

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        with patch("builtins.open", mock_open()) as mock_file:
            result = _download_single_archive(
                "http://example.com/test.zip", "/tmp/test.zip", "test.zip"
            )

        assert result == "/tmp/test.zip"
        mock_file.assert_called_once()

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.makedirs")
    @patch("src.utils.data_downloader.os.path.exists")
    @patch("src.utils.data_downloader.os.remove")
    def test_download_single_archive_failure(
        self, mock_remove, mock_exists, mock_makedirs, mock_get
    ):
        """Test single archive download failure."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Network error")
        mock_get.return_value = mock_response

        # Mock that file exists for cleanup
        mock_exists.return_value = True

        with pytest.raises(Exception):
            _download_single_archive("http://example.com/test.zip", "/tmp/test.zip", "test.zip")

        # Should clean up partial download
        mock_remove.assert_called_once_with("/tmp/test.zip")

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.makedirs")
    def test_download_single_archive_large_file_progress(self, mock_makedirs, mock_get):
        """Test progress logging for large files."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {"content-length": str(20 * 1024 * 1024)}  # 20MB
        # Create chunks that sum to more than 10MB
        large_chunk = b"x" * (11 * 1024 * 1024)  # 11MB
        mock_response.iter_content.return_value = [large_chunk]
        mock_get.return_value = mock_response

        with patch("builtins.open", mock_open()):
            _download_single_archive("http://example.com/large.zip", "/tmp/large.zip", "large.zip")

        # Should have logged progress (can't easily test logger calls without more complex mocking)


class TestDownloadMultipleFiles:
    """Test download_multiple_files function."""

    @patch("src.utils.data_downloader.TaskQueue")
    @patch("src.utils.data_downloader.create_task")
    @patch("src.utils.data_downloader.os.path.exists")
    def test_download_multiple_files_success(self, mock_exists, mock_create_task, mock_task_queue):
        """Test successful multiple file download."""
        mock_exists.return_value = False

        # Mock task creation
        mock_task = Mock()
        mock_create_task.return_value = mock_task

        # Mock task queue
        mock_queue_instance = Mock()
        mock_task_queue.return_value = mock_queue_instance
        mock_queue_instance.execute_tasks.return_value = [
            Mock(task_id="download_file1.txt", success=True),
            Mock(task_id="download_file2.txt", success=True),
        ]

        file_urls = {
            "file1.txt": "http://example.com/file1.txt",
            "file2.txt": "http://example.com/file2.txt",
        }
        result = download_multiple_files(file_urls)

        assert len(result) == 2
        assert "file1.txt" in result
        assert "file2.txt" in result

    @patch("src.utils.data_downloader.os.path.exists")
    def test_download_multiple_files_already_exist(self, mock_exists):
        """Test skipping download when files already exist."""
        mock_exists.return_value = True

        file_urls = {"file1.txt": "http://example.com/file1.txt"}
        result = download_multiple_files(file_urls)

        assert len(result) == 1
        assert "file1.txt" in result


class TestDownloadSingleFile:
    """Test _download_single_file function."""

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.makedirs")
    def test_download_single_file_success(self, mock_makedirs, mock_get):
        """Test successful single file download."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_get.return_value = mock_response

        with patch("builtins.open", mock_open()) as mock_file:
            result = _download_single_file(
                "http://example.com/test.txt", "/tmp/test.txt", "test.txt"
            )

        assert result == "/tmp/test.txt"
        mock_file.assert_called_once()

    @patch("src.utils.data_downloader.requests.get")
    @patch("src.utils.data_downloader.os.makedirs")
    @patch("src.utils.data_downloader.os.path.exists")
    @patch("src.utils.data_downloader.os.remove")
    def test_download_single_file_failure(self, mock_remove, mock_exists, mock_makedirs, mock_get):
        """Test single file download failure."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("Network error")
        mock_get.return_value = mock_response

        # Mock that file exists for cleanup
        mock_exists.return_value = True

        with pytest.raises(Exception):
            _download_single_file("http://example.com/test.txt", "/tmp/test.txt", "test.txt")

        # Should clean up partial download
        mock_remove.assert_called_once_with("/tmp/test.txt")


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_custom_archives_config(self):
        """Test with custom archives configuration."""
        custom_config = {"test_cancer": {"5_years": "http://example.com/test.zip"}}

        with patch("src.utils.data_downloader.requests.get") as mock_get, patch(
            "src.utils.data_downloader.os.makedirs"
        ), patch("src.utils.data_downloader.os.path.exists", return_value=False):

            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"data"]
            mock_get.return_value = mock_response

            with patch("builtins.open", mock_open()):
                result = download_synthetic_patient_archives(archives_config=custom_config)

            assert len(result) == 1
            assert "test_cancer_5_years.zip" in result

    def test_force_download(self):
        """Test force download option."""
        with patch("src.utils.data_downloader.requests.get") as mock_get, patch(
            "src.utils.data_downloader.os.makedirs"
        ), patch("src.utils.data_downloader.os.path.exists", return_value=True):

            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"data"]
            mock_get.return_value = mock_response

            with patch("builtins.open", mock_open()):
                download_synthetic_patient_archives(force_download=True)

            # Should download even though files exist
            assert mock_get.call_count == 4

    def test_directory_creation(self):
        """Test directory creation."""
        with patch("src.utils.data_downloader.requests.get") as mock_get, patch(
            "src.utils.data_downloader.os.makedirs"
        ) as mock_makedirs, patch("src.utils.data_downloader.os.path.exists", return_value=False):

            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_content.return_value = [b"data"]
            mock_get.return_value = mock_response

            with patch("builtins.open", mock_open()):
                download_synthetic_patient_archives()

            # Should create directories for each archive
            assert mock_makedirs.call_count == 4

    def test_empty_file_urls(self):
        """Test download_multiple_files with empty URLs."""
        result = download_multiple_files({})
        assert result == {}

    def test_concurrent_download_with_custom_workers(self):
        """Test concurrent download with custom worker count."""
        with patch("src.utils.data_downloader.os.path.exists", return_value=True):
            result = download_synthetic_patient_archives_concurrent(max_workers=2)
            assert len(result) == 4


if __name__ == "__main__":
    pytest.main([__file__])
