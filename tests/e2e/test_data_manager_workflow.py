#!/usr/bin/env python3
"""
End-to-End Tests for Data Manager Workflow

Tests the complete data manager workflow from bulk data import to storage:
1. Bulk Data Import (download-data) → 2. Validation → 3. Processing (process-patients) → 4. Storage (summarize-patients)
"""

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from mcode_translate import main as mcode_translate_main
from src.cli.patients_processor import main as patients_processor_main
from src.cli.patients_summarizer import main as patients_summarizer_main


class TestDataManagerWorkflowE2E:
    """End-to-end tests for the complete data manager workflow."""

    @pytest.fixture
    def sample_patient_data(self):
        """Sample patient data for testing."""
        return {
            "resourceType": "Bundle",
            "id": "bundle-12345",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "12345",
                        "name": [{"family": "Smith", "given": ["Jane"]}],
                        "gender": "female",
                        "birthDate": "1975-03-15",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "condition-1",
                        "code": {
                            "coding": [
                                {
                                    "system": "http://snomed.info/sct",
                                    "code": "254837009",
                                    "display": "Malignant neoplasm of breast",
                                }
                            ]
                        },
                    }
                },
            ],
        }

    @pytest.fixture
    def mock_download_response(self):
        """Mock HTTP response for data downloads."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [
            b"fake zip content" * 1000
        ]  # Mock zip content
        mock_response.headers = {"content-length": "10000"}
        return mock_response

    @pytest.fixture
    def mock_memory_storage(self):
        """Mock CORE Memory storage."""
        mock_storage = MagicMock()
        mock_storage.store_patient_mcode_summary.return_value = True
        return mock_storage

    @patch("src.utils.data_downloader.requests.get")
    def test_data_manager_workflow_bulk_import(
        self, mock_get, mock_download_response, tmp_path
    ):
        """Test the bulk data import phase of data manager workflow."""
        # Setup mock
        mock_get.return_value = mock_download_response

        # Create temporary directory for downloads
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        # Test CLI execution - simulate download-data command
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Simulate command line args for data download
            test_args = [
                "data",
                "download",
                "--archives",
                "breast_cancer_10_years",
                "--output-dir",
                str(download_dir),
                "--workers",
                "1",
                "--force",
            ]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            except SystemExit:
                pass  # Expected for CLI commands
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Verify download was attempted
        assert "📦 Downloading:" in output
        assert "breast_cancer_10_years" in output

        # Verify mock was called
        mock_get.assert_called()

    def test_data_manager_workflow_validation_and_processing(
        self, sample_patient_data, tmp_path
    ):
        """Test the validation and processing phase of data manager workflow."""
        # Create input file with patient data
        input_file = tmp_path / "patients.ndjson"
        with open(input_file, "w") as f:
            json.dump(sample_patient_data, f)
            f.write("\n")

        # Create output file
        output_file = tmp_path / "mcode_patients.ndjson"

        # Test CLI execution
        args = argparse.Namespace(
            input_file=str(input_file),
            output_file=str(output_file),
            trials=None,
            ingest=False,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            verbose=False,
            log_level="INFO",
            config=None,
        )

        patients_processor_main(args)

        # Verify output file was created
        assert output_file.exists()

        # Verify output contains expected mCODE data structure
        with open(output_file, "r") as f:
            content = f.read().strip()
            assert content
            mcode_data = json.loads(content)
            assert "trial_id" in mcode_data or "patient_id" in mcode_data
            assert "mcode_elements" in mcode_data
            assert (
                "original_trial_data" in mcode_data
                or "original_patient_data" in mcode_data
            )

    def test_data_manager_workflow_storage(self, sample_patient_data, tmp_path):
        """Test the storage phase of data manager workflow."""
        # Create input file with mCODE patient data
        input_file = tmp_path / "mcode_input.ndjson"
        mcode_patient_data = {
            "patient_id": "12345",
            "mcode_elements": {
                "Patient": {"name": "Jane Smith", "gender": "female"},
                "CancerCondition": {"display": "Malignant neoplasm of breast"},
            },
            "original_patient_data": sample_patient_data,
        }
        with open(input_file, "w") as f:
            json.dump(mcode_patient_data, f)
            f.write("\n")

        # Create output file
        output_file = tmp_path / "patient_summaries.ndjson"

        # Test CLI execution
        args = argparse.Namespace(
            input_file=str(input_file),
            output_file=str(output_file),
            ingest=True,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            dry_run=False,
            verbose=False,
            log_level="INFO",
            config=None,
        )

        patients_summarizer_main(args)

        # Verify output file was created
        assert output_file.exists()

        # Verify output contains expected summary data structure
        with open(output_file, "r") as f:
            content = f.read().strip()
            assert content
            summary_data = json.loads(content)
            assert "patient_id" in summary_data
            assert "summary" in summary_data
            assert "mcode_elements" in summary_data

    @patch("src.utils.data_downloader.requests.get")
    def test_complete_data_manager_workflow_integration(
        self, mock_get, sample_patient_data, tmp_path
    ):
        """Test the complete end-to-end data manager workflow integration."""
        # Setup download mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b"fake zip content" * 1000]
        mock_response.headers = {"content-length": "10000"}
        mock_get.return_value = mock_response

        # Create temporary directories
        download_dir = tmp_path / "data" / "synthetic_patients"
        download_dir.mkdir(parents=True)
        patients_file = tmp_path / "patients.ndjson"
        mcode_file = tmp_path / "mcode_patients.ndjson"
        summary_file = tmp_path / "summaries.ndjson"

        # Step 1: Bulk Data Import (simulate download completion)
        # Create a mock downloaded file
        archive_file = (
            download_dir / "breast_cancer" / "10_years" / "breast_cancer_10_years.zip"
        )
        archive_file.parent.mkdir(parents=True)
        with open(archive_file, "wb") as f:
            f.write(b"fake zip content" * 1000)

        # Create mock patient data file (simulating extraction from archive)
        with open(patients_file, "w") as f:
            json.dump(sample_patient_data, f)
            f.write("\n")

        # Step 2: Validation and Processing
        processor_args = argparse.Namespace(
            input_file=str(patients_file),
            output_file=str(mcode_file),
            trials=None,
            ingest=False,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            verbose=False,
            log_level="INFO",
            config=None,
        )

        patients_processor_main(processor_args)
        assert mcode_file.exists()

        # Step 3: Storage
        # Create mock mCODE data file
        mcode_data = {
            "patient_id": "12345",
            "mcode_elements": {
                "Patient": {"name": "Jane Smith", "gender": "female"},
                "CancerCondition": {"display": "Malignant neoplasm of breast"},
            },
            "original_patient_data": sample_patient_data,
        }
        with open(mcode_file, "w") as f:
            json.dump(mcode_data, f)
            f.write("\n")

        summarizer_args = argparse.Namespace(
            input_file=str(mcode_file),
            output_file=str(summary_file),
            ingest=True,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            dry_run=False,
            verbose=False,
            log_level="INFO",
            config=None,
        )

        patients_summarizer_main(summarizer_args)
        assert summary_file.exists()

        # Verify data flow between steps
        with open(summary_file, "r") as f:
            final_output = json.loads(f.read().strip())
            assert "patient_id" in final_output
            assert "summary" in final_output
            assert "mcode_elements" in final_output

    def test_data_manager_workflow_validation_errors(self, tmp_path):
        """Test validation error handling in data manager workflow."""
        # Test with non-existent input file for processor
        nonexistent_file = tmp_path / "nonexistent.ndjson"

        import argparse

        args = argparse.Namespace(
            input_file=str(nonexistent_file),
            output_file=None,
            trials=None,
            ingest=False,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            verbose=False,
            log_level="INFO",
            config=None,
        )

        with pytest.raises(SystemExit):
            patients_processor_main(args)

    @patch("src.utils.data_downloader.requests.get")
    def test_data_manager_workflow_download_failure(self, mock_get):
        """Test handling of download failures in data manager workflow."""
        # Setup mock to simulate download failure
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        # This would normally be tested through the CLI, but for simplicity
        # we'll test the underlying function
        from src.utils.data_downloader import _download_single_archive

        with pytest.raises(Exception, match="404 Not Found"):
            _download_single_archive(
                "http://fake.url/archive.zip", "/tmp/test.zip", "test_archive"
            )

    def test_data_manager_workflow_empty_data_validation(self, tmp_path):
        """Test validation of empty or invalid data files."""
        # Create empty input file
        empty_file = tmp_path / "empty.ndjson"
        empty_file.touch()

        import argparse

        args = argparse.Namespace(
            input_file=str(empty_file),
            output_file=None,
            trials=None,
            ingest=False,
            memory_source="test",
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            workers=1,
            verbose=False,
            log_level="INFO",
            config=None,
        )

        # Should handle empty file gracefully or raise appropriate error
        with pytest.raises(SystemExit):  # Expected behavior for invalid input
            patients_processor_main(args)
