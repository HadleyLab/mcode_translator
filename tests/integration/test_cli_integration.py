"""
Integration tests for CLI functionality.
Tests end-to-end CLI operations and workflows.
"""

import argparse
import io
from unittest.mock import Mock, patch
import pytest

from src.cli.patients_fetcher import (
    main as patients_fetcher_main,
    create_parser as create_patients_parser,
)
from src.cli.trials_fetcher import (
    main as trials_fetcher_main,
    create_parser as create_trials_parser,
)
from src.shared.cli_utils import McodeCLI


class TestPatientsFetcherCLI:
    """Test patients fetcher CLI integration."""

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_patients_fetcher_list_archives(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test listing available archives."""
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.list_available_archives.return_value = ["archive1", "archive2"]
        mock_workflow_class.return_value = mock_workflow

        # Mock config
        mock_config = Mock()
        mock_create_config.return_value = mock_config

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            # Create args for list-archives
            parser = create_patients_parser()
            args = parser.parse_args(["--list-archives"])

            patients_fetcher_main(args)

        # Check output
        output = captured_output.getvalue()
        assert "Available patient archives:" in output
        assert "archive1" in output
        assert "archive2" in output

        # Verify workflow was called correctly
        mock_workflow_class.assert_called_once_with(mock_config)
        mock_workflow.list_available_archives.assert_called_once()

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_patients_fetcher_fetch_patients_success(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test successful patient fetching."""
        # Mock workflow
        mock_workflow = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.metadata = {
            "total_fetched": 5,
            "fetch_type": "bulk",
            "archive_path": "/path/to/archive",
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock config
        mock_config = Mock()
        mock_create_config.return_value = mock_config

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            # Create args for fetching
            parser = create_patients_parser()
            args = parser.parse_args(["--archive", "test_archive", "--limit", "5"])

            patients_fetcher_main(args)

        # Check output
        output = captured_output.getvalue()
        assert "Patients fetch completed successfully!" in output
        assert "Total patients fetched: 5" in output
        assert "Results written to stdout" in output

        # Verify workflow was called correctly
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.execute.assert_called_once_with(
            archive_path="test_archive", limit=5
        )

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_patients_fetcher_fetch_with_output_file(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test patient fetching with output file."""
        # Mock workflow
        mock_workflow = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.metadata = {"total_fetched": 3, "archive_path": "/path/to/archive"}
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock config
        mock_config = Mock()
        mock_create_config.return_value = mock_config

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            # Create args for fetching with output file
            parser = create_patients_parser()
            args = parser.parse_args(
                ["--archive", "test_archive", "--out", "output.ndjson"]
            )

            patients_fetcher_main(args)

        # Check output
        output = captured_output.getvalue()
        assert "Results saved to: output.ndjson" in output

        # Verify workflow was called with output_path
        mock_workflow.execute.assert_called_once_with(
            archive_path="test_archive", limit=10, output_path="output.ndjson"
        )

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_patients_fetcher_fetch_failure(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test patient fetching failure."""
        # Mock workflow
        mock_workflow = Mock()
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "Archive not found"
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock config
        mock_config = Mock()
        mock_create_config.return_value = mock_config

        # Capture stdout and stderr
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output), patch(
            "sys.stderr", captured_output
        ), pytest.raises(SystemExit, match="1"):
            # Create args for fetching
            parser = create_patients_parser()
            args = parser.parse_args(["--archive", "missing_archive"])

            patients_fetcher_main(args)

        # Check output
        output = captured_output.getvalue()
        assert "Patients fetch failed: Archive not found" in output

    def test_patients_fetcher_missing_archive_arg(self):
        """Test error when archive argument is missing."""
        from unittest.mock import patch

        with patch("src.shared.cli_utils.McodeCLI.setup_logging"), patch(
            "src.shared.cli_utils.McodeCLI.create_config"
        ), pytest.raises(SystemExit):
            # Create args with no archive
            args = argparse.Namespace(
                verbose=False,
                log_level="INFO",
                config=None,
                archive=None,
                patient_id=None,
                limit=10,
                list_archives=False,
                output_file=None,
            )
            patients_fetcher_main(args)


class TestTrialsFetcherCLI:
    """Test trials fetcher CLI integration."""

    @patch("src.cli.trials_fetcher.TrialsFetcherWorkflow")
    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_trials_fetcher_search_success(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test successful trial search."""
        # Mock workflow
        mock_workflow = Mock()
        mock_result = Mock()
        mock_result.success = True
        mock_result.metadata = {"total_found": 25, "query": "breast cancer"}
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock config
        mock_config = Mock()
        mock_create_config.return_value = mock_config

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            # Create args for searching
            parser = create_trials_parser()
            args = parser.parse_args(["--condition", "breast cancer", "--limit", "25"])

            trials_fetcher_main(args)

        # Check output
        output = captured_output.getvalue()
        assert "Trials fetch completed successfully!" in output

        # Verify workflow was called correctly
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.execute.assert_called_once()


class TestCLIErrors:
    """Test CLI error handling."""

    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_keyboard_interrupt_handling(self, mock_setup_logging):
        """Test handling of keyboard interrupt."""
        with patch(
            "src.cli.patients_fetcher.PatientsFetcherWorkflow"
        ) as mock_workflow_class:
            mock_workflow_class.side_effect = KeyboardInterrupt()

            # Capture stdout
            captured_output = io.StringIO()
            with patch("sys.stdout", captured_output), pytest.raises(
                SystemExit, match="130"
            ):
                # Create args
                parser = create_patients_parser()
                args = parser.parse_args(["--archive", "test"])

                patients_fetcher_main(args)

            # Check output
            output = captured_output.getvalue()
            assert "Operation cancelled by user" in output

    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_unexpected_error_handling(self, mock_setup_logging):
        """Test handling of unexpected errors."""
        with patch(
            "src.cli.patients_fetcher.PatientsFetcherWorkflow"
        ) as mock_workflow_class:
            mock_workflow_class.side_effect = Exception("Unexpected error")

            # Capture stdout
            captured_output = io.StringIO()
            with patch("sys.stdout", captured_output), pytest.raises(
                SystemExit, match="1"
            ):
                # Create args
                parser = create_patients_parser()
                args = parser.parse_args(["--archive", "test"])

                patients_fetcher_main(args)

            # Check output
            output = captured_output.getvalue()
            assert "Unexpected error: Unexpected error" in output


class TestMcodeCLIIntegration:
    """Test McodeCLI utility integration."""

    def test_cli_utils_add_core_args(self):
        """Test that core args are added to parser."""
        parser = argparse.ArgumentParser()
        McodeCLI.add_core_args(parser)

        # Check that common args are present
        help_text = parser.format_help()
        assert "--verbose" in help_text or "-v" in help_text
        assert "--config" in help_text

    @patch("src.utils.config.Config")
    def test_cli_utils_create_config(self, mock_config_class):
        """Test config creation."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        args = Mock()
        args.config = None

        result = McodeCLI.create_config(args)

        assert result == mock_config
        mock_config_class.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
