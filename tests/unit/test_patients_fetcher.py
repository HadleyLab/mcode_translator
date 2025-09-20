#!/usr/bin/env python3
"""
Unit tests for patients_fetcher CLI module.

Tests the command-line interface for fetching synthetic patient data,
including argument parsing, validation, and workflow execution.
"""

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.cli.patients_fetcher import create_parser, main


class TestPatientsFetcherCLI:
    """Test the patients_fetcher CLI module."""

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description == "Fetch synthetic patient data from archives"

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "archive" in actions
        assert "patient_id" in actions
        assert "limit" in actions
        assert "list_archives" in actions
        assert "output_file" in actions

    def test_create_parser_archive_argument(self):
        """Test the archive argument configuration."""
        parser = create_parser()

        # Find the archive action
        archive_action = None
        for action in parser._actions:
            if action.dest == "archive":
                archive_action = action
                break

        assert archive_action is not None
        assert archive_action.help == "Patient archive identifier (e.g., breast_cancer_10_years)"

    def test_create_parser_limit_argument(self):
        """Test the limit argument configuration."""
        parser = create_parser()

        # Find the limit action
        limit_action = None
        for action in parser._actions:
            if action.dest == "limit":
                limit_action = action
                break

        assert limit_action is not None
        assert limit_action.type == int
        assert limit_action.default == 10
        assert "Maximum number of patients to fetch" in limit_action.help

    def test_create_parser_list_archives_flag(self):
        """Test the list-archives flag configuration."""
        parser = create_parser()

        # Find the list_archives action
        list_action = None
        for action in parser._actions:
            if action.dest == "list_archives":
                list_action = action
                break

        assert list_action is not None
        assert isinstance(list_action, argparse._StoreTrueAction)
        assert "List available patient archives" in list_action.help

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_list_archives_success(self, mock_workflow_class, capsys):
        """Test successful listing of archives."""
        # Mock the workflow
        mock_workflow = MagicMock()
        mock_workflow.list_available_archives.return_value = [
            "breast_cancer_10_years",
            "breast_cancer_lifetime",
            "mixed_cancer_10_years"
        ]
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with all required attributes
        args = argparse.Namespace(
            list_archives=True,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Call main
        main(args)

        # Verify workflow was created and called correctly
        mock_workflow_class.assert_called_once()
        mock_workflow.list_available_archives.assert_called_once()

        # Check output
        captured = capsys.readouterr()
        assert "📚 Available patient archives:" in captured.out
        assert "breast_cancer_10_years" in captured.out
        assert "breast_cancer_lifetime" in captured.out

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_fetch_single_patient_success(self, mock_workflow_class, capsys):
        """Test successful fetching of a single patient."""
        # Mock the workflow
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = {
            "total_fetched": 1,
            "fetch_type": "single_patient",
            "archive_path": "breast_cancer_10_years"
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with all required attributes
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            patient_id="patient_123",
            limit=10,
            output_file=None,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Call main
        main(args)

        # Verify workflow was called with correct parameters
        mock_workflow_class.assert_called_once()
        mock_workflow.execute.assert_called_once_with(
            archive_path="breast_cancer_10_years",
            patient_id="patient_123",
            limit=10
        )

        # Check output
        captured = capsys.readouterr()
        assert "✅ Patients fetch completed successfully!" in captured.out
        assert "📊 Total patients fetched: 1" in captured.out
        assert "🔍 Fetch type: single_patient" in captured.out

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_fetch_multiple_patients_with_output_file(self, mock_workflow_class, capsys):
        """Test fetching multiple patients with output file."""
        # Mock the workflow
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = {
            "total_fetched": 5,
            "fetch_type": "multiple_patients",
            "archive_path": "mixed_cancer_10_years",
            "output_path": "patients.ndjson"
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with all required attributes
        args = argparse.Namespace(
            archive="mixed_cancer_10_years",
            patient_id=None,
            limit=5,
            output_file="patients.ndjson",
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Call main
        main(args)

        # Verify workflow was called with correct parameters
        mock_workflow_class.assert_called_once()
        mock_workflow.execute.assert_called_once_with(
            archive_path="mixed_cancer_10_years",
            limit=5,
            output_path="patients.ndjson"
        )

        # Check output
        captured = capsys.readouterr()
        assert "✅ Patients fetch completed successfully!" in captured.out
        assert "📊 Total patients fetched: 5" in captured.out
        assert "💾 Results saved to: patients.ndjson" in captured.out

    def test_parser_creation_with_core_args(self):
        """Test that parser includes all core arguments from McodeCLI."""
        parser = create_parser()

        # Check that core arguments are present
        actions = {action.dest for action in parser._actions if action.dest}
        expected_core_args = {"verbose", "log_level", "config"}

        # Verify core arguments are included
        for arg in expected_core_args:
            assert arg in actions, f"Core argument '{arg}' not found in parser"

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_workflow_execution_failure(self, mock_workflow_class, capsys):
        """Test handling of workflow execution failure."""
        # Mock the workflow to return failure
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Archive not found"
        mock_result.metadata = {}
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args
        args = argparse.Namespace(
            archive="invalid_archive",
            patient_id=None,
            limit=10,
            output_file=None,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            main(args)

            # Check output
            captured = capsys.readouterr()
            assert "❌ Patients fetch failed: Archive not found" in captured.out

            # Verify sys.exit was called with error code 1
            mock_exit.assert_called_once_with(1)

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_unexpected_exception_handling(self, mock_workflow_class, capsys):
        """Test handling of unexpected exceptions."""
        # Mock the workflow to raise an exception
        mock_workflow = MagicMock()
        mock_workflow.execute.side_effect = Exception("Unexpected error")
        mock_workflow_class.return_value = mock_workflow

        # Create mock args
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            patient_id=None,
            limit=10,
            output_file=None,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            main(args)

            # Check output
            captured = capsys.readouterr()
            assert "❌ Unexpected error: Unexpected error" in captured.out

            # Verify sys.exit was called with error code 1
            mock_exit.assert_called_once_with(1)

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_keyboard_interrupt_handling(self, mock_workflow_class, capsys):
        """Test handling of keyboard interrupt."""
        # Mock the workflow to raise KeyboardInterrupt
        mock_workflow = MagicMock()
        mock_workflow.execute.side_effect = KeyboardInterrupt()
        mock_workflow_class.return_value = mock_workflow

        # Create mock args
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            patient_id=None,
            limit=10,
            output_file=None,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            main(args)

            # Check output
            captured = capsys.readouterr()
            assert "⏹️  Operation cancelled by user" in captured.out

            # Verify sys.exit was called with code 130 (standard for SIGINT)
            mock_exit.assert_called_once_with(130)

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_verbose_exception_traceback(self, mock_workflow_class, capsys):
        """Test that verbose mode shows exception traceback."""
        # Mock the workflow to raise an exception
        mock_workflow = MagicMock()
        mock_workflow.execute.side_effect = ValueError("Test error")
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with verbose=True
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            patient_id=None,
            limit=10,
            output_file=None,
            list_archives=False,
            verbose=True,
            log_level="DEBUG",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Mock traceback.print_exc
        with patch("sys.exit") as mock_exit:
            with patch("traceback.print_exc") as mock_traceback:
                main(args)

                # Verify traceback was printed in verbose mode
                mock_traceback.assert_called_once()

    def test_main_without_args_uses_sys_argv(self):
        """Test that main() without args uses sys.argv."""
        with patch("src.cli.patients_fetcher.create_parser") as mock_parser:
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parser_instance = MagicMock()
                mock_parser.return_value = mock_parser_instance

                # Call main without args
                main()

                # Verify parser was created and parse_args was called
                mock_parser.assert_called_once()
                mock_parser_instance.parse_args.assert_called_once()

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_stdout_output_mode(self, mock_workflow_class, capsys):
        """Test output to stdout when no output file is specified."""
        # Mock the workflow
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = {
            "total_fetched": 3,
            "fetch_type": "multiple_patients",
            "archive_path": "breast_cancer_10_years"
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args without output file
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            patient_id=None,
            limit=3,
            output_file=None,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Call main
        main(args)

        # Check output
        captured = capsys.readouterr()
        assert "📤 Results written to stdout: 3 records (NDJSON format)" in captured.out

    def test_parser_help_text_formatting(self):
        """Test that parser help text is properly formatted."""
        parser = create_parser()

        # Check that epilog contains examples
        assert "Examples:" in parser.epilog
        assert "breast_cancer_10_years" in parser.epilog
        assert "--list-archives" in parser.epilog

    @patch("src.cli.patients_fetcher.McodeCLI.create_config")
    @patch("src.cli.patients_fetcher.McodeCLI.setup_logging")
    def test_main_config_and_logging_setup(self, mock_setup_logging, mock_create_config, capsys):
        """Test that config and logging are properly set up."""
        # Mock the workflow
        with patch("src.cli.patients_fetcher.PatientsFetcherWorkflow") as mock_workflow_class:
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.metadata = {"total_fetched": 1}
            mock_workflow.execute.return_value = mock_result
            mock_workflow_class.return_value = mock_workflow

            # Create mock args
            args = argparse.Namespace(
                archive="test_archive",
                patient_id=None,
                limit=1,
                output_file=None,
                list_archives=False,
                verbose=False,
                log_level="INFO",
                config=None,
                debug=False,
                quiet=False,
                log_file=None,
                config_file=None
            )

            # Call main
            main(args)

            # Verify setup functions were called
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    def test_main_memory_storage_disabled(self, mock_workflow_class):
        """Test that CORE memory is disabled for fetcher operations."""
        # Mock the workflow
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = {"total_fetched": 1}
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args
        args = argparse.Namespace(
            archive="test_archive",
            patient_id=None,
            limit=1,
            output_file=None,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            debug=False,
            quiet=False,
            log_file=None,
            config_file=None
        )

        # Call main
        main(args)

        # Verify workflow was created with memory_storage=False
        mock_workflow_class.assert_called_once()
        call_args = mock_workflow_class.call_args
        assert call_args[1]["memory_storage"] == False