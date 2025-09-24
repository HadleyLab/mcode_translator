#!/usr/bin/env python3
"""
Unit tests for patients_fetcher CLI module.

Tests the command-line interface for fetching synthetic patient data,
including argument parsing, workflow execution, data validation, and error handling.
"""

import argparse
from unittest.mock import MagicMock, patch


from src.cli.patients_fetcher import (
    create_parser,
    main,
)


class TestPatientsFetcherCLI:
    """Test the patients_fetcher CLI module."""

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Fetch synthetic patient data" in parser.description

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "archive" in actions
        assert "output_file" in actions
        assert "limit" in actions

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
        assert "Patient archive identifier" in archive_action.help

    def test_create_parser_output_file_argument(self):
        """Test the output file argument configuration."""
        parser = create_parser()

        # Find the output_file action
        output_action = None
        for action in parser._actions:
            if action.dest == "output_file":
                output_action = action
                break

        assert output_action is not None
        assert "Output file for patient data" in output_action.help

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

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.cli.patients_fetcher.McodeCLI.create_config")
    @patch("src.cli.patients_fetcher.McodeCLI.setup_logging")
    def test_main_successful_fetch(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test successful patient data fetching."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            output_file="patients.ndjson",
            patient_id=None,
            limit=5,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            batch_size=10,
            workers=0,
            worker_pool="custom",
            max_queue_size=1000,
            task_timeout=None,
            memory_source="mcode_translator",
        )

        # Mock workflow and result
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = {
            "total_fetched": 5,
            "fetch_type": "archive",
            "archive_path": "breast_cancer_10_years",
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Call main
        main(args)

        # Verify workflow was created and executed
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.execute.assert_called_once_with(
            archive_path="breast_cancer_10_years",
            limit=5,
            output_path="patients.ndjson",
        )

        # Verify logging was set up
        mock_setup_logging.assert_called_once_with(args)
        mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.cli.patients_fetcher.McodeCLI.create_config")
    @patch("src.cli.patients_fetcher.McodeCLI.setup_logging")
    def test_main_list_archives_flag(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test the list archives functionality."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args with list_archives=True
        args = argparse.Namespace(
            archive=None,
            output_file=None,
            patient_id=None,
            limit=10,
            list_archives=True,
            verbose=False,
            log_level="INFO",
            config=None,
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            batch_size=10,
            workers=0,
            worker_pool="custom",
            max_queue_size=1000,
            task_timeout=None,
            memory_source="mcode_translator",
        )

        # Mock workflow for list_archives
        mock_workflow = MagicMock()
        mock_workflow.list_available_archives.return_value = [
            "breast_cancer_10_years",
            "mixed_cancer_lifetime",
        ]
        mock_workflow_class.return_value = mock_workflow

        # Call main
        main(args)

        # Verify workflow was created and list_archives was called
        mock_workflow_class.assert_called_once_with(mock_config)
        mock_workflow.list_available_archives.assert_called_once()

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.cli.patients_fetcher.McodeCLI.create_config")
    @patch("src.cli.patients_fetcher.McodeCLI.setup_logging")
    def test_main_workflow_failure(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test handling of workflow execution failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            output_file="patients.ndjson",
            patient_id=None,
            limit=5,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            batch_size=10,
            workers=0,
            worker_pool="custom",
            max_queue_size=1000,
            task_timeout=None,
            memory_source="mcode_translator",
        )

        # Mock workflow and failed result
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Archive not found"
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            # Call main - should exit with error
            try:
                main(args)
            except SystemExit:
                pass  # Expected when workflow fails

            # Verify sys.exit was called
            assert mock_exit.call_count >= 1
            mock_exit.assert_any_call(1)

    @patch("src.cli.patients_fetcher.PatientsFetcherWorkflow")
    @patch("src.cli.patients_fetcher.McodeCLI.create_config")
    @patch("src.cli.patients_fetcher.McodeCLI.setup_logging")
    def test_main_keyboard_interrupt_handling(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test handling of keyboard interrupt during execution."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            archive="breast_cancer_10_years",
            output_file="patients.ndjson",
            patient_id=None,
            limit=5,
            list_archives=False,
            verbose=False,
            log_level="INFO",
            config=None,
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            batch_size=10,
            workers=0,
            worker_pool="custom",
            max_queue_size=1000,
            task_timeout=None,
            memory_source="mcode_translator",
        )

        # Mock workflow to raise KeyboardInterrupt
        mock_workflow = MagicMock()
        mock_workflow.execute.side_effect = KeyboardInterrupt()
        mock_workflow_class.return_value = mock_workflow

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            # Call main - should handle KeyboardInterrupt gracefully
            try:
                main(args)
            except SystemExit:
                pass  # Expected when KeyboardInterrupt occurs

            # Verify sys.exit was called with code 130 (standard for SIGINT)
            assert mock_exit.call_count >= 1
            mock_exit.assert_any_call(130)
