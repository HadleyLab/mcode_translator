#!/usr/bin/env python3
"""
Unit tests for trials_fetcher CLI module.

Tests the command-line interface for fetching clinical trials from ClinicalTrials.gov,
including argument parsing, workflow execution, data validation, and error handling.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from src.cli.trials_fetcher import (
    create_parser,
    main,
    print_fetch_summary,
)


class TestTrialsFetcherCLI:
    """Test the trials_fetcher CLI module."""

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Fetch clinical trials" in parser.description

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "condition" in actions
        assert "nct_id" in actions
        assert "nct_ids" in actions
        assert "limit" in actions
        assert "output_file" in actions

    def test_create_parser_condition_argument(self):
        """Test the condition argument configuration."""
        parser = create_parser()

        # Find the condition action
        condition_action = None
        for action in parser._actions:
            if action.dest == "condition":
                condition_action = action
                break

        assert condition_action is not None
        assert "Medical condition to search for" in condition_action.help

    def test_create_parser_nct_id_argument(self):
        """Test the nct_id argument configuration."""
        parser = create_parser()

        # Find the nct_id action
        nct_id_action = None
        for action in parser._actions:
            if action.dest == "nct_id":
                nct_id_action = action
                break

        assert nct_id_action is not None
        assert "Specific NCT ID to fetch" in nct_id_action.help

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

    @patch("src.cli.trials_fetcher.TrialsFetcherWorkflow")
    @patch("src.cli.trials_fetcher.McodeCLI.create_config")
    @patch("src.cli.trials_fetcher.McodeCLI.setup_logging")
    def test_main_successful_fetch_by_condition(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test successful trial fetching by medical condition."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            condition="breast cancer",
            nct_id=None,
            nct_ids=None,
            limit=5,
            output_file="trials.ndjson",
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
            "fetch_type": "condition_search",
            "duration_seconds": 2.34
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Call main
        main(args)

        # Verify workflow was created and executed
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.execute.assert_called_once_with(
            cli_args=args,
            condition="breast cancer",
            limit=5,
            output_path="trials.ndjson"
        )

        # Verify logging was set up
        mock_setup_logging.assert_called_once_with(args)
        mock_create_config.assert_called_once_with(args)

    @patch("src.cli.trials_fetcher.TrialsFetcherWorkflow")
    @patch("src.cli.trials_fetcher.McodeCLI.create_config")
    @patch("src.cli.trials_fetcher.McodeCLI.setup_logging")
    def test_main_successful_fetch_by_nct_id(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test successful trial fetching by specific NCT ID."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            condition=None,
            nct_id="NCT12345678",
            nct_ids=None,
            limit=10,
            output_file="trial.ndjson",
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
            "total_fetched": 1,
            "fetch_type": "single_trial",
            "duration_seconds": 1.23
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Call main
        main(args)

        # Verify workflow was executed with NCT ID
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.execute.assert_called_once_with(
            cli_args=args,
            nct_id="NCT12345678",
            output_path="trial.ndjson"
        )

    @patch("src.cli.trials_fetcher.TrialsFetcherWorkflow")
    @patch("src.cli.trials_fetcher.McodeCLI.create_config")
    @patch("src.cli.trials_fetcher.McodeCLI.setup_logging")
    def test_main_successful_fetch_by_nct_ids(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test successful trial fetching by multiple NCT IDs."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            condition=None,
            nct_id=None,
            nct_ids="NCT12345678,NCT87654321,NCT11111111",
            limit=10,
            output_file=None,
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
            "total_fetched": 3,
            "fetch_type": "multiple_trials",
            "duration_seconds": 3.45
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Call main
        main(args)

        # Verify workflow was executed with NCT IDs list
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.execute.assert_called_once_with(
            cli_args=args,
            nct_ids=["NCT12345678", "NCT87654321", "NCT11111111"]
        )

    @patch("src.cli.trials_fetcher.McodeCLI.create_config")
    @patch("src.cli.trials_fetcher.McodeCLI.setup_logging")
    def test_main_missing_required_args(self, mock_setup_logging, mock_create_config):
        """Test error when required arguments are missing."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args without any search criteria
        args = argparse.Namespace(
            condition=None,
            nct_id=None,
            nct_ids=None,
            limit=10,
            output_file=None,
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

        # Mock parser.error to prevent actual exit
        with patch("src.cli.trials_fetcher.create_parser") as mock_create_parser, \
             patch("sys.exit") as mock_exit:

            mock_parser = MagicMock()
            mock_parser.error.side_effect = SystemExit(2)
            mock_create_parser.return_value = mock_parser

            # Call main - should exit with error
            try:
                main(args)
            except SystemExit:
                pass  # Expected when required args are missing

            # Verify parser.error was called
            mock_parser.error.assert_called_once_with("Must specify one of: --condition, --nct-id, or --nct-ids")

    @patch("src.cli.trials_fetcher.TrialsFetcherWorkflow")
    @patch("src.cli.trials_fetcher.McodeCLI.create_config")
    @patch("src.cli.trials_fetcher.McodeCLI.setup_logging")
    def test_main_workflow_failure(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test handling of workflow execution failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            condition="lung cancer",
            nct_id=None,
            nct_ids=None,
            limit=5,
            output_file="trials.ndjson",
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
        mock_result.error_message = "API connection failed"
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

    @patch("src.cli.trials_fetcher.TrialsFetcherWorkflow")
    @patch("src.cli.trials_fetcher.McodeCLI.create_config")
    @patch("src.cli.trials_fetcher.McodeCLI.setup_logging")
    def test_main_keyboard_interrupt_handling(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test handling of keyboard interrupt during execution."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            condition="prostate cancer",
            nct_id=None,
            nct_ids=None,
            limit=10,
            output_file=None,
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

    def test_print_fetch_summary_with_metadata(self):
        """Test printing fetch summary with complete metadata."""
        metadata = {
            "total_fetched": 15,
            "fetch_type": "condition_search",
            "duration_seconds": 5.67
        }

        with patch("builtins.print") as mock_print:
            print_fetch_summary(metadata, "trials.ndjson")

            # Verify all summary information was printed
            calls = [call.args[0] for call in mock_print.call_args_list]
            assert "📊 Total trials fetched: 15" in calls
            assert "💾 Results saved to: trials.ndjson" in calls
            assert "🔍 Fetch type: condition_search" in calls
            assert "⏱️  Duration: 5.67s" in calls

    def test_print_fetch_summary_stdout_output(self):
        """Test printing fetch summary for stdout output."""
        metadata = {
            "total_fetched": 8,
            "fetch_type": "single_trial"
        }

        with patch("builtins.print") as mock_print:
            print_fetch_summary(metadata, None)

            # Verify stdout message was printed
            calls = [call.args[0] for call in mock_print.call_args_list]
            assert "📤 Results written to stdout" in calls

    def test_print_fetch_summary_empty_metadata(self):
        """Test printing fetch summary with empty metadata."""
        with patch("builtins.print") as mock_print:
            print_fetch_summary(None, "trials.ndjson")

            # Verify no prints occurred
            assert mock_print.call_count == 0

    def test_print_fetch_summary_minimal_metadata(self):
        """Test printing fetch summary with minimal metadata."""
        metadata = {"total_fetched": 3}

        with patch("builtins.print") as mock_print:
            print_fetch_summary(metadata, None)

            # Verify only total fetched was printed
            calls = [call.args[0] for call in mock_print.call_args_list]
            assert "📊 Total trials fetched: 3" in calls
            assert "📤 Results written to stdout" in calls
            assert len([call for call in calls if "🔍" in call or "⏱️" in call]) == 0

    @patch("src.cli.trials_fetcher.TrialsFetcherWorkflow")
    @patch("src.cli.trials_fetcher.McodeCLI.create_config")
    @patch("src.cli.trials_fetcher.McodeCLI.setup_logging")
    def test_main_with_concurrency_args(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test that concurrency arguments are properly passed."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args with concurrency settings
        args = argparse.Namespace(
            condition="diabetes",
            nct_id=None,
            nct_ids=None,
            limit=20,
            output_file="diabetes_trials.ndjson",
            verbose=True,
            log_level="DEBUG",
            config=None,
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            batch_size=15,
            workers=4,
            worker_pool="thread",
            max_queue_size=2000,
            task_timeout=300,
            memory_source="mcode_translator",
        )

        # Mock workflow and result
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = {
            "total_fetched": 20,
            "fetch_type": "condition_search",
            "duration_seconds": 8.90
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Call main
        main(args)

        # Verify workflow was created with correct config
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.execute.assert_called_once_with(
            cli_args=args,
            condition="diabetes",
            limit=20,
            output_path="diabetes_trials.ndjson"
        )

        # Verify logging was set up with verbose level
        mock_setup_logging.assert_called_once_with(args)
        mock_create_config.assert_called_once_with(args)