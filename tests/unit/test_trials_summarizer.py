#!/usr/bin/env python3
"""
Unit tests for trials_summarizer CLI module.

Tests the command-line interface for generating natural language summaries from mCODE trial data,
including file handling, data parsing, workflow execution, summary extraction, and CORE Memory integration.
"""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


from src.cli.trials_summarizer import (
    create_parser,
    main,
    load_mcode_trials,
    save_summaries,
    extract_summaries,
    print_processing_summary,
)


class TestTrialsSummarizerCLI:
    """Test the trials_summarizer CLI module."""

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Generate natural language summaries" in parser.description

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "input_file" in actions
        assert "output_file" in actions

    def test_create_parser_input_file_argument(self):
        """Test the input_file argument configuration."""
        parser = create_parser()

        # Find the input_file action
        input_action = None
        for action in parser._actions:
            if action.dest == "input_file":
                input_action = action
                break

        assert input_action is not None
        assert "Input file with mCODE trial data" in input_action.help

    def test_create_parser_output_file_argument(self):
        """Test the output_file argument configuration."""
        parser = create_parser()

        # Find the output_file action
        output_action = None
        for action in parser._actions:
            if action.dest == "output_file":
                output_action = action
                break

        assert output_action is not None
        assert "Output file for summaries" in output_action.help

    @patch("src.cli.trials_summarizer.TrialsSummarizerWorkflow")
    @patch("src.cli.trials_summarizer.McodeCLI.create_config")
    @patch("src.cli.trials_summarizer.McodeCLI.setup_logging")
    def test_main_successful_summarization_without_ingest(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test successful trial summarization without CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_mcode_data = [
            {
                "trial_id": "NCT12345678",
                "McodeResults": {
                    "mcode_mappings": [{"element_type": "CancerCondition"}]
                },
                "original_trial_data": {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT12345678"}
                    }
                },
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_trials.ndjson",
            output_file="summaries.ndjson",
            ingest=False,
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
        mock_result.data = [
            {
                "McodeResults": {
                    "natural_language_summary": "This is a test summary",
                    "mcode_mappings": [{"element_type": "CancerCondition"}],
                },
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            }
        ]
        mock_result.metadata = {
            "total_trials": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
            "model_used": "deepseek-coder",
            "prompt_used": "direct_mcode_evidence_based_concise",
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock file reading
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(mock_mcode_data[0])
            mock_open.return_value.__enter__.return_value = mock_file

            # Call main
            main(args)

        # Verify workflow was created and executed
        mock_workflow_class.assert_called_once_with(mock_config, None)
        mock_workflow.execute.assert_called_once_with(
            trials_data=[mock_mcode_data[0]["original_trial_data"]],
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            store_in_memory=False,
            workers=0,
        )

    @patch("src.cli.trials_summarizer.TrialsSummarizerWorkflow")
    @patch("src.cli.trials_summarizer.McodeCLI.create_config")
    @patch("src.cli.trials_summarizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_summarizer.McodeMemoryStorage")
    def test_main_successful_summarization_with_ingest(
        self,
        mock_memory_class,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test successful trial summarization with CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock memory storage
        mock_memory = MagicMock()
        mock_memory_class.return_value = mock_memory

        # Mock trial data
        mock_mcode_data = [
            {
                "trial_id": "NCT12345678",
                "McodeResults": {"mcode_mappings": []},
                "original_trial_data": {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT12345678"}
                    }
                },
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_trials.ndjson",
            output_file=None,
            ingest=True,
            verbose=False,
            log_level="INFO",
            config=None,
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            batch_size=10,
            workers=4,
            worker_pool="thread",
            max_queue_size=1000,
            task_timeout=None,
            memory_source="test_memory",
        )

        # Mock workflow and result
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = []
        mock_result.metadata = {
            "total_trials": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
            "stored_in_memory": True,
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock file reading
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(mock_mcode_data[0])
            mock_open.return_value.__enter__.return_value = mock_file

            # Call main
            main(args)

        # Verify memory storage was initialized
        mock_memory_class.assert_called_once_with(source="test_memory")

        # Verify workflow was created with memory storage
        mock_workflow_class.assert_called_once_with(mock_config, mock_memory)

    def test_main_no_input_file_uses_stdin(self):
        """Test that main uses stdin when no input file is provided."""
        # Create mock args without input file
        args = argparse.Namespace(
            input_file=None,
            output_file=None,
            ingest=False,
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

        # Mock stdin
        mock_mcode_data = {
            "trial_id": "NCT12345678",
            "McodeResults": {"mcode_mappings": []},
        }
        with patch("sys.stdin.read", return_value=json.dumps(mock_mcode_data)), patch(
            "src.cli.trials_summarizer.McodeCLI.setup_logging"
        ), patch("src.cli.trials_summarizer.McodeCLI.create_config"), patch(
            "src.cli.trials_summarizer.TrialsSummarizerWorkflow"
        ) as mock_workflow_class, patch(
            "sys.exit"
        ):

            # Mock workflow to avoid actual execution
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False  # Exit early to avoid full execution
            mock_result.error_message = "Test exit"
            mock_workflow.execute.return_value = mock_result
            mock_workflow_class.return_value = mock_workflow

            # Call main - should attempt to read from stdin
            try:
                main(args)
            except SystemExit:
                pass  # Expected

            # Verify stdin was read
            import sys

            sys.stdin.read.assert_called_once()

    @patch("src.cli.trials_summarizer.McodeCLI.create_config")
    @patch("src.cli.trials_summarizer.McodeCLI.setup_logging")
    def test_main_empty_input_data(self, mock_setup_logging, mock_create_config):
        """Test error when input file contains no valid data."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            input_file="empty.ndjson",
            output_file=None,
            ingest=False,
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

        # Mock empty file
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = ""
            mock_open.return_value.__enter__.return_value = mock_file

            # Mock sys.exit to prevent actual exit
            with patch("sys.exit") as mock_exit:
                # Call main - should exit with error
                try:
                    main(args)
                except SystemExit:
                    pass  # Expected when no data found

                # Verify sys.exit was called
                assert mock_exit.call_count >= 1

    @patch("src.cli.trials_summarizer.TrialsSummarizerWorkflow")
    @patch("src.cli.trials_summarizer.McodeCLI.create_config")
    @patch("src.cli.trials_summarizer.McodeCLI.setup_logging")
    def test_main_workflow_failure(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test handling of workflow execution failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_mcode_data = [
            {
                "trial_id": "NCT12345678",
                "McodeResults": {"mcode_mappings": []},
                "original_trial_data": {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT12345678"}
                    }
                },
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_trials.ndjson",
            output_file="summaries.ndjson",
            ingest=False,
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
        mock_result.error_message = "Summarization failed"
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock file reading
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(mock_mcode_data[0])
            mock_open.return_value.__enter__.return_value = mock_file

            # Mock sys.exit to prevent actual exit
            with patch("sys.exit") as mock_exit:
                # Call main - should exit with error
                try:
                    main(args)
                except SystemExit:
                    pass  # Expected when workflow fails

                # Verify sys.exit was called
                assert mock_exit.call_count >= 1

    @patch("src.cli.trials_summarizer.TrialsSummarizerWorkflow")
    @patch("src.cli.trials_summarizer.McodeCLI.create_config")
    @patch("src.cli.trials_summarizer.McodeCLI.setup_logging")
    def test_main_keyboard_interrupt_handling(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test handling of keyboard interrupt during execution."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_mcode_data = [
            {
                "trial_id": "NCT12345678",
                "McodeResults": {"mcode_mappings": []},
                "original_trial_data": {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT12345678"}
                    }
                },
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_trials.ndjson",
            output_file=None,
            ingest=False,
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

        # Mock file reading
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(mock_mcode_data[0])
            mock_open.return_value.__enter__.return_value = mock_file

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

    def test_load_mcode_trials_from_file(self):
        """Test loading mCODE trials from file."""
        mock_data = [
            {"trial_id": "NCT12345678", "McodeResults": {"mcode_mappings": []}},
            {"trial_id": "NCT87654321", "McodeResults": {"mcode_mappings": []}},
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".ndjson"
        ) as temp_file:
            for item in mock_data:
                json.dump(item, temp_file)
                temp_file.write("\n")
            temp_path = temp_file.name

        try:
            # Call load_mcode_trials
            result = load_mcode_trials(temp_path)

            # Verify data was loaded correctly
            assert len(result) == 2
            assert result[0]["trial_id"] == "NCT12345678"
            assert result[1]["trial_id"] == "NCT87654321"

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_load_mcode_trials_from_stdin(self):
        """Test loading mCODE trials from stdin."""
        mock_data = {"trial_id": "NCT12345678", "McodeResults": {"mcode_mappings": []}}

        # Mock stdin
        with patch("sys.stdin.read", return_value=json.dumps(mock_data)):
            # Call load_mcode_trials with no input file
            result = load_mcode_trials(None)

            # Verify data was loaded correctly
            assert len(result) == 1
            assert result[0]["trial_id"] == "NCT12345678"

    def test_load_mcode_trials_invalid_json(self):
        """Test handling of invalid JSON in input."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".ndjson"
        ) as temp_file:
            temp_file.write('{"invalid": json}\n')
            temp_file.write('{"valid": "data"}\n')
            temp_path = temp_file.name

        try:
            # Call load_mcode_trials
            result = load_mcode_trials(temp_path)

            # Verify only valid JSON was loaded
            assert len(result) == 1
            assert result[0]["valid"] == "data"

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_save_summaries_to_file(self):
        """Test saving summaries to output file."""
        mock_summaries = [
            {
                "trial_id": "NCT12345678",
                "summary": "Test summary",
                "mcode_elements": [{"element_type": "CancerCondition"}],
            }
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".ndjson"
        ) as temp_file:
            temp_path = temp_file.name

        try:
            # Call save_summaries
            save_summaries(mock_summaries, temp_path)

            # Verify file was written
            with open(temp_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1

                # Parse JSON and verify structure
                saved_data = json.loads(lines[0])
                assert saved_data["trial_id"] == "NCT12345678"
                assert saved_data["summary"] == "Test summary"
                assert len(saved_data["mcode_elements"]) == 1

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_save_summaries_to_stdout(self):
        """Test saving summaries to stdout."""
        mock_summaries = [
            {"trial_id": "NCT12345678", "summary": "Test summary", "mcode_elements": []}
        ]

        # Mock stdout
        with patch("sys.stdout") as mock_stdout:
            # Call save_summaries with no output file
            save_summaries(mock_summaries, None)

            # Verify stdout was written to
            assert mock_stdout.write.call_count >= 1
            assert mock_stdout.flush.called

    def test_extract_summaries_with_valid_data(self):
        """Test extracting summaries from workflow results."""
        mock_data = [
            {
                "McodeResults": {
                    "natural_language_summary": "This is a test summary",
                    "mcode_mappings": [{"element_type": "CancerCondition"}],
                },
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            }
        ]

        # Call extract_summaries
        result = extract_summaries(mock_data)

        # Verify summary was extracted
        assert len(result) == 1
        assert result[0]["trial_id"] == "NCT12345678"
        assert result[0]["summary"] == "This is a test summary"
        assert len(result[0]["mcode_elements"]) == 1

    def test_extract_summaries_with_missing_data(self):
        """Test extracting summaries when data is missing."""
        mock_data = [
            {"McodeResults": {"mcode_mappings": []}},  # Missing summary
            {
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}}
            },  # Missing McodeResults
            {},  # Empty data
        ]

        # Call extract_summaries
        result = extract_summaries(mock_data)

        # Verify no summaries were extracted
        assert len(result) == 0

    def test_extract_summaries_empty_data(self):
        """Test extracting summaries from empty data."""
        # Call extract_summaries with None
        result = extract_summaries(None)

        # Verify empty list returned
        assert result == []

    def test_print_processing_summary_complete_metadata(self):
        """Test printing processing summary with complete metadata."""
        metadata = {
            "total_trials": 10,
            "successful": 8,
            "failed": 2,
            "success_rate": 0.8,
            "stored_in_memory": True,
        }

        mock_logger = MagicMock()

        # Call print_processing_summary
        print_processing_summary(metadata, True, mock_logger)

        # Verify all summary information was logged
        calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "📊 Total trials: 10" in calls
        assert "✅ Successful: 8" in calls
        assert "❌ Failed: 2" in calls
        assert "📈 Success rate: 0.8%" in calls
        assert "🧠 Stored in CORE Memory" in calls

    def test_print_processing_summary_without_ingest(self):
        """Test printing processing summary without CORE Memory ingestion."""
        metadata = {
            "total_trials": 5,
            "successful": 5,
            "failed": 0,
            "success_rate": 1.0,
        }

        mock_logger = MagicMock()

        # Call print_processing_summary
        print_processing_summary(metadata, False, mock_logger)

        # Verify storage disabled message
        calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "💾 Storage disabled" in calls

    def test_print_processing_summary_empty_metadata(self):
        """Test printing processing summary with empty metadata."""
        mock_logger = MagicMock()

        # Call print_processing_summary with None metadata
        print_processing_summary(None, False, mock_logger)

        # Verify no logging occurred
        assert mock_logger.info.call_count == 0

    def test_print_processing_summary_minimal_metadata(self):
        """Test printing processing summary with minimal metadata."""
        metadata = {"total_trials": 3}

        mock_logger = MagicMock()

        # Call print_processing_summary
        print_processing_summary(metadata, False, mock_logger)

        # Verify only total trials was logged and default values were used
        calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "📊 Total trials: 3" in calls
        assert "✅ Successful: 0" in calls  # Default value
        assert "❌ Failed: 0" in calls  # Default value
        assert "📈 Success rate: 0.0%" in calls  # Default value

    @patch("src.cli.trials_summarizer.TrialsSummarizerWorkflow")
    @patch("src.cli.trials_summarizer.McodeCLI.create_config")
    @patch("src.cli.trials_summarizer.McodeCLI.setup_logging")
    def test_main_with_concurrency_settings(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test that concurrency settings are properly passed to workflow."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_mcode_data = [
            {
                "trial_id": "NCT12345678",
                "McodeResults": {"mcode_mappings": []},
                "original_trial_data": {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT12345678"}
                    }
                },
            }
        ]

        # Create mock args with concurrency settings
        args = argparse.Namespace(
            input_file="mcode_trials.ndjson",
            output_file="summaries.ndjson",
            ingest=False,
            verbose=True,
            log_level="DEBUG",
            config=None,
            model="gpt-4",
            prompt="direct_mcode_evidence_based_concise",
            batch_size=20,
            workers=8,
            worker_pool="process",
            max_queue_size=5000,
            task_timeout=600,
            memory_source="mcode_translator",
        )

        # Mock workflow and result
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = []
        mock_result.metadata = {
            "total_trials": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Mock file reading
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(mock_mcode_data[0])
            mock_open.return_value.__enter__.return_value = mock_file

            # Call main
            main(args)

        # Verify workflow was executed with correct parameters
        mock_workflow_class.assert_called_once_with(mock_config, None)
        mock_workflow.execute.assert_called_once_with(
            trials_data=[mock_mcode_data[0]["original_trial_data"]],
            model="gpt-4",
            prompt="direct_mcode_evidence_based_concise",
            store_in_memory=False,
            workers=8,
        )

    @patch("src.cli.trials_summarizer.McodeCLI.create_config")
    @patch("src.cli.trials_summarizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_summarizer.McodeMemoryStorage")
    def test_main_memory_storage_initialization_failure(
        self, mock_memory_class, mock_setup_logging, mock_create_config
    ):
        """Test handling of CORE Memory storage initialization failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_mcode_data = [
            {
                "trial_id": "NCT12345678",
                "McodeResults": {"mcode_mappings": []},
                "original_trial_data": {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT12345678"}
                    }
                },
            }
        ]

        # Create mock args with ingest enabled
        args = argparse.Namespace(
            input_file="mcode_trials.ndjson",
            output_file=None,
            ingest=True,
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

        # Mock McodeMemoryStorage to raise exception
        mock_memory_class.side_effect = Exception("Storage connection failed")

        # Mock file reading
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_file.read.return_value = json.dumps(mock_mcode_data[0])
            mock_open.return_value.__enter__.return_value = mock_file

            # Mock sys.exit to prevent actual exit
            with patch("sys.exit") as mock_exit:
                # Call main - should exit with error
                try:
                    main(args)
                except SystemExit:
                    pass  # Expected when memory storage fails

                # Verify sys.exit was called
                assert mock_exit.call_count >= 1
                mock_exit.assert_any_call(1)
