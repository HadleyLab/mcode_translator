#!/usr/bin/env python3
"""
Base test classes for CLI modules.

Provides common test patterns and utilities for CLI testing, reducing redundancy
across individual CLI test files.
"""

import argparse
import json
import tempfile
from abc import ABC
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


class BaseCLITest(ABC):
    """Base class for CLI module tests with common patterns."""

    def get_cli_module(self) -> Any:
        """Return the CLI module being tested."""
        raise NotImplementedError

    def get_workflow_class_name(self) -> str:
        """Return the workflow class name for this CLI module."""
        raise NotImplementedError

    def get_default_args(self):
        """Return default arguments for this CLI module."""
        return argparse.Namespace(
            input_file="test_input.ndjson",
            output_file="test_output.ndjson",
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

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = self.get_cli_module().create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description is not None

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "input_file" in actions
        assert "output_file" in actions

    def test_create_parser_input_file_argument(self):
        """Test the input_file argument configuration."""
        parser = self.get_cli_module().create_parser()

        # Find the input_file action
        input_action = None
        for action in parser._actions:
            if action.dest == "input_file":
                input_action = action
                break

        assert input_action is not None
        assert input_action.help is not None

    def test_create_parser_output_file_argument(self):
        """Test the output_file argument configuration."""
        parser = self.get_cli_module().create_parser()

        # Find the output_file action
        output_action = None
        for action in parser._actions:
            if action.dest == "output_file":
                output_action = action
                break

        assert output_action is not None
        assert output_action.help is not None

    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_main_successful_processing_without_ingest(
        self, mock_setup_logging, mock_create_config
    ):
        """Test successful processing without CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = self.get_default_args()

        # Mock workflow and result
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
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

            # Mock file operations
            with patch("pathlib.Path.exists", return_value=True), patch(
                "src.cli.load_ndjson_data"
            ) as mock_load:
                mock_load.return_value = [{"test": "data"}]

                # Call main
                self.get_cli_module().main(args)

                # Verify workflow was created and executed
                mock_workflow_class.assert_called_once_with(mock_config, None)
                mock_workflow.execute.assert_called_once()

    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    @patch("src.storage.mcode_memory_storage.McodeMemoryStorage")
    def test_main_successful_processing_with_ingest(
        self, mock_memory_class, mock_setup_logging, mock_create_config
    ):
        """Test successful processing with CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock memory storage
        mock_memory = MagicMock()
        mock_memory_class.return_value = mock_memory

        # Create mock args with ingest enabled
        args = self.get_default_args()
        args.ingest = True
        args.output_file = None
        args.memory_source = "test_memory"

        # Mock workflow and result
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
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

            # Mock file operations
            with patch("pathlib.Path.exists", return_value=True), patch(
                "src.cli.load_ndjson_data"
            ) as mock_load:
                mock_load.return_value = [{"test": "data"}]

                # Call main
                self.get_cli_module().main(args)

                # Verify memory storage was initialized
                mock_memory_class.assert_called_once_with(source="test_memory")

                # Verify workflow was created with memory storage
                mock_workflow_class.assert_called_once_with(mock_config, mock_memory)

    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    @patch("src.cli.trials_processor.handle_cli_error")
    @patch("src.cli.trials_processor.sys.exit")
    def test_main_missing_input_file(
        self, mock_exit, mock_handle_cli_error, mock_setup_logging, mock_create_config
    ):
        """Test error when input file is not specified."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Make handle_cli_error mock raise SystemExit to simulate real behavior
        def handle_cli_error_side_effect(*args, **kwargs):
            # Raise SystemExit to simulate the actual behavior
            raise SystemExit(1)

        mock_handle_cli_error.side_effect = handle_cli_error_side_effect

        # Create mock args without input file
        args = self.get_default_args()
        args.input_file = None

        # Call main - should handle error and exit
        import pytest

        with pytest.raises(SystemExit):
            self.get_cli_module().main(args)

        # Verify handle_cli_error was called with the correct error
        mock_handle_cli_error.assert_called_once()
        call_args = mock_handle_cli_error.call_args
        assert isinstance(call_args[0][0], ValueError)
        assert "Input file is required" in str(call_args[0][0])

    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_main_input_file_not_found(self, mock_setup_logging, mock_create_config):
        """Test error when input file does not exist."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args with non-existent input file
        args = self.get_default_args()
        args.input_file = "nonexistent.ndjson"

        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False), patch("sys.exit") as mock_exit:
            # Call main - should exit with error
            self.get_cli_module().main(args)

            # Verify sys.exit was called at least once
            assert mock_exit.call_count >= 1

    @patch("src.shared.cli_utils.McodeCLI.create_config")
    @patch("src.shared.cli_utils.McodeCLI.setup_logging")
    def test_main_workflow_failure(self, mock_setup_logging, mock_create_config):
        """Test handling of workflow execution failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = self.get_default_args()

        # Mock workflow and failed result
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error_message = "Processing failed"
            mock_result.metadata = {}
            mock_workflow.execute.return_value = mock_result
            mock_workflow_class.return_value = mock_workflow

            # Mock file operations
            with patch("pathlib.Path.exists", return_value=True), patch(
                "src.cli.load_ndjson_data"
            ) as mock_load, patch("sys.exit") as mock_exit:
                mock_load.return_value = [{"test": "data"}]

                # Call main - should exit with error
                self.get_cli_module().main(args)

                # Verify sys.exit was called
                mock_exit.assert_called_once_with(1)

    @patch("src.cli.trials_processor.McodeCLI.create_config")
    @patch("src.cli.trials_processor.McodeCLI.setup_logging")
    def test_main_keyboard_interrupt_handling(self, mock_setup_logging, mock_create_config):
        """Test handling of keyboard interrupt."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = self.get_default_args()

        # Mock file operations and workflow to raise KeyboardInterrupt
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.load_ndjson_data"
        ) as mock_load, patch("sys.exit") as mock_exit:
            mock_load.return_value = [{"test": "data"}]

            with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
                mock_workflow = MagicMock()
                mock_workflow.execute.side_effect = KeyboardInterrupt()
                mock_workflow_class.return_value = mock_workflow

                # Call main - should handle KeyboardInterrupt
                self.get_cli_module().main(args)

                # Verify sys.exit was called
                mock_exit.assert_called_once_with(130)

    def test_main_without_args_uses_sys_argv(self):
        """Test that main() without args uses sys.argv."""
        with patch("sys.argv", ["cli_module", "--help"]), patch(
            "argparse.ArgumentParser.print_help"
        ) as mock_help, patch("sys.exit"), patch(
            "src.cli.McodeCLI.create_config"
        ) as mock_create_config, patch(
            "src.cli.McodeCLI.setup_logging"
        ):
            # Mock configuration and logging
            mock_config = MagicMock()
            mock_create_config.return_value = mock_config

            # Call main without args - should parse sys.argv and show help
            try:
                self.get_cli_module().main()
            except (SystemExit, TypeError):
                pass  # Expected when --help is used or Path(None) occurs

            # Verify help was printed (since --help was in argv)
            mock_help.assert_called_once()

    def test_save_processed_data_to_file(self):
        """Test saving processed data to a file."""
        # Mock processed data
        mock_data = [{"test": "data"}]

        # Mock logger
        mock_logger = MagicMock()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".ndjson") as temp_file:
            temp_path = temp_file.name

        try:
            # Call save_processed_data
            self.get_cli_module().save_processed_data(mock_data, temp_path, mock_logger)

            # Verify file was written
            with open(temp_path, "r") as f:
                content = f.read()
                assert len(content) > 0

            # Verify logger was called
            mock_logger.info.assert_called()

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_save_processed_data_to_stdout(self):
        """Test saving processed data to stdout."""
        # Mock processed data
        mock_data = [{"test": "data"}]

        # Mock logger
        mock_logger = MagicMock()

        # Mock stdout
        with patch("sys.stdout") as mock_stdout:
            # Call save_processed_data with no output file
            self.get_cli_module().save_processed_data(mock_data, None, mock_logger)

            # Verify stdout was written to
            assert mock_stdout.write.call_count >= 1
            assert mock_stdout.flush.called

            # Verify logger was called
            mock_logger.info.assert_called()

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
        self.get_cli_module().print_processing_summary(metadata, True, mock_logger)

        # Verify all summary information was logged
        calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Total" in call and "10" in call for call in calls)
        assert any("Successful" in call and "8" in call for call in calls)
        assert any("Failed" in call and "2" in call for call in calls)
        assert any("Success rate" in call and "0.8%" in call for call in calls)
        assert any("Stored in CORE Memory" in call for call in calls)

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
        self.get_cli_module().print_processing_summary(metadata, False, mock_logger)

        # Verify storage disabled message
        calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Storage disabled" in call for call in calls)

    def test_print_processing_summary_empty_metadata(self):
        """Test printing processing summary with empty metadata."""
        mock_logger = MagicMock()

        # Call print_processing_summary with None metadata
        self.get_cli_module().print_processing_summary(None, False, mock_logger)

        # Verify no logging occurred
        assert mock_logger.info.call_count == 0

    def test_print_processing_summary_minimal_metadata(self):
        """Test printing processing summary with minimal metadata."""
        metadata = {"total_trials": 3}

        mock_logger = MagicMock()

        # Call print_processing_summary
        self.get_cli_module().print_processing_summary(metadata, False, mock_logger)

        # Verify total trials was logged and default values are used for missing fields
        calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Total" in call and "3" in call for call in calls)
        assert any("Successful: 0" in call for call in calls)  # Default value
        assert any("Failed: 0" in call for call in calls)  # Default value
        assert any("Success rate: 0.0%" in call for call in calls)  # Default value

    @patch("src.cli.trials_processor.McodeCLI.create_config")
    @patch("src.cli.trials_processor.McodeCLI.setup_logging")
    def test_main_with_concurrency_settings(self, mock_setup_logging, mock_create_config):
        """Test that concurrency settings are properly passed to workflow."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args with concurrency settings
        args = self.get_default_args()
        args.workers = 8
        args.worker_pool = "process"
        args.batch_size = 20
        args.max_queue_size = 5000
        args.task_timeout = 600

        # Mock workflow and result
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
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

            # Mock file operations
            with patch("pathlib.Path.exists", return_value=True), patch(
                "src.cli.load_ndjson_data"
            ) as mock_load:
                mock_load.return_value = [{"test": "data"}]

                # Call main
                self.get_cli_module().main(args)

                # Verify workflow was executed with correct parameters
                mock_workflow_class.assert_called_once_with(mock_config, None)
                mock_workflow.execute.assert_called_once()

    @patch("src.cli.trials_processor.McodeCLI.create_config")
    @patch("src.cli.trials_processor.McodeCLI.setup_logging")
    @patch("src.cli.trials_processor.McodeMemoryStorage")
    def test_main_memory_storage_initialization_failure(
        self, mock_memory_class, mock_setup_logging, mock_create_config
    ):
        """Test handling of CORE Memory storage initialization failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args with ingest enabled
        args = self.get_default_args()
        args.ingest = True
        args.output_file = None
        args.memory_source = "mcode_translator"

        # Mock McodeMemoryStorage to raise exception
        mock_memory_class.side_effect = Exception("Storage connection failed")

        # Mock file operations
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.load_ndjson_data"
        ) as mock_load, patch("sys.exit") as mock_exit:
            mock_load.return_value = [{"test": "data"}]

            # Call main - should exit with error
            try:
                self.get_cli_module().main(args)
            except SystemExit:
                pass  # Expected when memory storage fails

            # Verify sys.exit was called
            assert mock_exit.call_count >= 1
            mock_exit.assert_any_call(1)

    def test_main_with_empty_input_data(self):
        """Test handling of empty input data."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()

        # Mock workflow and empty result
        with patch(
            "src.cli.trials_processor.ClinicalTrialsProcessorWorkflow"
        ) as mock_workflow_class:
            with patch("src.cli.trials_processor.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.trials_processor.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("src.cli.trials_processor.load_ndjson_data", return_value=[]):
                            with patch("sys.exit") as mock_exit:
                                mock_workflow = MagicMock()
                                mock_result = MagicMock()
                                mock_result.success = False
                                mock_result.error_message = "No data found"
                                mock_workflow.execute.return_value = mock_result
                                mock_workflow_class.return_value = mock_workflow

                                # Call main - should handle empty data
                                self.get_cli_module().main(args)

                                # Verify sys.exit was called (may be called multiple times due to multiple error conditions)
                                assert mock_exit.call_count >= 1
                                mock_exit.assert_any_call(1)

    def test_main_with_malformed_json_data(self):
        """Test handling of malformed JSON data."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()

        # Mock workflow - patch the actual import location
        with patch(
            "src.workflows.trials_processor_workflow.ClinicalTrialsProcessorWorkflow"
        ) as mock_workflow_class:
            with patch("src.shared.cli_utils.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch(
                            "src.cli.load_ndjson_data",
                            side_effect=json.JSONDecodeError("Invalid JSON", "", 0),
                        ):
                            with patch("sys.exit") as mock_exit:
                                mock_workflow = MagicMock()
                                mock_workflow_class.return_value = mock_workflow

                                # Call main - should handle JSON decode error
                                self.get_cli_module().main(args)

                                # Verify sys.exit was called
                                mock_exit.assert_called_once_with(1)

    def test_main_with_file_permission_error(self):
        """Test handling of file permission errors."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()

        # Mock workflow
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
            with patch("src.cli.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch(
                            "src.cli.load_ndjson_data",
                            side_effect=PermissionError("Permission denied"),
                        ):
                            with patch("sys.exit") as mock_exit:
                                mock_workflow = MagicMock()
                                mock_workflow_class.return_value = mock_workflow

                                # Call main - should handle permission error
                                self.get_cli_module().main(args)

                                # Verify sys.exit was called
                                mock_exit.assert_called_once_with(1)

    def test_main_with_network_timeout_error(self):
        """Test handling of network timeout errors."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()

        # Mock workflow to raise timeout error
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
            with patch("src.cli.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("src.cli.load_ndjson_data", return_value=[{"test": "data"}]):
                            with patch("sys.exit") as mock_exit:
                                mock_workflow = MagicMock()
                                mock_workflow.execute.side_effect = TimeoutError("Network timeout")
                                mock_workflow_class.return_value = mock_workflow

                                # Call main - should handle timeout error
                                self.get_cli_module().main(args)

                                # Verify sys.exit was called
                                mock_exit.assert_called_once_with(1)

    def test_main_with_memory_error(self):
        """Test handling of memory errors during processing."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()

        # Mock workflow to raise memory error
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
            with patch("src.cli.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("src.cli.load_ndjson_data", return_value=[{"test": "data"}]):
                            with patch("sys.exit") as mock_exit:
                                mock_workflow = MagicMock()
                                mock_workflow.execute.side_effect = MemoryError("Out of memory")
                                mock_workflow_class.return_value = mock_workflow

                                # Call main - should handle memory error
                                self.get_cli_module().main(args)

                                # Verify sys.exit was called
                                mock_exit.assert_called_once_with(1)

    def test_main_with_invalid_configuration(self):
        """Test handling of invalid configuration."""
        # Create mock args
        args = self.get_default_args()

        # Mock configuration to raise exception
        with patch("src.cli.McodeCLI.create_config", side_effect=ValueError("Invalid config")):
            with patch("src.cli.McodeCLI.setup_logging"):
                with patch("sys.exit") as mock_exit:
                    # Call main - should handle configuration error
                    self.get_cli_module().main(args)

                    # Verify sys.exit was called
                    mock_exit.assert_called_once_with(1)

    def test_main_with_large_dataset_performance(self):
        """Test processing with large dataset (performance test)."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()
        args.batch_size = 100  # Large batch size for performance

        # Create large mock dataset
        large_dataset = [{"nct_id": f"NCT{i:06d}", "title": f"Trial {i}"} for i in range(1000)]

        # Mock workflow
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
            with patch("src.cli.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("src.cli.load_ndjson_data", return_value=large_dataset):
                            with patch("sys.exit"):
                                mock_workflow = MagicMock()
                                mock_result = MagicMock()
                                mock_result.success = True
                                mock_result.data = []
                                mock_result.metadata = {
                                    "total_trials": 1000,
                                    "successful": 1000,
                                    "failed": 0,
                                    "success_rate": 1.0,
                                }
                                mock_workflow.execute.return_value = mock_result
                                mock_workflow_class.return_value = mock_workflow

                                # Call main - should handle large dataset
                                self.get_cli_module().main(args)

                                # Verify workflow was called with large dataset
                                call_args = mock_workflow.execute.call_args
                                assert len(call_args[1]["trials_data"]) == 1000

    def test_main_with_unicode_characters(self):
        """Test processing with Unicode characters in data."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()

        # Create mock data with Unicode characters
        unicode_data = [
            {
                "nct_id": "NCT12345678",
                "title": "Trial with émojis 🎉 and spëcial chärs",
                "description": "测试 with 中文 characters",
            }
        ]

        # Mock workflow
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
            with patch("src.cli.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("src.cli.load_ndjson_data", return_value=unicode_data):
                            with patch("sys.exit"):
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

                                # Call main - should handle Unicode data
                                self.get_cli_module().main(args)

                                # Verify workflow was called with Unicode data
                                call_args = mock_workflow.execute.call_args
                                assert "🎉" in call_args[1]["trials_data"][0]["title"]
                                assert "中文" in call_args[1]["trials_data"][0]["description"]

    def test_main_with_concurrent_access(self):
        """Test handling of concurrent file access."""
        # Mock configuration and logging
        mock_config = MagicMock()

        # Create mock args
        args = self.get_default_args()

        # Mock workflow
        with patch(f"src.cli.{self.get_workflow_class_name()}") as mock_workflow_class:
            with patch("src.cli.McodeCLI.create_config", return_value=mock_config):
                with patch("src.cli.McodeCLI.setup_logging"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch(
                            "src.cli.load_ndjson_data",
                            side_effect=OSError("File is locked by another process"),
                        ):
                            with patch("sys.exit") as mock_exit:
                                mock_workflow = MagicMock()
                                mock_workflow_class.return_value = mock_workflow

                                # Call main - should handle concurrent access error
                                self.get_cli_module().main(args)

                                # Verify sys.exit was called
                                mock_exit.assert_called_once_with(1)


class TestDataHelper:
    """Helper class for creating test data."""

    @staticmethod
    def create_sample_trial_data():
        """Create sample trial data for testing."""
        return {
            "nct_id": "NCT12345678",
            "title": "Sample Clinical Trial",
            "eligibility": {"criteria": "Inclusion: Age >= 18\nExclusion: Prior chemotherapy"},
            "conditions": ["Breast Cancer"],
            "phases": ["Phase 2"],
            "interventions": [{"type": "Drug", "name": "Sample Drug"}],
        }

    @staticmethod
    def create_sample_patient_data():
        """Create sample patient data for testing."""
        return {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "12345",
                        "gender": "female",
                        "birthDate": "1980-01-01",
                    }
                }
            ],
        }

    @staticmethod
    def create_mcode_result():
        """Create sample mCODE result data."""
        return {
            "trial_id": "NCT12345678",
            "McodeResults": {"mcode_mappings": [{"element_type": "CancerCondition"}]},
            "original_trial_data": TestDataHelper.create_sample_trial_data(),
        }

    @staticmethod
    def create_workflow_result(success=True, data=None, metadata=None):
        """Create a mock workflow result."""
        mock_result = MagicMock()
        mock_result.success = success
        mock_result.data = data or []
        mock_result.metadata = metadata or {
            "total_trials": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
        }
        return mock_result
