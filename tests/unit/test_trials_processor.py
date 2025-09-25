#!/usr/bin/env python3
"""
Unit tests for trials_processor CLI module.

Tests the command-line interface for processing clinical trials with mCODE mapping,
including file handling, workflow execution, data saving, error handling, and CORE Memory integration.
"""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cli.trials_processor import (
    create_parser,
    main,
    save_processed_data,
    print_processing_summary,
)
from tests.unit.test_cli_base import BaseCLITest, TestDataHelper


class TestTrialsProcessorCLI(BaseCLITest):
    """Test the trials_processor CLI module."""

    def get_cli_module(self):
        """Return the CLI module being tested."""
        from src.cli import trials_processor
        return trials_processor

    def get_workflow_class_name(self):
        """Return the workflow class name for this CLI module."""
        return "ClinicalTrialsProcessorWorkflow"

    def get_default_args(self):
        """Return default arguments for this CLI module."""
        args = super().get_default_args()
        args.trials = None  # trials_processor doesn't have trials argument
        return args

    @patch("src.cli.trials_processor.ClinicalTrialsProcessorWorkflow")
    @patch("src.cli.trials_processor.McodeCLI.create_config")
    @patch("src.cli.trials_processor.McodeCLI.setup_logging")
    @patch("src.cli.trials_processor.load_ndjson_data")
    @patch("pathlib.Path.exists", return_value=True)
    def test_main_successful_processing_without_ingest(
        self,
        mock_path_exists,
        mock_load_data,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test successful trial processing without CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trial_data = [
            {"nct_id": "NCT12345678", "title": "Test Trial 1"},
            {"nct_id": "NCT87654321", "title": "Test Trial 2"},
        ]
        mock_load_data.return_value = mock_trial_data

        # Create mock args
        args = argparse.Namespace(
            input_file="trials.ndjson",
            output_file="mcode_trials.ndjson",
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
                "trial_id": "NCT12345678",
                "McodeResults": {
                    "mcode_mappings": [{"element_type": "CancerCondition"}]
                },
                "original_trial_data": mock_trial_data[0],
            }
        ]
        mock_result.metadata = {
            "total_trials": 2,
            "successful": 2,
            "failed": 0,
            "success_rate": 1.0,
            "model_used": "deepseek-coder",
            "prompt_used": "direct_mcode_evidence_based_concise",
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Call main
        main(args)

        # Verify workflow was created and executed
        mock_workflow_class.assert_called_once_with(mock_config, None)
        mock_workflow.execute.assert_called_once_with(
            trials_data=mock_trial_data,
            model="deepseek-coder",
            prompt="direct_mcode_evidence_based_concise",
            store_in_memory=False,
            workers=0,
            cli_args=args,
        )

        # Verify data loading was called
        mock_load_data.assert_called_once_with(Path("trials.ndjson"), "trials")

    @patch("src.cli.trials_processor.ClinicalTrialsProcessorWorkflow")
    @patch("src.cli.trials_processor.McodeCLI.create_config")
    @patch("src.cli.trials_processor.McodeCLI.setup_logging")
    @patch("src.cli.trials_processor.load_ndjson_data")
    @patch("src.cli.trials_processor.McodeMemoryStorage")
    @patch("pathlib.Path.exists", return_value=True)
    def test_main_successful_processing_with_ingest(
        self,
        mock_path_exists,
        mock_memory_class,
        mock_load_data,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test successful trial processing with CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock memory storage
        mock_memory = MagicMock()
        mock_memory_class.return_value = mock_memory

        # Mock trial data
        mock_trial_data = [{"nct_id": "NCT12345678", "title": "Test Trial"}]
        mock_load_data.return_value = mock_trial_data

        # Create mock args
        args = argparse.Namespace(
            input_file="trials.ndjson",
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

        # Call main
        main(args)

        # Verify memory storage was initialized
        mock_memory_class.assert_called_once_with(source="test_memory")

        # Verify workflow was created with memory storage
        mock_workflow_class.assert_called_once_with(mock_config, mock_memory)


    def test_save_processed_data_to_file(self):
        """Test saving processed data to output file."""
        # Mock processed data with proper ClinicalTrials.gov structure for extract_trial_id
        mock_data = [
            {
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
                "McodeResults": {
                    "mcode_mappings": [
                        {"element_type": "CancerCondition", "value": "Breast Cancer"},
                        {"element_type": "CancerTreatment", "value": "Chemotherapy"},
                    ]
                },
                "original_trial_data": {
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT12345678"}
                    },
                    "title": "Test Trial",
                },
            }
        ]

        # Mock logger
        mock_logger = MagicMock()

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".ndjson"
        ) as temp_file:
            temp_path = temp_file.name

        try:
            # Call save_processed_data
            save_processed_data(mock_data, temp_path, mock_logger)

            # Verify file was written
            with open(temp_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1

                # Parse JSON and verify structure
                saved_data = json.loads(lines[0])
                assert saved_data["trial_id"] == "NCT12345678"
                assert len(saved_data["mcode_elements"]["mcode_mappings"]) == 2
                assert (
                    saved_data["original_trial_data"]["protocolSection"][
                        "identificationModule"
                    ]["nctId"]
                    == "NCT12345678"
                )

            # Verify logger was called
            mock_logger.info.assert_called_with(f"💾 mCODE data saved to: {temp_path}")

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_save_processed_data_to_stdout(self):
        """Test saving processed data to stdout."""
        # Mock processed data
        mock_data = [
            {
                "trial_id": "NCT12345678",
                "McodeResults": {"mcode_mappings": []},
                "original_trial_data": {"nct_id": "NCT12345678"},
            }
        ]

        # Mock logger
        mock_logger = MagicMock()

        # Mock stdout
        with patch("sys.stdout") as mock_stdout:
            # Call save_processed_data with no output file
            save_processed_data(mock_data, None, mock_logger)

            # Verify stdout was written to
            assert mock_stdout.write.call_count >= 1
            assert mock_stdout.flush.called

            # Verify logger was called
            mock_logger.info.assert_called_with("📤 mCODE data written to stdout")

    def test_print_processing_summary_complete_metadata(self):
        """Test printing processing summary with complete metadata."""
        metadata = {
            "total_trials": 10,
            "successful": 8,
            "failed": 2,
            "success_rate": 0.8,
            "stored_in_memory": True,  # Add this to match the expected output
            "model_used": "deepseek-coder",
            "prompt_used": "direct_mcode_evidence_based_concise",
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
        assert "🤖 Model: deepseek-coder" in calls
        assert "📝 Prompt: direct_mcode_evidence_based_concise" in calls

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

        # Verify total trials was logged and default values are used for missing fields
        calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert "📊 Total trials: 3" in calls
        assert "✅ Successful: 0" in calls  # Default value when not provided
        assert "❌ Failed: 0" in calls  # Default value when not provided
        assert "📈 Success rate: 0.0%" in calls  # Default value when not provided

    @patch("src.cli.trials_processor.ClinicalTrialsProcessorWorkflow")
    @patch("src.cli.trials_processor.McodeCLI.create_config")
    @patch("src.cli.trials_processor.McodeCLI.setup_logging")
    @patch("src.cli.trials_processor.load_ndjson_data")
    @patch("pathlib.Path.exists", return_value=True)
    def test_main_with_concurrency_settings(
        self,
        mock_path_exists,
        mock_load_data,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test that concurrency settings are properly passed to workflow."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trial_data = [{"nct_id": "NCT12345678", "title": "Test Trial"}]
        mock_load_data.return_value = mock_trial_data

        # Create mock args with concurrency settings
        args = argparse.Namespace(
            input_file="trials.ndjson",
            output_file="mcode_trials.ndjson",
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

        # Call main
        main(args)

        # Verify workflow was executed with correct parameters
        mock_workflow_class.assert_called_once_with(mock_config, None)
        mock_workflow.execute.assert_called_once_with(
            trials_data=mock_trial_data,
            model="gpt-4",
            prompt="direct_mcode_evidence_based_concise",
            store_in_memory=False,
            workers=8,
            cli_args=args,
        )

    @patch("src.cli.trials_processor.McodeCLI.create_config")
    @patch("src.cli.trials_processor.McodeCLI.setup_logging")
    @patch("src.cli.trials_processor.load_ndjson_data")
    @patch("pathlib.Path.exists", return_value=True)
    def test_main_memory_storage_initialization_failure(
        self, mock_path_exists, mock_load_data, mock_setup_logging, mock_create_config
    ):
        """Test handling of CORE Memory storage initialization failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trial_data = [{"nct_id": "NCT12345678", "title": "Test Trial"}]
        mock_load_data.return_value = mock_trial_data

        # Create mock args with ingest enabled
        args = argparse.Namespace(
            input_file="trials.ndjson",
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
        with patch("src.cli.trials_processor.McodeMemoryStorage") as mock_memory_class:
            mock_memory_class.side_effect = Exception("Storage connection failed")

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
