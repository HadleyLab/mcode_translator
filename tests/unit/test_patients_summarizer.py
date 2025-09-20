#!/usr/bin/env python3
"""
Unit tests for patients_summarizer CLI module.

Tests the command-line interface for generating natural language summaries from mCODE patient data,
including file handling, data parsing, workflow execution, summary extraction, and CORE Memory integration.
"""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.patients_summarizer import (
    create_parser,
    main,
    load_mcode_patients,
    save_summaries,
)


class TestPatientsSummarizerCLI:
    """Test the patients_summarizer CLI module."""

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert "Generate natural language summaries" in parser.description

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "input_file" in actions
        assert "output_file" in actions
        assert "dry_run" in actions

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
        assert "Input file with mCODE patient data" in input_action.help

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

    def test_create_parser_dry_run_argument(self):
        """Test the dry_run argument configuration."""
        parser = create_parser()

        # Find the dry_run action
        dry_run_action = None
        for action in parser._actions:
            if action.dest == "dry_run":
                dry_run_action = action
                break

        assert dry_run_action is not None
        assert "Run summarization without storing results" in dry_run_action.help

    @patch("src.cli.patients_summarizer.PatientsSummarizerWorkflow")
    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
    def test_main_successful_summarization_without_ingest(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test successful patient summarization without CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock patient data
        mock_mcode_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": [{"element_type": "CancerCondition"}]},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_patients.ndjson",
            output_file="summaries.ndjson",
            ingest=False,
            dry_run=False,
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
                    "natural_language_summary": "This is a test patient summary",
                    "mcode_mappings": [{"element_type": "CancerCondition"}]
                },
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]
        mock_result.metadata = {
            "total_patients": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
            "model_used": "deepseek-coder",
            "prompt_used": "direct_mcode_evidence_based_concise"
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
            patients_data=mock_mcode_data,
            store_in_memory=False,
            workers=0
        )

    @patch("src.cli.patients_summarizer.PatientsSummarizerWorkflow")
    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
    @patch("src.cli.patients_summarizer.McodeMemoryStorage")
    def test_main_successful_summarization_with_ingest(self, mock_memory_class, mock_setup_logging,
                                                      mock_create_config, mock_workflow_class):
        """Test successful patient summarization with CORE Memory ingestion."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock memory storage
        mock_memory = MagicMock()
        mock_memory_class.return_value = mock_memory

        # Mock patient data
        mock_mcode_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_patients.ndjson",
            output_file=None,
            ingest=True,
            dry_run=False,
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
            "total_patients": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
            "stored_in_memory": True
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

    @patch("src.cli.patients_summarizer.PatientsSummarizerWorkflow")
    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
    @patch("src.cli.patients_summarizer.McodeMemoryStorage")
    def test_main_dry_run_mode(self, mock_memory_class, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test dry run mode prevents storage in CORE Memory."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock memory storage
        mock_memory = MagicMock()
        mock_memory_class.return_value = mock_memory

        # Mock patient data
        mock_mcode_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]

        # Create mock args with dry run enabled
        args = argparse.Namespace(
            input_file="mcode_patients.ndjson",
            output_file="summaries.ndjson",
            ingest=True,
            dry_run=True,
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
                    "natural_language_summary": "Test summary",
                    "mcode_mappings": []
                },
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]
        mock_result.metadata = {
            "total_patients": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
            "stored_in_memory": False  # Should be False due to dry run
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

        # Verify workflow was executed with memory storage but store_in_memory=False due to dry run
        mock_workflow_class.assert_called_once_with(mock_config, mock_memory)  # Memory storage is initialized
        mock_workflow.execute.assert_called_once_with(
            patients_data=mock_mcode_data,
            store_in_memory=False,  # Should be False due to dry run
            workers=0
        )

    def test_main_no_input_file_uses_stdin(self):
        """Test that main uses stdin when no input file is provided."""
        # Create mock args without input file
        args = argparse.Namespace(
            input_file=None,
            output_file=None,
            ingest=False,
            dry_run=False,
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
        mock_mcode_data = {"patient_id": "PATIENT123", "McodeResults": {"mcode_mappings": []}}
        with patch("sys.stdin.read", return_value=json.dumps(mock_mcode_data)), \
             patch("src.cli.patients_summarizer.McodeCLI.setup_logging"), \
             patch("src.cli.patients_summarizer.McodeCLI.create_config"), \
             patch("src.cli.patients_summarizer.PatientsSummarizerWorkflow") as mock_workflow_class, \
             patch("sys.exit") as mock_exit:

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

    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
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
            dry_run=False,
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

    @patch("src.cli.patients_summarizer.PatientsSummarizerWorkflow")
    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
    def test_main_workflow_failure(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test handling of workflow execution failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock patient data
        mock_mcode_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_patients.ndjson",
            output_file="summaries.ndjson",
            ingest=False,
            dry_run=False,
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

    @patch("src.cli.patients_summarizer.PatientsSummarizerWorkflow")
    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
    def test_main_keyboard_interrupt_handling(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test handling of keyboard interrupt during execution."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock patient data
        mock_mcode_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]

        # Create mock args
        args = argparse.Namespace(
            input_file="mcode_patients.ndjson",
            output_file=None,
            ingest=False,
            dry_run=False,
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

    def test_load_mcode_patients_from_file(self):
        """Test loading mCODE patients from file."""
        mock_data = [
            {"patient_id": "PATIENT123", "McodeResults": {"mcode_mappings": []}},
            {"patient_id": "PATIENT456", "McodeResults": {"mcode_mappings": []}}
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ndjson') as temp_file:
            for item in mock_data:
                json.dump(item, temp_file)
                temp_file.write('\n')
            temp_path = temp_file.name

        try:
            # Call load_mcode_patients
            result = load_mcode_patients(temp_path)

            # Verify data was loaded correctly
            assert len(result) == 2
            assert result[0]["patient_id"] == "PATIENT123"
            assert result[1]["patient_id"] == "PATIENT456"

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_load_mcode_patients_from_stdin(self):
        """Test loading mCODE patients from stdin."""
        mock_data = {"patient_id": "PATIENT123", "McodeResults": {"mcode_mappings": []}}

        # Mock stdin
        with patch("sys.stdin.read", return_value=json.dumps(mock_data)):
            # Call load_mcode_patients with no input file
            result = load_mcode_patients(None)

            # Verify data was loaded correctly
            assert len(result) == 1
            assert result[0]["patient_id"] == "PATIENT123"

    def test_load_mcode_patients_invalid_json(self):
        """Test handling of invalid JSON in input."""
        # Create temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ndjson') as temp_file:
            temp_file.write('{"invalid": json}\n')
            temp_file.write('{"valid": "data"}\n')
            temp_path = temp_file.name

        try:
            # Call load_mcode_patients
            result = load_mcode_patients(temp_path)

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
                "patient_id": "PATIENT123",
                "summary": "Test patient summary",
                "mcode_elements": [{"element_type": "CancerCondition"}]
            }
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ndjson') as temp_file:
            temp_path = temp_file.name

        try:
            # Call save_summaries
            save_summaries(mock_summaries, temp_path)

            # Verify file was written
            with open(temp_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1

                # Parse JSON and verify structure
                saved_data = json.loads(lines[0])
                assert saved_data["patient_id"] == "PATIENT123"
                assert saved_data["summary"] == "Test patient summary"
                assert len(saved_data["mcode_elements"]) == 1

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_save_summaries_to_stdout(self):
        """Test saving summaries to stdout."""
        mock_summaries = [
            {
                "patient_id": "PATIENT123",
                "summary": "Test patient summary",
                "mcode_elements": []
            }
        ]

        # Mock stdout
        with patch("sys.stdout") as mock_stdout:
            # Call save_summaries with no output file
            save_summaries(mock_summaries, None)

            # Verify stdout was written to
            assert mock_stdout.write.call_count >= 1
            assert mock_stdout.flush.called

    @patch("src.cli.patients_summarizer.PatientsSummarizerWorkflow")
    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
    def test_main_with_concurrency_settings(self, mock_setup_logging, mock_create_config, mock_workflow_class):
        """Test that concurrency settings are properly passed to workflow."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock patient data
        mock_mcode_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]

        # Create mock args with concurrency settings
        args = argparse.Namespace(
            input_file="mcode_patients.ndjson",
            output_file="summaries.ndjson",
            ingest=False,
            dry_run=False,
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
            "total_patients": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0
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
            patients_data=mock_mcode_data,
            store_in_memory=False,
            workers=8
        )

    @patch("src.cli.patients_summarizer.McodeCLI.create_config")
    @patch("src.cli.patients_summarizer.McodeCLI.setup_logging")
    @patch("src.cli.patients_summarizer.McodeMemoryStorage")
    def test_main_memory_storage_initialization_failure(self, mock_memory_class, mock_setup_logging, mock_create_config):
        """Test handling of CORE Memory storage initialization failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock patient data
        mock_mcode_data = [
            {
                "patient_id": "PATIENT123",
                "McodeResults": {"mcode_mappings": []},
                "entry": [{"resource": {"resourceType": "Patient", "id": "PATIENT123"}}]
            }
        ]

        # Create mock args with ingest enabled
        args = argparse.Namespace(
            input_file="mcode_patients.ndjson",
            output_file=None,
            ingest=True,
            dry_run=False,
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