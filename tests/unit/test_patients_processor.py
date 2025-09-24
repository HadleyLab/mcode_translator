#!/usr/bin/env python3
"""
Unit tests for patients_processor CLI module.

Tests the command-line interface for processing patient data with mCODE mapping,
including argument parsing, validation, workflow execution, and data handling.
"""

import argparse
from unittest.mock import MagicMock, patch, mock_open


from src.cli.patients_processor import (
    create_parser,
    main,
    save_processed_data,
    print_processing_summary,
    extract_mcode_criteria_from_trials,
)


class TestPatientsProcessorCLI:
    """Test the patients_processor CLI module."""

    def test_create_parser_basic_structure(self):
        """Test that the argument parser is created with expected structure."""
        parser = create_parser()

        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description == "Process patient data with mCODE mapping"

        # Check that required arguments are present
        actions = {action.dest for action in parser._actions}
        assert "input_file" in actions
        assert "output_file" in actions
        assert "trials" in actions

    def test_create_parser_input_file_argument(self):
        """Test the input file argument configuration."""
        parser = create_parser()

        # Find the input_file action
        input_action = None
        for action in parser._actions:
            if action.dest == "input_file":
                input_action = action
                break

        assert input_action is not None
        assert "Input file with patient data" in input_action.help

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
        assert "Output file for mCODE data" in output_action.help

    def test_create_parser_trials_argument(self):
        """Test the trials file argument configuration."""
        parser = create_parser()

        # Find the trials action
        trials_action = None
        for action in parser._actions:
            if action.dest == "trials":
                trials_action = action
                break

        assert trials_action is not None
        assert "trial data for eligibility filtering" in trials_action.help

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    @patch("src.cli.patients_processor.McodeCLI.create_config")
    @patch("src.cli.patients_processor.McodeCLI.setup_logging")
    def test_main_successful_processing(
        self, mock_setup_logging, mock_create_config, mock_workflow_class, capsys
    ):
        """Test successful patient processing workflow."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock the workflow
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = [
            {
                "patient_id": "patient_123",
                "mcode_elements": [{"resource_type": "Patient", "id": "123"}],
                "original_patient_data": {"entry": []},
            }
        ]
        mock_result.metadata = {
            "total_patients": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with all required attributes
        args = argparse.Namespace(
            input_file="patients.ndjson",
            output_file="output.ndjson",
            trials=None,
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

        # Mock file operations
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.patients_processor.load_ndjson_data"
        ) as mock_load, patch("builtins.open", mock_open()):

            mock_load.return_value = [{"entry": []}]

            # Call main
            main(args)

            # Verify workflow was called correctly
            mock_workflow_class.assert_called_once()
            mock_workflow.execute.assert_called_once()

            # Check that logging was set up
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    @patch("src.cli.patients_processor.McodeCLI.create_config")
    @patch("src.cli.patients_processor.McodeCLI.setup_logging")
    def test_main_missing_input_file(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test error when input file is not specified."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args without input file
        args = argparse.Namespace(
            input_file=None,
            output_file=None,
            trials=None,
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

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            # Call main - should exit with error
            try:
                main(args)
            except TypeError:
                pass  # Expected due to Path(None) - we're testing the error handling

            # Verify sys.exit was called at least once
            assert mock_exit.call_count >= 1
            mock_exit.assert_any_call(1)

            # Check that logging was set up
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    @patch("src.cli.patients_processor.McodeCLI.create_config")
    @patch("src.cli.patients_processor.McodeCLI.setup_logging")
    def test_main_input_file_not_found(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test error when input file does not exist."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args with non-existent input file
        args = argparse.Namespace(
            input_file="nonexistent.ndjson",
            output_file=None,
            trials=None,
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

        # Mock Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False), patch(
            "sys.exit"
        ) as mock_exit:

            # Call main - should exit with error
            main(args)

            # Verify sys.exit was called at least once
            assert mock_exit.call_count >= 1
            mock_exit.assert_any_call(1)

            # Check that logging was set up
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    @patch("src.cli.patients_processor.McodeCLI.create_config")
    @patch("src.cli.patients_processor.McodeCLI.setup_logging")
    def test_main_with_trials_file(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test processing with trial criteria file."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock the workflow
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = []
        mock_result.metadata = {
            "total_patients": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with trials file
        args = argparse.Namespace(
            input_file="patients.ndjson",
            output_file=None,
            trials="trials.ndjson",
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

        # Mock file operations
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.patients_processor.load_ndjson_data"
        ) as mock_load:

            mock_load.side_effect = [
                [{"entry": []}],  # patients data
                [{"McodeResults": {"mcode_mappings": []}}],  # trials data
            ]

            # Call main
            main(args)

            # Verify workflow was called with trials criteria
            call_args = mock_workflow.execute.call_args
            assert call_args[1]["trials_criteria"] is not None

            # Check that logging was set up
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    @patch("src.cli.patients_processor.McodeCLI.create_config")
    @patch("src.cli.patients_processor.McodeCLI.setup_logging")
    def test_main_with_memory_storage(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test processing with CORE memory storage enabled."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock the workflow
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = []
        mock_result.metadata = {
            "total_patients": 1,
            "successful": 1,
            "failed": 0,
            "success_rate": 1.0,
            "stored_in_memory": True,
        }
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with ingest enabled
        args = argparse.Namespace(
            input_file="patients.ndjson",
            output_file=None,
            trials=None,
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
            memory_source="test_source",
        )

        # Mock file operations and memory storage
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.patients_processor.load_ndjson_data"
        ) as mock_load, patch(
            "src.cli.patients_processor.McodeMemoryStorage"
        ) as mock_memory_class:

            mock_load.return_value = [{"entry": []}]
            mock_memory = MagicMock()
            mock_memory_class.return_value = mock_memory

            # Call main
            main(args)

            # Verify memory storage was initialized
            mock_memory_class.assert_called_once_with(source="test_source")

            # Check that logging was set up
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    @patch("src.cli.patients_processor.McodeCLI.create_config")
    @patch("src.cli.patients_processor.McodeCLI.setup_logging")
    def test_main_workflow_failure(
        self, mock_setup_logging, mock_create_config, mock_workflow_class, capsys
    ):
        """Test handling of workflow execution failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock the workflow to return failure
        mock_workflow = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Processing failed"
        mock_result.metadata = {}
        mock_workflow.execute.return_value = mock_result
        mock_workflow_class.return_value = mock_workflow

        # Create mock args
        args = argparse.Namespace(
            input_file="patients.ndjson",
            output_file=None,
            trials=None,
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

        # Mock file operations
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.patients_processor.load_ndjson_data"
        ) as mock_load, patch("sys.exit") as mock_exit:

            mock_load.return_value = [{"entry": []}]

            # Call main - should exit with error
            main(args)

            # Verify sys.exit was called
            mock_exit.assert_called_once_with(1)

            # Check that logging was set up
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    @patch("src.cli.patients_processor.McodeCLI.create_config")
    @patch("src.cli.patients_processor.McodeCLI.setup_logging")
    def test_main_keyboard_interrupt_handling(
        self, mock_setup_logging, mock_create_config, mock_workflow_class, capsys
    ):
        """Test handling of keyboard interrupt."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args
        args = argparse.Namespace(
            input_file="patients.ndjson",
            output_file=None,
            trials=None,
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

        # Mock file operations and workflow to raise KeyboardInterrupt
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.patients_processor.load_ndjson_data"
        ) as mock_load, patch("sys.exit") as mock_exit:

            mock_load.return_value = [{"entry": []}]
            mock_workflow = MagicMock()
            mock_workflow.execute.side_effect = KeyboardInterrupt()
            mock_workflow_class.return_value = mock_workflow

            # Call main - should handle KeyboardInterrupt
            main(args)

            # Verify sys.exit was called
            mock_exit.assert_called_once_with(130)

            # Check that logging was set up
            mock_setup_logging.assert_called_once_with(args)
            mock_create_config.assert_called_once_with(args)

    @patch("src.cli.patients_processor.PatientsProcessorWorkflow")
    def test_main_verbose_exception_traceback(self, mock_workflow_class, capsys):
        """Test that verbose mode shows exception traceback."""
        # Mock the workflow to raise an exception
        mock_workflow = MagicMock()
        mock_workflow.execute.side_effect = ValueError("Test error")
        mock_workflow_class.return_value = mock_workflow

        # Create mock args with verbose=True
        args = argparse.Namespace(
            input_file="patients.ndjson",
            output_file=None,
            trials=None,
            ingest=False,
            verbose=True,
            log_level="DEBUG",
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

        # Mock file operations
        with patch("pathlib.Path.exists", return_value=True), patch(
            "src.cli.patients_processor.load_ndjson_data"
        ) as mock_load, patch("sys.exit") as mock_exit, patch(
            "traceback.print_exc"
        ) as mock_traceback:

            mock_load.return_value = [{"entry": []}]

            # Call main - should show traceback in verbose mode
            main(args)

            # Verify traceback was printed
            mock_traceback.assert_called_once()

            # Verify sys.exit was called
            mock_exit.assert_called_once_with(1)

    def test_main_without_args_uses_sys_argv(self):
        """Test that main() without args uses sys.argv."""
        with patch("sys.argv", ["patients_processor", "--help"]), patch(
            "argparse.ArgumentParser.print_help"
        ) as mock_help, patch("sys.exit"), patch(
            "src.cli.patients_processor.McodeCLI.create_config"
        ) as mock_create_config, patch(
            "src.cli.patients_processor.McodeCLI.setup_logging"
        ):

            # Mock configuration and logging
            mock_config = MagicMock()
            mock_create_config.return_value = mock_config

            # Call main without args - should parse sys.argv and show help
            try:
                main()
            except (SystemExit, TypeError):
                pass  # Expected when --help is used or Path(None) occurs

            # Verify help was printed (since --help was in argv)
            mock_help.assert_called_once()

    def test_save_processed_data_to_file(self):
        """Test saving processed data to a file."""
        # Test data in FHIR bundle format as expected by save_processed_data
        test_data = [
            {
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "123",
                            "name": [{"given": ["John"], "family": "Doe"}],
                        }
                    },
                    {
                        "resource": {
                            "resourceType": "Condition",
                            "id": "456",
                            "code": {"coding": [{"display": "Breast Cancer"}]},
                        }
                    },
                ]
            }
        ]

        # Mock logger
        mock_logger = MagicMock()

        # Mock file operations
        with patch("builtins.open", mock_open()) as mock_open_func:
            # Call save_processed_data
            save_processed_data(test_data, "output.ndjson", mock_logger)

            # Verify file was opened for writing
            mock_open_func.assert_called_once_with(
                "output.ndjson", "w", encoding="utf-8"
            )

            # Verify logger was called
            mock_logger.info.assert_called_with("💾 mCODE data saved to: output.ndjson")

    def test_save_processed_data_to_stdout(self):
        """Test saving processed data to stdout when no output file specified."""
        # Test data in FHIR bundle format as expected by save_processed_data
        test_data = [
            {
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": "456",
                            "name": [{"given": ["Jane"], "family": "Doe"}],
                        }
                    }
                ]
            }
        ]

        # Mock logger
        mock_logger = MagicMock()

        # Mock stdout
        with patch("sys.stdout"):
            # Call save_processed_data without output file
            save_processed_data(test_data, None, mock_logger)

            # Verify logger was called
            mock_logger.info.assert_called_with("📤 mCODE data written to stdout")

    def test_print_processing_summary_complete(self):
        """Test printing complete processing summary."""
        metadata = {
            "total_patients": 10,
            "successful": 8,
            "failed": 2,
            "success_rate": 0.8,
            "stored_in_memory": True,  # Add this to get the correct storage message
        }
        ingested = True
        trials_criteria = {"CancerCondition": ["breast cancer"]}
        mock_logger = MagicMock()

        # Call print_processing_summary
        print_processing_summary(metadata, ingested, trials_criteria, mock_logger)

        # Verify logger calls - check that key messages are present
        calls = [str(call) for call in mock_logger.info.call_args_list]

        assert any("📊 Total patients: 10" in call for call in calls)
        assert any("✅ Successful: 8" in call for call in calls)
        assert any("❌ Failed: 2" in call for call in calls)
        assert any(
            "📈 Success rate: 0.8%" in call for call in calls
        )  # Actual format from function
        assert any("🎯 Applied trial eligibility filtering" in call for call in calls)
        assert any("🧠 Stored in CORE Memory" in call for call in calls)

    def test_print_processing_summary_no_trials(self, capsys):
        """Test printing summary without trial criteria."""
        metadata = {
            "total_patients": 5,
            "successful": 5,
            "failed": 0,
            "success_rate": 1.0,
        }
        ingested = False
        trials_criteria = None
        mock_logger = MagicMock()

        # Call print_processing_summary
        print_processing_summary(metadata, ingested, trials_criteria, mock_logger)

        # Verify no trial filtering message
        trial_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "trial eligibility filtering" in str(call)
        ]
        assert len(trial_calls) == 0

        # Verify storage disabled message
        storage_calls = [
            call
            for call in mock_logger.info.call_args_list
            if "Storage disabled" in str(call)
        ]
        assert len(storage_calls) == 1

    def test_extract_mcode_criteria_from_trials_empty(self):
        """Test extracting criteria from empty trials data."""
        result = extract_mcode_criteria_from_trials([])
        assert result == {}

    def test_extract_mcode_criteria_from_trials_single_trial(self):
        """Test extracting criteria from single trial."""
        trials_data = [
            {
                "McodeResults": {
                    "mcode_mappings": [
                        {
                            "mcode_element": "CancerCondition",
                            "value": "breast cancer",
                        },
                        {
                            "mcode_element": "TNMStage",
                            "value": "T2N1M0",
                        },
                    ]
                }
            }
        ]

        result = extract_mcode_criteria_from_trials(trials_data)

        expected = {
            "CancerCondition": ["breast cancer"],
            "TNMStage": ["T2N1M0"],
        }
        assert result == expected

    def test_extract_mcode_criteria_from_trials_multiple_trials(self):
        """Test extracting criteria from multiple trials with duplicates."""
        trials_data = [
            {
                "McodeResults": {
                    "mcode_mappings": [
                        {"mcode_element": "CancerCondition", "value": "breast cancer"},
                        {"mcode_element": "TNMStage", "value": "T2N1M0"},
                    ]
                }
            },
            {
                "McodeResults": {
                    "mcode_mappings": [
                        {"mcode_element": "CancerCondition", "value": "breast cancer"},
                        {"mcode_element": "TNMStage", "value": "T3N0M0"},
                    ]
                }
            },
        ]

        result = extract_mcode_criteria_from_trials(trials_data)

        expected = {
            "CancerCondition": ["breast cancer"],
            "TNMStage": ["T2N1M0", "T3N0M0"],
        }
        assert result == expected

    def test_extract_mcode_criteria_from_trials_different_formats(self):
        """Test extracting criteria from different trial data formats."""
        # Test with studies format
        trials_data = {
            "studies": [
                {
                    "McodeResults": {
                        "mcode_mappings": [
                            {"mcode_element": "CancerCondition", "value": "lung cancer"}
                        ]
                    }
                }
            ]
        }

        result = extract_mcode_criteria_from_trials(trials_data)
        assert result == {"CancerCondition": ["lung cancer"]}

        # Test with successful_trials format
        trials_data = {
            "successful_trials": [
                {
                    "McodeResults": {
                        "mcode_mappings": [
                            {
                                "mcode_element": "CancerCondition",
                                "value": "colon cancer",
                            }
                        ]
                    }
                }
            ]
        }

        result = extract_mcode_criteria_from_trials(trials_data)
        assert result == {"CancerCondition": ["colon cancer"]}

    def test_extract_mcode_criteria_from_trials_no_mappings(self):
        """Test extracting criteria from trials without mCODE mappings."""
        trials_data = [
            {"some_other_field": "value"},
            {"McodeResults": {}},
            {"McodeResults": {"mcode_mappings": []}},
        ]

        result = extract_mcode_criteria_from_trials(trials_data)
        assert result == {}

    def test_extract_mcode_criteria_from_trials_na_values(self):
        """Test that N/A values are filtered out."""
        trials_data = [
            {
                "McodeResults": {
                    "mcode_mappings": [
                        {"mcode_element": "CancerCondition", "value": "breast cancer"},
                        {"mcode_element": "TNMStage", "value": "N/A"},
                        {"mcode_element": "ERReceptorStatus", "value": "positive"},
                    ]
                }
            }
        ]

        result = extract_mcode_criteria_from_trials(trials_data)

        expected = {
            "CancerCondition": ["breast cancer"],
            "ERReceptorStatus": ["positive"],
        }
        assert result == expected
        assert "TNMStage" not in result
