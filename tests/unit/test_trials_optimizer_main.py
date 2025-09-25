#!/usr/bin/env python3
"""
Unit tests for trials_optimizer main function.
"""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.cli.trials_optimizer import main


class TestTrialsOptimizerMain:
    """Test the trials_optimizer main function."""

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    def test_main_list_prompts(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test listing available prompts."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock workflow
        mock_workflow = MagicMock()
        mock_workflow.get_available_prompts.return_value = [
            "prompt1",
            "prompt2",
            "prompt3",
        ]
        mock_workflow_class.return_value = mock_workflow

        # Create mock args
        args = argparse.Namespace(
            list_prompts=True,
            list_models=False,
            trials_file=None,
            cv_folds=None,
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
            async_queue=False,
            prompts=None,
            models=None,
            max_combinations=None,
            save_config=None,
            save_mcode_elements=None,
        )

        # Call main
        main(args)

        # Verify workflow was created and prompts were retrieved
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.get_available_prompts.assert_called_once()

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    def test_main_list_models(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test listing available models."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock workflow
        mock_workflow = MagicMock()
        mock_workflow.get_available_models.return_value = ["model1", "model2", "model3"]
        mock_workflow_class.return_value = mock_workflow

        # Create mock args
        args = argparse.Namespace(
            list_prompts=False,
            list_models=True,
            trials_file=None,
            cv_folds=None,
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
            async_queue=False,
            prompts=None,
            models=None,
            max_combinations=None,
            save_config=None,
            save_mcode_elements=None,
        )

        # Call main
        main(args)

        # Verify workflow was created and models were retrieved
        mock_workflow_class.assert_called_once_with(mock_config, memory_storage=False)
        mock_workflow.get_available_models.assert_called_once()

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    def test_main_missing_trials_file(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test error when trials_file is not provided."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args without trials_file
        args = argparse.Namespace(
            list_prompts=False,
            list_models=False,
            trials_file=None,
            cv_folds=3,
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
            async_queue=False,
            prompts=None,
            models=None,
            max_combinations=None,
            save_config=None,
            save_mcode_elements=None,
        )

        # Call main - should exit with error when trials_file is None
        with pytest.raises(SystemExit):
            main(args)

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    def test_main_missing_cv_folds(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test error when cv_folds is not provided."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args without cv_folds
        args = argparse.Namespace(
            list_prompts=False,
            list_models=False,
            trials_file="trials.ndjson",
            cv_folds=None,
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
            async_queue=False,
            prompts=None,
            models=None,
            max_combinations=None,
            save_config=None,
            save_mcode_elements=None,
        )

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            # Call main - should exit with error
            try:
                main(args)
            except SystemExit:
                pass  # Expected when cv_folds is missing

            # Verify sys.exit was called
            assert mock_exit.call_count >= 1

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    def test_main_trials_file_not_found(
        self, mock_setup_logging, mock_create_config, mock_workflow_class
    ):
        """Test error when trials file does not exist."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create mock args with non-existent file
        args = argparse.Namespace(
            list_prompts=False,
            list_models=False,
            trials_file="nonexistent.ndjson",
            cv_folds=3,
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
            async_queue=False,
            prompts=None,
            models=None,
            max_combinations=None,
            save_config=None,
            save_mcode_elements=None,
        )

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit") as mock_exit:
            # Call main - should exit with error
            try:
                main(args)
            except SystemExit:
                pass  # Expected when file not found

            # Verify sys.exit was called
            assert mock_exit.call_count >= 1

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.Observer")
    def test_main_successful_optimization_single_json(
        self,
        mock_observer_class,
        mock_asyncio_run,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test successful optimization with single JSON file format."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data in single JSON format
        mock_trials_data = [
            {
                "trial_id": "NCT12345678",
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            },
            {
                "trial_id": "NCT87654321",
                "protocolSection": {"identificationModule": {"nctId": "NCT87654321"}},
            },
        ]
        mock_json_data = {"successful_trials": mock_trials_data}

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump(mock_json_data, temp_file)
            temp_path = temp_file.name

        try:
            # Create mock args
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=False,
                prompts="prompt1,prompt2",
                models=["model1", "model2"],
                max_combinations=None,
                save_config="optimal_config.json",
                save_mcode_elements="mcode_elements.json",
            )

            # Mock workflow and result
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_workflow.execute = AsyncMock(return_value=mock_result)
            mock_workflow.validate_combination.return_value = True
            mock_workflow_class.return_value = mock_workflow

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Call main
            main(args)

            # Verify workflow was executed with correct parameters
            mock_workflow_class.assert_called_with(mock_config, memory_storage=False)
            mock_workflow.execute.assert_called_once()
            call_kwargs = mock_workflow.execute.call_args[1]

            assert call_kwargs["trials_data"] == mock_trials_data
            assert call_kwargs["cv_folds"] == 3
            assert call_kwargs["prompts"] == ["prompt1", "prompt2"]
            assert call_kwargs["models"] == ["model1", "model2"]
            assert call_kwargs["output_config"] == "optimal_config.json"
            assert call_kwargs["save_mcode_elements"] == "mcode_elements.json"
            assert call_kwargs["cli_args"] == args

        finally:
            # Clean up
            Path(temp_path).unlink()

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.Observer")
    def test_main_successful_optimization_ndjson(
        self,
        mock_observer_class,
        mock_asyncio_run,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test successful optimization with NDJSON file format."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data in NDJSON format
        mock_trials_data = [
            {
                "trial_id": "NCT12345678",
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            },
            {
                "trial_id": "NCT87654321",
                "protocolSection": {"identificationModule": {"nctId": "NCT87654321"}},
            },
        ]

        # Create temporary NDJSON file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".ndjson"
        ) as temp_file:
            for trial in mock_trials_data:
                json.dump(trial, temp_file)
                temp_file.write("\n")
            temp_path = temp_file.name

        try:
            # Create mock args
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=False,
                prompts=None,
                models=None,
                max_combinations=None,
                save_config=None,
                save_mcode_elements=None,
            )

            # Mock workflow and result
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_workflow.execute = AsyncMock(return_value=mock_result)
            mock_workflow_class.return_value = mock_workflow

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Call main
            main(args)

            # Verify workflow was executed with correct parameters
            mock_workflow_class.assert_called_with(mock_config, memory_storage=False)
            mock_workflow.execute.assert_called_once()
            call_kwargs = mock_workflow.execute.call_args[1]

            assert call_kwargs["trials_data"] == mock_trials_data
            assert call_kwargs["cv_folds"] == 3
            assert call_kwargs["prompts"] == [
                "direct_mcode_evidence_based_concise"
            ]  # Default
            assert call_kwargs["models"] == ["deepseek-coder"]  # Default

        finally:
            # Clean up
            Path(temp_path).unlink()

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.Observer")
    def test_main_invalid_combination(
        self,
        mock_observer_class,
        mock_asyncio_run,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test handling of invalid prompt×model combinations."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trials_data = [
            {
                "trial_id": "NCT12345678",
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            }
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump({"successful_trials": mock_trials_data}, temp_file)
            temp_path = temp_file.name

        try:
            # Create mock args with invalid combination
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=False,
                prompts="invalid_prompt",
                models=["invalid_model"],
                max_combinations=None,
                save_config=None,
                save_mcode_elements=None,
            )

            # Mock workflow
            mock_workflow = MagicMock()
            mock_workflow.validate_combination.return_value = (
                False  # Invalid combination
            )
            mock_workflow_class.return_value = mock_workflow

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Mock sys.exit to prevent actual exit
            with patch("sys.exit") as mock_exit:
                # Call main - should exit with error
                try:
                    main(args)
                except SystemExit:
                    pass  # Expected when invalid combination

                # Verify sys.exit was called
                assert mock_exit.call_count >= 1

        finally:
            # Clean up
            Path(temp_path).unlink()

    @patch("src.cli.trials_optimizer.Observer")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    def test_main_workflow_failure(
        self,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
        mock_asyncio_run,
        mock_observer_class,
    ):
        """Test handling of workflow execution failure."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trials_data = [
            {
                "trial_id": "NCT12345678",
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            }
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump({"successful_trials": mock_trials_data}, temp_file)
            temp_path = temp_file.name

        try:
            # Create mock args
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=False,
                prompts=None,
                models=None,
                max_combinations=None,
                save_config=None,
                save_mcode_elements=None,
            )

            # Mock workflow and failed result
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.error_message = "Optimization failed"
            mock_workflow.execute = AsyncMock(return_value=mock_result)
            mock_workflow.validate_combination.return_value = (
                True  # Ensure validation passes
            )
            mock_workflow_class.return_value = mock_workflow

            # Mock asyncio.run to return the failed result
            mock_asyncio_run.return_value = mock_result

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Mock sys.exit to prevent actual exit
            with patch("sys.exit") as mock_exit:
                # Call main - should exit with error
                try:
                    main(args)
                except SystemExit:
                    pass  # Expected when workflow fails

                # Verify sys.exit was called
                assert mock_exit.call_count >= 1

        finally:
            # Clean up
            Path(temp_path).unlink()

    @patch("src.cli.trials_optimizer.Observer")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    def test_main_keyboard_interrupt_handling(
        self,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
        mock_asyncio_run,
        mock_observer_class,
    ):
        """Test handling of keyboard interrupt during execution."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trials_data = [
            {
                "trial_id": "NCT12345678",
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            }
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump({"successful_trials": mock_trials_data}, temp_file)
            temp_path = temp_file.name

        try:
            # Create mock args
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=False,
                prompts=None,
                models=None,
                max_combinations=None,
                save_config=None,
                save_mcode_elements=None,
            )

            # Mock workflow to raise KeyboardInterrupt
            mock_workflow = MagicMock()
            mock_workflow.execute = AsyncMock(side_effect=KeyboardInterrupt())
            mock_workflow.validate_combination.return_value = (
                True  # Ensure validation passes
            )
            mock_workflow_class.return_value = mock_workflow

            # Mock asyncio.run to raise KeyboardInterrupt
            mock_asyncio_run.side_effect = KeyboardInterrupt()

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

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

        finally:
            # Clean up
            Path(temp_path).unlink()

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.Observer")
    def test_main_with_max_combinations(
        self,
        mock_observer_class,
        mock_asyncio_run,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test that max_combinations parameter is properly passed to workflow."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trials_data = [
            {
                "trial_id": "NCT12345678",
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            }
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump({"successful_trials": mock_trials_data}, temp_file)
            temp_path = temp_file.name

        try:
            # Create mock args with max_combinations
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=False,
                prompts=None,
                models=None,
                max_combinations=50,
                save_config=None,
                save_mcode_elements=None,
            )

            # Mock workflow and result
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_workflow.execute = AsyncMock(return_value=mock_result)
            mock_workflow_class.return_value = mock_workflow

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Call main
            main(args)

            # Verify max_combinations was passed to workflow
            mock_workflow.execute.assert_called_once()
            call_kwargs = mock_workflow.execute.call_args[1]
            assert call_kwargs["max_combinations"] == 50

        finally:
            # Clean up
            Path(temp_path).unlink()

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.Observer")
    def test_main_empty_trials_data(
        self,
        mock_observer_class,
        mock_asyncio_run,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test handling of empty trials data."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Create temporary file with empty data
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump({"successful_trials": []}, temp_file)
            temp_path = temp_file.name

        try:
            # Create mock args
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=False,
                prompts=None,
                models=None,
                max_combinations=None,
                save_config=None,
                save_mcode_elements=None,
            )

            # Mock workflow
            mock_workflow = MagicMock()
            mock_workflow_class.return_value = mock_workflow

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Mock sys.exit to prevent actual exit
            with patch("sys.exit") as mock_exit:
                # Call main - should exit with error due to empty data
                try:
                    main(args)
                except SystemExit:
                    pass  # Expected when no trial data found

                # Verify sys.exit was called
                assert mock_exit.call_count >= 1

        finally:
            # Clean up
            Path(temp_path).unlink()

    @patch("src.cli.trials_optimizer.TrialsOptimizerWorkflow")
    @patch("src.cli.trials_optimizer.McodeCLI.create_config")
    @patch("src.cli.trials_optimizer.McodeCLI.setup_logging")
    @patch("src.cli.trials_optimizer.asyncio.run")
    @patch("src.cli.trials_optimizer.Observer")
    def test_main_async_queue_mode(
        self,
        mock_observer_class,
        mock_asyncio_run,
        mock_setup_logging,
        mock_create_config,
        mock_workflow_class,
    ):
        """Test async queue mode configuration."""
        # Mock configuration and logging
        mock_config = MagicMock()
        mock_create_config.return_value = mock_config

        # Mock trial data
        mock_trials_data = [
            {
                "trial_id": "NCT12345678",
                "protocolSection": {"identificationModule": {"nctId": "NCT12345678"}},
            }
        ]

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            json.dump({"successful_trials": mock_trials_data}, temp_file)
            temp_path = temp_file.name

        try:
            # Create mock args with async_queue enabled
            args = argparse.Namespace(
                list_prompts=False,
                list_models=False,
                trials_file=temp_path,
                cv_folds=3,
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
                async_queue=True,  # Enable async queue
                prompts=None,
                models=None,
                max_combinations=None,
                save_config=None,
                save_mcode_elements=None,
            )

            # Mock workflow and result
            mock_workflow = MagicMock()
            mock_result = MagicMock()
            mock_result.success = True
            mock_workflow.execute = AsyncMock(return_value=mock_result)
            mock_workflow_class.return_value = mock_workflow

            # Mock observer
            mock_observer = MagicMock()
            mock_observer_class.return_value = mock_observer

            # Call main
            main(args)

            # Verify workflow was executed (async_queue affects internal processing but not main flow)
            mock_workflow.execute.assert_called_once()
            call_kwargs = mock_workflow.execute.call_args[1]
            assert (
                call_kwargs["cli_args"] == args
            )  # CLI args are passed for internal configuration

        finally:
            # Clean up
            Path(temp_path).unlink()