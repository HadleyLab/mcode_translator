"""
Tests for Click-based CLI.

Uses Click's CliRunner for testing CLI commands.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch

from src.cli.click_cli import cli


class TestClickCLI:
    """Test the Click-based CLI interface."""

    @pytest.fixture
    def runner(self):
        """Create a Click test runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that CLI shows help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "mCODE Translator CLI" in result.output
        assert "fetch-trials" in result.output
        assert "process-trials" in result.output

    def test_fetch_trials_missing_required_args(self, runner):
        """Test fetch-trials command with missing required arguments."""
        result = runner.invoke(cli, ["fetch-trials"])
        assert result.exit_code == 2  # Click usage error
        assert "Must specify one of" in result.output

    @patch("src.cli.trials_fetcher.main")
    def test_fetch_trials_with_condition(self, mock_main, runner):
        """Test fetch-trials command with condition."""
        result = runner.invoke(cli, ["fetch-trials", "--condition", "breast cancer"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
        args = mock_main.call_args[0][0]
        assert args.condition == "breast cancer"

    @patch("src.cli.trials_fetcher.main")
    def test_fetch_trials_with_nct_id(self, mock_main, runner):
        """Test fetch-trials command with NCT ID."""
        result = runner.invoke(cli, ["fetch-trials", "--nct-id", "NCT12345678"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
        args = mock_main.call_args[0][0]
        assert args.nct_id == "NCT12345678"

    @patch("src.cli.trials_processor.main")
    def test_process_trials(self, mock_main, runner):
        """Test process-trials command."""
        result = runner.invoke(cli, ["process-trials", "input.json"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
        args = mock_main.call_args[0][0]
        assert args.input_file == "input.json"

    def test_process_trials_missing_input(self, runner):
        """Test process-trials command with missing input file."""
        result = runner.invoke(cli, ["process-trials"])
        assert result.exit_code == 2
        assert "Missing argument" in result.output

    @patch("src.cli.trials_summarizer.main")
    def test_summarize_trials(self, mock_main, runner):
        """Test summarize-trials command."""
        result = runner.invoke(cli, ["summarize-trials", "--in", "input.json"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
        args = mock_main.call_args[0][0]
        assert args.input_file == "input.json"

    def test_summarize_trials_missing_input(self, runner):
        """Test summarize-trials command with missing input file."""
        result = runner.invoke(cli, ["summarize-trials"])
        assert result.exit_code == 2
        assert "Must specify input file" in result.output

    @patch("src.cli.patients_fetcher.main")
    def test_fetch_patients(self, mock_main, runner):
        """Test fetch-patients command."""
        result = runner.invoke(
            cli, ["fetch-patients", "--archive", "breast_cancer_10_years"]
        )
        assert result.exit_code == 0
        mock_main.assert_called_once()
        args = mock_main.call_args[0][0]
        assert args.archive == "breast_cancer_10_years"

    @patch("src.cli.patients_processor.main")
    def test_process_patients(self, mock_main, runner):
        """Test process-patients command."""
        result = runner.invoke(cli, ["process-patients", "--in", "patients.json"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
        args = mock_main.call_args[0][0]
        assert args.input_file == "patients.json"

    def test_process_patients_missing_input(self, runner):
        """Test process-patients command with missing input file."""
        result = runner.invoke(cli, ["process-patients"])
        assert result.exit_code == 2
        assert "Must specify input file" in result.output

    @patch("src.cli.patients_summarizer.main")
    def test_summarize_patients(self, mock_main, runner):
        """Test summarize-patients command."""
        result = runner.invoke(cli, ["summarize-patients", "--in", "patients.json"])
        assert result.exit_code == 0
        mock_main.assert_called_once()
        args = mock_main.call_args[0][0]
        assert args.input_file == "patients.json"

    def test_summarize_patients_missing_input(self, runner):
        """Test summarize-patients command with missing input file."""
        result = runner.invoke(cli, ["summarize-patients"])
        assert result.exit_code == 2
        assert "Must specify input file" in result.output

    @patch("src.cli.data_downloader.list_available_archives")
    def test_download_data_list(self, mock_list, runner):
        """Test download-data command with --list option."""
        result = runner.invoke(cli, ["download-data", "--list"])
        assert result.exit_code == 0
        mock_list.assert_called_once()

    def test_download_data_missing_args(self, runner):
        """Test download-data command with missing required arguments."""
        result = runner.invoke(cli, ["download-data"])
        assert result.exit_code == 2
        assert "Must specify --archives or --all" in result.output

    @patch("src.cli.data_downloader.download_archives")
    @patch("src.cli.data_downloader.get_default_archives_config")
    def test_download_data_all(self, mock_config, mock_download, runner):
        """Test download-data command with --all option."""
        mock_config.return_value = {"test": {"archive": "url"}}
        mock_download.return_value = {"test_archive": "/path/to/archive"}

        result = runner.invoke(cli, ["download-data", "--all"])
        assert result.exit_code == 0
        mock_download.assert_called_once()

    @patch("src.cli.test_runner.run_unit_tests")
    def test_run_tests_unit(self, mock_run, runner):
        """Test run-tests command with unit suite."""
        mock_run.return_value = True
        result = runner.invoke(cli, ["run-tests", "unit"])
        assert result.exit_code == 0
        assert "All tests passed!" in result.output
        mock_run.assert_called_once()

    @patch("src.cli.test_runner.run_unit_tests")
    def test_run_tests_unit_failure(self, mock_run, runner):
        """Test run-tests command with unit suite failure."""
        mock_run.return_value = False
        result = runner.invoke(cli, ["run-tests", "unit"])
        assert result.exit_code == 1
        assert "Some tests failed!" in result.output

    def test_run_tests_invalid_suite(self, runner):
        """Test run-tests command with invalid suite."""
        result = runner.invoke(cli, ["run-tests", "invalid"])
        assert result.exit_code == 2
        assert "Invalid value for" in result.output

    def test_run_tests_missing_src_directory(self, runner, tmp_path):
        """Test run-tests command when not in project root."""
        # Change to a temporary directory without src/
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["run-tests", "unit"])
            assert result.exit_code == 1
            assert "project root directory" in result.output

    def test_cli_verbose_flag(self, runner):
        """Test that verbose flag is accepted."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0
        # The help should still work with verbose flag
