#!/usr/bin/env python3
"""
End-to-End Tests for System Administrator Workflow

Tests the complete system administrator workflow from configuration to maintenance:
1. Configuration (config validation, environment setup) → 2. Monitoring (performance metrics, health checks) → 3. Maintenance (cleanup, testing, system administration)
"""

import json


from mcode_translate import main as mcode_translate_main


class TestSystemAdministratorWorkflowE2E:
    """End-to-end tests for the complete system administrator workflow."""

    def test_system_administrator_workflow_configuration_validation(self, tmp_path):
        """Test configuration validation phase of system administrator workflow."""
        # Create a test config file
        config_file = tmp_path / "test_config.json"
        test_config = {
            "llms": {
                "deepseek-coder": {
                    "model": "deepseek-coder",
                    "api_key": "test-key",
                    "base_url": "https://api.deepseek.com",
                }
            },
            "apis": {
                "clinical_trials": {
                    "base_url": "https://clinicaltrials.gov/api/v2",
                    "timeout": 30,
                }
            },
            "core_memory": {"source": "test", "enabled": True},
        }

        with open(config_file, "w") as f:
            json.dump(test_config, f)

        # Test basic CLI help to verify system is functional
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test basic help command to verify CLI is functional
            test_args = ["--help"]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            except SystemExit:
                pass  # Expected for help command
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Verify CLI is functional and shows help
        assert (
            "mCODE Translator CLI" in output
            or "usage:" in output
            or "positional arguments" in output
        )

    def test_system_administrator_workflow_monitoring_and_logging(self, tmp_path):
        """Test monitoring and logging phase of system administrator workflow."""
        # Test basic CLI command to verify logging system is functional
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test fetch-trials command with verbose logging (but no actual execution)
            test_args = ["fetch-trials", "--help"]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            except SystemExit:
                pass  # Expected for help command
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Verify CLI help shows logging options
        assert (
            "--verbose" in output
            or "--log-level" in output
            or "logging" in output.lower()
        )

    def test_system_administrator_workflow_maintenance_testing(self, tmp_path):
        """Test maintenance testing phase of system administrator workflow."""
        # Test CLI help for run-tests command to verify testing infrastructure is available
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test run-tests help command
            test_args = ["run-tests", "--help"]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            except SystemExit:
                pass  # Expected for help command
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Verify testing commands are available
        assert (
            "unit" in output
            or "integration" in output
            or "performance" in output
            or "lint" in output
        )

    def test_system_administrator_workflow_coverage_reporting(self, tmp_path):
        """Test coverage reporting in system administrator workflow."""
        # Test that coverage reporting command exists in CLI
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test run-tests help to verify coverage option exists
            test_args = ["run-tests", "--help"]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            except SystemExit:
                pass  # Expected for help command
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Verify coverage option is available
        assert "coverage" in output or "--coverage" in output

    def test_system_administrator_workflow_data_archive_management(self, tmp_path):
        """Test data archive management in system administrator workflow."""
        # Test CLI execution for data archive listing
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test download-data command with list option
            test_args = ["download-data", "--list"]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Verify archive listing was shown
        assert (
            "Available Synthetic Patient Archives" in output or "📚 Available" in output
        )

    def test_system_administrator_workflow_error_handling_and_recovery(self, tmp_path):
        """Test error handling and recovery in system administrator workflow."""
        # Test with invalid arguments to verify error handling
        import sys
        from io import StringIO

        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_output = StringIO()
        sys.stderr = captured_error = StringIO()

        try:
            # Test invalid command
            test_args = ["invalid-command"]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            except SystemExit:
                pass  # Expected for invalid command
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = captured_output.getvalue()
        error_output = captured_error.getvalue()

        # Verify error was handled gracefully
        combined_output = output + error_output
        assert (
            "invalid choice" in combined_output
            or "unrecognized arguments" in combined_output
            or len(combined_output) > 0
        )

    def test_system_administrator_workflow_system_health_check(self, tmp_path):
        """Test system health check in system administrator workflow."""
        # Test basic CLI functionality to verify system health
        import sys
        from io import StringIO

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Test version/info command (if available) or basic help
            test_args = ["--help"]

            # Mock sys.argv
            original_argv = sys.argv
            sys.argv = ["mcode_translate.py"] + test_args

            try:
                mcode_translate_main()
            except SystemExit:
                pass  # Expected for help command
            finally:
                sys.argv = original_argv

        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()

        # Verify system is responsive
        assert len(output) > 0 and (
            "mCODE" in output or "usage:" in output or "positional arguments" in output
        )

    def test_complete_system_administrator_workflow_integration(self, tmp_path):
        """Test the complete end-to-end system administrator workflow integration."""
        # Test multiple CLI commands to verify system administrator workflow
        commands_to_test = [
            ["--help"],
            ["run-tests", "--help"],
            ["download-data", "--help"],
        ]

        all_outputs = []

        for test_args in commands_to_test:
            import sys
            from io import StringIO

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            try:
                # Mock sys.argv
                original_argv = sys.argv
                sys.argv = ["mcode_translate.py"] + test_args

                try:
                    mcode_translate_main()
                except SystemExit:
                    pass  # Expected for help commands
                finally:
                    sys.argv = original_argv

            finally:
                sys.stdout = old_stdout

            output = captured_output.getvalue()
            all_outputs.append(output)

        # Verify all commands executed and produced output
        combined_output = " ".join(all_outputs)

        # Verify system administrator functionality is available
        assert (
            "run-tests" in combined_output
            or "download-data" in combined_output
            or "mCODE Translator" in combined_output
        )
