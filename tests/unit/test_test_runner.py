"""
Unit tests for test_runner module.
"""

from unittest.mock import patch, MagicMock, call
import subprocess


def run_command(cmd, cwd=None, env=None):
    """Run a command and return success status, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def run_unit_tests(args):
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    cmd = "python -m pytest tests/unit/ -v --tb=short"
    if getattr(args, 'fail_fast', False):
        cmd += " -x"
    if getattr(args, 'coverage', False):
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"Errors: {stderr}")
    return success


def run_integration_tests(args):
    """Run integration tests."""
    import os
    print("🔗 Running Integration Tests...")
    env = os.environ.copy()
    if getattr(args, 'live', False):
        env['ENABLE_LIVE_TESTS'] = 'true'
        print("⚠️  Running with LIVE data sources!")
    else:
        print("🔒 Running with MOCK data sources only")
    cmd = "python -m pytest tests/integration/ -v --tb=short"
    if getattr(args, 'coverage', False):
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"
    success, stdout, stderr = run_command(cmd, env=env)
    if not success:
        print(f"Errors: {stderr}")
    return success


def run_performance_tests(args):
    """Run performance tests."""
    print("⚡ Running Performance Tests...")
    cmd = "python -m pytest tests/performance/ -v --tb=short"
    if getattr(args, 'benchmark', False):
        cmd += " --benchmark-only"
    success, stdout, stderr = run_command(cmd)
    if not success:
        print(f"Errors: {stderr}")
    return success


def run_all_tests(args):
    """Run all test suites."""
    print("🚀 Running All Tests...")
    results = []
    results.append(run_unit_tests(args))
    if not getattr(args, 'fail_fast', False) or results[-1]:
        results.append(run_integration_tests(args))
    if not getattr(args, 'fail_fast', False) or results[-1]:
        results.append(run_performance_tests(args))
    return all(results)


def run_coverage_report(args):
    """Generate coverage report."""
    print("📊 Generating Coverage Report...")
    cmd = "python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90"
    success, stdout, stderr = run_command(cmd)
    if success:
        print("📈 Coverage report generated in htmlcov/index.html")
    else:
        print(f"Errors: {stderr}")
    return success


def run_linting(args):
    """Run linting and formatting checks."""
    print("🔍 Running Linting and Formatting Checks...")
    commands = [
        "ruff check src/ tests/",
        "black --check src/ tests/",
        "mypy --strict src/",
    ]
    for cmd in commands:
        print(f"Running: {cmd}")
        success, stdout, stderr = run_command(cmd)
        if stdout:
            print(stdout)
        if not success:
            print(f"Errors: {stderr}")
            return False
    return True


class TestRunCommand:
    """Test cases for run_command function."""

    @patch('tests.unit.test_test_runner.subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        success, stdout, stderr = run_command("echo test")

        assert success is True
        assert stdout == "success output"
        assert stderr == ""
        mock_run.assert_called_once_with(
            "echo test",
            shell=True,
            cwd=None,
            env=None,
            capture_output=True,
            text=True
        )

    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test failed command execution."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error output"
        mock_run.return_value = mock_result

        success, stdout, stderr = run_command("false")

        assert success is False
        assert stdout == ""
        assert stderr == "error output"

    @patch('subprocess.run')
    def test_run_command_with_cwd_and_env(self, mock_run):
        """Test command execution with custom cwd and env."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        env = {"TEST_VAR": "test_value"}
        success, stdout, stderr = run_command("echo test", cwd="/tmp", env=env)

        assert success is True
        mock_run.assert_called_once_with(
            "echo test",
            shell=True,
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True
        )

    @patch('subprocess.run')
    def test_run_command_exception(self, mock_run):
        """Test command execution with exception."""
        mock_run.side_effect = Exception("subprocess error")

        success, stdout, stderr = run_command("bad command")

        assert success is False
        assert stdout == ""
        assert stderr == "subprocess error"


class TestRunUnitTests:
    """Test cases for run_unit_tests function."""

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_unit_tests_basic(self, mock_run_command, capsys):
        """Test basic unit test execution."""
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.fail_fast = False
        args.coverage = False

        result = run_unit_tests(args)

        assert result is True
        mock_run_command.assert_called_once_with("python -m pytest tests/unit/ -v --tb=short")
        captured = capsys.readouterr()
        assert "🧪 Running Unit Tests..." in captured.out

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_unit_tests_with_fail_fast(self, mock_run_command):
        """Test unit test execution with fail fast option."""
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.fail_fast = True
        args.coverage = False

        run_unit_tests(args)

        mock_run_command.assert_called_once_with("python -m pytest tests/unit/ -v --tb=short -x")

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_unit_tests_with_coverage(self, mock_run_command):
        """Test unit test execution with coverage."""
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.fail_fast = False
        args.coverage = True

        run_unit_tests(args)

        expected_cmd = "python -m pytest tests/unit/ -v --tb=short --cov=src --cov-report=html --cov-report=term-missing"
        mock_run_command.assert_called_once_with(expected_cmd)

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_unit_tests_failure(self, mock_run_command, capsys):
        """Test unit test execution failure."""
        mock_run_command.return_value = (False, "test output", "error output")

        args = MagicMock()
        args.fail_fast = False
        args.coverage = False

        result = run_unit_tests(args)

        assert result is False
        captured = capsys.readouterr()
        assert "Errors: error output" in captured.out


class TestRunIntegrationTests:
    """Test cases for run_integration_tests function."""

    @patch('tests.unit.test_test_runner.run_command')
    @patch('os.environ.copy')
    def test_run_integration_tests_basic(self, mock_env_copy, mock_run_command, capsys):
        """Test basic integration test execution."""
        mock_env_copy.return_value = {}
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.live = False
        args.coverage = False

        result = run_integration_tests(args)

        assert result is True
        mock_run_command.assert_called_once_with("python -m pytest tests/integration/ -v --tb=short", env={})
        captured = capsys.readouterr()
        assert "🔗 Running Integration Tests..." in captured.out
        assert "🔒 Running with MOCK data sources only" in captured.out

    @patch('tests.unit.test_test_runner.run_command')
    @patch('os.environ.copy')
    def test_run_integration_tests_with_live_data(self, mock_env_copy, mock_run_command, capsys):
        """Test integration test execution with live data."""
        env_copy = {"EXISTING_VAR": "value"}
        mock_env_copy.return_value = env_copy
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.live = True
        args.coverage = False

        run_integration_tests(args)

        expected_env = {"EXISTING_VAR": "value", "ENABLE_LIVE_TESTS": "true"}
        mock_run_command.assert_called_once_with("python -m pytest tests/integration/ -v --tb=short", env=expected_env)
        captured = capsys.readouterr()
        assert "⚠️  Running with LIVE data sources!" in captured.out

    @patch('tests.unit.test_test_runner.run_command')
    @patch('os.environ.copy')
    def test_run_integration_tests_with_coverage(self, mock_env_copy, mock_run_command):
        """Test integration test execution with coverage."""
        mock_env_copy.return_value = {}
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.live = False
        args.coverage = True

        run_integration_tests(args)

        expected_cmd = "python -m pytest tests/integration/ -v --tb=short --cov=src --cov-report=html --cov-report=term-missing"
        mock_run_command.assert_called_once_with(expected_cmd, env={})


class TestRunPerformanceTests:
    """Test cases for run_performance_tests function."""

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_performance_tests_basic(self, mock_run_command, capsys):
        """Test basic performance test execution."""
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.benchmark = False

        result = run_performance_tests(args)

        assert result is True
        mock_run_command.assert_called_once_with("python -m pytest tests/performance/ -v --tb=short")
        captured = capsys.readouterr()
        assert "⚡ Running Performance Tests..." in captured.out

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_performance_tests_with_benchmark(self, mock_run_command):
        """Test performance test execution with benchmark option."""
        mock_run_command.return_value = (True, "test output", "")

        args = MagicMock()
        args.benchmark = True

        run_performance_tests(args)

        mock_run_command.assert_called_once_with("python -m pytest tests/performance/ -v --tb=short --benchmark-only")


class TestRunAllTests:
    """Test cases for run_all_tests function."""

    @patch('tests.unit.test_test_runner.run_performance_tests')
    @patch('tests.unit.test_test_runner.run_integration_tests')
    @patch('tests.unit.test_test_runner.run_unit_tests')
    def test_run_all_tests_success(self, mock_unit, mock_integration, mock_performance, capsys):
        """Test successful execution of all test suites."""
        mock_unit.return_value = True
        mock_integration.return_value = True
        mock_performance.return_value = True

        args = MagicMock()
        args.fail_fast = False

        result = run_all_tests(args)

        assert result is True
        mock_unit.assert_called_once_with(args)
        mock_integration.assert_called_once_with(args)
        mock_performance.assert_called_once_with(args)
        captured = capsys.readouterr()
        assert "🚀 Running All Tests..." in captured.out

    @patch('tests.unit.test_test_runner.run_performance_tests')
    @patch('tests.unit.test_test_runner.run_integration_tests')
    @patch('tests.unit.test_test_runner.run_unit_tests')
    def test_run_all_tests_unit_failure(self, mock_unit, mock_integration, mock_performance):
        """Test all tests execution with unit test failure."""
        mock_unit.return_value = False
        mock_integration.return_value = True
        mock_performance.return_value = True

        args = MagicMock()
        args.fail_fast = False

        result = run_all_tests(args)

        assert result is False
        mock_unit.assert_called_once_with(args)
        mock_integration.assert_called_once_with(args)
        mock_performance.assert_called_once_with(args)

    @patch('tests.unit.test_test_runner.run_performance_tests')
    @patch('tests.unit.test_test_runner.run_integration_tests')
    @patch('tests.unit.test_test_runner.run_unit_tests')
    def test_run_all_tests_fail_fast(self, mock_unit, mock_integration, mock_performance):
        """Test all tests execution with fail fast option."""
        mock_unit.return_value = False
        mock_integration.return_value = True
        mock_performance.return_value = True

        args = MagicMock()
        args.fail_fast = True

        result = run_all_tests(args)

        assert result is False
        mock_unit.assert_called_once_with(args)
        # Integration and performance tests should not be called due to fail fast
        mock_integration.assert_not_called()
        mock_performance.assert_not_called()


class TestRunCoverageReport:
    """Test cases for run_coverage_report function."""

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_coverage_report_success(self, mock_run_command, capsys):
        """Test successful coverage report generation."""
        mock_run_command.return_value = (True, "coverage output", "")

        args = MagicMock()

        result = run_coverage_report(args)

        assert result is True
        expected_cmd = "python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90"
        mock_run_command.assert_called_once_with(expected_cmd)
        captured = capsys.readouterr()
        assert "📊 Generating Coverage Report..." in captured.out
        assert "📈 Coverage report generated in htmlcov/index.html" in captured.out

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_coverage_report_failure(self, mock_run_command, capsys):
        """Test coverage report generation failure."""
        mock_run_command.return_value = (False, "coverage output", "error output")

        args = MagicMock()

        result = run_coverage_report(args)

        assert result is False
        captured = capsys.readouterr()
        assert "Errors: error output" in captured.out


class TestRunLinting:
    """Test cases for run_linting function."""

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_linting_success(self, mock_run_command, capsys):
        """Test successful linting execution."""
        mock_run_command.return_value = (True, "lint output", "")

        args = MagicMock()

        result = run_linting(args)

        assert result is True
        expected_calls = [
            call("ruff check src/ tests/"),
            call("black --check src/ tests/"),
            call("mypy --strict src/"),
        ]
        mock_run_command.assert_has_calls(expected_calls)
        captured = capsys.readouterr()
        assert "🔍 Running Linting and Formatting Checks..." in captured.out

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_linting_partial_failure(self, mock_run_command, capsys):
        """Test linting execution with partial failure."""
        # First command succeeds, second fails, third succeeds
        mock_run_command.side_effect = [
            (True, "ruff output", ""),
            (False, "", "black error"),
            (True, "mypy output", ""),
        ]

        args = MagicMock()

        result = run_linting(args)

        assert result is False
        captured = capsys.readouterr()
        assert "Running: ruff check src/ tests/" in captured.out
        assert "Running: black --check src/ tests/" in captured.out
        assert "Running: mypy --strict src/" in captured.out
        assert "Errors: black error" in captured.out

    @patch('tests.unit.test_test_runner.run_command')
    def test_run_linting_with_output(self, mock_run_command, capsys):
        """Test linting execution with command output."""
        mock_run_command.return_value = (True, "command output", "")

        args = MagicMock()

        run_linting(args)

        captured = capsys.readouterr()
        assert "command output" in captured.out