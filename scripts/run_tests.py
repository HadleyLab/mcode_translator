#!/usr/bin/env python3
"""
Test runner script for mcode_translator project.

This script provides convenient commands to run different test suites
with proper environment setup and configuration.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None, env=None):
    """Run a command and return the result."""
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


def activate_conda_env():
    """Get the command to activate conda environment."""
    return "source activate mcode_translator && "


def run_unit_tests(args):
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    cmd = f"{activate_conda_env()}python -m pytest tests/unit/ -v --tb=short"
    if args.coverage:
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"
    if args.fail_fast:
        cmd += " --exitfirst"

    success, stdout, stderr = run_command(cmd)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    return success


def run_integration_tests(args):
    """Run integration tests."""
    print("🔗 Running Integration Tests...")

    # Set environment variable for live tests if requested
    env = os.environ.copy()
    if args.live:
        env["ENABLE_LIVE_TESTS"] = "true"
        print("⚠️  Running with LIVE data sources!")
    else:
        print("🔒 Running with MOCK data sources only")

    cmd = f"{activate_conda_env()}python -m pytest tests/integration/ -v --tb=short"
    if args.coverage:
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"

    success, stdout, stderr = run_command(cmd, env=env)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    return success


def run_performance_tests(args):
    """Run performance tests."""
    print("⚡ Running Performance Tests...")
    cmd = f"{activate_conda_env()}python -m pytest tests/performance/ -v --tb=short"
    if args.benchmark:
        cmd += " --benchmark-only"

    success, stdout, stderr = run_command(cmd)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    return success


def run_all_tests(args):
    """Run all test suites."""
    print("🚀 Running All Tests...")

    success = True

    # Run unit tests
    if not run_unit_tests(args):
        success = False
        if args.fail_fast:
            return False

    # Run integration tests
    if not run_integration_tests(args):
        success = False
        if args.fail_fast:
            return False

    # Run performance tests
    if not run_performance_tests(args):
        success = False

    return success


def run_coverage_report(args):
    """Generate coverage report."""
    print("📊 Generating Coverage Report...")
    cmd = f"{activate_conda_env()}python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90"

    success, stdout, stderr = run_command(cmd)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")

    if success:
        print("\n📈 Coverage report generated in htmlcov/index.html")

    return success


def run_linting(args):
    """Run linting and formatting checks."""
    print("🔍 Running Linting and Formatting Checks...")

    commands = [
        f"{activate_conda_env()}ruff check src/ tests/",
        f"{activate_conda_env()}black --check src/ tests/",
        f"{activate_conda_env()}mypy --strict src/"
    ]

    success = True
    for cmd in commands:
        print(f"Running: {cmd}")
        cmd_success, stdout, stderr = run_command(cmd)
        if stdout:
            print(stdout)
        if stderr:
            print(f"Errors: {stderr}")
        if not cmd_success:
            success = False

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for mcode_translator project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py unit                    # Run unit tests
  python scripts/run_tests.py integration --live      # Run integration tests with live data
  python scripts/run_tests.py performance             # Run performance tests
  python scripts/run_tests.py all --coverage          # Run all tests with coverage
  python scripts/run_tests.py coverage                # Generate coverage report
  python scripts/run_tests.py lint                    # Run linting checks
        """
    )

    parser.add_argument(
        "suite",
        choices=["unit", "integration", "performance", "all", "coverage", "lint"],
        help="Test suite to run"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run integration tests with live data sources (requires ENABLE_LIVE_TESTS=true)"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage reports"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run only benchmark tests in performance suite"
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Run the appropriate test suite
    success = False

    if args.suite == "unit":
        success = run_unit_tests(args)
    elif args.suite == "integration":
        success = run_integration_tests(args)
    elif args.suite == "performance":
        success = run_performance_tests(args)
    elif args.suite == "all":
        success = run_all_tests(args)
    elif args.suite == "coverage":
        success = run_coverage_report(args)
    elif args.suite == "lint":
        success = run_linting(args)

    # Exit with appropriate code
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()