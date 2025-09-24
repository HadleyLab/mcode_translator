"""
Test runner functions for mCODE Translator CLI.

This module provides functions to run various test suites
and generate coverage reports.
"""

import os
import subprocess
from typing import Optional, Tuple


def run_command(
    cmd: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None
) -> Tuple[bool, str, str]:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def run_unit_tests(args) -> bool:
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    cmd = "python -m pytest tests/unit/ -v --tb=short"
    if getattr(args, "fail_fast", False):
        cmd += " -x"
    if getattr(args, "coverage", False):
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"

    success, stdout, stderr = run_command(cmd)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    return success


def run_integration_tests(args) -> bool:
    """Run integration tests."""
    print("🔗 Running Integration Tests...")

    # Set environment variable for live tests if requested
    env = os.environ.copy()
    if getattr(args, "live", False):
        env["ENABLE_LIVE_TESTS"] = "true"
        print("⚠️  Running with LIVE data sources!")
    else:
        print("🔒 Running with MOCK data sources only")

    cmd = "python -m pytest tests/integration/ -v --tb=short"
    if getattr(args, "coverage", False):
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"

    success, stdout, stderr = run_command(cmd, env=env)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    return success


def run_performance_tests(args) -> bool:
    """Run performance tests."""
    print("⚡ Running Performance Tests...")
    cmd = "python -m pytest tests/performance/ -v --tb=short"
    if getattr(args, "benchmark", False):
        cmd += " --benchmark-only"

    success, stdout, stderr = run_command(cmd)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    return success


def run_all_tests(args) -> bool:
    """Run all test suites."""
    print("🚀 Running All Tests...")

    success = True

    # Run unit tests
    if not run_unit_tests(args):
        success = False
        if getattr(args, "fail_fast", False):
            return False

    # Run integration tests
    if not run_integration_tests(args):
        success = False
        if getattr(args, "fail_fast", False):
            return False

    # Run performance tests
    if not run_performance_tests(args):
        success = False

    return success


def run_coverage_report(args) -> bool:
    """Generate coverage report."""
    print("📊 Generating Coverage Report...")
    cmd = "python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90"

    success, stdout, stderr = run_command(cmd)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")

    if success:
        print("\n📈 Coverage report generated in htmlcov/index.html")

    return success


def run_linting(args) -> bool:
    """Run linting and formatting checks."""
    print("🔍 Running Linting and Formatting Checks...")

    commands = [
        "ruff check src/ tests/",
        "black --check src/ tests/",
        "mypy --strict src/",
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
