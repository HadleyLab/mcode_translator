"""
Test execution CLI commands for mCODE Translator.

This module contains commands for running various test suites.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import click

from .test_runner import (
    run_all_tests,
    run_coverage_report,
    run_integration_tests,
    run_linting,
    run_performance_tests,
    run_unit_tests,
)


@click.group()
def test():
    """Test execution commands."""
    pass


@test.command()
@click.argument(
    "suite",
    type=click.Choice(
        ["unit", "integration", "performance", "all", "coverage", "lint"]
    ),
)
@click.option(
    "--live", is_flag=True, help="Run integration tests with live data sources"
)
@click.option("--coverage", is_flag=True, help="Generate coverage reports")
@click.option(
    "--benchmark", is_flag=True, help="Run only benchmark tests in performance suite"
)
@click.option("--fail-fast", is_flag=True, help="Stop on first failure")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
def run(suite, live, coverage, benchmark, fail_fast, log_level, config):
    """Run tests."""
    # Check if we're in the right directory
    if not Path("src").exists():
        click.echo("❌ Error: Please run this script from the project root directory")
        sys.exit(1)

    # Create args object for compatibility
    args = SimpleNamespace(
        live=live,
        coverage=coverage,
        benchmark=benchmark,
        fail_fast=fail_fast,
    )

    # Run the appropriate test suite
    success = False

    if suite == "unit":
        success = run_unit_tests(args)
    elif suite == "integration":
        success = run_integration_tests(args)
    elif suite == "performance":
        success = run_performance_tests(args)
    elif suite == "all":
        success = run_all_tests(args)
    elif suite == "coverage":
        success = run_coverage_report(args)
    elif suite == "lint":
        success = run_linting(args)

    # Exit with appropriate code
    if success:
        click.echo("✅ All tests passed!")
        sys.exit(0)
    else:
        click.echo("❌ Some tests failed!")
        sys.exit(1)
