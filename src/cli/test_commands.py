"""
Test execution CLI commands for mCODE Translator.

This module contains commands for running various test suites.

Migrated from Click to Typer with full type hints and heysol_api_client integration.
"""

from pathlib import Path
from types import SimpleNamespace
import typer

from ..config.heysol_config import get_config
from .test_runner import (run_all_tests, run_coverage_report,
                           run_integration_tests, run_linting,
                           run_performance_tests, run_unit_tests)

# Create the test Typer app
app = typer.Typer()


@app.command()
def run(
    suite: str = typer.Argument(..., help="Test suite to run"),
    live: bool = typer.Option(
        False, help="Run integration tests with live data sources"
    ),
    coverage: bool = typer.Option(False, help="Generate coverage reports"),
    benchmark: bool = typer.Option(
        False, help="Run only benchmark tests in performance suite"
    ),
    fail_fast: bool = typer.Option(False, help="Stop on first failure"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: str = typer.Option(None, help="Path to configuration file"),
) -> None:
    """Run tests."""
    # Check if we're in the right directory
    if not Path("src").exists():
        typer.echo(
            "❌ Error: Please run this script from the project root directory", err=True
        )
        raise typer.Exit(1)

    # Get global configuration
    get_config()

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
        typer.echo("✅ All tests passed!")
        raise typer.Exit(0)
    else:
        typer.echo("❌ Some tests failed!")
        raise typer.Exit(1)


# For backward compatibility, expose the app as 'test'
test = app
