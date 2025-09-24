"""
Click-based CLI for mCODE Translator.

This module provides a Click-based command-line interface for testing purposes.
Click provides better testing utilities compared to argparse.
"""

import click

# Import command groups
from .trials_commands import trials
from .patients_commands import patients
from .data_commands import data
from .test_commands import test


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, verbose):
    """mCODE Translator CLI - Click-based interface for testing."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# Add command groups
cli.add_command(trials)
cli.add_command(patients)
cli.add_command(data)
cli.add_command(test)


if __name__ == "__main__":
    cli()
