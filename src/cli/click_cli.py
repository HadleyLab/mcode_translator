"""
Click-based CLI for mCODE Translator.

This module provides a Click-based command-line interface for testing purposes.
Click provides better testing utilities compared to argparse.
"""

import click

from .data_commands import data
from .patients_commands import patients
from .test_commands import test

# Import command groups
from .trials_commands import trials


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
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
