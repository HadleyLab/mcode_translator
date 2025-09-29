"""
Patient-related CLI commands for mCODE Translator.

This module contains all commands related to patients:
- fetch_patients
- process_patients
- summarize_patients

Migrated from Click to Typer with full type hints and heysol_api_client integration.
"""

from types import SimpleNamespace
from typing import Optional

import typer

from ..config.heysol_config import get_config
from . import patients_fetcher, patients_processor, patients_summarizer

# Create the patients Typer app
app = typer.Typer()


@app.command()
def fetch(
    archive: Optional[str] = typer.Option(
        None, help="Patient archive identifier (e.g., breast_cancer_10_years)"
    ),
    patient_id: Optional[str] = typer.Option(None, help="Specific patient ID to fetch"),
    limit: int = typer.Option(
        10, help="Maximum number of patients to fetch (default: 10)"
    ),
    list_archives: bool = typer.Option(
        False, help="List available patient archives and exit"
    ),
    output_file: Optional[str] = typer.Option(
        None, help="Output file for patient data (NDJSON format)"
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
) -> None:
    """Fetch synthetic patients."""

    # Get global configuration
    get_config()

    args = SimpleNamespace(
        archive=archive,
        patient_id=patient_id,
        limit=limit,
        list_archives=list_archives,
        output_file=output_file,
        log_level=log_level,
        config=config_file,
    )
    patients_fetcher.main(args)  # type: ignore


@app.command()
def process(
    input_file: str = typer.Option(
        ..., "--in", help="Input file containing patient data"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--out", help="Output file for processed mCODE data"
    ),
    trials: Optional[str] = typer.Option(
        None, help="Path to NDJSON file containing trial data for eligibility filtering"
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
    store_in_memory: bool = typer.Option(False, help="Store results in CORE memory"),
) -> None:
    """Process patients to mCODE."""
    if not input_file:
        typer.echo("Error: Must specify input file with --in", err=True)
        raise typer.Exit(1)

    # Get global configuration
    get_config()

    args = SimpleNamespace(
        input_file=input_file,
        output_file=output_file,
        trials=trials,
        log_level=log_level,
        config=config_file,
        store_in_memory=store_in_memory,
    )
    patients_processor.main(args)  # type: ignore


@app.command()
def summarize(
    input_file: str = typer.Option(
        ..., "--in", help="Input file containing mCODE patient data"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--out", help="Output file for summarized data"
    ),
    dry_run: bool = typer.Option(
        False, help="Run summarization without storing results in CORE Memory"
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
    store_in_memory: bool = typer.Option(False, help="Store results in CORE memory"),
) -> None:
    """Summarize mCODE patients."""
    if not input_file:
        typer.echo("Error: Must specify input file with --in", err=True)
        raise typer.Exit(1)

    # Get global configuration
    get_config()

    args = SimpleNamespace(
        input_file=input_file,
        output_file=output_file,
        dry_run=dry_run,
        log_level=log_level,
        config=config_file,
        store_in_memory=store_in_memory,
    )
    patients_summarizer.main(args)  # type: ignore


# For backward compatibility, expose the app as 'patients'
patients = app
