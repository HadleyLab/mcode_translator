"""
Patient-related CLI commands for mCODE Translator.

This module contains all Click commands related to patients:
- fetch_patients
- process_patients
- summarize_patients
"""

import click
from types import SimpleNamespace

from . import patients_fetcher, patients_processor, patients_summarizer


@click.group()
def patients():
    """Patient data commands."""
    pass


@patients.command()
@click.option(
    "--archive", help="Patient archive identifier (e.g., breast_cancer_10_years)"
)
@click.option("--patient-id", help="Specific patient ID to fetch")
@click.option(
    "--limit",
    type=int,
    default=10,
    help="Maximum number of patients to fetch (default: 10)",
)
@click.option(
    "--list-archives", is_flag=True, help="List available patient archives and exit"
)
@click.option(
    "--out", "output_file", help="Output file for patient data (NDJSON format)"
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
def fetch(archive, patient_id, limit, list_archives, output_file, log_level, config):
    """Fetch synthetic patients."""

    args = SimpleNamespace(
        archive=archive,
        patient_id=patient_id,
        limit=limit,
        list_archives=list_archives,
        output_file=output_file,
        log_level=log_level,
        config=config,
    )
    patients_fetcher.main(args)


@patients.command()
@click.option("--in", "input_file", help="Input file containing patient data")
@click.option("--out", "output_file", help="Output file for processed mCODE data")
@click.option(
    "--trials",
    help="Path to NDJSON file containing trial data for eligibility filtering",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.option(
    "--store-in-memory/--no-store-in-memory",
    default=False,
    help="Store results in CORE memory",
)
def process(input_file, output_file, trials, log_level, config, store_in_memory):
    """Process patients to mCODE."""
    if not input_file:
        raise click.UsageError("Must specify input file with --in")

    args = SimpleNamespace(
        input_file=input_file,
        output_file=output_file,
        trials=trials,
        log_level=log_level,
        config=config,
        store_in_memory=store_in_memory,
    )
    patients_processor.main(args)


@patients.command()
@click.option("--in", "input_file", help="Input file containing mCODE patient data")
@click.option("--out", "output_file", help="Output file for summarized data")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run summarization without storing results in CORE Memory",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
@click.option(
    "--store-in-memory/--no-store-in-memory",
    default=False,
    help="Store results in CORE memory",
)
def summarize(input_file, output_file, dry_run, log_level, config, store_in_memory):
    """Summarize mCODE patients."""
    if not input_file:
        raise click.UsageError("Must specify input file with --in")

    args = SimpleNamespace(
        input_file=input_file,
        output_file=output_file,
        dry_run=dry_run,
        log_level=log_level,
        config=config,
        store_in_memory=store_in_memory,
    )
    patients_summarizer.main(args)