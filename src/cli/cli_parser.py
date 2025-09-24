"""
CLI argument parser for mCODE Translator.

This module provides the main argument parser setup for all CLI commands.
"""

import argparse

from shared.cli_utils import McodeCLI


def setup_trials_fetcher_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up trials fetcher parser."""
    trials_fetcher_parser = subparsers.add_parser(
        "fetch-trials", help="Fetch clinical trials"
    )
    # Add arguments directly to subparser
    trials_fetcher_parser.add_argument(
        "--condition", help="Medical condition to search for (e.g., 'breast cancer')"
    )
    trials_fetcher_parser.add_argument(
        "--nct-id", help="Specific NCT ID to fetch (e.g., NCT12345678)"
    )
    trials_fetcher_parser.add_argument(
        "--nct-ids", help="Comma-separated list of NCT IDs to fetch"
    )
    trials_fetcher_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of trials to fetch (default: 10)",
    )
    trials_fetcher_parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for trial data (NDJSON format). If not specified, writes to stdout",
    )
    McodeCLI.add_core_args(trials_fetcher_parser)
    McodeCLI.add_concurrency_args(trials_fetcher_parser)


def setup_trials_processor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up trials processor parser."""
    trials_processor_parser = subparsers.add_parser(
        "process-trials", help="Process clinical trials to mCODE"
    )
    trials_processor_parser.add_argument(
        "input_file", help="Input file containing trial data"
    )
    trials_processor_parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file containing trial data (alternative to positional argument)",
    )
    trials_processor_parser.add_argument(
        "--out", dest="output_file", help="Output file for processed mCODE data"
    )
    McodeCLI.add_core_args(trials_processor_parser)
    McodeCLI.add_memory_args(trials_processor_parser)
    McodeCLI.add_processor_args(trials_processor_parser)


def setup_trials_summarizer_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up trials summarizer parser."""
    trials_summarizer_parser = subparsers.add_parser(
        "summarize-trials", help="Summarize mCODE trials"
    )
    trials_summarizer_parser.add_argument(
        "--in", dest="input_file", help="Input file containing mCODE trial data"
    )
    trials_summarizer_parser.add_argument(
        "--out", dest="output_file", help="Output file for summarized data"
    )
    McodeCLI.add_core_args(trials_summarizer_parser)
    McodeCLI.add_memory_args(trials_summarizer_parser)
    McodeCLI.add_processor_args(trials_summarizer_parser)


def setup_trials_optimizer_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up trials optimizer parser."""
    trials_optimizer_parser = subparsers.add_parser(
        "optimize-trials", help="Optimize trial processing parameters"
    )
    trials_optimizer_parser.add_argument(
        "--trials-file", help="Path to trials data file for optimization"
    )
    trials_optimizer_parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of cross-validation folds (default: 3)",
    )
    trials_optimizer_parser.add_argument(
        "--list-models", action="store_true", help="List available AI models and exit"
    )
    trials_optimizer_parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt templates and exit",
    )
    trials_optimizer_parser.add_argument(
        "--prompts", help="Comma-separated list of prompt templates to test"
    )
    trials_optimizer_parser.add_argument(
        "--models",
        type=lambda s: [item.strip() for item in s.split(",")],
        help="Comma-separated list of LLM models to test",
    )
    McodeCLI.add_core_args(trials_optimizer_parser)
    McodeCLI.add_optimizer_args(trials_optimizer_parser)


def setup_patients_fetcher_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up patients fetcher parser."""
    patients_fetcher_parser = subparsers.add_parser(
        "fetch-patients", help="Fetch synthetic patients"
    )
    patients_fetcher_parser.add_argument(
        "--archive", help="Patient archive identifier (e.g., breast_cancer_10_years)"
    )
    patients_fetcher_parser.add_argument(
        "--patient-id", help="Specific patient ID to fetch"
    )
    patients_fetcher_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of patients to fetch (default: 10)",
    )
    patients_fetcher_parser.add_argument(
        "--list-archives",
        action="store_true",
        help="List available patient archives and exit",
    )
    patients_fetcher_parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for patient data (NDJSON format). If not specified, writes to stdout",
    )
    McodeCLI.add_core_args(patients_fetcher_parser)


def setup_patients_processor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up patients processor parser."""
    patients_processor_parser = subparsers.add_parser(
        "process-patients", help="Process patients to mCODE"
    )
    patients_processor_parser.add_argument(
        "--in", dest="input_file", help="Input file containing patient data"
    )
    patients_processor_parser.add_argument(
        "--out", dest="output_file", help="Output file for processed mCODE data"
    )
    patients_processor_parser.add_argument(
        "--trials",
        help="Path to NDJSON file containing trial data for eligibility filtering",
    )
    McodeCLI.add_core_args(patients_processor_parser)
    McodeCLI.add_memory_args(patients_processor_parser)
    McodeCLI.add_processor_args(patients_processor_parser)


def setup_patients_summarizer_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up patients summarizer parser."""
    patients_summarizer_parser = subparsers.add_parser(
        "summarize-patients", help="Summarize mCODE patients"
    )
    patients_summarizer_parser.add_argument(
        "--in", dest="input_file", help="Input file containing mCODE patient data"
    )
    patients_summarizer_parser.add_argument(
        "--out", dest="output_file", help="Output file for summarized data"
    )
    patients_summarizer_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run summarization without storing results in CORE Memory",
    )
    McodeCLI.add_core_args(patients_summarizer_parser)
    McodeCLI.add_memory_args(patients_summarizer_parser)
    McodeCLI.add_processor_args(patients_summarizer_parser)


def setup_download_data_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up download data parser."""
    download_data_parser = subparsers.add_parser(
        "download-data", help="Download data archives"
    )
    # Archive selection
    archive_group = download_data_parser.add_mutually_exclusive_group(required=True)
    archive_group.add_argument(
        "--archives",
        help="Comma-separated list of archive names (e.g., breast_cancer_10_years,mixed_cancer_lifetime)",
    )
    archive_group.add_argument(
        "--all", action="store_true", help="Download all available archives"
    )
    archive_group.add_argument(
        "--list", action="store_true", help="List available archives and exit"
    )

    # Download options
    download_data_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (default: 4)",
    )

    download_data_parser.add_argument(
        "--force", action="store_true", help="Force re-download of existing archives"
    )

    download_data_parser.add_argument(
        "--output-dir",
        default="data/synthetic_patients",
        help="Output directory for downloaded archives (default: data/synthetic_patients)",
    )

    McodeCLI.add_core_args(download_data_parser)


def setup_run_tests_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up run tests parser."""
    run_tests_parser = subparsers.add_parser("run-tests", help="Run tests")
    run_tests_parser.add_argument(
        "suite",
        choices=["unit", "integration", "performance", "all", "coverage", "lint"],
        help="Test suite to run",
    )

    run_tests_parser.add_argument(
        "--live",
        action="store_true",
        help="Run integration tests with live data sources (requires ENABLE_LIVE_TESTS=true)",
    )

    run_tests_parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage reports"
    )

    run_tests_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run only benchmark tests in performance suite",
    )

    run_tests_parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )
    McodeCLI.add_core_args(run_tests_parser)


def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="mCODE Translator CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Set up all subparsers
    setup_trials_fetcher_parser(subparsers)
    setup_trials_processor_parser(subparsers)
    setup_trials_summarizer_parser(subparsers)
    setup_trials_optimizer_parser(subparsers)
    setup_patients_fetcher_parser(subparsers)
    setup_patients_processor_parser(subparsers)
    setup_patients_summarizer_parser(subparsers)
    setup_download_data_parser(subparsers)
    setup_run_tests_parser(subparsers)

    return parser
