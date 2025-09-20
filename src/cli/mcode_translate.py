#!/usr/bin/env python3
"""
Main CLI entry point for the mCODE Translator.

This script unifies all the different CLI tools into a single interface.
"""

import argparse
import sys

from src.shared.cli_utils import McodeCLI
from src.cli import (
    patients_fetcher,
    patients_processor,
    patients_summarizer,
    trials_fetcher,
    trials_optimizer,
    trials_processor,
    trials_summarizer,
)
from scripts import download_data, run_tests


def main():
    """Main entry point for the unified CLI."""
    parser = argparse.ArgumentParser(
        description="mCODE Translator CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subparsers for each command
    # Trials
    trials_fetcher_parser = subparsers.add_parser(
        "fetch-trials", help="Fetch clinical trials"
    )
    trials_fetcher.create_parser(trials_fetcher_parser)

    trials_processor_parser = subparsers.add_parser(
        "process-trials", help="Process clinical trials to mCODE"
    )
    trials_processor.create_parser(trials_processor_parser)

    trials_summarizer_parser = subparsers.add_parser(
        "summarize-trials", help="Summarize mCODE trials"
    )
    trials_summarizer.create_parser(trials_summarizer_parser)

    trials_optimizer_parser = subparsers.add_parser(
        "optimize-trials", help="Optimize trial processing parameters"
    )
    trials_optimizer.create_parser(trials_optimizer_parser)

    # Patients
    patients_fetcher_parser = subparsers.add_parser(
        "fetch-patients", help="Fetch synthetic patients"
    )
    patients_fetcher.create_parser(patients_fetcher_parser)

    patients_processor_parser = subparsers.add_parser(
        "process-patients", help="Process patients to mCODE"
    )
    patients_processor.create_parser(patients_processor_parser)

    patients_summarizer_parser = subparsers.add_parser(
        "summarize-patients", help="Summarize mCODE patients"
    )
    patients_summarizer.create_parser(patients_summarizer_parser)

    # Data
    download_data_parser = subparsers.add_parser(
        "download-data", help="Download data archives"
    )
    download_data.create_parser(download_data_parser)

    # Tests
    run_tests_parser = subparsers.add_parser("run-tests", help="Run tests")
    run_tests.create_parser(run_tests_parser)

    args = parser.parse_args()

    # Execute the corresponding command's main function
    if args.command == "fetch-trials":
        trials_fetcher.main(args)
    elif args.command == "process-trials":
        trials_processor.main(args)
    elif args.command == "summarize-trials":
        trials_summarizer.main(args)
    elif args.command == "optimize-trials":
        trials_optimizer.main(args)
    elif args.command == "fetch-patients":
        patients_fetcher.main(args)
    elif args.command == "process-patients":
        patients_processor.main(args)
    elif args.command == "summarize-patients":
        patients_summarizer.main(args)
    elif args.command == "download-data":
        download_data.main(args)
    elif args.command == "run-tests":
        run_tests.main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()