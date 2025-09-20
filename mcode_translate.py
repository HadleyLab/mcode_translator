#!/usr/bin/env python3
"""
Main CLI entry point for the mCODE Translator.

This script unifies all the different CLI tools into a single interface.
"""

import argparse
import sys

sys.path.append('src')

from shared.cli_utils import McodeCLI
from cli import (
    patients_fetcher,
    patients_processor,
    patients_summarizer,
    trials_fetcher,
    trials_optimizer,
    trials_processor,
    trials_summarizer,
)
# Import script functionality directly
from src.utils.data_downloader import (
    download_synthetic_patient_archives_concurrent,
    get_archive_paths,
)
from pathlib import Path
import time
import os
import subprocess


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


def run_unit_tests(args):
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    cmd = f"python -m pytest tests/unit/ -v --tb=short"
    if getattr(args, 'coverage', False):
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"
    if getattr(args, 'fail_fast', False):
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
    if getattr(args, 'live', False):
        env["ENABLE_LIVE_TESTS"] = "true"
        print("⚠️  Running with LIVE data sources!")
    else:
        print("🔒 Running with MOCK data sources only")

    cmd = f"python -m pytest tests/integration/ -v --tb=short"
    if getattr(args, 'coverage', False):
        cmd += " --cov=src --cov-report=html --cov-report=term-missing"

    success, stdout, stderr = run_command(cmd, env=env)
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    return success


def run_performance_tests(args):
    """Run performance tests."""
    print("⚡ Running Performance Tests...")
    cmd = f"python -m pytest tests/performance/ -v --tb=short"
    if getattr(args, 'benchmark', False):
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
        if getattr(args, 'fail_fast', False):
            return False

    # Run integration tests
    if not run_integration_tests(args):
        success = False
        if getattr(args, 'fail_fast', False):
            return False

    # Run performance tests
    if not run_performance_tests(args):
        success = False

    return success


def run_coverage_report(args):
    """Generate coverage report."""
    print("📊 Generating Coverage Report...")
    cmd = f"python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=90"

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
        "ruff check src/ tests/",
        "black --check src/ tests/",
        "mypy --strict src/"
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


def get_default_archives_config():
    """Get the default archives configuration."""
    return {
        "mixed_cancer": {
            "10_years": "https://mitre.box.com/shared/static/7k7lk7wmza4m17916xnvc2uszidyv6vm.zip",
            "lifetime": "https://mitre.box.com/shared/static/mn6kpk56zvvk2o0lvjv55n7rnyajbnm4.zip",
        },
        "breast_cancer": {
            "10_years": "https://mitre.box.com/shared/static/c6ca6y2jfumrhw4nu20kztktxdlhhzo8.zip",
            "lifetime": "https://mitre.box.com/shared/static/59n7mcm8si0qk3p36ud0vmrcdv7pr0s7.zip",
        },
    }


def parse_archive_list(archive_str: str) -> dict:
    """Parse comma-separated archive list into config format."""
    archives_config = get_default_archives_config()
    requested_archives = [a.strip() for a in archive_str.split(",")]

    filtered_config = {}

    for archive_name in requested_archives:
        # Try to match archive name to config
        found = False
        for cancer_type, durations in archives_config.items():
            for duration, url in durations.items():
                config_name = f"{cancer_type}_{duration}"
                if archive_name.lower() == config_name.lower():
                    if cancer_type not in filtered_config:
                        filtered_config[cancer_type] = {}
                    filtered_config[cancer_type][duration] = url
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"⚠️  Archive '{archive_name}' not found in available archives")
            print("   Use --list to see available archives")

    return filtered_config


def list_available_archives():
    """List all available archives."""
    archives_config = get_default_archives_config()
    existing_archives = get_archive_paths()

    print("📚 Available Synthetic Patient Archives:")
    print("=" * 50)

    for cancer_type, durations in archives_config.items():
        print(f"\n🧬 {cancer_type.replace('_', ' ').title()}:")
        for duration, url in durations.items():
            archive_name = f"{cancer_type}_{duration}.zip"
            status = "✅ Downloaded" if archive_name in existing_archives else "⬇️  Available"
            size_info = ""
            if archive_name in existing_archives:
                path = existing_archives[archive_name]
                if Path(path).exists():
                    size = Path(path).stat().st_size
                    size_info = f" ({size / (1024*1024):.1f} MB)"

            print(f"   • {archive_name}{size_info} - {status}")

    print(f"\n📊 Total archives: {sum(len(durations) for durations in archives_config.values())}")
    print(f"📦 Downloaded: {len(existing_archives)}")


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

    trials_processor_parser = subparsers.add_parser(
        "process-trials", help="Process clinical trials to mCODE"
    )
    trials_processor_parser.add_argument(
        "input_file", help="Input file containing trial data"
    )
    trials_processor_parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file containing trial data (alternative to positional argument)"
    )
    trials_processor_parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for processed mCODE data"
    )
    McodeCLI.add_core_args(trials_processor_parser)
    McodeCLI.add_memory_args(trials_processor_parser)
    McodeCLI.add_processor_args(trials_processor_parser)

    trials_summarizer_parser = subparsers.add_parser(
        "summarize-trials", help="Summarize mCODE trials"
    )
    trials_summarizer_parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file containing mCODE trial data"
    )
    trials_summarizer_parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for summarized data"
    )
    McodeCLI.add_core_args(trials_summarizer_parser)
    McodeCLI.add_memory_args(trials_summarizer_parser)
    McodeCLI.add_processor_args(trials_summarizer_parser)

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
        help="Number of cross-validation folds (default: 3)"
    )
    trials_optimizer_parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available AI models and exit"
    )
    trials_optimizer_parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt templates and exit"
    )
    McodeCLI.add_core_args(trials_optimizer_parser)
    McodeCLI.add_optimizer_args(trials_optimizer_parser)

    # Patients
    patients_fetcher_parser = subparsers.add_parser(
        "fetch-patients", help="Fetch synthetic patients"
    )
    patients_fetcher_parser.add_argument(
        "--archive", help="Patient archive identifier (e.g., breast_cancer_10_years)"
    )
    patients_fetcher_parser.add_argument("--patient-id", help="Specific patient ID to fetch")
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

    patients_processor_parser = subparsers.add_parser(
        "process-patients", help="Process patients to mCODE"
    )
    patients_processor_parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file containing patient data"
    )
    patients_processor_parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for processed mCODE data"
    )
    McodeCLI.add_core_args(patients_processor_parser)
    McodeCLI.add_processor_args(patients_processor_parser)

    patients_summarizer_parser = subparsers.add_parser(
        "summarize-patients", help="Summarize mCODE patients"
    )
    patients_summarizer_parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file containing mCODE patient data"
    )
    patients_summarizer_parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for summarized data"
    )
    McodeCLI.add_core_args(patients_summarizer_parser)
    McodeCLI.add_memory_args(patients_summarizer_parser)

    # Data
    download_data_parser = subparsers.add_parser(
        "download-data", help="Download data archives"
    )
    # Archive selection
    archive_group = download_data_parser.add_mutually_exclusive_group(required=True)
    archive_group.add_argument(
        "--archives",
        help="Comma-separated list of archive names (e.g., breast_cancer_10_years,mixed_cancer_lifetime)"
    )
    archive_group.add_argument(
        "--all",
        action="store_true",
        help="Download all available archives"
    )
    archive_group.add_argument(
        "--list",
        action="store_true",
        help="List available archives and exit"
    )

    # Download options
    download_data_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (default: 4)"
    )

    download_data_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing archives"
    )

    download_data_parser.add_argument(
        "--output-dir",
        default="data/synthetic_patients",
        help="Output directory for downloaded archives (default: data/synthetic_patients)"
    )

    McodeCLI.add_core_args(download_data_parser)

    # Tests
    run_tests_parser = subparsers.add_parser("run-tests", help="Run tests")
    run_tests_parser.add_argument(
        "suite",
        choices=["unit", "integration", "performance", "all", "coverage", "lint"],
        help="Test suite to run"
    )

    run_tests_parser.add_argument(
        "--live",
        action="store_true",
        help="Run integration tests with live data sources (requires ENABLE_LIVE_TESTS=true)"
    )

    run_tests_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage reports"
    )

    run_tests_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run only benchmark tests in performance suite"
    )

    run_tests_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    McodeCLI.add_core_args(run_tests_parser)

    args = parser.parse_args()

    # Execute the corresponding command's main function
    if args.command == "fetch-trials":
        # Validate arguments for fetch-trials
        if not any([getattr(args, 'condition', None), getattr(args, 'nct_id', None), getattr(args, 'nct_ids', None)]):
            parser.error("Must specify one of: --condition, --nct-id, or --nct-ids")
        trials_fetcher.main(args)
    elif args.command == "process-trials":
        if not hasattr(args, 'input_file') or not args.input_file:
            parser.error("Must specify input file for process-trials")
        trials_processor.main(args)
    elif args.command == "summarize-trials":
        if not hasattr(args, 'input_file') or not args.input_file:
            parser.error("Must specify input file for summarize-trials")
        trials_summarizer.main(args)
    elif args.command == "optimize-trials":
        trials_optimizer.main(args)
    elif args.command == "fetch-patients":
        patients_fetcher.main(args)
    elif args.command == "process-patients":
        if not hasattr(args, 'input_file') or not args.input_file:
            parser.error("Must specify input file for process-patients")
        patients_processor.main(args)
    elif args.command == "summarize-patients":
        if not hasattr(args, 'input_file') or not args.input_file:
            parser.error("Must specify input file for summarize-patients")
        patients_summarizer.main(args)
    elif args.command == "download-data":
        # Handle download-data command
        if getattr(args, 'list', False):
            list_available_archives()
            return

        # Determine which archives to download
        if getattr(args, 'all', False):
            archives_config = get_default_archives_config()
            archive_desc = "all available archives"
        else:
            archives_config = parse_archive_list(getattr(args, 'archives', ''))
            archive_count = sum(len(durations) for durations in archives_config.values())
            archive_desc = f"{archive_count} specified archives"

        if not archives_config:
            print("❌ No valid archives specified for download")
            sys.exit(1)

        print("🚀 mCODE Data Downloader")
        print("=" * 30)
        print(f"📦 Downloading: {archive_desc}")
        print(f"🔄 Workers: {getattr(args, 'workers', 4)}")
        print(f"💾 Output: {getattr(args, 'output_dir', 'data/synthetic_patients')}")
        print(f"🔄 Force: {'Yes' if getattr(args, 'force', False) else 'No'}")
        print()

        try:
            # Perform concurrent download
            start_time = time.time()

            downloaded_paths = download_synthetic_patient_archives_concurrent(
                base_dir=getattr(args, 'output_dir', 'data/synthetic_patients'),
                archives_config=archives_config,
                force_download=getattr(args, 'force', False),
                max_workers=getattr(args, 'workers', 4)
            )

            end_time = time.time()
            duration = end_time - start_time

            # Report results
            successful = len(downloaded_paths)
            total_requested = sum(len(durations) for durations in archives_config.values())

            print("\n" + "=" * 50)
            print("✅ Download Summary:")
            print(f"   📊 Archives requested: {total_requested}")
            print(f"   ✅ Successfully processed: {successful}")
            print(f"   ⏱️  Total time: {duration:.2f} seconds")
            if total_requested > 0:
                avg_time_per_archive = duration / total_requested
                print(f"   📈 Avg time per archive: {avg_time_per_archive:.2f} seconds")

            print("\n📁 Downloaded Archives:")
            for archive_name, path in downloaded_paths.items():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"   ✅ {archive_name}: {size_mb:.1f} MB")

            if successful < total_requested:
                print(f"\n⚠️  {total_requested - successful} archives were skipped (already exist or failed)")
                print("   Use --force to re-download existing archives")

        except KeyboardInterrupt:
            print("\n⏹️  Download cancelled by user")
            sys.exit(130)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            sys.exit(1)
    elif args.command == "run-tests":
        # Check if we're in the right directory
        if not Path("src").exists():
            print("❌ Error: Please run this script from the project root directory")
            sys.exit(1)

        # Run the appropriate test suite
        success = False

        suite = getattr(args, 'suite', 'all')
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
            print("✅ All tests passed!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()