#!/usr/bin/env python3
"""
Main CLI entry point for the mCODE Translator.

This script provides both argparse-based and Click-based CLI interfaces.
The Click-based interface is recommended for testing.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the Click CLI for testing
from src.cli.click_cli import cli as click_cli

# Import argparse-based CLI components for backward compatibility
from src.cli.cli_parser import create_main_parser
from src.cli import (
    patients_fetcher,
    patients_processor,
    patients_summarizer,
    trials_fetcher,
    trials_optimizer,
    trials_processor,
    trials_summarizer,
)
from src.cli.data_downloader import (
    download_archives,
    list_available_archives,
    parse_archive_list,
    get_default_archives_config,
)
from src.cli.test_runner import (
    run_unit_tests,
    run_integration_tests,
    run_performance_tests,
    run_all_tests,
    run_coverage_report,
    run_linting,
)


def main():
    """Main entry point - uses argparse for backward compatibility."""
    parser = create_main_parser()
    args = parser.parse_args()

    # Execute the corresponding command's main function
    if args.command == "fetch-trials":
        # Validate arguments for fetch-trials
        if not any(
            [
                getattr(args, "condition", None),
                getattr(args, "nct_id", None),
                getattr(args, "nct_ids", None),
            ]
        ):
            parser.error("Must specify one of: --condition, --nct-id, or --nct-ids")
        trials_fetcher.main(args)
    elif args.command == "process-trials":
        if not hasattr(args, "input_file") or not args.input_file:
            parser.error("Must specify input file for process-trials")
        trials_processor.main(args)
    elif args.command == "summarize-trials":
        if not hasattr(args, "input_file") or not args.input_file:
            parser.error("Must specify input file for summarize-trials")
        trials_summarizer.main(args)
    elif args.command == "optimize-trials":
        trials_optimizer.main(args)
    elif args.command == "fetch-patients":
        patients_fetcher.main(args)
    elif args.command == "process-patients":
        if not hasattr(args, "input_file") or not args.input_file:
            parser.error("Must specify input file for process-patients")
        patients_processor.main(args)
    elif args.command == "summarize-patients":
        if not hasattr(args, "input_file") or not args.input_file:
            parser.error("Must specify input file for summarize-patients")
        patients_summarizer.main(args)
    elif args.command == "download-data":
        # Handle download-data command
        if getattr(args, "list", False):
            list_available_archives()
            return

        # Determine which archives to download
        if getattr(args, "all", False):
            archives_config = get_default_archives_config()
            archive_desc = "all available archives"
        else:
            archives_config = parse_archive_list(getattr(args, "archives", ""))
            archive_count = sum(
                len(durations) for durations in archives_config.values()
            )
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
            import time

            # Perform concurrent download
            start_time = time.time()

            downloaded_paths = download_archives(
                archives_config=archives_config,
                output_dir=getattr(args, "output_dir", "data/synthetic_patients"),
                workers=getattr(args, "workers", 4),
                force=getattr(args, "force", False),
            )

            end_time = time.time()
            duration = end_time - start_time

            # Report results
            successful = len(downloaded_paths)
            total_requested = sum(
                len(durations) for durations in archives_config.values()
            )

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
                print(
                    f"\n⚠️  {total_requested - successful} archives were skipped (already exist or failed)"
                )
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

        suite = getattr(args, "suite", "all")
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
    # Check if we should use Click CLI (for testing)
    if len(sys.argv) > 1 and sys.argv[1] == "--click":
        # Remove --click from args and run Click CLI
        sys.argv.pop(1)
        click_cli()
    else:
        main()
