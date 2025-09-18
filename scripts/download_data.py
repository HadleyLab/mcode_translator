#!/usr/bin/env python3
"""
Data Downloader CLI - Concurrent Download of Synthetic Patient Archives

A command-line interface for downloading synthetic patient data archives
with concurrent processing for improved performance.

Usage:
    python scripts/download_data.py --archives breast_cancer_10_years,mixed_cancer_lifetime
    python scripts/download_data.py --all --workers 4
    python scripts/download_data.py --list
"""

import argparse
import sys
from pathlib import Path

from src.utils.data_downloader import (
    download_synthetic_patient_archives_concurrent,
    get_archive_paths,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for data downloader."""
    parser = argparse.ArgumentParser(
        description="Download synthetic patient data archives with concurrent processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific archives concurrently
  python scripts/download_data.py --archives breast_cancer_10_years,mixed_cancer_lifetime

  # Download all available archives with 4 concurrent workers
  python scripts/download_data.py --all --workers 4

  # List available archives
  python scripts/download_data.py --list

  # Force re-download existing archives
  python scripts/download_data.py --all --force

  # Download to custom directory
  python scripts/download_data.py --all --output-dir /path/to/custom/dir

The script supports concurrent downloading of multiple archives simultaneously
for improved performance compared to sequential downloads.
        """,
    )

    # Archive selection
    archive_group = parser.add_mutually_exclusive_group(required=True)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download workers (default: 4)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing archives"
    )

    parser.add_argument(
        "--output-dir",
        default="data/synthetic_patients",
        help="Output directory for downloaded archives (default: data/synthetic_patients)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser


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
            logger.warning(f"Archive '{archive_name}' not found in available archives")
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


def main() -> None:
    """Main entry point for data downloader CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle list command
    if args.list:
        list_available_archives()
        return

    # Determine which archives to download
    if args.all:
        archives_config = get_default_archives_config()
        archive_desc = "all available archives"
    else:
        archives_config = parse_archive_list(args.archives)
        archive_count = sum(len(durations) for durations in archives_config.values())
        archive_desc = f"{archive_count} specified archives"

    if not archives_config:
        print("❌ No valid archives specified for download")
        sys.exit(1)

    print("🚀 mCODE Data Downloader")
    print("=" * 30)
    print(f"📦 Downloading: {archive_desc}")
    print(f"🔄 Workers: {args.workers}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔄 Force: {'Yes' if args.force else 'No'}")
    print()

    try:
        # Perform concurrent download
        import time
        start_time = time.time()

        downloaded_paths = download_synthetic_patient_archives_concurrent(
            base_dir=args.output_dir,
            archives_config=archives_config,
            force_download=args.force,
            max_workers=args.workers
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
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()