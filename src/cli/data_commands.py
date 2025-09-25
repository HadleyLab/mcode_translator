"""
Data management CLI commands for mCODE Translator.

This module contains commands for downloading and managing data archives.
"""

import sys
import time
from pathlib import Path

import click

from .data_downloader import (
    download_archives,
    get_default_archives_config,
    list_available_archives,
    parse_archive_list,
)


@click.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.option(
    "--archives",
    help="Comma-separated list of archive names (e.g., breast_cancer_10_years,mixed_cancer_lifetime)",
)
@click.option(
    "--all", "download_all", is_flag=True, help="Download all available archives"
)
@click.option(
    "--list", "list_archives", is_flag=True, help="List available archives and exit"
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Number of concurrent download workers (default: 4)",
)
@click.option("--force", is_flag=True, help="Force re-download of existing archives")
@click.option(
    "--output-dir",
    default="data/synthetic_patients",
    help="Output directory for downloaded archives",
)
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config", help="Path to configuration file")
def download(
    archives,
    download_all,
    list_archives,
    workers,
    force,
    output_dir,
    log_level,
    config,
):
    """Download data archives."""
    if list_archives:
        list_available_archives()
        return

    # Determine which archives to download
    if download_all:
        archives_config = get_default_archives_config()
        archive_desc = "all available archives"
    else:
        if not archives:
            raise click.UsageError("Must specify --archives or --all")
        archives_config = parse_archive_list(archives)
        archive_count = sum(len(durations) for durations in archives_config.values())
        archive_desc = f"{archive_count} specified archives"

    if not archives_config:
        click.echo("❌ No valid archives specified for download")
        sys.exit(1)

    click.echo("🚀 mCODE Data Downloader")
    click.echo("=" * 30)
    click.echo(f"📦 Downloading: {archive_desc}")
    click.echo(f"🔄 Workers: {workers}")
    click.echo(f"💾 Output: {output_dir}")
    click.echo(f"🔄 Force: {'Yes' if force else 'No'}")
    click.echo()

    try:
        # Perform concurrent download
        start_time = time.time()

        downloaded_paths = download_archives(
            archives_config=archives_config,
            output_dir=output_dir,
            workers=workers,
            force=force,
        )

        end_time = time.time()
        duration = end_time - start_time

        # Report results
        successful = len(downloaded_paths)
        total_requested = sum(len(durations) for durations in archives_config.values())

        click.echo("\n" + "=" * 50)
        click.echo("✅ Download Summary:")
        click.echo(f"   📊 Archives requested: {total_requested}")
        click.echo(f"   ✅ Successfully processed: {successful}")
        click.echo(f"   ⏱️  Total time: {duration:.2f} seconds")
        if total_requested > 0:
            avg_time_per_archive = duration / total_requested
            click.echo(
                f"   📈 Avg time per archive: {avg_time_per_archive:.2f} seconds"
            )

        click.echo("\n📁 Downloaded Archives:")
        for archive_name, path in downloaded_paths.items():
            try:
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                click.echo(f"   ✅ {archive_name}: {size_mb:.1f} MB")
            except (OSError, FileNotFoundError):
                click.echo(f"   ✅ {archive_name}: {path}")

        if successful < total_requested:
            click.echo(
                f"\n⚠️  {total_requested - successful} archives were skipped (already exist or failed)"
            )
            click.echo("   Use --force to re-download existing archives")

    except KeyboardInterrupt:
        click.echo("\n⏹️  Download cancelled by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"❌ Download failed: {e}")
        sys.exit(1)
