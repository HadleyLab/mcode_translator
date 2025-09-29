"""
Data management CLI commands for mCODE Translator.

This module contains commands for downloading and managing data archives.

Migrated from Click to Typer with full type hints and heysol_api_client integration.
"""

import time
from pathlib import Path
from typing import Optional

import typer

from ..config.heysol_config import get_config
from .data_downloader import (download_archives, get_default_archives_config,
                              list_available_archives, parse_archive_list)

# Create the data Typer app
app = typer.Typer()


@app.command()
def download(
    archives: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of archive names (e.g., breast_cancer_10_years,mixed_cancer_lifetime)",
    ),
    download_all: bool = typer.Option(
        False, "--all", help="Download all available archives"
    ),
    list_archives: bool = typer.Option(
        False, "--list", help="List available archives and exit"
    ),
    workers: int = typer.Option(
        4, help="Number of concurrent download workers (default: 4)"
    ),
    force: bool = typer.Option(False, help="Force re-download of existing archives"),
    output_dir: str = typer.Option(
        "data/synthetic_patients", help="Output directory for downloaded archives"
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
) -> None:
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
            typer.echo("Error: Must specify --archives or --all", err=True)
            raise typer.Exit(1)
        archives_config = parse_archive_list(archives)
        archive_count = sum(len(durations) for durations in archives_config.values())
        archive_desc = f"{archive_count} specified archives"

    if not archives_config:
        typer.echo("❌ No valid archives specified for download")
        raise typer.Exit(1)

    # Get global configuration
    get_config()

    typer.echo("🚀 mCODE Data Downloader")
    typer.echo("=" * 30)
    typer.echo(f"📦 Downloading: {archive_desc}")
    typer.echo(f"🔄 Workers: {workers}")
    typer.echo(f"💾 Output: {output_dir}")
    typer.echo(f"🔄 Force: {'Yes' if force else 'No'}")
    typer.echo()

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

        typer.echo("\n" + "=" * 50)
        typer.echo("✅ Download Summary:")
        typer.echo(f"   📊 Archives requested: {total_requested}")
        typer.echo(f"   ✅ Successfully processed: {successful}")
        typer.echo(f"   ⏱️  Total time: {duration:.2f} seconds")
        if total_requested > 0:
            avg_time_per_archive = duration / total_requested
            typer.echo(
                f"   📈 Avg time per archive: {avg_time_per_archive:.2f} seconds"
            )

        typer.echo("\n📁 Downloaded Archives:")
        for archive_name, path in downloaded_paths.items():
            try:
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                typer.echo(f"   ✅ {archive_name}: {size_mb:.1f} MB")
            except (OSError, FileNotFoundError):
                typer.echo(f"   ✅ {archive_name}: {path}")

        if successful < total_requested:
            typer.echo(
                f"\n⚠️  {total_requested - successful} archives were skipped (already exist or failed)"
            )
            typer.echo("   Use --force to re-download existing archives")

    except KeyboardInterrupt:
        typer.echo("\n⏹️  Download cancelled by user")
        raise typer.Exit(130)
    except Exception as e:
        typer.echo(f"❌ Download failed: {e}")
        raise typer.Exit(1)


# For backward compatibility, expose the app as 'data'
data = app
