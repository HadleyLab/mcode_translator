"""
Data downloader helper functions for CLI.

This module provides helper functions for downloading synthetic patient data archives.
"""

import click


def list_available_archives():
    """List available synthetic patient archives."""
    archives_config = get_default_archives_config()
    click.echo("Available synthetic patient archives:")
    for cancer_type, durations in archives_config.items():
        click.echo(f"  {cancer_type}:")
        for duration in durations.keys():
            click.echo(f"    - {duration}")


def get_default_archives_config():
    """Get default archive configuration."""
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


def parse_archive_list(archives_str):
    """Parse comma-separated archive list into config dict."""
    if not archives_str:
        return {}

    archive_names = [name.strip() for name in archives_str.split(",")]
    default_config = get_default_archives_config()
    result = {}

    for name in archive_names:
        if "_" in name:
            # Parse format like "breast_cancer_10_years"
            parts = name.split("_")
            if len(parts) >= 3:
                cancer_type = f"{parts[0]}_{parts[1]}"
                duration = f"{parts[2]}_{parts[3]}" if len(parts) > 3 else parts[2]
                if cancer_type in default_config and duration in default_config[cancer_type]:
                    if cancer_type not in result:
                        result[cancer_type] = {}
                    result[cancer_type][duration] = default_config[cancer_type][duration]

    return result


def download_archives(archives_config, output_dir="data/synthetic_patients", workers=4, force=False):
    """Download archives using the data downloader."""
    from src.utils.data_downloader import download_synthetic_patient_archives_concurrent
    return download_synthetic_patient_archives_concurrent(
        base_dir=output_dir,
        archives_config=archives_config,
        force_download=force,
        max_workers=workers
    )
