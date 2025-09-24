"""
Data downloader utilities for mCODE Translator CLI.

This module provides functions for downloading and managing
synthetic patient data archives.
"""

from pathlib import Path
from typing import Dict, List

from src.utils.data_downloader import (
    download_synthetic_patient_archives_concurrent,
    get_archive_paths,
)


def get_default_archives_config() -> Dict[str, Dict[str, str]]:
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


def parse_archive_list(archive_str: str) -> Dict[str, Dict[str, str]]:
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


def list_available_archives() -> None:
    """List all available archives."""
    archives_config = get_default_archives_config()
    existing_archives = get_archive_paths()

    print("📚 Available Synthetic Patient Archives:")
    print("=" * 50)

    for cancer_type, durations in archives_config.items():
        print(f"\n🧬 {cancer_type.replace('_', ' ').title()}:")
        for duration, url in durations.items():
            archive_name = f"{cancer_type}_{duration}.zip"
            status = (
                "✅ Downloaded" if archive_name in existing_archives else "⬇️  Available"
            )
            size_info = ""
            if archive_name in existing_archives:
                path = existing_archives[archive_name]
                if Path(path).exists():
                    size = Path(path).stat().st_size
                    size_info = f" ({size / (1024*1024):.1f} MB)"

            print(f"   • {archive_name}{size_info} - {status}")

    print(
        f"\n📊 Total archives: {sum(len(durations) for durations in archives_config.values())}"
    )
    print(f"📦 Downloaded: {len(existing_archives)}")


def download_archives(
    archives_config: Dict[str, Dict[str, str]],
    output_dir: str = "data/synthetic_patients",
    workers: int = 4,
    force: bool = False,
) -> List[str]:
    """Download synthetic patient archives concurrently."""
    try:
        # Perform concurrent download
        downloaded_paths = download_synthetic_patient_archives_concurrent(
            base_dir=output_dir,
            archives_config=archives_config,
            force_download=force,
            max_workers=workers,
        )

        return downloaded_paths

    except KeyboardInterrupt:
        print("\n⏹️  Download cancelled by user")
        raise SystemExit(130)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise
