import os
from typing import Dict, List
from urllib.parse import urlparse

import requests


def download_synthetic_patient_archives(
    base_dir: str = "data/synthetic_patients",
    archives_config: Dict[str, Dict[str, str]] = None,
    force_download: bool = False,
) -> Dict[str, str]:
    """
    Download synthetic patient ZIP archives from MITRE/Synthea mCODE test data sources.

    Args:
        base_dir: Base directory to store archives.
        archives_config: Configuration dict with cancer_type -> {duration: url} mappings.
        force_download: If True, download even if files exist.

    Returns:
        Dict of archive paths: {archive_name: full_path}
    """
    if archives_config is None:
        archives_config = {
            "mixed_cancer": {
                "10_years": "https://mitre.box.com/shared/static/7k7lk7wmza4m17916xnvc2uszidyv6vm.zip",
                "lifetime": "https://mitre.box.com/shared/static/mn6kpk56zvvk2o0lvjv55n7rnyajbnm4.zip",
            },
            "breast_cancer": {
                "10_years": "https://mitre.box.com/shared/static/c6ca6y2jfumrhw4nu20kztktxdlhhzo8.zip",
                "lifetime": "https://mitre.box.com/shared/static/59n7mcm8si0qk3p36ud0vmrcdv7pr0s7.zip",
            },
        }

    downloaded_archives = {}

    for cancer_type, durations in archives_config.items():
        for duration, url in durations.items():
            # Create the duration subdirectory
            duration_dir = os.path.join(base_dir, cancer_type, duration)
            os.makedirs(duration_dir, exist_ok=True)

            archive_name = f"{cancer_type}_{duration}.zip"
            archive_path = os.path.join(duration_dir, archive_name)

            if os.path.exists(archive_path) and not force_download:
                print(f"Archive already exists: {archive_path}")
                downloaded_archives[archive_name] = archive_path
                continue

            print(f"Downloading {archive_name} from {url}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(archive_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Downloaded: {archive_path}")
                downloaded_archives[archive_name] = archive_path

            except requests.RequestException as e:
                print(f"Failed to download {archive_name}: {e}")
                if os.path.exists(archive_path):
                    os.remove(archive_path)

    return downloaded_archives


def get_archive_paths(base_dir: str = "data/synthetic_patients") -> Dict[str, str]:
    """
    Get paths to existing synthetic patient archives.

    Returns:
        Dict of archive paths: {archive_name: full_path}
    """
    archives = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".zip"):
                full_path = os.path.join(root, file)
                archives[file] = full_path
    return archives
