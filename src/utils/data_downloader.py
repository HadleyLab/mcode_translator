import os
from typing import Dict

import requests

from src.utils.concurrency import TaskQueue, create_task
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


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


def download_synthetic_patient_archives_concurrent(
    base_dir: str = "data/synthetic_patients",
    archives_config: Dict[str, Dict[str, str]] = None,
    force_download: bool = False,
    max_workers: int = 4,
) -> Dict[str, str]:
    """
    Download synthetic patient ZIP archives concurrently for improved performance.

    Args:
        base_dir: Base directory to store archives.
        archives_config: Configuration dict with cancer_type -> {duration: url} mappings.
        force_download: If True, download even if files exist.
        max_workers: Maximum number of concurrent downloads.

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

    logger.info(
        f"🔄 Starting concurrent download of {len(archives_config)} archive types with {max_workers} workers"
    )

    # Prepare download tasks
    download_tasks = []
    archive_paths = {}

    for cancer_type, durations in archives_config.items():
        for duration, url in durations.items():
            # Create the duration subdirectory
            duration_dir = os.path.join(base_dir, cancer_type, duration)
            os.makedirs(duration_dir, exist_ok=True)

            archive_name = f"{cancer_type}_{duration}.zip"
            archive_path = os.path.join(duration_dir, archive_name)

            if os.path.exists(archive_path) and not force_download:
                logger.debug(f"Archive already exists: {archive_path}")
                archive_paths[archive_name] = archive_path
                continue

            # Create download task
            task = create_task(
                task_id=f"download_{archive_name}",
                func=_download_single_archive,
                url=url,
                dest_path=archive_path,
                archive_name=archive_name,
            )
            download_tasks.append(task)
            archive_paths[archive_name] = archive_path

    if not download_tasks:
        logger.info("✅ All archives already exist, skipping downloads")
        return archive_paths

    # Execute downloads concurrently
    task_queue = TaskQueue(max_workers=max_workers, name="ArchiveDownloader")

    def progress_callback(completed, total, result):
        archive_name = result.task_id.replace("download_", "")
        if result.success:
            logger.info(f"✅ Downloaded: {archive_name}")
        else:
            logger.error(f"❌ Failed to download {archive_name}: {result.error}")

    task_results = task_queue.execute_tasks(
        download_tasks, progress_callback=progress_callback
    )

    # Process results
    successful_downloads = 0
    failed_downloads = 0

    for result in task_results:
        archive_name = result.task_id.replace("download_", "")
        if result.success:
            successful_downloads += 1
        else:
            failed_downloads += 1
            # Remove failed downloads
            archive_path = archive_paths.get(archive_name)
            if archive_path and os.path.exists(archive_path):
                os.remove(archive_path)
                del archive_paths[archive_name]

    logger.info(
        f"📊 Concurrent download complete: {successful_downloads} successful, {failed_downloads} failed"
    )
    return archive_paths


def _download_single_archive(url: str, dest_path: str, archive_name: str) -> str:
    """
    Download a single archive file with streaming and error handling.

    Args:
        url: Download URL
        dest_path: Destination file path
        archive_name: Name of the archive for logging

    Returns:
        Destination path on success

    Raises:
        Exception: If download fails
    """
    try:
        logger.debug(f"Downloading {archive_name} from {url}")

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Stream download to avoid loading large files into memory
        response = requests.get(url, stream=True, timeout=300)  # 5 minute timeout
        response.raise_for_status()

        # Get file size for progress tracking
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Log progress for large files
                    if total_size > 10 * 1024 * 1024:  # > 10MB
                        (downloaded / total_size) * 100 if total_size > 0 else 0
                        logger.debug(".1f")

        logger.debug(f"Successfully downloaded {archive_name} ({downloaded} bytes)")
        return dest_path

    except Exception as e:
        logger.error(f"Failed to download {archive_name}: {e}")
        # Clean up partial downloads
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise


def download_multiple_files(
    file_urls: Dict[str, str],
    dest_dir: str = ".",
    max_workers: int = 4,
    force_download: bool = False,
) -> Dict[str, str]:
    """
    Download multiple files concurrently.

    Args:
        file_urls: Dict mapping filename to URL
        dest_dir: Destination directory
        max_workers: Maximum concurrent downloads
        force_download: Whether to overwrite existing files

    Returns:
        Dict mapping filename to local path
    """
    logger.info(
        f"🔄 Downloading {len(file_urls)} files concurrently with {max_workers} workers"
    )

    # Prepare download tasks
    download_tasks = []
    downloaded_paths = {}

    for filename, url in file_urls.items():
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path) and not force_download:
            logger.debug(f"File already exists: {dest_path}")
            downloaded_paths[filename] = dest_path
            continue

        task = create_task(
            task_id=f"download_{filename}",
            func=_download_single_file,
            url=url,
            dest_path=dest_path,
            filename=filename,
        )
        download_tasks.append(task)
        downloaded_paths[filename] = dest_path

    if not download_tasks:
        logger.info("✅ All files already exist, skipping downloads")
        return downloaded_paths

    # Execute downloads concurrently
    task_queue = TaskQueue(max_workers=max_workers, name="FileDownloader")

    def progress_callback(completed, total, result):
        filename = result.task_id.replace("download_", "")
        if result.success:
            logger.info(f"✅ Downloaded: {filename}")
        else:
            logger.error(f"❌ Failed to download {filename}: {result.error}")

    task_results = task_queue.execute_tasks(
        download_tasks, progress_callback=progress_callback
    )

    # Process results and clean up failed downloads
    successful_downloads = 0
    failed_downloads = 0

    for result in task_results:
        filename = result.task_id.replace("download_", "")
        if result.success:
            successful_downloads += 1
        else:
            failed_downloads += 1
            # Remove failed downloads
            dest_path = downloaded_paths.get(filename)
            if dest_path and os.path.exists(dest_path):
                os.remove(dest_path)
                del downloaded_paths[filename]

    logger.info(
        f"📊 Multi-file download complete: {successful_downloads} successful, {failed_downloads} failed"
    )
    return downloaded_paths


def _download_single_file(url: str, dest_path: str, filename: str) -> str:
    """
    Download a single file with streaming.

    Args:
        url: Download URL
        dest_path: Destination file path
        filename: Name of the file for logging

    Returns:
        Destination path on success

    Raises:
        Exception: If download fails
    """
    try:
        logger.debug(f"Downloading {filename} from {url}")

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.debug(f"Successfully downloaded {filename}")
        return dest_path

    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        # Clean up partial downloads
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
