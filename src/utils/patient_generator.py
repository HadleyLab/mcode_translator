#!/usr/bin/env python3
"""
Patient Generator - Iterator for Synthetic Patient Data Archives

A robust iterator class for streaming FHIR patient bundles from synthetic patient ZIP archives
without extraction. Supports randomization, specific patient lookup, and multiple archive management.

Author: mCODE Translation Team
Version: 1.0.0
License: MIT
"""

import json
import os
import random
import zipfile
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .config import Config
from .logging_config import get_logger


class PatientNotFoundError(Exception):
    """Raised when a specific patient ID is not found in the archive."""

    pass


class ArchiveLoadError(Exception):
    """Raised when there's an error loading the archive."""

    pass


class PatientGenerator:
    """
    Iterator for streaming synthetic patient FHIR bundles from ZIP archives.

    This class provides:
    - Streaming access to patient bundles without extracting the ZIP
    - Random patient selection
    - Specific patient lookup by ID
    - Support for multiple archives
    - Automatic handling of NDJSON and JSON formats

    Example:
        >>> generator = PatientGenerator("path/to/archive.zip")
        >>> for patient_bundle in generator:
        ...     process_patient(patient_bundle)
        >>> random_patient = generator.get_random_patient()
        >>> specific_patient = generator.get_patient_by_id("patient-123")
    """

    def __init__(
        self,
        archive_path: str,
        config: Optional[Config] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the PatientGenerator.

        Args:
            archive_path: Path to ZIP archive containing patient data
            config: Optional Config instance for archive path resolution
            shuffle: Whether to randomize patient order (default: False)
            seed: Random seed for reproducible shuffling (optional)

        Raises:
            ArchiveLoadError: If the archive cannot be loaded
        """
        self.logger = get_logger(__name__)
        self.config = config or Config()
        self.archive_path = self._resolve_archive_path(archive_path)
        self.shuffle = shuffle
        self.seed = seed
        self._patient_files: List[str] = []
        self._current_index = 0
        self._loaded = False

        # Setup random seed if provided
        if shuffle and seed is not None:
            random.seed(seed)
            self.logger.debug(f"Random seed set to {seed} for reproducible shuffling")

        self._load_file_list()

    def _resolve_archive_path(self, archive_path: str) -> str:
        """Resolve archive path using configuration if it's a named archive."""
        if not os.path.exists(archive_path):
            # Check if it's a named archive from config
            synthetic_config = self.config.synthetic_data_config.get(
                "synthetic_data", {}
            )
            archives = synthetic_config.get("archives", {})

            # Try to find matching archive name
            for cancer_type, durations in archives.items():
                for duration, archive_info in durations.items():
                    archive_name = f"{cancer_type}_{duration}.zip"
                    full_path = os.path.join(
                        synthetic_config.get(
                            "base_directory", "data/synthetic_patients"
                        ),
                        cancer_type,
                        duration,
                        archive_name,
                    )
                    if (
                        archive_path.lower() == archive_name.lower()
                        or archive_path.lower() == f"{cancer_type}/{duration}"
                    ):
                        if os.path.exists(full_path):
                            self.logger.info(
                                f"Resolved named archive '{archive_path}' to: {full_path}"
                            )
                            return full_path
                        else:
                            raise ArchiveLoadError(
                                f"Named archive '{archive_path}' not found at expected path: {full_path}"
                            )

            # If still not found, try direct path resolution
            resolved_path = Path(archive_path).resolve()
            if not resolved_path.exists():
                raise ArchiveLoadError(
                    f"Archive not found: {archive_path} (resolved: {resolved_path})"
                )
            return str(resolved_path)

        return archive_path

    def _load_file_list(self) -> None:
        """Load the list of patient files from the archive (lazy loading)."""
        if self._loaded:
            return

        self.logger.info(f"Scanning patient data archive: {self.archive_path}")

        if not os.path.exists(self.archive_path):
            raise ArchiveLoadError(f"Archive file not found: {self.archive_path}")

        try:
            with zipfile.ZipFile(self.archive_path, "r") as zf:
                file_count = len(zf.namelist())
                self.logger.debug(f"Archive contains {file_count} files")

                # Find patient data files
                patient_files = self._find_patient_files(zf)
                self._patient_files = list(patient_files.keys())
                self.logger.info(f"Found {len(self._patient_files)} patient data files")

                if self.shuffle:
                    random.shuffle(self._patient_files)

                self._loaded = True

        except zipfile.BadZipFile as e:
            raise ArchiveLoadError(
                f"Invalid ZIP archive: {self.archive_path} - {str(e)}"
            )
        except Exception as e:
            raise ArchiveLoadError(
                f"Failed to scan archive {self.archive_path}: {str(e)}"
            )

    def _find_patient_files(self, zf: zipfile.ZipFile) -> Dict[str, Dict[str, Any]]:
        """Find patient data files in the archive."""
        patient_files = {}
        archive_files = zf.namelist()

        for fname in archive_files:
            # Skip directories and non-data files
            if fname.endswith("/") or fname.startswith("__MACOSX/"):
                continue

            # Look for common patient data file patterns
            if fname.lower().endswith((".ndjson", ".json")) and any(
                keyword in fname.lower()
                for keyword in [
                    "patient",
                    "bundle",
                    "synthea",
                    "fhir",
                    "10yearsmcodebreast",
                ]
            ):
                patient_files[fname] = {
                    "format": "ndjson" if fname.endswith(".ndjson") else "json",
                    "size": zf.getinfo(fname).file_size,
                }
            elif any(
                pattern in fname.lower()
                for pattern in ["patient", "synthea", "fhir", "10yearsmcodebreast"]
            ):
                # Potential patient data file - will try to parse
                patient_files[fname] = {
                    "format": "auto",
                    "size": zf.getinfo(fname).file_size,
                }
            # Also include files that look like patient records (contain underscores and UUIDs)
            elif (
                fname.endswith(".json") and "_" in fname and len(fname.split("_")) >= 3
            ):
                patient_files[fname] = {
                    "format": "json",
                    "size": zf.getinfo(fname).file_size,
                }

        return patient_files

    def _load_file_from_archive(
        self, zf: zipfile.ZipFile, fname: str, file_info: Dict[str, Any]
    ) -> None:
        """Load patient bundles from a specific file in the archive."""
        try:
            with zf.open(fname) as file_like:
                content = file_like.read().decode("utf-8").strip()

                if not content:
                    self.logger.debug(f"Empty file {fname}, skipping")
                    return

                if file_info["format"] == "ndjson" or (
                    file_info["format"] == "auto" and "\n" in content
                ):
                    # Handle NDJSON format
                    self._load_ndjson(content, fname)
                else:
                    # Handle single JSON file
                    self._load_json(content, fname)

        except Exception as e:
            self.logger.warning(f"Failed to load {fname}: {str(e)}")

    def _load_ndjson(self, content: str, source_file: str) -> None:
        """Load patients from NDJSON content."""
        lines = [line for line in content.split("\n") if line.strip()]
        self.logger.debug(f"Processing {len(lines)} NDJSON lines from {source_file}")

        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                bundle = self._normalize_to_bundle(data)

                if bundle:
                    patient_id = self._extract_patient_id(bundle)
                    self._patients.append(bundle)
                    if patient_id:
                        self._patient_index[patient_id] = len(self._patients) - 1

                    if (i + 1) % 100 == 0:  # Progress logging every 100 patients
                        self.logger.info(
                            f"Loaded patient {i+1}/{len(lines)} from {source_file}"
                        )

            except json.JSONDecodeError as e:
                self.logger.debug(
                    f"Skipping invalid JSON line {i+1} in {source_file}: {str(e)}"
                )
                continue

    def _load_json(self, content: str, source_file: str) -> None:
        """Load patients from JSON content."""
        try:
            data = json.loads(content)
            bundle = self._normalize_to_bundle(data)

            if bundle:
                patient_id = self._extract_patient_id(bundle)
                self._patients.append(bundle)
                if patient_id:
                    self._patient_index[patient_id] = len(self._patients) - 1

                self.logger.debug(f"Loaded single bundle from {source_file}")

        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON in {source_file}: {str(e)}")

    def _normalize_to_bundle(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize data to FHIR Bundle format."""
        if isinstance(data, dict):
            if data.get("resourceType") == "Bundle":
                # Already a bundle
                return data
            elif data.get("resourceType") == "Patient":
                # Single patient, wrap in bundle
                return {
                    "resourceType": "Bundle",
                    "type": "collection",
                    "entry": [{"resource": data}],
                }
            elif "entry" in data and isinstance(data["entry"], list):
                # Bundle-like structure
                data["resourceType"] = "Bundle"
                data["type"] = data.get("type", "collection")
                return data

        return None

    def _extract_patient_id(self, bundle: Dict[str, Any]) -> Optional[str]:
        """Extract patient ID from bundle."""
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                # Try ID first
                patient_id = resource.get("id")
                if patient_id:
                    return patient_id

                # Try identifier
                for identifier in resource.get("identifier", []):
                    if identifier.get("use") == "usual" or identifier.get("system"):
                        return identifier.get("value")

                # Fallback to name-based ID
                name = resource.get("name", [{}])[0]
                return f"{name.get('family', 'unknown')}_{name.get('given', [''])[0] or 'unknown'}"

        return None

    def _shuffle_patients(self) -> None:
        """Shuffle patient order for randomization."""
        random.shuffle(self._patients)
        # Rebuild index after shuffling
        self._patient_index = {}
        for i, bundle in enumerate(self._patients):
            patient_id = self._extract_patient_id(bundle)
            if patient_id:
                self._patient_index[patient_id] = i
        self.logger.info(f"Shuffled {len(self._patients)} patients")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return iterator for streaming patients."""
        self._current_index = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """Get next patient bundle (lazy loading)."""
        if self._current_index >= len(self._patient_files):
            raise StopIteration

        # Load patient from file on demand
        fname = self._patient_files[self._current_index]
        patient = self._load_patient_from_file(fname)
        self._current_index += 1
        return patient

    def _load_patient_from_file(self, fname: str) -> Dict[str, Any]:
        """Load a single patient from a file in the archive."""
        try:
            with zipfile.ZipFile(self.archive_path, "r") as zf:
                with zf.open(fname) as file_like:
                    content = file_like.read().decode("utf-8").strip()

                    if not content:
                        raise ValueError(f"Empty file {fname}")

                    # Assume JSON format for patient files
                    data = json.loads(content)
                    bundle = self._normalize_to_bundle(data)

                    if not bundle:
                        raise ValueError(
                            f"Could not normalize data to FHIR bundle in {fname}"
                        )

                    return bundle

        except Exception as e:
            self.logger.error(f"Failed to load patient from {fname}: {str(e)}")
            raise

    def __len__(self) -> int:
        """Get total number of patients."""
        return len(self._patient_files)

    def get_random_patient(
        self, exclude_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get a random patient bundle.

        Args:
            exclude_ids: List of patient IDs to exclude from random selection

        Returns:
            Random patient bundle
        """
        if not self._patient_files:
            raise ArchiveLoadError("No patient files found in archive")

        # Select random file
        available_files = self._patient_files.copy()

        if exclude_ids:
            # For simplicity, just pick a random file and check if it's excluded
            # In a real implementation, you'd want to be more efficient
            max_attempts = min(len(available_files), 10)
            for _ in range(max_attempts):
                fname = random.choice(available_files)
                patient = self._load_patient_from_file(fname)
                patient_id = self._extract_patient_id(patient)
                if patient_id not in exclude_ids:
                    return patient
            raise ValueError("Could not find available patient after exclusions")

        # Pick random file
        fname = random.choice(available_files)
        return self._load_patient_from_file(fname)

    def get_patient_by_id(self, patient_id: str) -> Dict[str, Any]:
        """
        Get specific patient by ID.

        Args:
            patient_id: Patient identifier

        Returns:
            Patient bundle matching the ID

        Raises:
            PatientNotFoundError: If patient ID not found
        """
        # Search through all patient files (this is expensive for large archives)
        self.logger.warning(
            f"Searching for patient {patient_id} - this may be slow for large archives"
        )

        for fname in self._patient_files:
            try:
                patient = self._load_patient_from_file(fname)
                if self._extract_patient_id(patient) == patient_id:
                    return patient
            except Exception as e:
                self.logger.debug(f"Error loading {fname}: {e}")
                continue

        raise PatientNotFoundError(
            f"Patient with ID '{patient_id}' not found in archive"
        )

    def get_patients(
        self, limit: Optional[int] = None, start: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get a slice of patients.

        Args:
            limit: Maximum number of patients to return (None for all)
            start: Starting index

        Returns:
            List of patient bundles
        """
        patients = []
        files_to_process = self._patient_files[start:]

        if limit is not None:
            files_to_process = files_to_process[:limit]

        for fname in files_to_process:
            try:
                patient = self._load_patient_from_file(fname)
                patients.append(patient)
            except Exception as e:
                self.logger.warning(f"Failed to load patient from {fname}: {e}")

        return patients

    def get_patient_ids(self) -> List[str]:
        """Get list of all patient IDs in the archive (loads all patients)."""
        self.logger.warning(
            "get_patient_ids() requires loading all patients - this may be slow"
        )
        patient_ids = []
        original_index = self._current_index
        self._current_index = 0

        try:
            for patient in self:
                patient_id = self._extract_patient_id(patient)
                if patient_id:
                    patient_ids.append(patient_id)
        finally:
            self._current_index = original_index

        return patient_ids

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_index = 0

    def close(self) -> None:
        """Clean up resources (no-op for this implementation)."""
        self._patient_files = []
        self._current_index = 0
        self._loaded = False


def create_patient_generator(
    archive_identifier: str,
    config: Optional[Config] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> PatientGenerator:
    """
    Factory function to create PatientGenerator from archive identifier.

    Args:
        archive_identifier: Can be full path, archive name, or "cancer_type/duration"
        config: Optional Config instance
        shuffle: Whether to randomize order
        seed: Random seed for reproducibility

    Returns:
        Initialized PatientGenerator instance
    """
    # If config is not provided, create a new one
    if config is None:
        config = Config()
    return PatientGenerator(archive_identifier, config, shuffle, seed)


if __name__ == "__main__":
    # Example usage and testing
    import sys

    from .config import Config

    if len(sys.argv) < 2:
        print("Usage: python patient_generator.py <archive_path_or_name> [options]")
        sys.exit(1)

    archive = sys.argv[1]
    config = Config()
    generator = create_patient_generator(archive, config)

    print(f"Loaded {len(generator)} patients from {archive}")
    print(
        f"Available patient IDs: {generator.get_patient_ids()[:5]}..."
    )  # Show first 5

    # Show first patient
    first_patient = next(iter(generator))
    patient_id = generator._extract_patient_id(first_patient)
    print(f"First patient ID: {patient_id}")

    # Show random patient
    try:
        random_patient = generator.get_random_patient()
        random_id = generator._extract_patient_id(random_patient)
        print(f"Random patient ID: {random_id}")
    except Exception as e:
        print(f"Could not get random patient: {e}")
