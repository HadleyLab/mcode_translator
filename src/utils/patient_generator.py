#!/usr/bin/env python3
"""
Patient Generator - Iterator for Synthetic Patient Data Archives

A robust iterator class for streaming FHIR patient bundles from synthetic patient ZIP archives
without extraction. Supports randomization, specific patient lookup, and multiple archive management.

Author: mCODE Translation Team
Version: 1.0.0
License: MIT
"""

import os
import json
import random
import zipfile
from typing import Iterator, Optional, List, Dict, Any, Union
from pathlib import Path
from itertools import islice

from .logging_config import get_logger
from .config import Config


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
        seed: Optional[int] = None
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
        self._patients: List[Dict[str, Any]] = []
        self._patient_index: Dict[str, int] = {}
        self._current_index = 0
        self._loaded = False
        
        # Setup random seed if provided
        if shuffle and seed is not None:
            random.seed(seed)
            self.logger.debug(f"Random seed set to {seed} for reproducible shuffling")
        
        self._load_archive()
    
    def _resolve_archive_path(self, archive_path: str) -> str:
        """Resolve archive path using configuration if it's a named archive."""
        if not os.path.exists(archive_path):
            # Check if it's a named archive from config
            synthetic_config = self.config.config_data.get('synthetic_data', {})
            archives = synthetic_config.get('archives', {})
            
            # Try to find matching archive name
            for cancer_type, durations in archives.items():
                for duration, archive_info in durations.items():
                    archive_name = f"{cancer_type}_{duration}.zip"
                    full_path = os.path.join(
                        synthetic_config.get('base_directory', 'data/synthetic_patients'),
                        cancer_type, duration, archive_name
                    )
                    if archive_path.lower() == archive_name.lower() or archive_path.lower() == f"{cancer_type}/{duration}":
                        if os.path.exists(full_path):
                            self.logger.info(f"Resolved named archive '{archive_path}' to: {full_path}")
                            return full_path
                        else:
                            raise ArchiveLoadError(f"Named archive '{archive_path}' not found at expected path: {full_path}")
            
            # If still not found, try direct path resolution
            resolved_path = Path(archive_path).resolve()
            if not resolved_path.exists():
                raise ArchiveLoadError(f"Archive not found: {archive_path} (resolved: {resolved_path})")
            return str(resolved_path)
        
        return archive_path
    
    def _load_archive(self) -> None:
        """Load all patient bundles from the archive without extraction."""
        if self._loaded:
            return
        
        self.logger.info(f"Loading patient data from archive: {self.archive_path}")
        
        if not os.path.exists(self.archive_path):
            raise ArchiveLoadError(f"Archive file not found: {self.archive_path}")
        
        try:
            with zipfile.ZipFile(self.archive_path, 'r') as zf:
                file_count = len(zf.namelist())
                self.logger.debug(f"Archive contains {file_count} files")
                
                # Find patient data files
                patient_files = self._find_patient_files(zf)
                self.logger.info(f"Found {len(patient_files)} patient data files in archive")
                
                # Load patients from each file
                for fname, file_info in patient_files.items():
                    self._load_file_from_archive(zf, fname, file_info)
                
                self.logger.info(f"Successfully loaded {len(self._patients)} patient bundles")
                self._loaded = True
                
                if self.shuffle:
                    self._shuffle_patients()
                
        except zipfile.BadZipFile as e:
            raise ArchiveLoadError(f"Invalid ZIP archive: {self.archive_path} - {str(e)}")
        except Exception as e:
            raise ArchiveLoadError(f"Failed to load archive {self.archive_path}: {str(e)}")
    
    def _find_patient_files(self, zf: zipfile.ZipFile) -> Dict[str, Dict[str, Any]]:
        """Find patient data files in the archive."""
        patient_files = {}
        archive_files = zf.namelist()
        
        for fname in archive_files:
            # Skip directories and non-data files
            if fname.endswith('/') or fname.startswith('__MACOSX/'):
                continue
            
            # Look for common patient data file patterns
            if (fname.lower().endswith(('.ndjson', '.json')) and 
                any(keyword in fname.lower() for keyword in ['patient', 'bundle', 'synthea', 'fhir'])):
                patient_files[fname] = {
                    'format': 'ndjson' if fname.endswith('.ndjson') else 'json',
                    'size': zf.getinfo(fname).file_size
                }
            elif any(pattern in fname.lower() for pattern in ['patient', 'synthea', 'fhir']):
                # Potential patient data file - will try to parse
                patient_files[fname] = {
                    'format': 'auto',
                    'size': zf.getinfo(fname).file_size
                }
        
        return patient_files
    
    def _load_file_from_archive(self, zf: zipfile.ZipFile, fname: str, file_info: Dict[str, Any]) -> None:
        """Load patient bundles from a specific file in the archive."""
        try:
            with zf.open(fname) as file_like:
                content = file_like.read().decode('utf-8').strip()
                
                if not content:
                    self.logger.debug(f"Empty file {fname}, skipping")
                    return
                
                if file_info['format'] == 'ndjson' or (file_info['format'] == 'auto' and '\n' in content):
                    # Handle NDJSON format
                    self._load_ndjson(content, fname)
                else:
                    # Handle single JSON file
                    self._load_json(content, fname)
                    
        except Exception as e:
            self.logger.warning(f"Failed to load {fname}: {str(e)}")
    
    def _load_ndjson(self, content: str, source_file: str) -> None:
        """Load patients from NDJSON content."""
        lines = [line for line in content.split('\n') if line.strip()]
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
                        
                    self.logger.debug(f"Loaded patient {i+1}/{len(lines)} from {source_file}")
                    
            except json.JSONDecodeError as e:
                self.logger.debug(f"Skipping invalid JSON line {i+1} in {source_file}: {str(e)}")
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
            if data.get('resourceType') == 'Bundle':
                # Already a bundle
                return data
            elif data.get('resourceType') == 'Patient':
                # Single patient, wrap in bundle
                return {
                    'resourceType': 'Bundle',
                    'type': 'collection',
                    'entry': [{'resource': data}]
                }
            elif 'entry' in data and isinstance(data['entry'], list):
                # Bundle-like structure
                data['resourceType'] = 'Bundle'
                data['type'] = data.get('type', 'collection')
                return data
        
        return None
    
    def _extract_patient_id(self, bundle: Dict[str, Any]) -> Optional[str]:
        """Extract patient ID from bundle."""
        for entry in bundle.get('entry', []):
            resource = entry.get('resource', {})
            if resource.get('resourceType') == 'Patient':
                # Try ID first
                patient_id = resource.get('id')
                if patient_id:
                    return patient_id
                
                # Try identifier
                for identifier in resource.get('identifier', []):
                    if identifier.get('use') == 'usual' or identifier.get('system'):
                        return identifier.get('value')
                
                # Fallback to name-based ID
                name = resource.get('name', [{}])[0]
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
        """Get next patient bundle."""
        if self._current_index >= len(self._patients):
            raise StopIteration
        
        patient = self._patients[self._current_index]
        self._current_index += 1
        return patient
    
    def __len__(self) -> int:
        """Get total number of patients."""
        return len(self._patients)
    
    def get_random_patient(self, exclude_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get a random patient bundle.
        
        Args:
            exclude_ids: List of patient IDs to exclude from random selection
        
        Returns:
            Random patient bundle
        """
        if not self._patients:
            raise ArchiveLoadError("No patients loaded in generator")
        
        available_patients = self._patients.copy()
        
        if exclude_ids:
            available_patients = [
                p for i, p in enumerate(self._patients)
                if self._extract_patient_id(p) not in exclude_ids
            ]
        
        if not available_patients:
            raise ValueError("No available patients after exclusions")
        
        return random.choice(available_patients)
    
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
        if patient_id in self._patient_index:
            return self._patients[self._patient_index[patient_id]]
        
        # Fallback search if index doesn't have the ID
        for bundle in self._patients:
            if self._extract_patient_id(bundle) == patient_id:
                return bundle
        
        raise PatientNotFoundError(f"Patient with ID '{patient_id}' not found in archive")
    
    def get_patients(self, limit: Optional[int] = None, start: int = 0) -> List[Dict[str, Any]]:
        """
        Get a slice of patients.
        
        Args:
            limit: Maximum number of patients to return (None for all)
            start: Starting index
        
        Returns:
            List of patient bundles
        """
        if limit is None:
            return self._patients[start:]
        else:
            return list(islice(islice(self._patients, start, None), limit))
    
    def get_patient_ids(self) -> List[str]:
        """Get list of all patient IDs in the archive."""
        return list(self._patient_index.keys())
    
    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._current_index = 0
    
    def close(self) -> None:
        """Clean up resources (no-op for this implementation)."""
        self._patients = []
        self._patient_index = {}
        self._current_index = 0
        self._loaded = False


def create_patient_generator(
    archive_identifier: str,
    config: Optional[Config] = None,
    shuffle: bool = False,
    seed: Optional[int] = None
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
    print(f"Available patient IDs: {generator.get_patient_ids()[:5]}...")  # Show first 5
    
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