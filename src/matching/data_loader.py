"""
Data loading utilities for patient-trial matching.
"""

import json
from typing import Dict, List, Any
from pathlib import Path


def load_ndjson_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from NDJSON file.

    Args:
        file_path: Path to the NDJSON file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_patients(file_path: str) -> List[Dict[str, Any]]:
    """
    Load patient data from NDJSON file.

    Args:
        file_path: Path to patient NDJSON file

    Returns:
        List of patient FHIR bundles
    """
    return load_ndjson_file(file_path)


def load_trials(file_path: str) -> List[Dict[str, Any]]:
    """
    Load trial data from NDJSON file.

    Args:
        file_path: Path to trial NDJSON file

    Returns:
        List of trial data dictionaries
    """
    return load_ndjson_file(file_path)


def create_patient_trial_pairs(patients: List[Dict[str, Any]], trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create all possible patient-trial pairs for matching.

    Args:
        patients: List of patient FHIR bundles
        trials: List of trial data

    Returns:
        List of dictionaries with 'patient' and 'trial' keys
    """
    pairs = []
    for patient in patients:
        for trial in trials:
            pairs.append({
                'patient': patient,
                'trial': trial
            })
    return pairs


def get_sample_data(patients_file: str, trials_file: str, sample_size: int = 5) -> Dict[str, Any]:
    """
    Load small sample of patient and trial data for testing.

    Args:
        patients_file: Path to patients NDJSON file
        trials_file: Path to trials NDJSON file
        sample_size: Number of items to sample from each file

    Returns:
        Dictionary with 'patients', 'trials', and 'pairs' keys
    """
    patients = load_patients(patients_file)[:sample_size]
    trials = load_trials(trials_file)[:sample_size]
    pairs = create_patient_trial_pairs(patients, trials)

    return {
        'patients': patients,
        'trials': trials,
        'pairs': pairs
    }