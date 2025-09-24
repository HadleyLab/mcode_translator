"""
Data Loading Utilities - Consolidated data loading functions.

This module provides shared utilities for loading various data formats
used throughout the mCODE Translator application.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_ndjson_data(input_path: Path, data_type: str = "data") -> List[Dict[str, Any]]:
    """
    Load data from NDJSON file.

    Args:
        input_path: Path to the NDJSON file
        data_type: Type of data being loaded (for logging)

    Returns:
        List of parsed JSON objects

    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON parsing fails
    """
    if not input_path.exists():
        raise FileNotFoundError(f"{data_type.title()} file not found: {input_path}")

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Invalid JSON at line {line_num} in {input_path}: {e}"
                    )
                    continue

    if not data:
        logger.warning(f"No valid {data_type} found in {input_path}")

    logger.info(f"📄 Loaded {len(data)} {data_type} from {input_path}")
    return data


def extract_trial_id(trial_data: Dict[str, Any]) -> Optional[str]:
    """Extract trial ID from trial data."""
    try:
        return trial_data["protocolSection"]["identificationModule"]["nctId"]
    except KeyError:
        return None


def extract_patient_id(patient_data: Dict[str, Any]) -> Optional[str]:
    """Extract patient ID from patient data."""
    try:
        # Try different possible ID locations
        entries = patient_data.get("entry", [])
        for entry in entries:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patient_id = resource.get("id")
                if patient_id:
                    return patient_id
        return None
    except (KeyError, TypeError):
        return None
