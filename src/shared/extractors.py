"""
Shared extractors for common data extraction patterns.
"""

from typing import Dict, Any, Optional


class DataExtractor:
    """Shared utility for extracting common data patterns."""

    @staticmethod
    def extract_trial_id(trial: Optional[Dict[str, Any]]) -> str:
        """Extract trial ID from trial data."""
        try:
            if trial is None:
                return ""
            return trial.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
        except (KeyError, AttributeError, TypeError):
            return ""

    @staticmethod
    def extract_patient_id(patient: Optional[Dict[str, Any]]) -> str:
        """Extract patient ID from patient data."""
        try:
            if patient is None:
                return ""
            # Try different possible ID fields
            if "id" in patient:
                return str(patient["id"])
            elif "resource" in patient and "id" in patient["resource"]:
                return str(patient["resource"]["id"])
            elif "identifier" in patient and patient["identifier"]:
                # Use first identifier
                identifier = patient["identifier"][0]
                if "value" in identifier:
                    return str(identifier["value"])
            return ""
        except (KeyError, AttributeError, IndexError, TypeError):
            return ""

    @staticmethod
    def extract_provider_from_model(model: Optional[str]) -> str:
        """Get provider name from model name."""
        if model is None:
            return "Other"
        if model.startswith("deepseek"):
            return "DeepSeek"
        elif model.startswith("gpt"):
            return "OpenAI"
        else:
            return "Other"