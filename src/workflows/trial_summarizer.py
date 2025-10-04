"""
Trial Summarizer - Generate natural language summaries for clinical trials.

This module provides functionality to create comprehensive summaries
of clinical trials using the McodeSummarizer service.
"""

from typing import Any, Dict

from src.services.summarizer import McodeSummarizer


class TrialSummarizer:
    """Generate natural language summaries for clinical trials."""

    def __init__(self, include_dates: bool = True):
        """Initialize the trial summarizer."""
        self.summarizer = McodeSummarizer(include_dates=include_dates)

    def generate_trial_natural_language_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
        """Generate comprehensive natural language summary for clinical trial using McodeSummarizer."""
        try:
            # Use the McodeSummarizer service with the original trial data
            summary = self.summarizer.create_trial_summary(trial_data)
            print(
                f"Generated comprehensive trial summary for {trial_id}: {summary[:200]}..."
            )
            print(f"Full trial summary length: {len(summary)} characters")
            return summary

        except Exception as e:
            print(f"Error generating trial natural language summary: {e}")
            print(f"Trial data for error: {trial_data}")
            return f"Clinical Trial {trial_id}: Error generating comprehensive summary - {str(e)}"

    def convert_trial_mcode_to_mappings_format(
        self, mcode_elements: Dict[str, Any]
    ) -> list[Dict[str, Any]]:
        """Convert trial mCODE elements to standardized mappings format for storage."""
        mappings = []

        try:
            for element_name, element_data in mcode_elements.items():
                print(
                    f"Converting trial element {element_name}: type={type(element_data)}, value={element_data}"
                )

                if isinstance(element_data, list):
                    # Handle multiple values (e.g., multiple conditions, interventions)
                    for item in element_data:
                        if isinstance(item, dict):
                            mapping = {
                                "mcode_element": element_name,
                                "value": item.get("display", str(item)),
                                "system": item.get("system"),
                                "code": item.get("code"),
                                "interpretation": item.get("interpretation"),
                            }
                        else:
                            mapping = {
                                "mcode_element": element_name,
                                "value": str(item),
                                "system": None,
                                "code": None,
                                "interpretation": None,
                            }
                        mappings.append(mapping)
                else:
                    # Handle single values
                    if isinstance(element_data, dict):
                        mapping = {
                            "mcode_element": element_name,
                            "value": element_data.get("display", str(element_data)),
                            "system": element_data.get("system"),
                            "code": element_data.get("code"),
                            "interpretation": element_data.get("interpretation"),
                        }
                    else:
                        mapping = {
                            "mcode_element": element_name,
                            "value": str(element_data),
                            "system": None,
                            "code": None,
                            "interpretation": None,
                        }
                    mappings.append(mapping)
        except Exception as e:
            print(f"Error converting trial mCODE elements to mappings format: {e}")
            print(f"mcode_elements: {mcode_elements}")
            raise

        return mappings

    def format_trial_mcode_element(
        self, element_name: str, system: str, code: str
    ) -> str:
        """Centralized function to format trial mCODE elements consistently."""
        # Clean up system URLs to standard names
        if "snomed" in system.lower():
            clean_system = "SNOMED"
        elif "loinc" in system.lower():
            clean_system = "LOINC"
        elif "cvx" in system.lower():
            clean_system = "CVX"
        elif "rxnorm" in system.lower():
            clean_system = "RxNorm"
        elif "icd" in system.lower():
            clean_system = "ICD"
        elif "clinicaltrials.gov" in system.lower():
            clean_system = "ClinicalTrials.gov"
        else:
            # Remove URLs and keep only the system identifier
            clean_system = system.split("/")[-1].split(":")[-1].upper()

        return f"(mCODE: {element_name}; {clean_system}:{code})"
