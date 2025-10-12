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
        """Generate comprehensive natural language summary for clinical trial using mCODE mappings."""
        try:
            # Convert mCODE mappings to standardized format
            mappings = self.convert_trial_mcode_to_mappings_format(mcode_elements)

            # Filter to only trial-related elements
            trial_element_types = {
                "Trial",
                "TrialTitle",
                "TrialStudyType",
                "TrialPhase",
                "TrialStatus",
                "TrialLeadSponsor",
                "TrialCollaborators",
                "TrialInterventionModel",
                "TrialPrimaryPurpose",
                "TrialEnrollment",
                "TrialCancerConditions",
                "TrialAgeCriteria",
                "TrialSexCriteria",
                "TrialHealthyVolunteers",
                "TrialMedicationInterventions",
                "TrialOtherInterventions",
                "TrialPrimaryOutcomes",
                "TrialStartDate",
                "TrialCompletionDate",
            }

            # Create elements list for summarizer, filtering to trial elements
            elements = []
            for mapping in mappings:
                element_type = mapping.get("mcode_element", "")
                if element_type in trial_element_types or element_type.startswith("Trial"):
                    elements.append(
                        {
                            "element_name": element_type,
                            "value": mapping.get("value", ""),
                            "codes": mapping.get("code", ""),
                            "date_qualifier": "",
                        }
                    )

            # Use summarizer to generate sentences from LLM mappings
            subject = trial_id
            sentences = self.summarizer._generate_sentences_from_elements(elements, subject)
            llm_summary = " ".join(sentences)

            # Always include basic trial summary from structured data
            basic_summary = self.summarizer.create_trial_summary(trial_data)

            # Combine: basic summary first, then LLM-extracted details
            if llm_summary.strip():
                summary = f"{basic_summary} {llm_summary}".strip()
            else:
                summary = basic_summary

            print(f"Generated comprehensive trial summary for {trial_id}: {summary[:200]}...")
            print(f"Full trial summary length: {len(summary)} characters")
            return summary

        except Exception as e:
            print(f"Error generating trial natural language summary: {e}")
            # Fallback to basic summary
            try:
                summary = self.summarizer.create_trial_summary(trial_data)
                return summary
            except Exception as fallback_e:
                return f"Clinical Trial {trial_id}: Error generating comprehensive summary - {str(e)}, fallback failed: {str(fallback_e)}"

    def convert_trial_mcode_to_mappings_format(self, mcode_elements: Any) -> list[Dict[str, Any]]:
        """Convert trial mCODE elements to standardized mappings format for storage."""
        mappings = []

        try:
            if isinstance(mcode_elements, list):
                # Handle list of McodElement objects or dicts
                for item in mcode_elements:
                    if hasattr(item, "element_type"):
                        # McodElement object
                        mapping = {
                            "mcode_element": getattr(item, "element_type", ""),
                            "value": getattr(item, "display", ""),
                            "system": getattr(item, "system", ""),
                            "code": getattr(item, "code", ""),
                            "interpretation": getattr(item, "evidence_text", ""),
                        }
                    elif isinstance(item, dict):
                        # Dict format
                        mapping = {
                            "mcode_element": item.get("element_type", ""),
                            "value": item.get("display", ""),
                            "system": item.get("system", ""),
                            "code": item.get("code", ""),
                            "interpretation": item.get("evidence_text", ""),
                        }
                    else:
                        # Fallback
                        mapping = {
                            "mcode_element": "Unknown",
                            "value": str(item),
                            "system": None,
                            "code": None,
                            "interpretation": None,
                        }
                    mappings.append(mapping)
            elif isinstance(mcode_elements, dict):
                # Handle dict format (legacy)
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
            print(f"mcode_elements type: {type(mcode_elements)}")
            if isinstance(mcode_elements, list) and len(mcode_elements) > 0:
                print(f"First element type: {type(mcode_elements[0])}")
                if hasattr(mcode_elements[0], "__dict__"):
                    print(f"First element attrs: {mcode_elements[0].__dict__}")
            raise

        return mappings

    def format_trial_mcode_element(self, element_name: str, system: str, code: str) -> str:
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
