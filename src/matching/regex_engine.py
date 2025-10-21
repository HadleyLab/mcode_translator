"""
RegexRulesEngine - A deterministic, pattern-based matching engine.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from matching.base import MatchingEngineBase
from utils.logging_config import get_logger


class RegexRulesEngine(MatchingEngineBase):
    """
    A matching engine that uses regex patterns for deterministic matching.
    """

    def __init__(self, rules: Dict[str, str], cache_enabled: bool = True, max_retries: int = 3):
        """
        Initializes the engine with a set of regex rules.

        Args:
            rules: A dictionary where keys are mCODE element types and
                   values are regex patterns.
            cache_enabled: Whether to enable caching for results.
            max_retries: Maximum number of retries on failure.
        """
        super().__init__(cache_enabled=cache_enabled, max_retries=max_retries)
        self.rules = rules
        self.logger = get_logger(__name__)
        self.logger.info(f"✅ RegexRulesEngine initialized with {len(rules)} rules, cache: {cache_enabled}, retries: {max_retries}.")

    async def match(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> bool:
        """
        Matches patient data against trial criteria using regex rules.

        Args:
            patient_data: The patient's data.
            trial_criteria: The trial's eligibility criteria.

        Returns:
            A boolean indicating if a match was found.
        """
        # Extract patient attributes
        patient_age = self._extract_patient_age(patient_data)
        patient_stage = self._extract_patient_stage(patient_data)
        patient_cancer_type = self._extract_patient_cancer_type(patient_data)
        patient_biomarkers = self._extract_patient_biomarkers(patient_data)

        # Extract trial criteria
        eligibility_text = trial_criteria.get("eligibilityCriteria", "")
        if not eligibility_text:
            eligibility_text = trial_criteria.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")

        trial_age_min = self._extract_age_min(eligibility_text)
        trial_stages = self._extract_stages(eligibility_text)
        trial_cancer_types = self._extract_cancer_types(eligibility_text)
        trial_biomarkers = self._extract_biomarkers(eligibility_text)

        # Perform matching
        age_match = patient_age is not None and trial_age_min is not None and patient_age >= trial_age_min
        stage_match = patient_stage and trial_stages and patient_stage.upper() in [s.upper() for s in trial_stages]
        cancer_type_match = patient_cancer_type and trial_cancer_types and any(ct.lower() in patient_cancer_type.lower() for ct in trial_cancer_types)

        biomarker_match = False
        for pb in patient_biomarkers:
            for tb in trial_biomarkers:
                if pb[0] == tb[0] and pb[1] == tb[1]:
                    biomarker_match = True
                    break
            if biomarker_match:
                break

        # A match is found if at least one criterion is met.
        is_match = any([age_match, stage_match, cancer_type_match, biomarker_match])

        self.logger.debug(f"RegexRulesEngine match result: {is_match}")
        return is_match

    def _extract_patient_age(self, patient_data: Dict[str, Any]) -> Optional[int]:
        """Extract patient age from FHIR bundle."""
        try:
            birth_date = None
            for entry in patient_data.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Patient":
                    birth_date = resource.get("birthDate")
                    break

            if birth_date:
                birth_year = int(birth_date[:4])
                current_year = datetime.now().year
                return current_year - birth_year
        except (ValueError, TypeError):
            pass
        return None

    def _extract_patient_stage(self, patient_data: Dict[str, Any]) -> Optional[str]:
        """Extract patient cancer stage from FHIR bundle."""
        for entry in patient_data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Observation":
                code = resource.get("code", {}).get("text", "")
                if "stage" in code.lower():
                    value = resource.get("valueString")
                    if isinstance(value, str):
                        return value
        return None

    def _extract_patient_cancer_type(self, patient_data: Dict[str, Any]) -> Optional[str]:
        """Extract patient cancer type from FHIR bundle."""
        for entry in patient_data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Condition":
                code = resource.get("code", {}).get("text", "")
                if isinstance(code, str):
                    return code
        return None

    def _extract_patient_biomarkers(self, patient_data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract patient biomarkers from FHIR bundle."""
        biomarkers = []
        for entry in patient_data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Observation":
                code = resource.get("code", {}).get("text", "")
                value = resource.get("valueString", "")
                if code and value and any(term in code.lower() for term in ["er", "pr", "her2", "triple negative"]):
                    biomarkers.append((code, value))
        return biomarkers

    def _extract_age_min(self, text: str) -> Optional[int]:
        """Extract minimum age requirement from eligibility text."""
        match = re.search(r'age\s*(?:>=|≥|greater than or equal to)\s*(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)\s*years?\s*of\s*age\s*or\s*older', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_stages(self, text: str) -> List[str]:
        """Extract cancer stages from eligibility text."""
        stages = []
        stage_patterns = [
            r'stage\s*([IV]+(?:[ABC]?\d*)*)',
            r'stage\s*(\d+(?:\.\d+)?)',
            r'stage\s*([IV]+[ABC]?)',
        ]
        for pattern in stage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            stages.extend(matches)
        return list(set(stages))

    def _extract_cancer_types(self, text: str) -> List[str]:
        """Extract cancer types from eligibility text."""
        cancer_types = []
        patterns = [
            r'breast cancer',
            r'lung cancer',
            r'prostate cancer',
            r'colon cancer',
            r'melanoma',
            r'leukemia',
            r'lymphoma',
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                cancer_types.append(pattern)
        return cancer_types

    def _extract_biomarkers(self, text: str) -> List[Tuple[str, str]]:
        """Extract biomarker requirements from eligibility text."""
        biomarkers = []
        patterns = [
            (r'ER\s*positive', 'ER', 'positive'),
            (r'ER\s*negative', 'ER', 'negative'),
            (r'PR\s*positive', 'PR', 'positive'),
            (r'PR\s*negative', 'PR', 'negative'),
            (r'HER2\s*positive', 'HER2', 'positive'),
            (r'HER2\s*negative', 'HER2', 'negative'),
            (r'triple\s*negative', 'Triple Negative', 'positive'),
        ]
        for pattern, biomarker, status in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                biomarkers.append((biomarker, status))
        return biomarkers
