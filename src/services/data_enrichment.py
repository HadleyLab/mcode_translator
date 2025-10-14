"""
Data Enrichment Service for mCODE Elements

This service provides post-processing enrichment for missing clinical elements in patient and trial data.
It analyzes processed data, identifies gaps, and applies various enrichment strategies to improve completeness.

Features:
- Missing element analysis against mCODE profiles
- LLM-based intelligent enrichment using clinical context
- Rule-based enrichment for common clinical patterns
- Laboratory reference ranges enrichment
- Default values for optional but commonly expected elements
- mCODE model validation before saving

Integration:
- Used by patients_processor and trials_processor workflows
- Leverages existing LLM service and validation infrastructure
- Follows fail-fast, lean code principles
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

from src.services.llm.service import LLMService
from src.shared.mcode_models import (
    McodeValidator,
    CancerCondition,
    ECOGPerformanceStatus,
    TNMStageGroup,
    TumorMarkerTest,
    CancerRelatedMedicationStatement,
    CancerRelatedSurgicalProcedure,
    CancerRelatedRadiationProcedure,
    McodePatient,
    AdministrativeGender,
    BirthSex,
    ReceptorStatus,
    HistologyMorphologyBehavior,
)
from src.shared.models import (
    McodeElement,
    ProcessingMetadata,
    ParsedLLMResponse,
)
from src.utils.config import Config
from src.utils.logging_config import get_logger


class DataEnrichmentService:
    """
    Service for enriching processed patient/trial data with missing mCODE elements.

    Uses multiple enrichment strategies:
    1. LLM-based enrichment for intelligent inference
    2. Rule-based patterns for common clinical relationships
    3. Reference data enrichment for standard values
    4. Default value assignment for optional elements
    """

    def __init__(self, config: Config, llm_service: Optional[LLMService] = None):
        """
        Initialize the data enrichment service.

        Args:
            config: Configuration instance
            llm_service: Optional LLM service for enrichment (created if not provided)
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.llm_service = llm_service
        self.mcode_validator = McodeValidator()

        # Enrichment configuration
        self.enrichment_config = self._load_enrichment_config()

        # Laboratory reference ranges
        self.lab_reference_ranges = self._load_lab_reference_ranges()

        # Default values for optional elements
        self.default_values = self._load_default_values()

        # Rule-based enrichment patterns
        self.enrichment_rules = self._load_enrichment_rules()

        self.logger.info("🔧 Data Enrichment Service initialized")

    def _load_enrichment_config(self) -> Dict[str, Any]:
        """Load enrichment configuration from config files."""
        # Default configuration - can be extended with config files
        return {
            "enable_llm_enrichment": True,
            "enable_rule_based_enrichment": True,
            "enable_lab_reference_ranges": True,
            "enable_default_values": True,
            "llm_model": "deepseek-coder",
            "llm_prompt": "direct_mcode_evidence_based_concise",
            "max_enrichment_attempts": 3,
            "confidence_threshold": 0.7,
        }

    def _load_lab_reference_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Load laboratory reference ranges for common tests."""
        return {
            "hemoglobin": {
                "male": {"low": 13.5, "high": 17.5, "unit": "g/dL"},
                "female": {"low": 12.0, "high": 15.5, "unit": "g/dL"},
            },
            "wbc": {"low": 4.0, "high": 11.0, "unit": "10^9/L"},
            "platelet_count": {"low": 150, "high": 450, "unit": "10^9/L"},
            "creatinine": {
                "male": {"low": 0.7, "high": 1.3, "unit": "mg/dL"},
                "female": {"low": 0.6, "high": 1.1, "unit": "mg/dL"},
            },
            "bilirubin_total": {"low": 0.3, "high": 1.2, "unit": "mg/dL"},
            "alt": {"low": 7, "high": 56, "unit": "U/L"},
            "ast": {"low": 10, "high": 40, "unit": "U/L"},
            "alkaline_phosphatase": {"low": 44, "high": 147, "unit": "U/L"},
            "albumin": {"low": 3.5, "high": 5.0, "unit": "g/dL"},
        }

    def _load_default_values(self) -> Dict[str, Any]:
        """Load default values for optional but commonly expected elements."""
        return {
            "ecog_performance_status": "0",  # Fully active
            "birth_sex": "UNK",  # Unknown
            "receptor_status_er": "unknown",
            "receptor_status_pr": "unknown",
            "receptor_status_her2": "unknown",
            "histology_behavior": "3",  # Malignant
        }

    def _load_enrichment_rules(self) -> Dict[str, Any]:
        """Load rule-based enrichment patterns."""
        return {
            "performance_status_from_treatment": {
                # Infer ECOG from treatment tolerance
                "chemotherapy_tolerance": {
                    "good": "0",
                    "moderate": "1",
                    "poor": "2",
                }
            },
            "stage_from_diagnosis": {
                # Infer likely stage from diagnosis context
                "early_diagnosis": "I",
                "advanced_diagnosis": "IV",
            },
            "receptor_status_from_histology": {
                # Common receptor patterns by histology
                "invasive_ductal": {
                    "er_positive_rate": 0.7,
                    "pr_positive_rate": 0.6,
                    "her2_positive_rate": 0.2,
                }
            }
        }

    async def enrich_patient_data(
        self,
        patient_data: Dict[str, Any],
        existing_mcode_elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich patient data with missing mCODE elements.

        Args:
            patient_data: Original patient data
            existing_mcode_elements: Currently extracted mCODE elements

        Returns:
            Enriched patient data with additional elements
        """
        self.logger.info("🔍 Analyzing patient data for enrichment opportunities")

        # Analyze missing elements
        missing_elements = self._analyze_missing_elements(
            existing_mcode_elements, data_type="patient"
        )

        if not missing_elements:
            self.logger.info("✅ No enrichment needed - all elements present")
            return patient_data

        self.logger.info(f"📊 Found {len(missing_elements)} missing elements for enrichment")

        # Apply enrichment strategies
        enriched_elements = await self._apply_enrichment_strategies(
            patient_data, existing_mcode_elements, missing_elements, data_type="patient"
        )

        # Validate enriched data
        validation_result = self._validate_enriched_data(
            existing_mcode_elements, enriched_elements, data_type="patient"
        )

        if not validation_result["valid"]:
            self.logger.warning(f"⚠️ Validation issues with enriched data: {validation_result['issues']}")

        # Merge enriched elements
        enriched_patient_data = patient_data.copy()
        enriched_patient_data["enriched_mcode_elements"] = enriched_elements
        enriched_patient_data["enrichment_metadata"] = {
            "missing_elements_analyzed": len(missing_elements),
            "elements_enriched": len(enriched_elements),
            "validation_result": validation_result,
            "enrichment_timestamp": datetime.utcnow().isoformat(),
        }

        self.logger.info(f"✅ Patient data enriched with {len(enriched_elements)} additional elements")
        return enriched_patient_data

    async def enrich_trial_data(
        self,
        trial_data: Dict[str, Any],
        existing_mcode_elements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich trial data with missing mCODE elements.

        Args:
            trial_data: Original trial data
            existing_mcode_elements: Currently extracted mCODE elements

        Returns:
            Enriched trial data with additional elements
        """
        self.logger.info("🔍 Analyzing trial data for enrichment opportunities")

        # Analyze missing elements
        missing_elements = self._analyze_missing_elements(
            existing_mcode_elements, data_type="trial"
        )

        if not missing_elements:
            self.logger.info("✅ No enrichment needed - all elements present")
            return trial_data

        self.logger.info(f"📊 Found {len(missing_elements)} missing elements for enrichment")

        # Apply enrichment strategies
        enriched_elements = await self._apply_enrichment_strategies(
            trial_data, existing_mcode_elements, missing_elements, data_type="trial"
        )

        # Validate enriched data
        validation_result = self._validate_enriched_data(
            existing_mcode_elements, enriched_elements, data_type="trial"
        )

        if not validation_result["valid"]:
            self.logger.warning(f"⚠️ Validation issues with enriched data: {validation_result['issues']}")

        # Merge enriched elements
        enriched_trial_data = trial_data.copy()
        enriched_trial_data["enriched_mcode_elements"] = enriched_elements
        enriched_trial_data["enrichment_metadata"] = {
            "missing_elements_analyzed": len(missing_elements),
            "elements_enriched": len(enriched_elements),
            "validation_result": validation_result,
            "enrichment_timestamp": datetime.utcnow().isoformat(),
        }

        self.logger.info(f"✅ Trial data enriched with {len(enriched_elements)} additional elements")
        return enriched_trial_data

    def _analyze_missing_elements(
        self,
        existing_elements: Dict[str, Any],
        data_type: str
    ) -> List[str]:
        """
        Analyze existing elements to identify missing mCODE elements.

        Args:
            existing_elements: Currently extracted elements
            data_type: "patient" or "trial"

        Returns:
            List of missing element names
        """
        # Define expected mCODE elements by data type
        expected_elements = self._get_expected_elements(data_type)

        missing_elements = []
        for element in expected_elements:
            if element not in existing_elements:
                # Check for variations and partial matches
                if not self._has_element_variation(element, existing_elements):
                    missing_elements.append(element)

        return missing_elements

    def _get_expected_elements(self, data_type: str) -> List[str]:
        """Get expected mCODE elements for the given data type."""
        if data_type == "patient":
            return [
                "CancerCondition",
                "ECOGPerformanceStatus",
                "TNMStageGroup",
                "TumorMarkerTest",
                "CancerRelatedMedicationStatement",
                "CancerRelatedSurgicalProcedure",
                "CancerRelatedRadiationProcedure",
                "BodyWeight",
                "BodyHeight",
                "BloodPressure",
                "Hemoglobin",
                "WhiteBloodCellCount",
                "PlateletCount",
                "Creatinine",
                "TotalBilirubin",
                "AlanineAminotransferase",
            ]
        elif data_type == "trial":
            return [
                "TrialCancerConditions",
                "TrialInterventions",
                "TrialEligibilityCriteria",
                "TrialPhase",
                "TrialStatus",
                "TrialDesign",
            ]
        else:
            return []

    def _has_element_variation(self, element: str, existing_elements: Dict[str, Any]) -> bool:
        """Check if element exists in a variation (case-insensitive, partial match)."""
        element_lower = element.lower()
        for existing in existing_elements.keys():
            if element_lower in existing.lower() or existing.lower() in element_lower:
                return True
        return False

    async def _apply_enrichment_strategies(
        self,
        data: Dict[str, Any],
        existing_elements: Dict[str, Any],
        missing_elements: List[str],
        data_type: str
    ) -> Dict[str, Any]:
        """
        Apply multiple enrichment strategies to fill missing elements.

        Args:
            data: Original data
            existing_elements: Currently extracted elements
            missing_elements: List of missing element names
            data_type: "patient" or "trial"

        Returns:
            Dictionary of enriched elements
        """
        enriched_elements = {}

        # Strategy 1: Rule-based enrichment
        if self.enrichment_config["enable_rule_based_enrichment"]:
            rule_based = self._apply_rule_based_enrichment(
                data, existing_elements, missing_elements, data_type
            )
            enriched_elements.update(rule_based)

        # Strategy 2: Laboratory reference ranges
        if self.enrichment_config["enable_lab_reference_ranges"]:
            lab_ranges = self._apply_lab_reference_ranges(
                data, existing_elements, missing_elements
            )
            enriched_elements.update(lab_ranges)

        # Strategy 3: Default values
        if self.enrichment_config["enable_default_values"]:
            defaults = self._apply_default_values(missing_elements)
            enriched_elements.update(defaults)

        # Strategy 4: LLM-based enrichment (most sophisticated, applied last)
        if self.enrichment_config["enable_llm_enrichment"] and self.llm_service:
            llm_enriched = await self._apply_llm_enrichment(
                data, existing_elements, missing_elements, data_type
            )
            enriched_elements.update(llm_enriched)

        return enriched_elements

    def _apply_rule_based_enrichment(
        self,
        data: Dict[str, Any],
        existing_elements: Dict[str, Any],
        missing_elements: List[str],
        data_type: str
    ) -> Dict[str, Any]:
        """Apply rule-based enrichment patterns."""
        enriched = {}

        # Performance status inference from treatment data
        if "ECOGPerformanceStatus" in missing_elements:
            if "CancerRelatedMedicationStatement" in existing_elements:
                # Infer from chemotherapy tolerance
                treatment_data = existing_elements["CancerRelatedMedicationStatement"]
                inferred_status = self._infer_performance_status_from_treatment(treatment_data)
                if inferred_status:
                    enriched["ECOGPerformanceStatus"] = {
                        "value": inferred_status,
                        "system": "http://snomed.info/sct",
                        "code": inferred_status,
                        "display": f"ECOG Performance Status {inferred_status}",
                        "enrichment_method": "rule_based_treatment_inference",
                        "confidence": 0.6,
                    }

        # Stage inference from diagnosis context
        if "TNMStageGroup" in missing_elements:
            if "CancerCondition" in existing_elements:
                condition_data = existing_elements["CancerCondition"]
                inferred_stage = self._infer_stage_from_condition(condition_data)
                if inferred_stage:
                    enriched["TNMStageGroup"] = {
                        "value": inferred_stage,
                        "system": "http://snomed.info/sct",
                        "code": inferred_stage,
                        "display": f"Stage {inferred_stage}",
                        "enrichment_method": "rule_based_condition_inference",
                        "confidence": 0.5,
                    }

        return enriched

    def _infer_performance_status_from_treatment(self, treatment_data: Any) -> Optional[str]:
        """Infer ECOG performance status from treatment tolerance."""
        # Simple rule: if patient is on active chemotherapy, assume they can tolerate it
        if isinstance(treatment_data, list) and treatment_data:
            # Check for chemotherapy agents
            chemo_indicators = ["chemotherapy", "chemo", "paclitaxel", "docetaxel", "doxorubicin"]
            for treatment in treatment_data:
                treatment_name = str(treatment).lower()
                if any(indicator in treatment_name for indicator in chemo_indicators):
                    return "1"  # Restricted activity but ambulatory

        return None

    def _infer_stage_from_condition(self, condition_data: Any) -> Optional[str]:
        """Infer TNM stage from condition description."""
        if isinstance(condition_data, dict):
            condition_text = str(condition_data.get("display", "")).lower()
        else:
            condition_text = str(condition_data).lower()

        # Simple heuristics based on terminology
        if any(term in condition_text for term in ["early", "localized", "stage i", "stage 1"]):
            return "I"
        elif any(term in condition_text for term in ["advanced", "metastatic", "stage iv", "stage 4"]):
            return "IV"
        elif any(term in condition_text for term in ["regional", "stage ii", "stage iii"]):
            return "II"

        return None

    def _apply_lab_reference_ranges(
        self,
        data: Dict[str, Any],
        existing_elements: Dict[str, Any],
        missing_elements: List[str]
    ) -> Dict[str, Any]:
        """Apply laboratory reference ranges for missing lab values."""
        enriched = {}

        # Extract patient gender for gender-specific ranges
        patient_gender = None
        if "PatientSex" in existing_elements:
            sex_data = existing_elements["PatientSex"]
            if isinstance(sex_data, dict):
                patient_gender = sex_data.get("display", "").lower()
            else:
                patient_gender = str(sex_data).lower()

        # Map missing elements to lab reference ranges
        lab_mapping = {
            "Hemoglobin": "hemoglobin",
            "WhiteBloodCellCount": "wbc",
            "PlateletCount": "platelet_count",
            "Creatinine": "creatinine",
            "TotalBilirubin": "bilirubin_total",
            "AlanineAminotransferase": "alt",
        }

        for element in missing_elements:
            if element in lab_mapping:
                range_key = lab_mapping[element]
                lab_range = self.lab_reference_ranges.get(range_key)

                if lab_range:
                    # Handle gender-specific ranges
                    if isinstance(lab_range, dict) and patient_gender:
                        if patient_gender in ["male", "m"]:
                            range_data = lab_range.get("male", lab_range)
                        elif patient_gender in ["female", "f"]:
                            range_data = lab_range.get("female", lab_range)
                        else:
                            continue  # Skip if gender unknown for gender-specific test
                    else:
                        range_data = lab_range

                    enriched[element] = {
                        "reference_range": range_data,
                        "enrichment_method": "lab_reference_ranges",
                        "confidence": 0.9,
                    }

        return enriched

    def _apply_default_values(self, missing_elements: List[str]) -> Dict[str, Any]:
        """Apply default values for optional but commonly expected elements."""
        enriched = {}

        for element in missing_elements:
            if element in self.default_values:
                default_value = self.default_values[element]

                # Format based on element type
                if element == "ECOGPerformanceStatus":
                    enriched[element] = {
                        "value": default_value,
                        "system": "http://snomed.info/sct",
                        "code": default_value,
                        "display": f"ECOG Performance Status {default_value}",
                        "enrichment_method": "default_value",
                        "confidence": 0.3,
                    }
                elif element in ["receptor_status_er", "receptor_status_pr", "receptor_status_her2"]:
                    enriched[element] = {
                        "value": default_value,
                        "system": "http://snomed.info/sct",
                        "code": default_value,
                        "display": default_value.title(),
                        "enrichment_method": "default_value",
                        "confidence": 0.3,
                    }

        return enriched

    async def _apply_llm_enrichment(
        self,
        data: Dict[str, Any],
        existing_elements: Dict[str, Any],
        missing_elements: List[str],
        data_type: str
    ) -> Dict[str, Any]:
        """Apply LLM-based enrichment for intelligent inference."""
        if not self.llm_service:
            return {}

        enriched = {}

        # Create enrichment prompt
        enrichment_prompt = self._create_enrichment_prompt(
            data, existing_elements, missing_elements, data_type
        )

        try:
            # Use LLM service for enrichment
            enrichment_response = await self.llm_service.map_to_mcode(enrichment_prompt)

            if enrichment_response.success and enrichment_response.raw_response.parsed_json:
                llm_suggestions = enrichment_response.raw_response.parsed_json

                # Process LLM suggestions
                for suggestion in llm_suggestions.get("enrichment_suggestions", []):
                    element_name = suggestion.get("element_name")
                    if element_name in missing_elements:
                        # Validate confidence threshold
                        confidence = suggestion.get("confidence", 0.0)
                        if confidence >= self.enrichment_config["confidence_threshold"]:
                            enriched[element_name] = {
                                **suggestion,
                                "enrichment_method": "llm_inference",
                            }

        except Exception as e:
            self.logger.warning(f"LLM enrichment failed: {e}")

        return enriched

    def _create_enrichment_prompt(
        self,
        data: Dict[str, Any],
        existing_elements: Dict[str, Any],
        missing_elements: List[str],
        data_type: str
    ) -> str:
        """Create a prompt for LLM-based enrichment."""
        context = f"Data Type: {data_type}\n"
        context += f"Existing Elements: {json.dumps(existing_elements, indent=2)}\n"
        context += f"Missing Elements: {', '.join(missing_elements)}\n"

        if data_type == "patient":
            context += "Patient Data Summary:\n"
            # Extract relevant patient context
            if "bundle" in data:
                bundle = data["bundle"]
                if "entry" in bundle:
                    for entry in bundle["entry"]:
                        resource = entry.get("resource", {})
                        if resource.get("resourceType") == "Patient":
                            context += f"- Gender: {resource.get('gender')}\n"
                            context += f"- Birth Date: {resource.get('birthDate')}\n"
                        elif resource.get("resourceType") == "Condition":
                            context += f"- Condition: {resource.get('code', {}).get('text', 'Unknown')}\n"

        prompt = f"""You are a clinical data enrichment specialist. Analyze the provided clinical data and suggest appropriate values for missing mCODE elements.

CONTEXT:
{context}

INSTRUCTIONS:
1. Analyze existing clinical data for patterns and relationships
2. Suggest clinically appropriate values for missing elements
3. Provide confidence scores based on clinical reasoning
4. Only suggest values supported by the existing data
5. Use standard clinical reference ranges and guidelines

RESPONSE FORMAT (JSON):
{{
  "enrichment_suggestions": [
    {{
      "element_name": "ECOGPerformanceStatus",
      "value": "1",
      "system": "http://snomed.info/sct",
      "code": "1",
      "display": "ECOG Performance Status 1",
      "confidence": 0.8,
      "reasoning": "Patient on active chemotherapy suggests good performance status"
    }}
  ]
}}

Analyze the data and provide enrichment suggestions:"""

        return prompt

    def _validate_enriched_data(
        self,
        existing_elements: Dict[str, Any],
        enriched_elements: Dict[str, Any],
        data_type: str
    ) -> Dict[str, Any]:
        """
        Validate enriched data against mCODE models.

        Args:
            existing_elements: Original elements
            enriched_elements: Newly enriched elements
            data_type: "patient" or "trial"

        Returns:
            Validation result
        """
        issues = []

        # Combine elements for validation
        combined_elements = {**existing_elements, **enriched_elements}

        # Validate based on data type
        if data_type == "patient":
            # Validate patient-specific elements
            if "CancerCondition" in combined_elements:
                condition_data = combined_elements["CancerCondition"]
                if isinstance(condition_data, dict):
                    try:
                        # Create CancerCondition instance for validation
                        condition = CancerCondition(
                            resourceType="Condition",
                            subject={"reference": "Patient/unknown"},
                            clinicalStatus={"coding": [{"code": "active"}]},
                            category=[{"coding": [{"code": "problem-list-item"}]}],
                            code={"coding": [{"code": condition_data.get("code", "unknown")}]}
                        )
                    except Exception as e:
                        issues.append(f"CancerCondition validation failed: {e}")

            if "ECOGPerformanceStatus" in combined_elements:
                ecog_data = combined_elements["ECOGPerformanceStatus"]
                if isinstance(ecog_data, dict):
                    value = ecog_data.get("value")
                    if value not in ["0", "1", "2", "3", "4", "5"]:
                        issues.append(f"Invalid ECOG Performance Status value: {value}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "elements_validated": len(enriched_elements),
        }