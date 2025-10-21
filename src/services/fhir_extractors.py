"""
FHIR Resource Extractors - Extract mCODE elements from FHIR resources.

This module provides specialized extractors for different FHIR resource types
to convert them into mCODE elements using the new Pydantic mCODE models.
"""

from typing import Any, Dict, Optional, List, Union

from utils.logging_config import get_logger
from shared.models import (
    # mCODE profile models
    CancerCondition,
    TumorMarkerTest,
    ECOGPerformanceStatusObservation,
    CancerRelatedMedicationStatement,
    CancerRelatedSurgicalProcedure,
    CancerRelatedRadiationProcedure,
    # Base FHIR models
    FHIRCodeableConcept,
    FHIRReference,
    # Value sets
    CancerConditionCode,
    ECOGPerformanceStatus,
    ReceptorStatus,
    HistologyMorphologyBehavior,
    # Extensions
    HistologyMorphologyBehaviorExtension,
    LateralityExtension,
)


class FHIRResourceExtractors:
    """
    Extractors for converting FHIR resources to mCODE elements.

    Provides specialized methods for extracting mCODE elements from
    different FHIR resource types.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def extract_condition_mcode(self, condition: Dict[str, Any]) -> Optional[CancerCondition]:
        """Extract mCODE CancerCondition from Condition resource."""
        try:
            # Use the new CancerCondition model for structured extraction
            code = condition.get("code", {})
            coding = code.get("coding", [])

            # Find cancer-related coding
            cancer_coding = None
            for c in coding:
                if isinstance(c, dict):
                    display = (c.get("display", "")).lower()
                    if "breast" in display or "cancer" in display or "carcinoma" in display:
                        cancer_coding = c
                        break

            if not cancer_coding:
                return None

            # Create CancerCondition with proper profile
            cancer_condition = CancerCondition(
                resourceType="Condition",
                subject=FHIRReference(reference="Patient/unknown"),  # Will be set by caller
                clinicalStatus=FHIRCodeableConcept(
                    coding=[{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
                ),
                category=[FHIRCodeableConcept(
                    coding=[{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]
                )],
                code=FHIRCodeableConcept(
                    coding=[{
                        "system": cancer_coding.get("system"),
                        "code": cancer_coding.get("code"),
                        "display": cancer_coding.get("display")
                    }]
                ),
                meta={"profile": ["http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-condition"]}
            )

            return cancer_condition

        except Exception as e:
            self.logger.error(f"Error extracting condition mCODE: {e}")
            self.logger.debug(f"Condition resource: {condition}")
            return None

    def extract_observation_mcode(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mCODE elements from Observation resource."""
        try:
            # Simplified observation extraction
            elements = {}

            code = observation.get("code", {})
            coding = code.get("coding", [{}])[0]
            display = coding.get("display", "").lower()

            # Map common observations to mCODE elements
            if "estrogen" in display and "receptor" in display:
                elements["ERReceptorStatus"] = self._extract_receptor_status(observation)
            elif "her2" in display.lower():
                elements["HER2ReceptorStatus"] = self._extract_receptor_status(observation)
            elif "stage" in display or "tnm" in display:
                elements["TNMStage"] = self._extract_stage_info(observation)

            return elements
        except Exception as e:
            self.logger.error(f"Error extracting observation mCODE: {e}")
            self.logger.debug(f"Observation resource: {observation}")
            return {}

    def extract_observation_mcode_comprehensive(
        self, observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive mCODE elements from Observation resource."""
        try:
            elements = {}

            code = observation.get("code", {})
            coding = code.get("coding", [{}])[0]
            display = coding.get("display", "").lower()
            system = coding.get("system", "")

            # Performance Status
            if any(term in display for term in ["ecog", "performance", "karnofsky"]):
                value_codeable = observation.get("valueCodeableConcept", {})
                coding_val = value_codeable.get("coding", [{}])[0]
                if "ecog" in display:
                    elements["ECOGPerformanceStatus"] = {
                        "system": system,
                        "code": coding_val.get("code"),
                        "display": coding_val.get("display"),
                        "interpretation": coding_val.get("display", "Unknown"),
                    }
                elif "karnofsky" in display:
                    elements["KarnofskyPerformanceStatus"] = {
                        "system": system,
                        "code": coding_val.get("code"),
                        "display": coding_val.get("display"),
                        "interpretation": coding_val.get("display", "Unknown"),
                    }

            # Vital Signs
            elif any(
                term in display
                for term in [
                    "weight",
                    "height",
                    "bmi",
                    "body mass index",
                    "blood pressure",
                ]
            ):
                if "weight" in display:
                    value_quantity = observation.get("valueQuantity", {})
                    elements["BodyWeight"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "kg"),
                        "system": system,
                    }
                elif "height" in display:
                    value_quantity = observation.get("valueQuantity", {})
                    elements["BodyHeight"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "cm"),
                        "system": system,
                    }
                elif "bmi" in display or "body mass index" in display:
                    value_quantity = observation.get("valueQuantity", {})
                    elements["BodyMassIndex"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "kg/m2"),
                        "system": system,
                    }
                elif "blood pressure" in display:
                    # Handle systolic/diastolic components
                    components = observation.get("component", [])
                    systolic = diastolic = None
                    for comp in components:
                        comp_code = (
                            comp.get("code", {}).get("coding", [{}])[0].get("display", "").lower()
                        )
                        comp_value = comp.get("valueQuantity", {}).get("value")
                        if "systolic" in comp_code:
                            systolic = comp_value
                        elif "diastolic" in comp_code:
                            diastolic = comp_value
                    if systolic and diastolic:
                        elements["BloodPressure"] = {
                            "systolic": systolic,
                            "diastolic": diastolic,
                            "unit": "mmHg",
                            "system": system,
                        }

            # Laboratory Results
            elif any(
                term in display
                for term in [
                    "hemoglobin",
                    "wbc",
                    "white blood cell",
                    "platelet",
                    "creatinine",
                    "bilirubin",
                    "alt",
                    "alanine aminotransferase",
                ]
            ):
                value_quantity = observation.get("valueQuantity", {})
                if "hemoglobin" in display:
                    elements["Hemoglobin"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "g/dL"),
                        "system": system,
                    }
                elif "wbc" in display or "white blood cell" in display:
                    elements["WhiteBloodCellCount"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "10^9/L"),
                        "system": system,
                    }
                elif "platelet" in display:
                    elements["PlateletCount"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "10^9/L"),
                        "system": system,
                    }
                elif "creatinine" in display:
                    elements["Creatinine"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "mg/dL"),
                        "system": system,
                    }
                elif "bilirubin" in display:
                    elements["TotalBilirubin"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "mg/dL"),
                        "system": system,
                    }
                elif "alt" in display or "alanine aminotransferase" in display:
                    elements["AlanineAminotransferase"] = {
                        "value": value_quantity.get("value"),
                        "unit": value_quantity.get("unit", "U/L"),
                        "system": system,
                    }

            return elements
        except Exception as e:
            self.logger.error(f"Error extracting comprehensive observation mCODE: {e}")
            self.logger.debug(f"Observation resource: {observation}")
            return {}

    def extract_procedure_mcode(self, procedure: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from Procedure resource."""
        try:
            code = procedure.get("code", {})
            coding = code.get("coding", [{}])[0]

            display = coding.get("display", "").lower()
            if any(term in display for term in ["biopsy", "mastectomy", "surgery", "resection"]):
                return {
                    "system": coding.get("system"),
                    "code": coding.get("code"),
                    "display": coding.get("display"),
                    "date": procedure.get("performedDateTime"),
                }
        except Exception as e:
            self.logger.error(f"Error extracting procedure mCODE: {e}")
            self.logger.debug(f"Procedure resource: {procedure}")

        return None

    def extract_allergy_mcode(self, allergy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from AllergyIntolerance resource."""
        try:
            code = allergy.get("code", {})
            coding = code.get("coding", [{}])[0]

            return {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display"),
                "criticality": allergy.get("criticality"),
                "recordedDate": allergy.get("recordedDate"),
            }
        except Exception as e:
            self.logger.error(f"Error extracting allergy mCODE: {e}")
            self.logger.debug(f"Allergy resource: {allergy}")
            return None

    def extract_immunization_mcode(self, immunization: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from Immunization resource."""
        try:
            vaccine_code = immunization.get("vaccineCode", {})
            coding = vaccine_code.get("coding", [{}])[0]

            return {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display"),
                "occurrenceDateTime": immunization.get("occurrenceDateTime"),
                "status": immunization.get("status"),
            }
        except Exception as e:
            self.logger.error(f"Error extracting immunization mCODE: {e}")
            self.logger.debug(f"Immunization resource: {immunization}")
            return None

    def extract_family_history_mcode(
        self, family_history: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from FamilyMemberHistory resource."""
        try:
            relationship = family_history.get("relationship", {})
            coding = relationship.get("coding", [{}])[0]

            conditions = []
            for condition in family_history.get("condition", []):
                condition_code = condition.get("code", {})
                condition_coding = condition_code.get("coding", [{}])[0]
                conditions.append(
                    {
                        "system": condition_coding.get("system"),
                        "code": condition_coding.get("code"),
                        "display": condition_coding.get("display"),
                    }
                )

            return {
                "relationship": {
                    "system": coding.get("system"),
                    "code": coding.get("code"),
                    "display": coding.get("display"),
                },
                "conditions": conditions,
                "born": family_history.get("born"),
            }
        except Exception as e:
            self.logger.error(f"Error extracting family history mCODE: {e}")
            self.logger.debug(f"Family history resource: {family_history}")
            return None

    def _extract_receptor_status(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract receptor status from observation."""
        try:
            value_codeable = observation.get("valueCodeableConcept", {})
            coding = value_codeable.get("coding", [{}])[0]

            return {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display"),
                "interpretation": coding.get("display", "Unknown"),
            }
        except Exception as e:
            self.logger.error(f"Error extracting receptor status: {e}")
            return {"interpretation": "Unknown"}

    def _extract_stage_info(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract stage information from observation."""
        try:
            value_codeable = observation.get("valueCodeableConcept", {})
            coding = value_codeable.get("coding", [{}])[0]

            return {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display"),
                "interpretation": coding.get("display", "Unknown"),
            }
        except Exception as e:
            self.logger.error(f"Error extracting stage info: {e}")
            return {"interpretation": "Unknown"}
