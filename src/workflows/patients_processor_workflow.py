"""
Patients Processor Workflow - Process patient data with mCODE mapping.

This workflow handles processing patient data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.logging_config import get_logger

from .base_workflow import ProcessorWorkflow, WorkflowResult


class PatientsProcessorWorkflow(ProcessorWorkflow):
    """
    Workflow for processing patient data with mCODE mapping.

    Processes patient data and stores mCODE summaries to CORE Memory.
    """

    def __init__(self, config, memory_storage: Optional[McodeMemoryStorage] = None):
        """
        Initialize the patients processor workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the patients processing workflow.

        Args:
            **kwargs: Workflow parameters including:
                - patients_data: List of patient data to process
                - trials_criteria: Optional trial criteria for filtering
                - store_in_memory: Whether to store results in core memory

        Returns:
            WorkflowResult: Processing results
        """
        try:
            # Extract parameters
            patients_data = kwargs.get("patients_data", [])
            trials_criteria = kwargs.get("trials_criteria")
            store_in_memory = kwargs.get("store_in_memory", True)

            if not patients_data:
                return self._create_result(
                    success=False,
                    error_message="No patient data provided for processing.",
                )

            # Process patients
            processed_patients = []
            successful_count = 0
            failed_count = 0

            self.logger.info(
                f"🔬 Processing {len(patients_data)} patients with mCODE mapping"
            )

            for i, patient in enumerate(patients_data):
                try:
                    self.logger.info(f"Processing patient {i+1}/{len(patients_data)}")

                    # Extract mCODE elements from patient data
                    patient_mcode = self._extract_patient_mcode_elements(patient)

                    # Filter based on trial criteria if provided
                    if trials_criteria:
                        filtered_mcode = self._filter_by_trial_criteria(
                            patient_mcode, trials_criteria
                        )
                        self.logger.info(
                            f"Filtered patient mCODE elements: {len(filtered_mcode)}/{len(patient_mcode)}"
                        )
                    else:
                        filtered_mcode = patient_mcode

                    # Create processed patient data
                    processed_patient = patient.copy()
                    processed_patient["filtered_mcode_elements"] = filtered_mcode
                    processed_patient["mcode_processing_metadata"] = {
                        "original_elements_count": len(patient_mcode),
                        "filtered_elements_count": len(filtered_mcode),
                        "trial_criteria_applied": trials_criteria is not None,
                    }

                    processed_patients.append(processed_patient)
                    successful_count += 1

                    # Store to core memory if requested
                    if store_in_memory and self.memory_storage:
                        patient_id = self._extract_patient_id(patient)

                        # Prepare mCODE data for storage
                        mcode_data = {
                            "mcode_mappings": self._convert_to_mappings_format(
                                filtered_mcode
                            ),
                            "demographics": self._extract_demographics(patient),
                            "metadata": processed_patient.get(
                                "mcode_processing_metadata", {}
                            ),
                        }

                        success = self.memory_storage.store_patient_mcode_summary(
                            patient_id, mcode_data
                        )
                        if success:
                            self.logger.info(
                                f"✅ Stored patient {patient_id} mCODE summary"
                            )
                        else:
                            self.logger.warning(
                                f"❌ Failed to store patient {patient_id} mCODE summary"
                            )

                except Exception as e:
                    self.logger.error(f"Failed to process patient {i+1}: {e}")
                    failed_count += 1

                    # Add error information to patient
                    error_patient = patient.copy()
                    error_patient["McodeProcessingError"] = str(e)
                    processed_patients.append(error_patient)

            # Calculate success rate
            total_count = len(patients_data)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Processing complete: {successful_count}/{total_count} patients successful"
            )

            return self._create_result(
                success=successful_count > 0,
                data=processed_patients,
                metadata={
                    "total_patients": total_count,
                    "successful": successful_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "trial_criteria_applied": trials_criteria is not None,
                    "stored_in_memory": store_in_memory
                    and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "patients processing")

    def _extract_patient_mcode_elements(
        self, patient: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract mCODE elements from patient FHIR Bundle.

        This is a simplified version - in practice, this would use
        the full mCODE extraction logic from the original codebase.
        """
        mcode_elements = {}

        # Process each entry in the patient bundle
        for entry in patient.get("entry", []):
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "Patient":
                # Extract demographics
                mcode_elements.update(self._extract_demographics(resource))
            elif resource_type == "Condition":
                # Extract conditions as mCODE CancerCondition
                condition_data = self._extract_condition_mcode(resource)
                if condition_data:
                    mcode_elements["CancerCondition"] = condition_data
            elif resource_type == "Observation":
                # Extract observations as various mCODE elements
                observation_data = self._extract_observation_mcode(resource)
                if observation_data:
                    mcode_elements.update(observation_data)
            elif resource_type == "Procedure":
                # Extract procedures as mCODE CancerRelatedSurgicalProcedure
                procedure_data = self._extract_procedure_mcode(resource)
                if procedure_data:
                    if "CancerRelatedSurgicalProcedure" not in mcode_elements:
                        mcode_elements["CancerRelatedSurgicalProcedure"] = []
                    mcode_elements["CancerRelatedSurgicalProcedure"].append(
                        procedure_data
                    )

        return mcode_elements

    def _extract_demographics(self, patient_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic demographics as mCODE elements."""
        demographics = {}

        # PatientSex
        gender = patient_resource.get("gender")
        if gender:
            demographics["PatientSex"] = {
                "value": gender,
                "system": "http://hl7.org/fhir/administrative-gender",
                "display": {
                    "male": "Male",
                    "female": "Female",
                    "other": "Other",
                    "unknown": "Unknown",
                }.get(gender, gender),
            }

        # Race/Ethnicity from US Core extensions
        extensions = patient_resource.get("extension", [])
        for ext in extensions:
            url = ext.get("url")
            if url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race":
                coding = ext.get("extension", [{}])[0].get("valueCoding", {})
                demographics["Race"] = {
                    "system": coding.get("system"),
                    "code": coding.get("code"),
                    "display": coding.get("display", "Unknown"),
                }
            elif (
                url
                == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
            ):
                coding = ext.get("extension", [{}])[0].get("valueCoding", {})
                demographics["Ethnicity"] = {
                    "system": coding.get("system"),
                    "code": coding.get("code"),
                    "display": coding.get("display", "Unknown"),
                }

        return demographics

    def _extract_condition_mcode(
        self, condition: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from Condition resource."""
        # Simplified condition extraction
        code = condition.get("code", {})
        coding = code.get("coding", [])

        for c in coding:
            if (
                "breast" in (c.get("display", "")).lower()
                or "cancer" in (c.get("display", "")).lower()
            ):
                return {
                    "system": c.get("system"),
                    "code": c.get("code"),
                    "display": c.get("display"),
                    "interpretation": "Confirmed",
                }

        return None

    def _extract_observation_mcode(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mCODE elements from Observation resource."""
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

    def _extract_receptor_status(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract receptor status from observation."""
        value_codeable = observation.get("valueCodeableConcept", {})
        coding = value_codeable.get("coding", [{}])[0]

        return {
            "system": coding.get("system"),
            "code": coding.get("code"),
            "display": coding.get("display"),
            "interpretation": coding.get("display", "Unknown"),
        }

    def _extract_stage_info(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract stage information from observation."""
        value_codeable = observation.get("valueCodeableConcept", {})
        coding = value_codeable.get("coding", [{}])[0]

        return {
            "system": coding.get("system"),
            "code": coding.get("code"),
            "display": coding.get("display"),
            "interpretation": coding.get("display", "Unknown"),
        }

    def _extract_procedure_mcode(
        self, procedure: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from Procedure resource."""
        code = procedure.get("code", {})
        coding = code.get("coding", [{}])[0]

        display = coding.get("display", "").lower()
        if any(
            term in display for term in ["biopsy", "mastectomy", "surgery", "resection"]
        ):
            return {
                "system": coding.get("system"),
                "code": coding.get("code"),
                "display": coding.get("display"),
                "date": procedure.get("performedDateTime"),
            }

        return None

    def _filter_by_trial_criteria(
        self, patient_mcode: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter patient mCODE elements based on trial eligibility criteria.

        Args:
            patient_mcode: Patient's mCODE elements
            trial_criteria: Trial eligibility criteria

        Returns:
            Filtered mCODE elements
        """
        filtered = {}

        # Get the set of mCODE element types from trial criteria
        trial_element_types = set(trial_criteria.keys())

        # Filter patient elements to keep only those types present in trial criteria
        for element_type, element_data in patient_mcode.items():
            if element_type in trial_element_types:
                filtered[element_type] = element_data
                self.logger.debug(f"Keeping patient mCODE element: {element_type}")
            else:
                self.logger.debug(
                    f"Filtering out patient mCODE element: {element_type}"
                )

        return filtered

    def _convert_to_mappings_format(
        self, mcode_elements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert mCODE elements dict to mappings format expected by storage."""
        mappings = []

        for element_name, element_data in mcode_elements.items():
            if isinstance(element_data, list):
                # Handle multiple values (e.g., multiple procedures)
                for item in element_data:
                    mapping = {
                        "mcode_element": element_name,
                        "value": item.get("display", str(item)),
                        "system": item.get("system"),
                        "code": item.get("code"),
                        "interpretation": item.get("interpretation"),
                    }
                    mappings.append(mapping)
            else:
                # Handle single values
                mapping = {
                    "mcode_element": element_name,
                    "value": element_data.get("display", str(element_data)),
                    "system": element_data.get("system"),
                    "code": element_data.get("code"),
                    "interpretation": element_data.get("interpretation"),
                }
                mappings.append(mapping)

        return mappings

    def _extract_patient_id(self, patient: Dict[str, Any]) -> str:
        """Extract patient ID from patient data."""
        # Try different ways to get patient ID
        for entry in patient.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                # Try identifier first
                identifiers = resource.get("identifier", [])
                if identifiers:
                    return str(identifiers[0].get("value", "unknown"))

                # Try ID field
                patient_id = resource.get("id")
                if patient_id:
                    return str(patient_id)

        # Fallback to hash of patient data
        return f"patient_{hash(str(patient)) % 10000}"
