"""
Patients Processor Workflow - Process patient data with mCODE mapping.

This workflow handles processing patient data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

from typing import Any, Dict, List, Optional

from src.services.clinical_note_generator import ClinicalNoteGenerator
from src.services.demographics_extractor import DemographicsExtractor
from src.services.fhir_extractors import FHIRResourceExtractors
from src.services.summarizer import McodeSummarizer
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.concurrency import TaskQueue, create_task

from .base_workflow import \
    PatientsProcessorWorkflow as BasePatientsProcessorWorkflow
from .base_workflow import WorkflowResult


class PatientsProcessorWorkflow(BasePatientsProcessorWorkflow):
    """
    Workflow for processing patient data with mCODE mapping.

    Processes patient data and stores mCODE summaries to CORE Memory.
    """

    def __init__(
        self, config: Any, memory_storage: Optional[McodeMemoryStorage] = None
    ):
        """
        Initialize the patients processor workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.summarizer = McodeSummarizer()
        self.clinical_note_generator = ClinicalNoteGenerator()
        self.demographics_extractor = DemographicsExtractor()
        self.fhir_extractors = FHIRResourceExtractors()

    def execute(self, **kwargs: Any) -> WorkflowResult:
        """
        Execute the patients processing workflow.

        By default, does NOT store results to CORE memory. Use store_in_memory=True to enable.

        Args:
            **kwargs: Workflow parameters including:
                - patients_data: List of patient data to process
                - trials_criteria: Optional trial criteria for filtering
                - store_in_memory: Whether to store results in CORE memory (default: False)

        Returns:
            WorkflowResult: Processing results
        """
        try:
            # Extract parameters
            patients_data = kwargs.get("patients_data", [])
            trials_criteria = kwargs.get("trials_criteria")

            # Default to NOT store in CORE memory - use --ingest to enable
            store_in_memory = False

            if not patients_data:
                return self._create_result(
                    success=False,
                    error_message="No patient data provided for processing.",
                )

            # Process patients concurrently using TaskQueue
            self.logger.info(
                f"🔬 Processing {len(patients_data)} patients with mCODE mapping (concurrent)"
            )

            # Create tasks for concurrent processing
            tasks = []
            for i, patient in enumerate(patients_data):
                task = create_task(
                    task_id=f"patient_{i}",
                    func=self._process_single_patient,
                    patient=patient,
                    patient_index=i,
                    trials_criteria=trials_criteria,
                    store_in_memory=store_in_memory,
                )
                tasks.append(task)

            # Execute tasks concurrently
            task_queue = TaskQueue(
                max_workers=8, name="PatientProcessor"
            )  # Use 8 workers for patient processing
            task_results = task_queue.execute_tasks(tasks)

            # Process results
            processed_patients = []
            successful_count = 0
            failed_count = 0

            for result in task_results:
                if result.success:
                    processed_patients.append(result.result)
                    successful_count += 1
                else:
                    self.logger.error(f"Task {result.task_id} failed: {result.error}")
                    # Create error patient for failed processing
                    patient_index = int(result.task_id.split("_")[1])
                    error_patient = patients_data[patient_index].copy()
                    error_patient["McodeProcessingError"] = str(result.error)
                    processed_patients.append(error_patient)
                    failed_count += 1

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

    def _process_single_patient(
        self,
        patient: Dict[str, Any],
        patient_index: int,
        trials_criteria: Optional[Dict[str, Any]],
        store_in_memory: bool,
    ) -> Dict[str, Any]:
        """
        Process a single patient with mCODE mapping.

        Args:
            patient: Patient data to process
            patient_index: Index of patient for logging
            trials_criteria: Optional trial criteria for filtering
            store_in_memory: Whether to store results in CORE memory

        Returns:
            Processed patient data with mCODE elements
        """
        try:
            self.logger.info(f"Processing patient {patient_index+1}")
            self.logger.debug(
                f"Patient type: {type(patient)}, keys: {patient.keys() if isinstance(patient, dict) else 'Not a dict'}"
            )

            # Debug: Check patient data integrity before processing
            entries = patient.get("entry", [])
            self.logger.debug(f"Patient {patient_index+1} has {len(entries)} entries")
            for j, entry in enumerate(entries[:3]):  # Check first 3 entries
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Patient":
                    name = resource.get("name", [])
                    self.logger.debug(
                        f"Patient {patient_index+1} name before processing: {name}"
                    )
                    break

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

            # Store to CORE memory if requested
            if store_in_memory and self.memory_storage:
                try:
                    patient_id = self._extract_patient_id(patient)
                    self.logger.debug(f"Extracted patient ID: {patient_id}")

                    # Prepare mCODE data for storage
                    # Extract demographics from the Patient resource specifically
                    patient_resource = None
                    for entry in patient.get("entry", []):
                        resource = entry.get("resource", {})
                        if resource.get("resourceType") == "Patient":
                            patient_resource = resource
                            break

                    if patient_resource:
                        demographics = self.demographics_extractor.extract_demographics(
                            patient_resource
                        )
                    else:
                        demographics = {}
                    self.logger.debug(f"Extracted demographics: {demographics}")
                except Exception as e:
                    self.logger.error(f"Error preparing data for memory storage: {e}")
                    self.logger.debug(f"Patient data: {patient}")
                    raise

                # Add demographic info from mCODE mappings if not already extracted
                try:
                    if "gender" not in demographics and "PatientSex" in filtered_mcode:
                        patient_sex = filtered_mcode["PatientSex"]
                        self.logger.debug(
                            f"PatientSex type: {type(patient_sex)}, value: {patient_sex}"
                        )
                        if isinstance(patient_sex, dict):
                            demographics["gender"] = patient_sex.get(
                                "display", "Unknown"
                            )
                        else:
                            demographics["gender"] = str(patient_sex)
                except Exception as e:
                    self.logger.error(f"Error processing PatientSex: {e}")
                    demographics["gender"] = "Unknown"

                mcode_data = {
                    "original_patient_data": patient,  # Include original patient data for summarizer
                    "mcode_mappings": self._convert_to_mappings_format(filtered_mcode),
                    "demographics": demographics,
                    "metadata": processed_patient.get("mcode_processing_metadata", {}),
                }

                success = self.memory_storage.store_patient_mcode_summary(
                    patient_id, mcode_data
                )
                if success:
                    self.logger.info(f"✅ Stored patient {patient_id} mCODE summary")
                else:
                    self.logger.warning(
                        f"❌ Failed to store patient {patient_id} mCODE summary"
                    )

            return processed_patient

        except Exception as e:
            self.logger.error(f"Failed to process patient {patient_index+1}: {e}")
            # Add error information to patient
            if isinstance(patient, dict):
                error_patient = patient.copy()
                error_patient["McodeProcessingError"] = str(e)
                return error_patient
            else:
                # For non-dict patients, return a dict with error info
                return {
                    "McodeProcessingError": str(e),
                    "original_patient": str(patient),
                }

    def _extract_patient_mcode_elements(
        self, patient: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract mCODE elements from patient FHIR Bundle.

        This is a simplified version - in practice, this would use
        the full mCODE extraction logic from the original codebase.
        """
        mcode_elements = {}

        # Debug: Check patient data at start of method
        self.logger.debug(
            f"_extract_patient_mcode_elements called with patient type: {type(patient)}"
        )
        if not isinstance(patient, dict):
            self.logger.error(f"Patient is not a dict: {type(patient)} - {patient}")
            return mcode_elements

        entries = patient.get("entry", [])
        self.logger.debug(f"Found {len(entries)} entries in patient bundle")

        # Process each entry in the patient bundle
        for i, entry in enumerate(entries):
            self.logger.debug(f"Processing entry {i} type: {type(entry)}")
            if not isinstance(entry, dict):
                self.logger.error(f"Entry {i} is not a dict: {type(entry)} - {entry}")
                continue

            resource = entry.get("resource", {})
            self.logger.debug(f"Entry {i} resource type: {type(resource)}")
            if not isinstance(resource, dict):
                self.logger.error(
                    f"Entry {i} resource is not a dict: {type(resource)} - {resource}"
                )
                continue

            resource_type = resource.get("resourceType")
            self.logger.debug(f"Entry {i} resource type: {resource_type}")
            if not isinstance(resource_type, str):
                self.logger.error(
                    f"Entry {i} resourceType is not a string: {type(resource_type)} - {resource_type}"
                )
                continue

            try:
                if resource_type == "Patient":
                    # Extract demographics
                    mcode_elements.update(
                        self.demographics_extractor.extract_demographics(resource)
                    )
                elif resource_type == "Condition":
                    # Extract conditions as mCODE CancerCondition or ComorbidCondition
                    condition_data = self.fhir_extractors.extract_condition_mcode(
                        resource
                    )
                    if condition_data:
                        # Check if this is a cancer condition or comorbidity
                        display = condition_data.get("display", "").lower()
                        if any(
                            term in display
                            for term in [
                                "cancer",
                                "carcinoma",
                                "neoplasm",
                                "tumor",
                                "malignant",
                            ]
                        ):
                            mcode_elements["CancerCondition"] = condition_data
                        else:
                            # This is a comorbidity
                            if "ComorbidCondition" not in mcode_elements:
                                mcode_elements["ComorbidCondition"] = []
                            mcode_elements["ComorbidCondition"].append(condition_data)

                elif resource_type == "AllergyIntolerance":
                    # Extract allergies as mCODE elements
                    allergy_data = self.fhir_extractors.extract_allergy_mcode(resource)
                    if allergy_data:
                        if "AllergyIntolerance" not in mcode_elements:
                            mcode_elements["AllergyIntolerance"] = []
                        mcode_elements["AllergyIntolerance"].append(allergy_data)

                elif resource_type == "Immunization":
                    # Extract immunizations as mCODE elements
                    immunization_data = self.fhir_extractors.extract_immunization_mcode(
                        resource
                    )
                    if immunization_data:
                        if "Immunization" not in mcode_elements:
                            mcode_elements["Immunization"] = []
                        mcode_elements["Immunization"].append(immunization_data)

                elif resource_type == "FamilyMemberHistory":
                    # Extract family history as mCODE elements
                    family_data = self.fhir_extractors.extract_family_history_mcode(
                        resource
                    )
                    if family_data:
                        if "FamilyMemberHistory" not in mcode_elements:
                            mcode_elements["FamilyMemberHistory"] = []
                        mcode_elements["FamilyMemberHistory"].append(family_data)

                elif resource_type == "Observation":
                    # Extract observations as various mCODE elements
                    observation_data = self.fhir_extractors.extract_observation_mcode(
                        resource
                    )
                    if observation_data:
                        mcode_elements.update(observation_data)

                    # Extract comprehensive observations (performance status, vitals, labs)
                    comprehensive_obs = (
                        self.fhir_extractors.extract_observation_mcode_comprehensive(
                            resource
                        )
                    )
                    if comprehensive_obs:
                        mcode_elements.update(comprehensive_obs)
                elif resource_type == "Procedure":
                    # Extract procedures as mCODE CancerRelatedSurgicalProcedure
                    procedure_data = self.fhir_extractors.extract_procedure_mcode(
                        resource
                    )
                    if procedure_data:
                        if "CancerRelatedSurgicalProcedure" not in mcode_elements:
                            mcode_elements["CancerRelatedSurgicalProcedure"] = []
                        mcode_elements["CancerRelatedSurgicalProcedure"].append(
                            procedure_data
                        )
            except Exception as e:
                self.logger.error(f"Error processing {resource_type} resource: {e}")
                self.logger.debug(f"Resource content: {resource}")
                raise

        return mcode_elements

    def _extract_observation_mcode_comprehensive(
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
                            comp.get("code", {})
                            .get("coding", [{}])[0]
                            .get("display", "")
                            .lower()
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

        try:
            for element_name, element_data in mcode_elements.items():
                self.logger.debug(
                    f"Converting element {element_name}: type={type(element_data)}, value={element_data}"
                )

                if isinstance(element_data, list):
                    # Handle multiple values (e.g., multiple procedures)
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
            self.logger.error(
                f"Error converting mCODE elements to mappings format: {e}"
            )
            self.logger.debug(f"mcode_elements: {mcode_elements}")
            raise

        return mappings

    def _generate_natural_language_summary(
        self,
        patient_id: str,
        mcode_elements: Dict[str, Any],
        demographics: Dict[str, Any],
    ) -> str:
        """Generate clinical note-style natural language summary for CORE knowledge graph entity extraction."""
        return self.clinical_note_generator.generate_summary(
            patient_id, mcode_elements, demographics
        )

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

        import hashlib

        # Fallback to hash of patient data
        return f"patient_{hashlib.md5(str(patient).encode('utf-8')).hexdigest()[:8]}"
