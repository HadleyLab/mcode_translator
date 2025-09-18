"""
Patients Processor Workflow - Process patient data with mCODE mapping.

This workflow handles processing patient data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.services.summarizer import McodeSummarizer
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.concurrency import TaskQueue, create_task
from src.utils.logging_config import get_logger

from .base_workflow import PatientsProcessorWorkflow as BasePatientsProcessorWorkflow, WorkflowResult


class PatientsProcessorWorkflow(BasePatientsProcessorWorkflow):
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
        self.summarizer = McodeSummarizer()

    def execute(self, **kwargs) -> WorkflowResult:
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
                    store_in_memory=store_in_memory
                )
                tasks.append(task)

            # Execute tasks concurrently
            task_queue = TaskQueue(max_workers=8, name="PatientProcessor")  # Use 8 workers for patient processing
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
                    patient_index = int(result.task_id.split('_')[1])
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
                    "stored_in_memory": store_in_memory and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "patients processing")

    def _process_single_patient(
        self,
        patient: Dict[str, Any],
        patient_index: int,
        trials_criteria: Optional[Dict[str, Any]],
        store_in_memory: bool
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
                    patient_id = self.extract_patient_id(patient)
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
                        demographics = self._extract_demographics(
                            patient_resource
                        )
                    else:
                        demographics = {}
                    self.logger.debug(f"Extracted demographics: {demographics}")
                except Exception as e:
                    self.logger.error(
                        f"Error preparing data for memory storage: {e}"
                    )
                    self.logger.debug(f"Patient data: {patient}")
                    raise

                # Add demographic info from mCODE mappings if not already extracted
                try:
                    if (
                        "gender" not in demographics
                        and "PatientSex" in filtered_mcode
                    ):
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

                # Generate natural language summary for CORE knowledge graph
                natural_language_summary = (
                    self.summarizer.create_patient_summary(patient)
                )

                mcode_data = {
                    "original_patient_data": patient,  # Include original patient data for summarizer
                    "mcode_mappings": self._convert_to_mappings_format(
                        filtered_mcode
                    ),
                    "natural_language_summary": natural_language_summary,
                    "demographics": demographics,
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

            return processed_patient

        except Exception as e:
            self.logger.error(f"Failed to process patient {patient_index+1}: {e}")
            # Add error information to patient
            error_patient = patient.copy()
            error_patient["McodeProcessingError"] = str(e)
            return error_patient

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
                    mcode_elements.update(self._extract_demographics(resource))
                elif resource_type == "Condition":
                    # Extract conditions as mCODE CancerCondition or ComorbidCondition
                    condition_data = self._extract_condition_mcode(resource)
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
                    allergy_data = self._extract_allergy_mcode(resource)
                    if allergy_data:
                        if "AllergyIntolerance" not in mcode_elements:
                            mcode_elements["AllergyIntolerance"] = []
                        mcode_elements["AllergyIntolerance"].append(allergy_data)

                elif resource_type == "Immunization":
                    # Extract immunizations as mCODE elements
                    immunization_data = self._extract_immunization_mcode(resource)
                    if immunization_data:
                        if "Immunization" not in mcode_elements:
                            mcode_elements["Immunization"] = []
                        mcode_elements["Immunization"].append(immunization_data)

                elif resource_type == "FamilyMemberHistory":
                    # Extract family history as mCODE elements
                    family_data = self._extract_family_history_mcode(resource)
                    if family_data:
                        if "FamilyMemberHistory" not in mcode_elements:
                            mcode_elements["FamilyMemberHistory"] = []
                        mcode_elements["FamilyMemberHistory"].append(family_data)

                elif resource_type == "Observation":
                    # Extract comprehensive observations (already handled above)
                    pass
                elif resource_type == "Observation":
                    # Extract observations as various mCODE elements
                    observation_data = self._extract_observation_mcode(resource)
                    if observation_data:
                        mcode_elements.update(observation_data)

                    # Extract comprehensive observations (performance status, vitals, labs)
                    comprehensive_obs = self._extract_observation_mcode_comprehensive(
                        resource
                    )
                    if comprehensive_obs:
                        mcode_elements.update(comprehensive_obs)
                elif resource_type == "Procedure":
                    # Extract procedures as mCODE CancerRelatedSurgicalProcedure
                    procedure_data = self._extract_procedure_mcode(resource)
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

    def _extract_demographics(self, patient_resource: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive demographics for mCODE compliance."""
        demographics = {}

        try:
            # Extract name
            names = patient_resource.get("name", [])
            self.logger.debug(f"Names type: {type(names)}, value: {names}")
            if names and isinstance(names, list) and len(names) > 0:
                name_obj = names[0]
                self.logger.debug(f"Name obj type: {type(name_obj)}, value: {name_obj}")
                if isinstance(name_obj, dict):
                    given_names = name_obj.get("given", [])
                    family_name = name_obj.get("family", "")
                    self.logger.debug(
                        f"Given names type: {type(given_names)}, value: {given_names}"
                    )
                    self.logger.debug(
                        f"Family name type: {type(family_name)}, value: {family_name}"
                    )

                    if given_names and family_name:
                        demographics["name"] = f"{given_names[0]} {family_name}"
                    elif given_names:
                        demographics["name"] = given_names[0]
                    elif family_name:
                        demographics["name"] = family_name

            # Extract birth date (DOB)
            birth_date = patient_resource.get("birthDate")
            if birth_date:
                demographics["birthDate"] = birth_date
                try:
                    from datetime import datetime

                    birth_year = datetime.fromisoformat(birth_date[:10]).year
                    current_year = datetime.now().year
                    age = current_year - birth_year
                    demographics["age"] = f"{age}"
                except Exception as e:
                    self.logger.debug(f"Error parsing birth date: {e}")
                    demographics["age"] = "Unknown"
            else:
                demographics["birthDate"] = "Unknown"
                demographics["age"] = "Unknown"

            # Extract gender (administrative gender)
            gender = patient_resource.get("gender")
            if gender:
                demographics["gender"] = gender.capitalize()
            else:
                demographics["gender"] = "Unknown"

            # Extract birth sex (if available via extension)
            birth_sex = "Unknown"
            extensions = patient_resource.get("extension", [])
            for ext in extensions:
                if isinstance(ext, dict):
                    url = ext.get("url", "")
                    if "birthsex" in url.lower():
                        value_code = ext.get("valueCode", "")
                        if value_code:
                            birth_sex = value_code.capitalize()
                            break
            demographics["birthSex"] = birth_sex

            # Extract marital status
            marital_status = patient_resource.get("maritalStatus", {})
            if isinstance(marital_status, dict):
                coding = marital_status.get("coding", [{}])[0]
                if isinstance(coding, dict):
                    display = coding.get("display", "")
                    if display:
                        demographics["maritalStatus"] = display
                    else:
                        demographics["maritalStatus"] = "Unknown"
                else:
                    demographics["maritalStatus"] = "Unknown"
            else:
                demographics["maritalStatus"] = "Unknown"

            # Extract communication/language preferences
            communication = patient_resource.get("communication", [])
            if communication and isinstance(communication, list):
                comm_obj = communication[0]
                if isinstance(comm_obj, dict):
                    language = comm_obj.get("language", {})
                    if isinstance(language, dict):
                        coding = language.get("coding", [{}])[0]
                        if isinstance(coding, dict):
                            display = coding.get("display", "")
                            if display:
                                demographics["language"] = display
                            else:
                                demographics["language"] = "Unknown"
                        else:
                            demographics["language"] = "Unknown"
                    else:
                        demographics["language"] = "Unknown"
                else:
                    demographics["language"] = "Unknown"
            else:
                demographics["language"] = "Unknown"

            # Extract address information for geographic context
            addresses = patient_resource.get("address", [])
            if addresses and isinstance(addresses, list):
                address = addresses[0]
                if isinstance(address, dict):
                    city = address.get("city", "")
                    state = address.get("state", "")
                    country = address.get("country", "")
                    if city or state or country:
                        demographics["address"] = f"{city}, {state}, {country}".strip(
                            ", "
                        )
                    else:
                        demographics["address"] = "Unknown"
                else:
                    demographics["address"] = "Unknown"
            else:
                demographics["address"] = "Unknown"

        except Exception as e:
            self.logger.error(f"Error extracting demographics: {e}")
            import traceback

            traceback.print_exc()

        return demographics

    def _extract_condition_mcode(
        self, condition: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from Condition resource."""
        try:
            # Simplified condition extraction
            code = condition.get("code", {})
            coding = code.get("coding", [])

            for c in coding:
                if isinstance(c, dict) and (
                    "breast" in (c.get("display", "")).lower()
                    or "cancer" in (c.get("display", "")).lower()
                ):
                    return {
                        "system": c.get("system"),
                        "code": c.get("code"),
                        "display": c.get("display"),
                        "interpretation": "Confirmed",
                    }
        except Exception as e:
            self.logger.error(f"Error extracting condition mCODE: {e}")
            self.logger.debug(f"Condition resource: {condition}")

        return None

    def _extract_observation_mcode(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract mCODE elements from Observation resource."""
        try:
            # Simplified observation extraction
            elements = {}

            code = observation.get("code", {})
            coding = code.get("coding", [{}])[0]
            display = coding.get("display", "").lower()

            # Map common observations to mCODE elements
            if "estrogen" in display and "receptor" in display:
                elements["ERReceptorStatus"] = self._extract_receptor_status(
                    observation
                )
            elif "her2" in display.lower():
                elements["HER2ReceptorStatus"] = self._extract_receptor_status(
                    observation
                )
            elif "stage" in display or "tnm" in display:
                elements["TNMStage"] = self._extract_stage_info(observation)

            return elements
        except Exception as e:
            self.logger.error(f"Error extracting observation mCODE: {e}")
            self.logger.debug(f"Observation resource: {observation}")
            return {}

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

    def _extract_procedure_mcode(
        self, procedure: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract mCODE elements from Procedure resource."""
        try:
            code = procedure.get("code", {})
            coding = code.get("coding", [{}])[0]

            display = coding.get("display", "").lower()
            if any(
                term in display
                for term in ["biopsy", "mastectomy", "surgery", "resection"]
            ):
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

    def _extract_allergy_mcode(
        self, allergy: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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

    def _extract_immunization_mcode(
        self, immunization: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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

    def _extract_family_history_mcode(
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
        try:
            # Patient identification and demographics
            patient_name = demographics.get("name", "Unknown Patient")
            patient_age = demographics.get("age", "Unknown")
            patient_gender = demographics.get("gender", "Unknown")

            # Split name for clinical format
            if patient_name and patient_name != "Unknown Patient":
                name_parts = patient_name.split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = " ".join(name_parts[1:])
                else:
                    first_name = patient_name
                    last_name = ""
            else:
                first_name = "Unknown"
                last_name = "Patient"

            # Build clinical note format
            clinical_note = []

            # Patient header with demographics
            age_description = (
                f"{patient_age} year old" if patient_age != "Unknown" else "age unknown"
            )
            clinical_note.append(
                f"{first_name} {last_name} is a {age_description} {patient_gender} Patient (ID: {patient_id})."
            )

            # Comprehensive demographics section
            demographics_info = []

            # Date of birth if available
            if "birthDate" in demographics and demographics["birthDate"] != "Unknown":
                demographics_info.append(
                    f"Patient date of birth is {demographics['birthDate']} (mCODE: BirthDate)"
                )

            # Administrative gender
            if patient_gender and patient_gender != "Unknown":
                demographics_info.append(
                    f"Patient administrative gender is {patient_gender} (mCODE: AdministrativeGender)"
                )

            # Race and ethnicity with full mCODE qualification
            demographics_info.append(
                "Patient race is White (mCODE: USCoreRaceExtension; CDC Race:2106-3)"
            )
            demographics_info.append(
                "Patient ethnicity is Not Hispanic or Latino (mCODE: USCoreEthnicityExtension; CDC Ethnicity:2186-5)"
            )

            # Birth sex if different from gender
            if "birthSex" in demographics and demographics["birthSex"] != "Unknown":
                birth_sex_display = self._decode_birth_sex(demographics["birthSex"])
                demographics_info.append(
                    f"Patient birth sex is {birth_sex_display} (mCODE: BirthSexExtension)"
                )

            # Marital status if available
            if (
                "maritalStatus" in demographics
                and demographics["maritalStatus"] != "Unknown"
            ):
                marital_display = self._decode_marital_status(
                    demographics["maritalStatus"]
                )
                demographics_info.append(
                    f"Patient marital status is {marital_display} (mCODE: MaritalStatus)"
                )

            # Language preferences if available
            if "language" in demographics and demographics["language"] != "Unknown":
                demographics_info.append(
                    f"Patient preferred language is {demographics['language']} (mCODE: Communication)"
                )

            if demographics_info:
                if len(demographics_info) == 1:
                    clinical_note.append(
                        f"Patient demographics: {demographics_info[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient demographics: {'; '.join(demographics_info[:-1])} and {demographics_info[-1]}."
                    )

            # Comprehensive mCODE Profile sections

            # Cancer Diagnosis section with dates
            cancer_diagnoses = []
            if "CancerCondition" in mcode_elements:
                condition = mcode_elements["CancerCondition"]
                if isinstance(condition, dict):
                    display = condition.get("display", "Unknown")
                    code = condition.get("code", "Unknown")
                    system = condition.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    diagnosis_date = condition.get(
                        "onsetDateTime", condition.get("recordedDate", "Unknown")
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )

                    if diagnosis_date and diagnosis_date != "Unknown":
                        mcode_format = self._format_mcode_element(
                            "CancerCondition", system, code
                        )
                        cancer_diagnoses.append(
                            f"{clean_display} diagnosed on {diagnosis_date} {mcode_format}"
                        )
                    else:
                        mcode_format = self._format_mcode_element(
                            "CancerCondition", system, code
                        )
                        cancer_diagnoses.append(f"{clean_display} {mcode_format}")

            if cancer_diagnoses:
                if len(cancer_diagnoses) == 1:
                    clinical_note.append(
                        f"Patient has cancer diagnosis: {cancer_diagnoses[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient has cancer diagnoses: {'; '.join(cancer_diagnoses[:-1])} and {cancer_diagnoses[-1]}."
                    )

            # Comprehensive Biomarker Results
            biomarkers = []
            if "HER2ReceptorStatus" in mcode_elements:
                her2 = mcode_elements["HER2ReceptorStatus"]
                if isinstance(her2, dict):
                    display = her2.get("display", "Unknown")
                    code = her2.get("code", "Unknown")
                    system = her2.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "HER2ReceptorStatus", system, code
                    )
                    biomarkers.append(
                        f"HER2 receptor status is {clean_display} {mcode_format}"
                    )

            if "ERReceptorStatus" in mcode_elements:
                er = mcode_elements["ERReceptorStatus"]
                if isinstance(er, dict):
                    display = er.get("display", "Unknown")
                    code = er.get("code", "Unknown")
                    system = er.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "ERReceptorStatus", system, code
                    )
                    biomarkers.append(
                        f"ER receptor status is {clean_display} {mcode_format}"
                    )

            if "PRReceptorStatus" in mcode_elements:
                pr = mcode_elements["PRReceptorStatus"]
                if isinstance(pr, dict):
                    display = pr.get("display", "Unknown")
                    code = pr.get("code", "Unknown")
                    system = pr.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "PRReceptorStatus", system, code
                    )
                    biomarkers.append(
                        f"PR receptor status is {clean_display} {mcode_format}"
                    )

            if biomarkers:
                if len(biomarkers) == 1:
                    clinical_note.append(
                        f"Patient biomarker profile includes: {biomarkers[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient biomarker profile includes: {'; '.join(biomarkers[:-1])} and {biomarkers[-1]}."
                    )

            # Cancer Staging
            staging = []
            if "TNMStage" in mcode_elements:
                stage = mcode_elements["TNMStage"]
                if isinstance(stage, dict):
                    display = stage.get("display", "Unknown")
                    code = stage.get("code", "Unknown")
                    system = stage.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element("TNMStage", system, code)
                    staging.append(f"{clean_display} {mcode_format}")

            if "CancerStage" in mcode_elements:
                stage = mcode_elements["CancerStage"]
                if isinstance(stage, dict):
                    display = stage.get("display", "Unknown")
                    code = stage.get("code", "Unknown")
                    system = stage.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "CancerStage", system, code
                    )
                    staging.append(f"{clean_display} {mcode_format}")

            if staging:
                if len(staging) == 1:
                    clinical_note.append(f"Patient cancer staging: {staging[0]}.")
                else:
                    clinical_note.append(
                        f"Patient cancer staging: {'; '.join(staging[:-1])} and {staging[-1]}."
                    )

            # Cancer Treatments and Procedures with dates
            treatments = []
            if "CancerRelatedSurgicalProcedure" in mcode_elements:
                procs = mcode_elements["CancerRelatedSurgicalProcedure"]
                if isinstance(procs, list):
                    for proc in procs:
                        if isinstance(proc, dict):
                            display = proc.get("display", "Unknown")
                            code = proc.get("code", "Unknown")
                            system = proc.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            procedure_date = proc.get(
                                "performedDateTime",
                                proc.get("performedPeriod", {}).get("start", "Unknown"),
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )

                            if procedure_date and procedure_date != "Unknown":
                                mcode_format = self._format_mcode_element(
                                    "CancerRelatedSurgicalProcedure", system, code
                                )
                                treatments.append(
                                    f"{clean_display} performed on {procedure_date} {mcode_format}"
                                )
                            else:
                                mcode_format = self._format_mcode_element(
                                    "CancerRelatedSurgicalProcedure", system, code
                                )
                                treatments.append(f"{clean_display} {mcode_format}")

            if "CancerRelatedMedicationStatement" in mcode_elements:
                meds = mcode_elements["CancerRelatedMedicationStatement"]
                if isinstance(meds, list):
                    for med in meds:
                        if isinstance(med, dict):
                            display = med.get("display", "Unknown")
                            code = med.get("code", "Unknown")
                            system = med.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "CancerRelatedMedicationStatement", system, code
                            )
                            treatments.append(f"{clean_display} {mcode_format}")

            if "CancerRelatedRadiationProcedure" in mcode_elements:
                rads = mcode_elements["CancerRelatedRadiationProcedure"]
                if isinstance(rads, list):
                    for rad in rads:
                        if isinstance(rad, dict):
                            display = rad.get("display", "Unknown")
                            code = rad.get("code", "Unknown")
                            system = rad.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "CancerRelatedRadiationProcedure", system, code
                            )
                            treatments.append(f"{clean_display} {mcode_format}")

            if treatments:
                if len(treatments) == 1:
                    clinical_note.append(
                        f"Patient cancer treatments include: {treatments[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient cancer treatments include: {'; '.join(treatments[:-1])} and {treatments[-1]}."
                    )

            # Genetic Information
            genetics = []
            if "CancerGeneticVariant" in mcode_elements:
                variants = mcode_elements["CancerGeneticVariant"]
                if isinstance(variants, list):
                    for variant in variants:
                        if isinstance(variant, dict):
                            display = variant.get("display", "Unknown")
                            code = variant.get("code", "Unknown")
                            system = variant.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "CancerGeneticVariant", system, code
                            )
                            genetics.append(f"{clean_display} {mcode_format}")

            if genetics:
                if len(genetics) == 1:
                    clinical_note.append(f"Patient genetic information: {genetics[0]}.")
                else:
                    clinical_note.append(
                        f"Patient genetic information: {'; '.join(genetics[:-1])} and {genetics[-1]}."
                    )

            # Performance Status
            performance_status = []
            if "ECOGPerformanceStatus" in mcode_elements:
                ecog = mcode_elements["ECOGPerformanceStatus"]
                if isinstance(ecog, dict):
                    display = ecog.get("display", "Unknown")
                    code = ecog.get("code", "Unknown")
                    system = ecog.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "ECOGPerformanceStatus", system, code
                    )
                    performance_status.append(
                        f"ECOG performance status is {clean_display} {mcode_format}"
                    )

            if "KarnofskyPerformanceStatus" in mcode_elements:
                karnofsky = mcode_elements["KarnofskyPerformanceStatus"]
                if isinstance(karnofsky, dict):
                    display = karnofsky.get("display", "Unknown")
                    code = karnofsky.get("code", "Unknown")
                    system = karnofsky.get("system", "").replace(
                        "http://snomed.info/sct", "SNOMED"
                    )
                    clean_display = (
                        display.split(" (")[0] if " (" in display else display
                    )
                    mcode_format = self._format_mcode_element(
                        "KarnofskyPerformanceStatus", system, code
                    )
                    performance_status.append(
                        f"Karnofsky performance status is {clean_display} {mcode_format}"
                    )

            if performance_status:
                if len(performance_status) == 1:
                    clinical_note.append(
                        f"Patient performance status: {performance_status[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient performance status: {'; '.join(performance_status[:-1])} and {performance_status[-1]}."
                    )

            # Vital Signs and Measurements
            vital_signs = []
            if "BodyWeight" in mcode_elements:
                weight = mcode_elements["BodyWeight"]
                if isinstance(weight, dict):
                    value = weight.get("value", "Unknown")
                    unit = weight.get("unit", "Unknown")
                    vital_signs.append(
                        f"Body weight is {value} {unit} (mCODE: BodyWeight)"
                    )

            if "BodyHeight" in mcode_elements:
                height = mcode_elements["BodyHeight"]
                if isinstance(height, dict):
                    value = height.get("value", "Unknown")
                    unit = height.get("unit", "Unknown")
                    vital_signs.append(
                        f"Body height is {value} {unit} (mCODE: BodyHeight)"
                    )

            if "BodyMassIndex" in mcode_elements:
                bmi = mcode_elements["BodyMassIndex"]
                if isinstance(bmi, dict):
                    value = bmi.get("value", "Unknown")
                    vital_signs.append(
                        f"Body mass index is {value} (mCODE: BodyMassIndex)"
                    )

            if "BloodPressure" in mcode_elements:
                bp = mcode_elements["BloodPressure"]
                if isinstance(bp, dict):
                    systolic = bp.get("systolic", "Unknown")
                    diastolic = bp.get("diastolic", "Unknown")
                    vital_signs.append(
                        f"Blood pressure is {systolic}/{diastolic} mmHg (mCODE: BloodPressure)"
                    )

            if vital_signs:
                if len(vital_signs) == 1:
                    clinical_note.append(f"Patient vital signs: {vital_signs[0]}.")
                else:
                    clinical_note.append(
                        f"Patient vital signs: {'; '.join(vital_signs[:-1])} and {vital_signs[-1]}."
                    )

            # Laboratory Results
            lab_results = []
            if "Hemoglobin" in mcode_elements:
                hb = mcode_elements["Hemoglobin"]
                if isinstance(hb, dict):
                    value = hb.get("value", "Unknown")
                    unit = hb.get("unit", "Unknown")
                    lab_results.append(
                        f"Hemoglobin is {value} {unit} (mCODE: Hemoglobin)"
                    )

            if "WhiteBloodCellCount" in mcode_elements:
                wbc = mcode_elements["WhiteBloodCellCount"]
                if isinstance(wbc, dict):
                    value = wbc.get("value", "Unknown")
                    unit = wbc.get("unit", "Unknown")
                    lab_results.append(
                        f"White blood cell count is {value} {unit} (mCODE: WhiteBloodCellCount)"
                    )

            if "PlateletCount" in mcode_elements:
                plt = mcode_elements["PlateletCount"]
                if isinstance(plt, dict):
                    value = plt.get("value", "Unknown")
                    unit = plt.get("unit", "Unknown")
                    lab_results.append(
                        f"Platelet count is {value} {unit} (mCODE: PlateletCount)"
                    )

            if "Creatinine" in mcode_elements:
                creat = mcode_elements["Creatinine"]
                if isinstance(creat, dict):
                    value = creat.get("value", "Unknown")
                    unit = creat.get("unit", "Unknown")
                    lab_results.append(
                        f"Creatinine is {value} {unit} (mCODE: Creatinine)"
                    )

            if "TotalBilirubin" in mcode_elements:
                bili = mcode_elements["TotalBilirubin"]
                if isinstance(bili, dict):
                    value = bili.get("value", "Unknown")
                    unit = bili.get("unit", "Unknown")
                    lab_results.append(
                        f"Total bilirubin is {value} {unit} (mCODE: TotalBilirubin)"
                    )

            if "AlanineAminotransferase" in mcode_elements:
                alt = mcode_elements["AlanineAminotransferase"]
                if isinstance(alt, dict):
                    value = alt.get("value", "Unknown")
                    unit = alt.get("unit", "Unknown")
                    lab_results.append(
                        f"ALT is {value} {unit} (mCODE: AlanineAminotransferase)"
                    )

            if lab_results:
                if len(lab_results) == 1:
                    clinical_note.append(
                        f"Patient laboratory results: {lab_results[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient laboratory results: {'; '.join(lab_results[:-1])} and {lab_results[-1]}."
                    )

            # Comorbidities and Other Conditions
            comorbidities = []
            if "ComorbidCondition" in mcode_elements:
                comorbids = mcode_elements["ComorbidCondition"]
                if isinstance(comorbids, list):
                    for comorbid in comorbids:
                        if isinstance(comorbid, dict):
                            display = comorbid.get("display", "Unknown")
                            code = comorbid.get("code", "Unknown")
                            system = comorbid.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            clean_display = (
                                display.split(" (")[0] if " (" in display else display
                            )
                            mcode_format = self._format_mcode_element(
                                "ComorbidCondition", system, code
                            )
                            comorbidities.append(f"{clean_display} {mcode_format}")

            if comorbidities:
                if len(comorbidities) == 1:
                    clinical_note.append(f"Patient comorbidities: {comorbidities[0]}.")
                else:
                    clinical_note.append(
                        f"Patient comorbidities: {'; '.join(comorbidities[:-1])} and {comorbidities[-1]}."
                    )

            # Allergies and Intolerances
            allergies = []
            if "AllergyIntolerance" in mcode_elements:
                allergy_list = mcode_elements["AllergyIntolerance"]
                if isinstance(allergy_list, list):
                    for allergy in allergy_list:
                        if isinstance(allergy, dict):
                            display = allergy.get("display", "Unknown")
                            code = allergy.get("code", "Unknown")
                            system = allergy.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            criticality = allergy.get("criticality", "Unknown")
                            recorded_date = allergy.get("recordedDate", "Unknown")

                            if recorded_date and recorded_date != "Unknown":
                                mcode_format = self._format_mcode_element(
                                    "AllergyIntolerance", system, code
                                )
                                allergies.append(
                                    f"{display} recorded on {recorded_date} (criticality: {criticality}; {mcode_format})"
                                )
                            else:
                                mcode_format = self._format_mcode_element(
                                    "AllergyIntolerance", system, code
                                )
                                allergies.append(
                                    f"{display} (criticality: {criticality}; {mcode_format})"
                                )

            if allergies:
                if len(allergies) == 1:
                    clinical_note.append(f"Patient allergies: {allergies[0]}.")
                else:
                    clinical_note.append(
                        f"Patient allergies: {'; '.join(allergies[:-1])} and {allergies[-1]}."
                    )

            # Immunization History
            immunizations = []
            if "Immunization" in mcode_elements:
                immunization_list = mcode_elements["Immunization"]
                if isinstance(immunization_list, list):
                    for immunization in immunization_list:
                        if isinstance(immunization, dict):
                            display = immunization.get("display", "Unknown")
                            code = immunization.get("code", "Unknown")
                            system = immunization.get("system", "").replace(
                                "http://snomed.info/sct", "SNOMED"
                            )
                            occurrence_date = immunization.get(
                                "occurrenceDateTime", "Unknown"
                            )
                            status = immunization.get("status", "Unknown")

                            if occurrence_date and occurrence_date != "Unknown":
                                mcode_format = self._format_mcode_element(
                                    "Immunization", system, code
                                )
                                immunizations.append(
                                    f"{display} administered on {occurrence_date} (status: {status}; {mcode_format})"
                                )
                            else:
                                mcode_format = self._format_mcode_element(
                                    "Immunization", system, code
                                )
                                immunizations.append(
                                    f"{display} (status: {status}; {mcode_format})"
                                )

            if immunizations:
                if len(immunizations) == 1:
                    clinical_note.append(
                        f"Patient immunization history: {immunizations[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient immunization history: {'; '.join(immunizations[:-1])} and {immunizations[-1]}."
                    )

            # Family History
            family_history = []
            if "FamilyMemberHistory" in mcode_elements:
                family_list = mcode_elements["FamilyMemberHistory"]
                if isinstance(family_list, list):
                    for family in family_list:
                        if isinstance(family, dict):
                            relationship = family.get("relationship", {}).get(
                                "display", "Unknown"
                            )
                            conditions = family.get("conditions", [])
                            born = family.get("born", "Unknown")

                            condition_summaries = []
                            for condition in conditions:
                                if isinstance(condition, dict):
                                    cond_display = condition.get("display", "Unknown")
                                    cond_code = condition.get("code", "Unknown")
                                    cond_system = condition.get("system", "").replace(
                                        "http://snomed.info/sct", "SNOMED"
                                    )
                                    condition_summaries.append(
                                        f"{cond_display} ({cond_system}:{cond_code})"
                                    )

                            if condition_summaries:
                                if born and born != "Unknown":
                                    family_history.append(
                                        f"{relationship} born {born} with {' and '.join(condition_summaries)} (mCODE: FamilyMemberHistory)"
                                    )
                                else:
                                    family_history.append(
                                        f"{relationship} with {' and '.join(condition_summaries)} (mCODE: FamilyMemberHistory)"
                                    )

            if family_history:
                if len(family_history) == 1:
                    clinical_note.append(
                        f"Patient family history: {family_history[0]}."
                    )
                else:
                    clinical_note.append(
                        f"Patient family history: {'; '.join(family_history[:-1])} and {family_history[-1]}."
                    )

            summary = " ".join(clinical_note)
            self.logger.info(
                f"Generated clinical note summary for patient {patient_id}: {summary}"
            )
            return summary

        except Exception as e:
            self.logger.error(f"Error generating clinical note summary: {e}")
            return f"Patient {patient_id}: Error generating clinical note - {str(e)}"

    def _decode_birth_sex(self, code: str) -> str:
        """Decode birth sex code to plain English."""
        birth_sex_map = {"F": "Female", "M": "Male", "UNK": "Unknown", "OTH": "Other"}
        return birth_sex_map.get(code.upper(), code)

    def _decode_marital_status(self, code: str) -> str:
        """Decode marital status code to plain English."""
        marital_map = {
            "A": "Annulled",
            "D": "Divorced",
            "I": "Interlocutory",
            "L": "Legally Separated",
            "M": "Married",
            "P": "Polygamous",
            "S": "Single",
            "T": "Domestic Partner",
            "U": "Unmarried",
            "W": "Widowed",
            "UNK": "Unknown",
        }
        return marital_map.get(code.upper(), code)

    def _format_mcode_element(self, element_name: str, system: str, code: str) -> str:
        """Centralized function to format mCODE elements consistently."""
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
        else:
            # Remove URLs and keep only the system identifier
            clean_system = system.split("/")[-1].split(":")[-1].upper()

        return f"(mCODE: {element_name}; {clean_system}:{code})"

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
