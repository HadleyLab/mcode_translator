"""
Patients Summarizer Workflow - Generate natural language summaries from mCODE patient data.

This workflow handles generating comprehensive natural language summaries
from processed mCODE patient data and stores them in CORE Memory.
"""

from typing import Any, Dict, List, Optional

from src.services.summarizer import McodeSummarizer
from src.storage.mcode_memory_storage import McodeMemoryStorage

from .base_workflow import PatientsProcessorWorkflow as BasePatientsProcessorWorkflow
from .base_workflow import WorkflowResult


class PatientsSummarizerWorkflow(BasePatientsProcessorWorkflow):
    """
    Workflow for generating natural language summaries from mCODE patient data.

    Takes processed mCODE patient data and generates comprehensive summaries
    for storage in CORE Memory.
    """

    def __init__(self, config, memory_storage: Optional[McodeMemoryStorage] = None):
        """
        Initialize the patients summarizer workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.summarizer = McodeSummarizer()

    def extract_patient_id(self, patient: Dict[str, Any]) -> str:
        """Extract patient ID from patient data."""
        try:
            # Try to extract from Patient resource
            for entry in patient.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Patient":
                    patient_id = resource.get("id")
                    if patient_id:
                        return str(patient_id)
        except Exception as e:
            self.logger.warning(f"Error extracting patient ID: {e}")

        # Fallback to hash-based ID
        import hashlib

        return f"patient_{hashlib.md5(str(patient).encode('utf-8')).hexdigest()[:8]}"

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the patients summarization workflow.

        Args:
            **kwargs: Workflow parameters including:
                - patients_data: List of patient data to summarize
                - store_in_memory: Whether to store results in CORE memory (default: False)

        Returns:
            WorkflowResult: Summarization results
        """
        try:
            self.logger.info("Starting patients summarizer workflow execution")

            # Extract parameters
            patients_data = kwargs.get("patients_data", [])
            store_in_memory = kwargs.get("store_in_memory", False)

            if not patients_data:
                return self._create_result(
                    success=False,
                    error_message="No patient data provided for summarization.",
                )

            # Generate summaries
            processed_patients = []
            successful_count = 0
            failed_count = 0

            self.logger.info(
                f"📝 Generating summaries for {len(patients_data)} patients"
            )

            for patient in patients_data:
                try:
                    # Check if this is already processed mCODE data with summary
                    if (
                        "mcode_elements" in patient
                        and "natural_language_summary"
                        in patient.get("mcode_elements", {})
                    ):
                        # Use existing summary
                        summary = patient["mcode_elements"]["natural_language_summary"]
                        patient_id = patient.get(
                            "patient_id", self.extract_patient_id(patient)
                        )
                        self.logger.info(
                            f"Using existing summary for patient {patient_id}"
                        )

                        # Create processed patient with existing summary
                        processed_patient = patient.copy()
                        if "McodeResults" not in processed_patient:
                            processed_patient["McodeResults"] = {}
                        processed_patient["McodeResults"][
                            "natural_language_summary"
                        ] = summary

                    else:
                        # Generate new summary from raw data
                        # Convert patient_bundle format to FHIR bundle format if needed
                        if "patient_bundle" in patient:
                            # Convert from custom format to FHIR bundle
                            patient = self._convert_patient_bundle_to_fhir(patient)
                            self.logger.info(
                                f"Converted bundle has {len(patient.get('entry', []))} entries"
                            )
                            for i, entry in enumerate(patient.get("entry", [])):
                                resource_type = entry.get("resource", {}).get(
                                    "resourceType"
                                )
                                self.logger.info(f"Entry {i}: {resource_type}")
                                if resource_type == "Patient":
                                    patient_res = entry["resource"]
                                    self.logger.info(
                                        f"Patient resource: name={patient_res.get('name')}, gender={patient_res.get('gender')}, birthDate={patient_res.get('birthDate')}"
                                    )

                        # Handle original_patient_data if present
                        elif "original_patient_data" in patient:
                            patient = patient["original_patient_data"]
                            self.logger.info(
                                f"Using original_patient_data, has {len(patient.get('entry', []))} entries"
                            )

                        # Debug: Check if patient has entry
                        if "entry" not in patient:
                            self.logger.error(
                                f"Patient data missing 'entry' key: {list(patient.keys())}"
                            )
                            raise ValueError("Patient data is missing 'entry' key")

                        patient_id = self.extract_patient_id(patient)

                        # Generate natural language summary
                        summary = self.summarizer.create_patient_summary(patient)
                        self.logger.debug(
                            f"Generated summary for patient {patient_id}: {summary[:100]}..."
                        )

                        # Create processed patient with summary
                        processed_patient = patient.copy()
                        if "McodeResults" not in processed_patient:
                            processed_patient["McodeResults"] = {}
                        processed_patient["McodeResults"][
                            "natural_language_summary"
                        ] = summary

                    # Debug: Check Patient resource structure
                    patient_resource = None
                    for entry in patient.get("entry", []):
                        if entry.get("resource", {}).get("resourceType") == "Patient":
                            patient_resource = entry["resource"]
                            break

                    if patient_resource:
                        self.logger.debug(
                            f"Patient resource keys: {list(patient_resource.keys())}"
                        )
                        self.logger.debug(
                            f"Patient name: {patient_resource.get('name')}"
                        )
                        self.logger.debug(
                            f"Patient gender: {patient_resource.get('gender')}"
                        )
                        self.logger.debug(
                            f"Patient birthDate: {patient_resource.get('birthDate')}"
                        )
                    else:
                        self.logger.error(
                            "No Patient resource found in converted bundle"
                        )

                    patient_id = self.extract_patient_id(patient)

                    # Generate natural language summary
                    summary = self.summarizer.create_patient_summary(patient)
                    self.logger.debug(
                        f"Generated summary for patient {patient_id}: {summary[:100]}..."
                    )

                    # Create processed patient with summary
                    processed_patient = patient.copy()
                    if "McodeResults" not in processed_patient:
                        processed_patient["McodeResults"] = {}
                    processed_patient["McodeResults"][
                        "natural_language_summary"
                    ] = summary

                    # Store in CORE Memory if requested
                    if store_in_memory and self.memory_storage:
                        # Handle storage for both existing and newly generated summaries
                        if "mcode_elements" in patient:
                            # Use existing mCODE data
                            mcode_elements = patient.get("mcode_elements", {}).get(
                                "mcode_mappings", []
                            )
                            demographics = patient.get("mcode_elements", {}).get(
                                "demographics", {}
                            )
                            metadata = patient.get("mcode_elements", {}).get(
                                "metadata", {}
                            )
                        else:
                            # Extract from newly processed data
                            mcode_elements = patient.get("filtered_mcode_elements", [])
                            demographics = self._extract_patient_demographics(patient)
                            metadata = patient.get("mcode_processing_metadata", {})

                        mcode_data = {
                            "original_patient_data": patient,
                            "mcode_mappings": self._convert_to_mappings_format(
                                mcode_elements
                            ),
                            "natural_language_summary": summary,
                            "demographics": demographics,
                            "metadata": metadata,
                        }

                        success = self.memory_storage.store_patient_mcode_summary(
                            patient_id, mcode_data
                        )
                        if success:
                            self.logger.info(f"✅ Stored patient {patient_id} summary")
                        else:
                            self.logger.warning(
                                f"❌ Failed to store patient {patient_id} summary"
                            )

                    processed_patients.append(processed_patient)
                    successful_count += 1

                except Exception as e:
                    self.logger.error(f"Failed to summarize patient: {e}")
                    # Add error information to patient
                    error_patient = patient.copy()
                    error_patient["SummaryError"] = str(e)
                    processed_patients.append(error_patient)
                    failed_count += 1

            # Calculate success rate
            total_count = len(patients_data)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Summarization complete: {successful_count}/{total_count} patients successful"
            )

            return self._create_result(
                success=successful_count > 0,
                data=processed_patients,
                metadata={
                    "total_patients": total_count,
                    "successful": successful_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "stored_in_memory": store_in_memory
                    and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "patients summarization")

    def _extract_patient_demographics(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient demographics for storage."""
        try:
            # Try to extract from Patient resource
            for entry in patient.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Patient":
                    return self._extract_demographics(resource)
        except Exception as e:
            self.logger.error(f"Error extracting patient demographics: {e}")

        return {}

    def process_single_patient(
        self, patient: Dict[str, Any], **kwargs
    ) -> WorkflowResult:
        """
        Process a single patient for summarization.

        Args:
            patient: Patient data to summarize
            **kwargs: Additional processing parameters

        Returns:
            WorkflowResult: Summarization result
        """
        result = self.execute(patients_data=[patient], **kwargs)

        # Return single patient result
        if result.success and result.data:
            return self._create_result(
                success=True, data=result.data[0], metadata=result.metadata
            )
        else:
            return result

    def _convert_patient_bundle_to_fhir(
        self, patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert custom patient_bundle format to FHIR bundle format."""
        patient_bundle = patient_data.get("patient_bundle", [])

        # Create FHIR bundle
        fhir_bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}

        for item in patient_bundle:
            if item.get("resource_type") == "Patient":
                # Convert Patient format
                entry = {
                    "resource": {
                        "resourceType": "Patient",
                        "id": item.get("id"),
                        "name": [item.get("name", {})],
                        "gender": item.get("gender") or "unknown",
                        "birthDate": item.get("birth_date") or "1900-01-01",
                    }
                }
                # Add extensions if present
                if "race" in item:
                    entry["resource"]["extension"] = entry["resource"].get(
                        "extension", []
                    )
                    entry["resource"]["extension"].append(
                        {
                            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                            "extension": [{"url": "text", "valueString": item["race"]}],
                        }
                    )
                if "ethnicity" in item:
                    entry["resource"]["extension"] = entry["resource"].get(
                        "extension", []
                    )
                    entry["resource"]["extension"].append(
                        {
                            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                            "extension": [
                                {"url": "text", "valueString": item["ethnicity"]}
                            ],
                        }
                    )
                fhir_bundle["entry"].append(entry)

            elif item.get("resource_type") in [
                "Observation",
                "Condition",
                "Procedure",
                "MedicationStatement",
                "MedicationRequest",
            ]:
                # Use clinical_data directly for other resources
                if "clinical_data" in item:
                    entry = {"resource": item["clinical_data"]}
                    # Add mCODE profiles for cancer-related observations
                    if item["resource_type"] == "Observation":
                        resource = entry["resource"]
                        code_display = (
                            resource.get("code", {})
                            .get("coding", [{}])[0]
                            .get("display", "")
                        )

                        # Add appropriate mCODE profiles based on content
                        if "estrogen receptor" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault(
                                "profile", []
                            ).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker-test"
                            )
                        elif "progesterone receptor" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault(
                                "profile", []
                            ).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker-test"
                            )
                        elif "her2" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault(
                                "profile", []
                            ).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker-test"
                            )
                        elif (
                            "tumor" in code_display.lower()
                            and "stage" in code_display.lower()
                        ):
                            resource.setdefault("meta", {}).setdefault(
                                "profile", []
                            ).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-stage-group"
                            )
                        elif "cause of death" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault(
                                "profile", []
                            ).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cause-of-death"
                            )

                    elif item["resource_type"] == "Condition":
                        resource = entry["resource"]
                        code_display = (
                            resource.get("code", {})
                            .get("coding", [{}])[0]
                            .get("display", "")
                        )

                        # Add mCODE primary cancer condition profile
                        if (
                            "cancer" in code_display.lower()
                            or "neoplasm" in code_display.lower()
                        ):
                            resource.setdefault("meta", {}).setdefault(
                                "profile", []
                            ).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-primary-cancer-condition"
                            )

                    fhir_bundle["entry"].append(entry)

        return fhir_bundle

    def _extract_demographics(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient demographics from FHIR bundle for CORE Memory storage."""
        demographics = {}

        # Find Patient resource
        for entry in patient_data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patient_resource = resource

                # Extract basic demographics
                demographics["patient_id"] = patient_resource.get("id", "")
                demographics["gender"] = patient_resource.get("gender", "")
                demographics["birth_date"] = patient_resource.get("birthDate", "")

                # Extract name
                name_data = patient_resource.get("name", [{}])[0]
                if name_data:
                    demographics["first_name"] = " ".join(name_data.get("given", []))
                    demographics["last_name"] = name_data.get("family", "")
                    demographics["full_name"] = (
                        f"{demographics['first_name']} {demographics['last_name']}".strip()
                    )

                break

        return demographics

    def _convert_to_mappings_format(self, summary) -> List[Dict[str, Any]]:
        """Convert natural language summary to mCODE mappings format for CORE Memory."""
        # Handle different summary formats
        if isinstance(summary, list):
            summary_text = " ".join(str(s) for s in summary)
        else:
            summary_text = str(summary)

        # This is a simplified conversion - in a real implementation, you might use
        # NLP or LLM to extract structured mCODE elements from the summary
        mappings = []

        # Extract basic patient information
        if "Patient" in summary_text:
            mappings.append(
                {
                    "element_type": "Patient",
                    "code": "",
                    "display": "Patient",
                    "system": "",
                    "confidence_score": 1.0,
                    "evidence_text": "Patient resource found in FHIR bundle",
                }
            )

        # Extract cancer conditions
        if "cancer" in summary_text.lower() or "neoplasm" in summary_text.lower():
            mappings.append(
                {
                    "element_type": "CancerCondition",
                    "code": "",
                    "display": "Malignant neoplasm",
                    "system": "SNOMED",
                    "confidence_score": 0.8,
                    "evidence_text": "Cancer condition mentioned in clinical summary",
                }
            )

        # Extract procedures/treatments
        if (
            "chemotherapy" in summary_text.lower()
            or "treatment" in summary_text.lower()
        ):
            mappings.append(
                {
                    "element_type": "CancerTreatment",
                    "code": "",
                    "display": "Chemotherapy",
                    "system": "",
                    "confidence_score": 0.7,
                    "evidence_text": "Treatment mentioned in clinical summary",
                }
            )

        return mappings
