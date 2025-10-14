"""
Patients Summarizer Workflow - Generate natural language summaries from mCODE patient data.

This workflow handles generating comprehensive natural language summaries
from processed mCODE patient data and stores them in CORE Memory.
"""

from typing import Any, Dict, List

from src.services.summarizer import McodeSummarizer
from src.storage.mcode_memory_storage import OncoCoreMemory

from .base_summarizer import BaseSummarizerWorkflow
from .base_workflow import WorkflowResult


class PatientsSummarizerWorkflow(BaseSummarizerWorkflow):
    """
    Workflow for generating natural language summaries from mCODE patient data.

    Takes processed mCODE patient data and generates comprehensive summaries
    for storage in CORE Memory.
    """

    @property
    def memory_space(self) -> str:
        """Patients summarizers use 'patients_summaries' space."""
        return "patients_summaries"

    def __init__(self, config: Any, memory_storage: OncoCoreMemory):
        """
        Initialize the patients summarizer workflow.

        Args:
            config: Configuration instance
            memory_storage: Core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.summarizer = McodeSummarizer()

    def extract_patient_id(self, patient: Dict[str, Any]) -> str:
        """Extract patient ID from patient data."""
        for entry in patient["entry"]:
            resource = entry["resource"]
            if resource["resourceType"] == "Patient":
                patient_id = resource["id"]
                return str(patient_id)
        raise ValueError("Patient resource not found in bundle")

    def execute(self, **kwargs: Any) -> WorkflowResult:
        """
        Execute the patients summarization workflow.

        Args:
            **kwargs: Workflow parameters including:
                - patients_data: List of patient data to summarize
                - store_in_memory: Whether to store results in CORE memory

        Returns:
            WorkflowResult: Summarization results
        """
        if "patients_data" not in kwargs:
            raise ValueError("patients_data parameter is required")
        patients_data = kwargs["patients_data"]
        store_in_memory = kwargs.get("store_in_memory", False)

        processed_patients = []
        successful_summaries = 0

        for patient in patients_data:
            try:
                if (
                    "mcode_elements" in patient
                    and "natural_language_summary" in patient["mcode_elements"]
                ):
                    summary = patient["mcode_elements"]["natural_language_summary"]
                    patient_id = patient.get("patient_id", self.extract_patient_id(patient))
                else:
                    if "patient_bundle" in patient:
                        patient = self._convert_patient_bundle_to_fhir(patient)
                    elif "original_patient_data" in patient:
                        patient = patient["original_patient_data"]

                    if "entry" not in patient:
                        patient_id = patient["protocolSection"]["identificationModule"]["nctId"]
                        title = patient["protocolSection"]["identificationModule"]["briefTitle"]
                        summary = f"Clinical Trial {patient_id}: {title}"
                    else:
                        patient_id = self.extract_patient_id(patient)
                        # Use the McodeSummarizer to generate the actual summary
                        summary = self.summarizer.create_patient_summary(patient)

                processed_patient = patient.copy()
                if "McodeResults" not in processed_patient:
                    processed_patient["McodeResults"] = {}
                processed_patient["McodeResults"]["natural_language_summary"] = summary

                if store_in_memory:
                    self.memory_storage.store_patient_summary(patient_id, summary)

                processed_patients.append(processed_patient)
                successful_summaries += 1

            except Exception as e:
                # Log error but continue processing other patients
                patient_id = patient.get("patient_id", "unknown")
                print(f"Failed to summarize patient {patient_id}: {e}")
                # Add patient with error summary
                processed_patient = patient.copy()
                if "McodeResults" not in processed_patient:
                    processed_patient["McodeResults"] = {}
                processed_patient["McodeResults"]["natural_language_summary"] = f"Patient {patient_id}: Summary failed - {str(e)}"
                processed_patients.append(processed_patient)

        total_count = len(patients_data)

        return self._create_result(
            success=True,
            data=processed_patients,
            metadata={
                "total_patients": total_count,
                "successful": successful_summaries,
                "stored_in_memory": store_in_memory,
            },
        )

    def _extract_patient_demographics(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient demographics for storage."""
        for entry in patient["entry"]:
            resource = entry["resource"]
            if resource["resourceType"] == "Patient":
                return self._extract_demographics(resource)
        return {}

    def process_single_patient(self, patient: Dict[str, Any], **kwargs: Any) -> WorkflowResult:
        """
        Process a single patient for summarization.

        Args:
            patient: Patient data to summarize
            **kwargs: Additional processing parameters

        Returns:
            WorkflowResult: Summarization result
        """
        result = self.execute(patients_data=[patient], **kwargs)
        if isinstance(result.data, list) and len(result.data) > 0:
            return self._create_result(success=True, data=result.data[0], metadata=result.metadata)
        raise ValueError("No patient data returned from summarization")

    def _convert_patient_bundle_to_fhir(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert custom patient_bundle format to FHIR bundle format."""
        patient_bundle = patient_data["patient_bundle"]

        entries: List[Dict[str, Any]] = []
        fhir_bundle: Dict[str, Any] = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": entries,
        }

        for item in patient_bundle:
            if item["resource_type"] == "Patient":
                entry = {
                    "resource": {
                        "resourceType": "Patient",
                        "id": item["id"],
                        "name": [item.get("name", {})],
                        "gender": item.get("gender", "unknown"),
                        "birthDate": item.get("birth_date", "1900-01-01"),
                    }
                }
                if "race" in item:
                    entry["resource"]["extension"] = []
                    entry["resource"]["extension"].append(
                        {
                            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                            "extension": [{"url": "text", "valueString": item["race"]}],
                        }
                    )
                if "ethnicity" in item:
                    if "extension" not in entry["resource"]:
                        entry["resource"]["extension"] = []
                    entry["resource"]["extension"].append(
                        {
                            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                            "extension": [{"url": "text", "valueString": item["ethnicity"]}],
                        }
                    )
                fhir_bundle["entry"].append(entry)

            elif item["resource_type"] in [
                "Observation",
                "Condition",
                "Procedure",
                "MedicationStatement",
                "MedicationRequest",
            ]:
                if "clinical_data" in item:
                    entry = {"resource": item["clinical_data"]}
                    if item["resource_type"] == "Observation":
                        resource = entry["resource"]
                        code_display = (
                            resource.get("code", {}).get("coding", [{}])[0].get("display", "")
                        )

                        if "estrogen receptor" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault("profile", []).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker-test"
                            )
                        elif "progesterone receptor" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault("profile", []).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker-test"
                            )
                        elif "her2" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault("profile", []).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker-test"
                            )
                        elif "tumor" in code_display.lower() and "stage" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault("profile", []).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-stage-group"
                            )
                        elif "cause of death" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault("profile", []).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cause-of-death"
                            )

                    elif item["resource_type"] == "Condition":
                        resource = entry["resource"]
                        code_display = (
                            resource.get("code", {}).get("coding", [{}])[0].get("display", "")
                        )

                        if "cancer" in code_display.lower() or "neoplasm" in code_display.lower():
                            resource.setdefault("meta", {}).setdefault("profile", []).append(
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-primary-cancer-condition"
                            )

                    fhir_bundle["entry"].append(entry)

        return fhir_bundle

    def _extract_demographics(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patient demographics from FHIR bundle for CORE Memory storage."""
        for entry in patient_data["entry"]:
            resource = entry["resource"]
            if resource["resourceType"] == "Patient":
                demographics = {}
                demographics["patient_id"] = resource["id"]
                demographics["gender"] = resource.get("gender", "")
                demographics["birth_date"] = resource.get("birthDate", "")

                name_data = resource.get("name", [{}])[0]
                if name_data:
                    demographics["first_name"] = " ".join(name_data.get("given", []))
                    demographics["last_name"] = name_data.get("family", "")
                    demographics["full_name"] = (
                        f"{demographics['first_name']} {demographics['last_name']}".strip()
                    )

                return demographics
        return {}

    def _convert_to_mappings_format(self, summary: Any) -> List[Dict[str, Any]]:
        """Convert natural language summary to mCODE mappings format for CORE Memory."""
        summary_text = (
            " ".join(str(s) for s in summary) if isinstance(summary, list) else str(summary)
        )

        mappings = []

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

        if "chemotherapy" in summary_text.lower() or "treatment" in summary_text.lower():
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
