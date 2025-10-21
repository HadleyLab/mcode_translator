"""
Patients Processor Workflow - Process patient data with mCODE mapping.

This workflow handles processing patient data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

from typing import Any, Dict, List, Optional

from services.clinical_note_generator import ClinicalNoteGenerator
from services.data_enrichment import DataEnrichmentService
from services.demographics_extractor import DemographicsExtractor
from services.fhir_extractors import FHIRResourceExtractors
from services.llm.service import LLMService
from services.summarizer import McodeSummarizer
from shared.data_quality_validator import DataQualityValidator
from shared.models import (
    McodeValidator,
)
from storage.mcode_memory_storage import OncoCoreMemory
from utils.concurrency import AsyncTaskQueue, create_task
from utils.logging_config import Loggable

from .base_workflow import PatientsProcessorWorkflow as BasePatientsProcessorWorkflow
from .base_workflow import WorkflowResult


class PatientsProcessorWorkflow(BasePatientsProcessorWorkflow, Loggable):
    """
    Workflow for processing patient data with mCODE mapping.

    Processes patient data and stores mCODE summaries to CORE Memory.
    """

    def __init__(self, config: Any, memory_storage: Optional[OncoCoreMemory] = None):
        """
        Initialize the patients processor workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        if memory_storage is None:
            raise ValueError("memory_storage is required")
        super().__init__(config, memory_storage)
        Loggable.__init__(self)
        self.summarizer = McodeSummarizer()
        self.clinical_note_generator = ClinicalNoteGenerator()
        self.demographics_extractor = DemographicsExtractor()
        self.fhir_extractors = FHIRResourceExtractors()
        self.mcode_validator = McodeValidator()  # Add structured validation
        self.quality_validator = DataQualityValidator()  # Add data quality validation

        # Initialize data enrichment service
        llm_service = LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")
        self.data_enrichment = DataEnrichmentService(config, llm_service)

    async def execute_async(self, **kwargs: Any) -> WorkflowResult:
        """
        Execute the patients processing workflow asynchronously.

        By default, does NOT store results to CORE memory. Use store_in_memory=True to enable.

        Args:
            **kwargs: Workflow parameters including:
                 - patients_data: List of patient data to process
                 - trials_criteria: Optional trial criteria for filtering
                 - store_in_memory: Whether to store results in CORE memory (default: False)

        Returns:
            WorkflowResult: Processing results
        """
        # Extract parameters
        patients_data = kwargs.get("patients_data", [])
        trials_criteria = kwargs.get("trials_criteria")

        # Default to NOT store in CORE memory - use --ingest to enable
        store_in_memory = False

        if not patients_data:
            raise ValueError("No patient data provided for processing.")

        # Process patients concurrently using AsyncTaskQueue
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
        task_queue = AsyncTaskQueue(
            max_concurrent=8, name="PatientProcessorQueue"
        )  # Use 8 workers for patient processing
        task_results = await task_queue.execute_tasks(tasks)

        # Process results and validate data quality
        processed_patients = []
        successful_count = 0
        failed_count = 0
        quality_reports = []

        for result in task_results:
            if result.success:
                patient_data = result.result
                patient_id = self._extract_patient_id(patient_data.get("original_patient_data", patient_data))

                # Validate data quality
                quality_report = self.quality_validator.validate_patient_data(
                    patient_data.get("original_patient_data", patient_data),
                    patient_data.get("filtered_mcode_elements", {})
                )
                quality_reports.append({
                    "patient_id": patient_id,
                    "report": quality_report
                })

                # Check if processing can proceed based on quality validation
                if not quality_report.can_proceed:
                    self.logger.warning(
                        f"❌ Patient {patient_id} failed quality validation. "
                        f"Critical issues: {quality_report.critical_issues}"
                    )
                    failed_count += 1
                    continue

                processed_patients.append(patient_data)
                successful_count += 1

                # Log quality metrics
                self.logger.info(
                    f"✅ Patient {patient_id} quality validated: "
                    f"{quality_report.coverage_percentage:.1f}% coverage, "
                    f"score: {quality_report.completeness_score:.2f}"
                )
            else:
                raise RuntimeError(f"Task {result.task_id} failed: {result.error}")

        # Calculate success rate
        total_count = len(patients_data)
        success_rate = successful_count / total_count if total_count > 0 else 0

        # Generate aggregate quality report
        if quality_reports:
            total_critical = sum(r["report"].critical_issues for r in quality_reports)
            total_warnings = sum(r["report"].warning_issues for r in quality_reports)
            avg_coverage = sum(r["report"].coverage_percentage for r in quality_reports) / len(quality_reports)
            avg_completeness = sum(r["report"].completeness_score for r in quality_reports) / len(quality_reports)

            self.logger.info(
                f"📊 Quality Summary: {successful_count}/{total_count} patients passed validation. "
                f"Avg coverage: {avg_coverage:.1f}%, Avg completeness: {avg_completeness:.2f}. "
                f"Total issues: {total_critical} critical, {total_warnings} warnings"
            )

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
                "quality_reports": quality_reports,
                "quality_summary": {
                    "total_critical_issues": sum(r["report"].critical_issues for r in quality_reports),
                    "total_warning_issues": sum(r["report"].warning_issues for r in quality_reports),
                    "average_coverage": sum(r["report"].coverage_percentage for r in quality_reports) / len(quality_reports) if quality_reports else 0,
                    "average_completeness": sum(r["report"].completeness_score for r in quality_reports) / len(quality_reports) if quality_reports else 0,
                } if quality_reports else None,
            },
        )

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
        import asyncio

        return asyncio.run(self.execute_async(**kwargs))

    async def _process_single_patient(
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
        self.logger.info(f"Processing patient {patient_index+1}")

        # Extract mCODE elements from patient data
        patient_mcode = self._extract_patient_mcode_elements(patient)

        # Filter based on trial criteria if provided
        if trials_criteria:
            filtered_mcode = self._filter_by_trial_criteria(patient_mcode, trials_criteria)
            self.logger.info(
                f"Filtered patient mCODE elements: {len(filtered_mcode)}/{len(patient_mcode)}"
            )
        else:
            filtered_mcode = patient_mcode

        # Apply data enrichment to improve completeness
        enriched_patient_data = await self.data_enrichment.enrich_patient_data(
            patient, filtered_mcode
        )

        # Create processed patient data
        processed_patient = enriched_patient_data.copy()
        processed_patient["filtered_mcode_elements"] = filtered_mcode
        processed_patient["mcode_processing_metadata"] = {
            "original_elements_count": len(patient_mcode),
            "filtered_elements_count": len(filtered_mcode),
            "enriched_elements_count": len(enriched_patient_data.get("enriched_mcode_elements", {})),
            "trial_criteria_applied": trials_criteria is not None,
        }

        # Store to CORE memory if requested
        if store_in_memory and self.memory_storage:
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

            if not patient_resource:
                raise ValueError("No Patient resource found in patient data")

            demographics = self.demographics_extractor.extract_demographics(patient_resource)
            self.logger.debug(f"Extracted demographics: {demographics}")

            # Add demographic info from mCODE mappings if not already extracted
            if "gender" not in demographics and "PatientSex" in filtered_mcode:
                patient_sex = filtered_mcode["PatientSex"]
                if isinstance(patient_sex, dict):
                    demographics["gender"] = patient_sex.get("display")
                else:
                    demographics["gender"] = str(patient_sex)

            mcode_data = {
                "original_patient_data": patient,  # Include original patient data for summarizer
                "mcode_mappings": self._convert_to_mappings_format(filtered_mcode),
                "demographics": demographics,
                "metadata": processed_patient.get("mcode_processing_metadata", {}),
            }

            success = self.memory_storage.store_patient_summary(patient_id, str(mcode_data))
            if not success:
                raise RuntimeError(f"Failed to store patient {patient_id} mCODE summary")

            self.logger.info(f"✅ Stored patient {patient_id} mCODE summary")

        return processed_patient

    def _extract_patient_mcode_elements(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract mCODE elements from patient FHIR Bundle.

        This is a simplified version - in practice, this would use
        the full mCODE extraction logic from the original codebase.
        """
        mcode_elements = {}

        entries = patient.get("entry", [])

        # Process each entry in the patient bundle
        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "Patient":
                # Extract demographics
                mcode_elements.update(self.demographics_extractor.extract_demographics(resource))
            elif resource_type == "Condition":
                # Extract conditions as mCODE CancerCondition or ComorbidCondition
                condition_data = self.fhir_extractors.extract_condition_mcode(resource)
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
                immunization_data = self.fhir_extractors.extract_immunization_mcode(resource)
                if immunization_data:
                    if "Immunization" not in mcode_elements:
                        mcode_elements["Immunization"] = []
                    mcode_elements["Immunization"].append(immunization_data)

            elif resource_type == "FamilyMemberHistory":
                # Extract family history as mCODE elements
                family_data = self.fhir_extractors.extract_family_history_mcode(resource)
                if family_data:
                    if "FamilyMemberHistory" not in mcode_elements:
                        mcode_elements["FamilyMemberHistory"] = []
                    mcode_elements["FamilyMemberHistory"].append(family_data)

            elif resource_type == "Observation":
                # Extract observations as various mCODE elements
                observation_data = self.fhir_extractors.extract_observation_mcode(resource)
                if observation_data:
                    mcode_elements.update(observation_data)

                # Extract comprehensive observations (performance status, vitals, labs)
                comprehensive_obs = self.fhir_extractors.extract_observation_mcode_comprehensive(
                    resource
                )
                if comprehensive_obs:
                    mcode_elements.update(comprehensive_obs)
            elif resource_type == "Procedure":
                # Extract procedures as mCODE CancerRelatedSurgicalProcedure
                procedure_data = self.fhir_extractors.extract_procedure_mcode(resource)
                if procedure_data:
                    if "CancerRelatedSurgicalProcedure" not in mcode_elements:
                        mcode_elements["CancerRelatedSurgicalProcedure"] = []
                    mcode_elements["CancerRelatedSurgicalProcedure"].append(procedure_data)

        return mcode_elements

    def _extract_observation_mcode_comprehensive(
        self, observation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract comprehensive mCODE elements from Observation resource."""
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
                    "interpretation": coding_val.get("display"),
                }
            elif "karnofsky" in display:
                elements["KarnofskyPerformanceStatus"] = {
                    "system": system,
                    "code": coding_val.get("code"),
                    "display": coding_val.get("display"),
                    "interpretation": coding_val.get("display"),
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
                self.logger.debug(f"Filtering out patient mCODE element: {element_type}")

        return filtered

    def _convert_to_mappings_format(self, mcode_elements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert mCODE elements dict to mappings format expected by storage."""
        mappings = []

        for element_name, element_data in mcode_elements.items():
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

        return mappings

    def _generate_natural_language_summary(
        self,
        patient_id: str,
        mcode_elements: Dict[str, Any],
        demographics: Dict[str, Any],
    ) -> str:
        """Generate clinical note-style natural language summary for CORE
        knowledge graph entity extraction."""
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
