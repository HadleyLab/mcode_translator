"""
Patients Summarizer Workflow - Generate natural language summaries from mCODE patient data.

This workflow handles generating comprehensive natural language summaries
from processed mCODE patient data and stores them in CORE Memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.services.summarizer import McodeSummarizer
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.logging_config import get_logger

from .base_workflow import PatientsProcessorWorkflow as BasePatientsProcessorWorkflow, WorkflowResult


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
                    processed_patient["McodeResults"]["natural_language_summary"] = summary

                    # Store in CORE Memory if requested
                    if store_in_memory and self.memory_storage:
                        # Extract mCODE elements if available
                        mcode_elements = patient.get("filtered_mcode_elements", [])
                        demographics = self._extract_patient_demographics(patient)

                        mcode_data = {
                            "original_patient_data": patient,
                            "mcode_mappings": self._convert_to_mappings_format(mcode_elements),
                            "natural_language_summary": summary,
                            "demographics": demographics,
                            "metadata": patient.get("mcode_processing_metadata", {}),
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
                    "stored_in_memory": store_in_memory and self.memory_storage is not None,
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

    def process_single_patient(self, patient: Dict[str, Any], **kwargs) -> WorkflowResult:
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