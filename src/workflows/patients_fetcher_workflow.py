"""
Patients Fetcher Workflow - Fetch synthetic patient data from archives.

This workflow handles fetching raw patient data from synthetic patient archives
without any processing or core memory storage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging_config import get_logger
from src.utils.patient_generator import create_patient_generator

from .base_workflow import FetcherWorkflow, WorkflowResult


class PatientsFetcherWorkflow(FetcherWorkflow):
    """
    Workflow for fetching synthetic patient data from archives.

    Fetches raw patient data without processing or storage to core memory.
    """

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the patients fetching workflow.

        Args:
            **kwargs: Workflow parameters including:
                - archive_path: Path to patient archive
                - patient_id: Specific patient ID to fetch
                - limit: Maximum number of patients to fetch
                - output_path: Where to save results

        Returns:
            WorkflowResult: Fetch results
        """
        try:
            # Extract parameters
            archive_path = kwargs.get("archive_path")
            patient_id = kwargs.get("patient_id")
            limit = kwargs.get("limit", 10)
            output_path = kwargs.get("output_path")

            # Validate inputs
            if not archive_path:
                return self._create_result(
                    success=False,
                    error_message="Archive path is required for patient fetching.",
                )

            # Execute fetch
            if patient_id:
                results = self._fetch_single_patient(archive_path, patient_id)
            else:
                results = self._fetch_multiple_patients(archive_path, limit)

            # Save results if output path provided, otherwise output to stdout
            if output_path and results["success"]:
                self._save_results(results["data"], output_path)
            elif results["success"]:
                self._output_to_stdout(results["data"])

            return self._create_result(
                success=results["success"],
                data=results["data"],
                error_message=results.get("error"),
                metadata={
                    "fetch_type": results.get("type", "unknown"),
                    "total_fetched": len(results.get("data", [])),
                    "archive_path": archive_path,
                    "output_path": str(output_path) if output_path else None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "patients fetching")

    def _fetch_single_patient(
        self, archive_path: str, patient_id: str
    ) -> Dict[str, Any]:
        """Fetch a single patient by ID."""
        try:
            self.logger.info(f"📥 Fetching patient {patient_id} from {archive_path}")

            # Create patient generator
            generator = create_patient_generator(
                archive_identifier=archive_path, config=self.config
            )

            # Get specific patient
            patient_data = generator.get_patient_by_id(patient_id)

            if not patient_data:
                return {
                    "success": False,
                    "error": f"Patient {patient_id} not found in archive",
                    "data": [],
                }

            self.logger.info(f"✅ Successfully fetched patient: {patient_id}")

            return {
                "success": True,
                "type": "single_patient",
                "data": [patient_data],
                "metadata": {"patient_id": patient_id, "archive_path": archive_path},
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch patient {patient_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch patient {patient_id}: {e}",
                "data": [],
            }

    def _fetch_multiple_patients(self, archive_path: str, limit: int) -> Dict[str, Any]:
        """Fetch multiple patients from archive."""
        try:
            self.logger.info(f"📥 Fetching up to {limit} patients from {archive_path}")

            # Create patient generator
            generator = create_patient_generator(
                archive_identifier=archive_path, config=self.config
            )

            # Generator is already initialized and file list is loaded

            # Get patients (limit the number)
            patients = []
            count = 0

            for patient in generator:
                if count >= limit:
                    break
                patients.append(patient)
                count += 1

                if count % 10 == 0:  # Progress logging
                    self.logger.info(f"📊 Fetched {count} patients...")

            if not patients:
                return {
                    "success": False,
                    "error": f"No patients found in archive: {archive_path}",
                    "data": [],
                }

            self.logger.info(f"✅ Successfully fetched {len(patients)} patients")

            return {
                "success": True,
                "type": "multiple_patients",
                "data": patients,
                "metadata": {
                    "archive_path": archive_path,
                    "requested_limit": limit,
                    "actual_count": len(patients),
                    "total_available": (
                        len(generator) if hasattr(generator, "__len__") else "unknown"
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch patients from {archive_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch patients: {e}",
                "data": [],
            }

    def _save_results(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save fetch results to file in NDJSON format."""
        try:
            output_file = Path(output_path)

            # Create output directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save as NDJSON (one JSON object per line)
            with open(output_file, "w", encoding="utf-8") as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

            self.logger.info(f"💾 Patient data saved to: {output_file} (NDJSON format)")

        except Exception as e:
            self.logger.error(f"Failed to save patient data to {output_path}: {e}")
            raise

    def _output_to_stdout(self, data: List[Dict[str, Any]]) -> None:
        """Output fetch results to stdout in NDJSON format."""
        try:
            import sys
            for item in data:
                json.dump(item, sys.stdout, ensure_ascii=False)
                sys.stdout.write('\n')
            sys.stdout.flush()

            self.logger.info(f"📤 Patient data written to stdout: {len(data)} records (NDJSON format)")

        except Exception as e:
            self.logger.error(f"Failed to output patient data to stdout: {e}")
            raise

    def list_available_archives(self) -> List[str]:
        """
        List available patient data archives.

        Returns:
            List of available archive identifiers
        """
        # This would typically scan the data directory for available archives
        # For now, return common archive types
        return [
            "breast_cancer_10_years",
            "breast_cancer_lifetime",
            "mixed_cancer_10_years",
            "mixed_cancer_lifetime",
        ]

    def get_archive_info(self, archive_path: str) -> Dict[str, Any]:
        """
        Get information about a patient archive.

        Args:
            archive_path: Path to the archive

        Returns:
            Dict with archive information
        """
        try:
            generator = create_patient_generator(
                archive_identifier=archive_path, config=self.config
            )

            return {
                "archive_path": archive_path,
                "total_patients": (
                    len(generator) if hasattr(generator, "__len__") else "unknown"
                ),
                "patient_generator_type": type(generator).__name__,
            }

        except Exception as e:
            self.logger.error(f"Failed to get archive info for {archive_path}: {e}")
            return {"archive_path": archive_path, "error": str(e)}
