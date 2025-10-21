"""
Patients Fetcher Workflow - Fetch synthetic patient data from archives.

This workflow handles fetching raw patient data from synthetic patient archives
without any processing or core memory storage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from utils.patient_generator import create_patient_generator

from .base_workflow import FetcherWorkflow, WorkflowResult


class PatientsFetcherWorkflow(FetcherWorkflow):
    """
    Workflow for fetching synthetic patient data from archives.

    Fetches raw patient data without processing or storage to core memory.
    """

    def execute(self, **kwargs: Any) -> WorkflowResult:
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
        archive_path = kwargs.get("archive_path")
        patient_id = kwargs.get("patient_id")
        limit = kwargs.get("limit")
        output_path = kwargs.get("output_path")

        if not archive_path:
            raise ValueError("Archive path is required for patient fetching.")

        if patient_id:
            if limit is not None:
                raise ValueError("Cannot specify both patient_id and limit")
            data = self._fetch_single_patient(archive_path, patient_id)
            fetch_type = "single_patient"
        else:
            if limit is None:
                raise ValueError("Limit is required when not fetching single patient")
            data = self._fetch_multiple_patients(archive_path, limit)
            fetch_type = "multiple_patients"

        if output_path:
            self._save_results(data, output_path)
        else:
            self._output_to_stdout(data)

        return self._create_result(
            success=True,
            data=data,
            metadata={
                "fetch_type": fetch_type,
                "total_fetched": len(data),
                "archive_path": archive_path,
                "output_path": str(output_path) if output_path else None,
            },
        )

    def _fetch_single_patient(self, archive_path: str, patient_id: str) -> List[Dict[str, Any]]:
        """Fetch a single patient by ID."""
        generator = create_patient_generator(archive_identifier=archive_path, config=self.config)

        patient_data = generator.get_patient_by_id(patient_id)

        if not patient_data:
            raise ValueError(f"Patient {patient_id} not found in archive")

        return [patient_data]

    def _fetch_multiple_patients(self, archive_path: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch multiple patients from archive."""
        generator = create_patient_generator(archive_identifier=archive_path, config=self.config)

        patients = []
        count = 0

        for patient in generator:
            if count >= limit:
                break
            patients.append(patient)
            count += 1

        if not patients:
            raise ValueError(f"No patients found in archive: {archive_path}")

        return patients

    def _save_results(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """Save fetch results to file in NDJSON format."""
        output_file = Path(output_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    def _output_to_stdout(self, data: List[Dict[str, Any]]) -> None:
        """Output fetch results to stdout in NDJSON format."""
        import sys

        for item in data:
            json.dump(item, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
        sys.stdout.flush()

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
        generator = create_patient_generator(archive_identifier=archive_path, config=self.config)

        return {
            "archive_path": archive_path,
            "total_patients": (len(generator) if hasattr(generator, "__len__") else "unknown"),
            "patient_generator_type": type(generator).__name__,
        }
