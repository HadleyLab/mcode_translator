"""
Data factories for generating test data with various scenarios.
Provides realistic test datasets for both valid and invalid cases.
"""

import json
import random
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class TrialFactory:
    """Factory for generating clinical trial test data."""

    @staticmethod
    def create_basic_trial(
        nct_id: str = None, title: str = None, status: str = "Recruiting", **kwargs
    ) -> Dict[str, Any]:
        """Create a basic clinical trial."""
        if nct_id is None:
            nct_id = f"NCT{random.randint(10000000, 99999999)}"

        trial = {
            "nct_id": nct_id,
            "title": title or f"Test Trial {nct_id}",
            "status": status,
            "phases": kwargs.get("phases", ["Phase 2"]),
            "conditions": kwargs.get("conditions", ["Cancer"]),
            "eligibility": {"criteria": kwargs.get("criteria", "Age >= 18 years")},
            "interventions": kwargs.get(
                "interventions",
                [
                    {
                        "type": "Drug",
                        "name": "Test Drug",
                        "description": "Experimental therapy",
                    }
                ],
            ),
            "locations": kwargs.get(
                "locations",
                [
                    {
                        "facility": "Test Hospital",
                        "city": "Test City",
                        "state": "Test State",
                        "country": "United States",
                    }
                ],
            ),
        }

        # Add any additional fields
        trial.update(kwargs)
        return trial

    @staticmethod
    def create_breast_cancer_trial(**kwargs) -> Dict[str, Any]:
        """Create a breast cancer specific trial."""
        return TrialFactory.create_basic_trial(
            title="Phase 2 Study of Targeted Therapy in Breast Cancer",
            conditions=["Breast Cancer", "HER2 Positive Breast Cancer"],
            criteria="Inclusion: Female patients with HER2+ breast cancer, Age 18-75\nExclusion: Prior chemotherapy, Brain metastases",
            interventions=[
                {
                    "type": "Drug",
                    "name": "Trastuzumab",
                    "description": "HER2-targeted monoclonal antibody",
                }
            ],
            **kwargs,
        )

    @staticmethod
    def create_lung_cancer_trial(**kwargs) -> Dict[str, Any]:
        """Create a lung cancer specific trial."""
        return TrialFactory.create_basic_trial(
            title="Immunotherapy Trial for Advanced Lung Cancer",
            conditions=["Non-Small Cell Lung Cancer", "Adenocarcinoma"],
            criteria="Inclusion: Stage IIIB/IV NSCLC, ECOG 0-1\nExclusion: EGFR mutation positive, Prior immunotherapy",
            interventions=[
                {
                    "type": "Drug",
                    "name": "Pembrolizumab",
                    "description": "PD-1 inhibitor immunotherapy",
                }
            ],
            **kwargs,
        )

    @staticmethod
    def create_invalid_trial(missing_field: str = None) -> Dict[str, Any]:
        """Create an invalid trial for testing error handling."""
        trial = TrialFactory.create_basic_trial()

        if missing_field:
            trial.pop(missing_field, None)

        return trial

    @staticmethod
    def create_large_trial(
        num_conditions: int = 50, num_locations: int = 100
    ) -> Dict[str, Any]:
        """Create a large trial for performance testing."""
        conditions = [f"Condition {i}" for i in range(num_conditions)]
        locations = [
            {
                "facility": f"Hospital {i}",
                "city": f"City {i}",
                "state": f"State {i % 10}",
                "country": "United States",
            }
            for i in range(num_locations)
        ]

        return TrialFactory.create_basic_trial(
            title="Large Multi-Center Clinical Trial",
            conditions=conditions,
            locations=locations,
            criteria="Complex eligibility criteria with multiple conditions",
        )


@dataclass
class PatientFactory:
    """Factory for generating patient test data."""

    @staticmethod
    def create_basic_patient(
        patient_id: str = None, gender: str = None, birth_date: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Create a basic patient bundle."""
        if patient_id is None:
            patient_id = f"patient_{random.randint(1000, 9999)}"

        if gender is None:
            gender = random.choice(["male", "female"])

        if birth_date is None:
            # Generate random birth date between 1920 and 2000
            year = random.randint(1920, 2000)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            birth_date = f"{year:04d}-{month:02d}-{day:02d}"

        patient = {
            "resourceType": "Bundle",
            "id": f"bundle_{patient_id}",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_id,
                        "gender": gender,
                        "birthDate": birth_date,
                        "extension": kwargs.get("extensions", []),
                    }
                }
            ],
        }

        # Add conditions if specified
        conditions = kwargs.get("conditions", [])
        for i, condition in enumerate(conditions):
            patient["entry"].append(
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": f"condition_{i}",
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "code": condition.get(
                            "code",
                            {
                                "coding": [
                                    {
                                        "system": "http://snomed.info/sct",
                                        "code": "12345",
                                        "display": "Test Condition",
                                    }
                                ]
                            },
                        ),
                        "clinicalStatus": condition.get(
                            "clinical_status",
                            {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                        "code": "active",
                                    }
                                ]
                            },
                        ),
                    }
                }
            )

        return patient

    @staticmethod
    def create_breast_cancer_patient(**kwargs) -> Dict[str, Any]:
        """Create a patient with breast cancer."""
        return PatientFactory.create_basic_patient(
            gender="female",
            conditions=[
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "254837009",
                                "display": "Malignant neoplasm of breast",
                            }
                        ]
                    },
                    "clinical_status": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                "code": "active",
                            }
                        ]
                    },
                }
            ],
            **kwargs,
        )

    @staticmethod
    def create_lung_cancer_patient(**kwargs) -> Dict[str, Any]:
        """Create a patient with lung cancer."""
        return PatientFactory.create_basic_patient(
            gender="male",
            conditions=[
                {
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "187875005",
                                "display": "Malignant neoplasm of lung",
                            }
                        ]
                    }
                }
            ],
            **kwargs,
        )

    @staticmethod
    def create_invalid_patient(missing_field: str = None) -> Dict[str, Any]:
        """Create an invalid patient for testing error handling."""
        patient = PatientFactory.create_basic_patient()

        if missing_field == "gender":
            patient["entry"][0]["resource"].pop("gender", None)
        elif missing_field == "birthDate":
            patient["entry"][0]["resource"].pop("birthDate", None)

        return patient

    @staticmethod
    def create_patient_bundle(num_patients: int = 10) -> Dict[str, Any]:
        """Create a bundle with multiple patients."""
        bundle = {
            "resourceType": "Bundle",
            "id": f"multi_patient_bundle_{random.randint(1000, 9999)}",
            "entry": [],
        }

        for i in range(num_patients):
            patient = PatientFactory.create_basic_patient(f"patient_{i}")
            bundle["entry"].extend(patient["entry"])

        return bundle


@dataclass
class McodeFactory:
    """Factory for generating mCODE test data."""

    @staticmethod
    def create_mcode_element(
        element_type: str = "PrimaryCancerCondition",
        system: str = "http://snomed.info/sct",
        code: str = "254837009",
        display: str = "Malignant neoplasm of breast",
    ) -> Dict[str, Any]:
        """Create a basic mCODE element."""
        return {
            "element": element_type,
            "system": system,
            "code": code,
            "display": display,
        }

    @staticmethod
    def create_mcode_response(
        elements: List[Dict[str, Any]] = None, confidence: float = 0.95, **kwargs
    ) -> Dict[str, Any]:
        """Create a complete mCODE mapping response."""
        if elements is None:
            elements = [McodeFactory.create_mcode_element()]

        return {
            "mcode_elements": elements,
            "metadata": {
                "confidence": confidence,
                "processing_time": kwargs.get("processing_time", 1.5),
                "model_version": kwargs.get("model_version", "1.0.0"),
                "token_usage": kwargs.get(
                    "token_usage",
                    {"input_tokens": 150, "output_tokens": 75, "total_tokens": 225},
                ),
            },
        }

    @staticmethod
    def create_invalid_mcode_response(missing_field: str = None) -> Dict[str, Any]:
        """Create an invalid mCODE response for testing error handling."""
        response = McodeFactory.create_mcode_response()

        if missing_field:
            if missing_field in response:
                response.pop(missing_field)
            elif missing_field in response.get("metadata", {}):
                response["metadata"].pop(missing_field)

        return response


class TestDataManager:
    """Manager for test data lifecycle and cleanup."""

    def __init__(self):
        self.created_files = []
        self.temp_data = {}

    def create_temp_trial_file(
        self, trial_data: Dict[str, Any], filename: str = None
    ) -> str:
        """Create a temporary trial data file."""
        if filename is None:
            filename = f"temp_trial_{random.randint(1000, 9999)}.json"

        import tempfile
        import os

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        with open(filepath, "w") as f:
            json.dump(trial_data, f, indent=2)

        self.created_files.append(filepath)
        return filepath

    def create_temp_patient_file(
        self, patient_data: Dict[str, Any], filename: str = None
    ) -> str:
        """Create a temporary patient data file."""
        if filename is None:
            filename = f"temp_patient_{random.randint(1000, 9999)}.json"

        import tempfile
        import os

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        with open(filepath, "w") as f:
            json.dump(patient_data, f, indent=2)

        self.created_files.append(filepath)
        return filepath

    def cleanup(self):
        """Clean up all created temporary files."""
        import os

        for filepath in self.created_files:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass  # Ignore cleanup errors

        self.created_files.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Global test data manager instance
test_data_manager = TestDataManager()
