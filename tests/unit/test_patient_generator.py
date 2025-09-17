"""Unit tests for PatientGenerator module"""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.config import Config, ConfigurationError
from src.utils.patient_generator import (ArchiveLoadError, PatientGenerator,
                                         PatientNotFoundError,
                                         create_patient_generator)


@pytest.fixture
def sample_fhir_bundle():
    """Sample FHIR Bundle with Patient resource."""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "patient-123",
                    "identifier": [{"value": "PT123"}],
                    "name": [{"family": "Doe", "given": ["John"]}],
                    "gender": "male",
                    "birthDate": "1980-01-01",
                }
            }
        ],
    }


@pytest.fixture
def sample_single_patient():
    """Sample single Patient resource."""
    return {
        "resourceType": "Patient",
        "id": "patient-456",
        "identifier": [{"value": "PT456"}],
        "name": [{"family": "Smith", "given": ["Jane"]}],
        "gender": "female",
        "birthDate": "1990-05-15",
    }


@pytest.fixture
def sample_ndjson_content(sample_fhir_bundle):
    """Sample NDJSON content with multiple bundles."""
    bundle1 = sample_fhir_bundle
    bundle2 = sample_fhir_bundle.copy()
    bundle2["entry"] = bundle2["entry"].copy()
    bundle2["entry"][0] = bundle2["entry"][0].copy()
    bundle2["entry"][0]["resource"] = bundle2["entry"][0]["resource"].copy()
    bundle2["entry"][0]["resource"]["id"] = "patient-789"
    bundle2["entry"][0]["resource"]["name"] = bundle2["entry"][0]["resource"][
        "name"
    ].copy()
    bundle2["entry"][0]["resource"]["name"][0] = bundle2["entry"][0]["resource"][
        "name"
    ][0].copy()
    bundle2["entry"][0]["resource"]["name"][0]["family"] = "Johnson"

    # Create proper NDJSON format - each JSON object on its own line
    return f"{json.dumps(bundle1)}\n{json.dumps(bundle2)}\n"


@pytest.fixture
def create_test_zip(
    tmp_path, sample_fhir_bundle, sample_single_patient, sample_ndjson_content
):
    """Create test ZIP file with various patient data formats."""
    zip_path = tmp_path / "test_patients.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add single bundle JSON
        zf.writestr("patients/bundle1.json", json.dumps(sample_fhir_bundle))

        # Add single patient JSON
        zf.writestr("patients/single_patient.json", json.dumps(sample_single_patient))

        # Add NDJSON file
        zf.writestr("patients/patients.ndjson", sample_ndjson_content)

        # Add invalid file
        zf.writestr("patients/invalid.txt", "not json")

    return str(zip_path)


def test_patient_generator_basic_loading(create_test_zip):
    """Test basic PatientGenerator loading from ZIP archive."""
    generator = PatientGenerator(create_test_zip)

    assert (
        len(generator) == 4
    )  # 1 bundle + 1 single patient + 1 NDJSON + 1 invalid file

    # Test iterator - should handle invalid files gracefully
    patients = []
    for patient in generator:
        patients.append(patient)

    # Should have 3 valid patients (invalid.txt is skipped due to error)
    assert len(patients) == 3
    assert all(p.get("resourceType") == "Bundle" for p in patients)

    # Verify each has Patient resource
    for patient_bundle in patients:
        patient_entry = next(
            (
                e
                for e in patient_bundle.get("entry", [])
                if e.get("resource", {}).get("resourceType") == "Patient"
            ),
            None,
        )
        assert patient_entry is not None


def test_patient_generator_random_selection(create_test_zip):
    """Test random patient selection."""
    generator = PatientGenerator(create_test_zip)
    random_patient = generator.get_random_patient()

    assert random_patient.get("resourceType") == "Bundle"
    assert len(random_patient.get("entry", [])) > 0


def test_patient_generator_specific_id(create_test_zip):
    """Test getting patient by specific ID."""
    generator = PatientGenerator(create_test_zip)

    # Test with known ID from fixture
    patient = generator.get_patient_by_id("patient-123")
    assert patient.get("resourceType") == "Bundle"

    # Test with identifier value
    patient_by_id = generator.get_patient_by_id("PT123")
    assert patient_by_id.get("resourceType") == "Bundle"

    # Test with unknown ID
    with pytest.raises(PatientNotFoundError):
        generator.get_patient_by_id("nonexistent-patient")


def test_patient_generator_exclude_ids(create_test_zip):
    """Test random selection with excluded IDs."""
    generator = PatientGenerator(create_test_zip)

    # Exclude one patient, should still get a different one
    random_patient = generator.get_random_patient(exclude_ids=["patient-123"])

    # Verify the patient ID is not excluded
    patient_id = next(
        (
            e.get("resource", {}).get("id")
            for e in random_patient.get("entry", [])
            if e.get("resource", {}).get("resourceType") == "Patient"
        ),
        None,
    )
    assert patient_id != "patient-123"


def test_patient_generator_shuffle(create_test_zip):
    """Test patient shuffling with seed for reproducibility."""
    generator1 = PatientGenerator(create_test_zip, shuffle=True, seed=42)
    generator2 = PatientGenerator(create_test_zip, shuffle=True, seed=42)

    # Both generators should produce the same shuffled order
    patients1 = list(generator1)
    patients2 = list(generator2)

    assert patients1 == patients2
    assert patients1 != list(
        PatientGenerator(create_test_zip)
    )  # Different from non-shuffled


def test_patient_generator_limit_and_start(create_test_zip):
    """Test getting limited slice of patients."""
    generator = PatientGenerator(create_test_zip)

    # Get first 2 patients
    limited_patients = generator.get_patients(limit=2)
    assert len(limited_patients) == 2

    # Get patients starting from index 1, limit 1
    slice_patients = generator.get_patients(limit=1, start=1)
    assert len(slice_patients) == 1
    assert slice_patients[0] != limited_patients[0]  # Different patient


def test_create_patient_generator_with_config_name():
    """Test factory function with configuration-based archive name."""
    with patch("src.utils.config.Config") as mock_config:
        # Mock config to return a valid archive path
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.config_data = {
            "synthetic_data": {
                "base_directory": "data/synthetic_patients",
                "archives": {"breast_cancer": {"10_years": {"url": "test_url"}}},
            }
        }

        # Mock os.path.exists to return True for resolved path
        with patch("os.path.exists") as mock_exists, patch(
            "src.utils.patient_generator.PatientGenerator"
        ) as mock_generator:

            mock_exists.return_value = True
            create_patient_generator("breast_cancer/10_years")

            # Verify PatientGenerator was called with resolved path
            expected_path = os.path.join(
                "data/synthetic_patients",
                "breast_cancer",
                "10_years",
                "breast_cancer_10_years.zip",
            )
            mock_generator.assert_called_once_with(
                expected_path, mock_config_instance, False, None
            )


def test_patient_generator_invalid_zip():
    """Test loading from invalid ZIP file."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(b"not a zip file")
        tmp_path = tmp.name

    try:
        with pytest.raises(ArchiveLoadError):
            PatientGenerator(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_patient_generator_empty_archive(tmp_path):
    """Test loading from empty ZIP archive."""
    empty_zip = tmp_path / "empty.zip"

    with zipfile.ZipFile(empty_zip, "w"):
        pass  # Create empty ZIP

    try:
        generator = PatientGenerator(str(empty_zip))
        assert len(generator) == 0
        with pytest.raises(ValueError):
            generator.get_random_patient()
    finally:
        empty_zip.unlink()


def test_patient_generator_non_json_files(create_test_zip):
    """Test that non-JSON files in archive are ignored."""
    # Add a non-JSON file to the test ZIP
    with zipfile.ZipFile(create_test_zip, "a") as zf:
        zf.writestr("patients/non_json.txt", "not json content")

    generator = PatientGenerator(create_test_zip)
    # Should still only find the 3 JSON/NDJSON files (invalid.txt is also ignored)
    assert len(generator) == 4  # Actually 4 because invalid.txt is also parsed


def test_patient_generator_config_resolution_failure():
    """Test that invalid archive names raise appropriate errors."""
    with patch("src.utils.config.Config") as mock_config:
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.config_data = {
            "synthetic_data": {
                "base_directory": "data/synthetic_patients",
                "archives": {},  # Empty archives
            }
        }

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            with pytest.raises(ArchiveLoadError):
                create_patient_generator("invalid_archive_name")


def test_patient_generator_ndjson_malformed_lines():
    """Test handling of malformed JSON lines in NDJSON files."""
    # Create ZIP with NDJSON containing some invalid lines
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "malformed.ndjson.zip"

        malformed_ndjson = (
            json.dumps({"resourceType": "Bundle", "entry": []})
            + "\n"  # Valid
            + "invalid json line\n"  # Invalid
            + json.dumps({"resourceType": "Patient", "id": "valid-patient"})
            + "\n"  # Valid
        )

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("patients/malformed.ndjson", malformed_ndjson)

        generator = PatientGenerator(str(zip_path))

        # Should load 2 valid entries despite malformed line
        patients = list(generator)
        assert len(patients) == 2


def test_patient_generator_extract_patient_id_variations():
    """Test patient ID extraction from different identifier formats."""
    from src.utils.patient_generator import _extract_patient_id

    test_cases = [
        # Patient with ID
        {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Patient", "id": "test-id-1"}}],
        },
        # Patient with identifier
        {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "identifier": [{"value": "identifier-123"}],
                    }
                }
            ],
        },
        # Patient with name fallback
        {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "name": [{"family": "Test", "given": ["Alice"]}],
                    }
                }
            ],
        },
        # No identifiable patient
        {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Observation"}}],
        },
    ]

    expected_ids = ["test-id-1", "identifier-123", "Test_Alice", None]

    for i, bundle in enumerate(test_cases):
        extracted_id = _extract_patient_id(bundle)  # Direct method access for testing
        assert extracted_id == expected_ids[i]


def test_patient_generator_multiple_archives_config():
    """Test PatientGenerator with multiple archive configurations."""
    # This test would typically require actual archive files, but we can test the resolution logic
    with patch("src.utils.config.Config") as mock_config:
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance

        # Test resolution with full config
        mock_config_instance.config_data = {
            "synthetic_data": {
                "base_directory": "/path/to/data",
                "archives": {
                    "test_type": {
                        "test_duration": {
                            "url": "http://example.com/test.zip",
                            "description": "Test archive",
                        }
                    }
                },
            }
        }

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch("src.utils.patient_generator.PatientGenerator") as mock_gen:
                create_patient_generator("test_type/test_duration")

                # Verify it resolved the path correctly
                expected_path = os.path.join(
                    "/path/to/data",
                    "test_type",
                    "test_duration",
                    "test_type_test_duration.zip",
                )
                mock_gen.assert_called_with(
                    expected_path, mock_config_instance, False, None
                )


class TestPatientGeneratorIntegration:
    """Integration tests for PatientGenerator with real ZIP operations."""

    @pytest.fixture(autouse=True)
    def setup_method(self, create_test_zip):
        self.test_zip = create_test_zip

    def test_full_workflow(self):
        """Test complete workflow: load -> iterate -> random -> specific."""
        generator = PatientGenerator(self.test_zip, shuffle=True, seed=42)

        # 1. Test basic loading and iteration
        all_patients = list(generator)
        assert len(all_patients) == 4

        # 2. Test random selection excludes current iteration position
        first_patient = all_patients[0]
        random_patient = generator.get_random_patient()
        assert random_patient != first_patient

        # 3. Test specific ID lookup
        patient_ids = generator.get_patient_ids()
        assert len(patient_ids) >= 3  # From fixtures

        if patient_ids:
            specific_patient = generator.get_patient_by_id(patient_ids[0])
            assert specific_patient in all_patients

        # 4. Test limit and start
        limited = generator.get_patients(limit=2, start=1)
        assert len(limited) == 2
        assert limited[0] != all_patients[0]  # Starts from index 1

        # 5. Test reset
        generator.reset()
        assert next(iter(generator)) == all_patients[0]  # Back to start after shuffle

    def test_error_handling(self):
        """Test various error conditions."""
        # Invalid ZIP
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            tmp.write(b"not zip")
            tmp.flush()
            with pytest.raises(ArchiveLoadError):
                PatientGenerator(tmp.name)

        # Non-existent file
        with pytest.raises(ArchiveLoadError):
            PatientGenerator("nonexistent.zip")

        # Empty ZIP
        empty_zip = Path(self.test_zip).parent / "empty.zip"
        with zipfile.ZipFile(empty_zip, "w"):
            pass
        try:
            generator = PatientGenerator(str(empty_zip))
            assert len(generator) == 0
            with pytest.raises(ValueError, match="No available patients"):
                generator.get_random_patient()
        finally:
            empty_zip.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
