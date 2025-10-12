"""
Performance tests for patient generator operations.
Tests speed and efficiency of patient data processing.
"""

import tempfile
import zipfile
from unittest.mock import patch

import pytest

from src.utils.patient_generator import PatientGenerator


@pytest.fixture
def sample_zip_archive():
    """Create a sample ZIP archive with test patient data."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, "w") as zf:
            # Create sample patient data
            for i in range(10):
                patient_data = {
                    "resourceType": "Bundle",
                    "type": "collection",
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Patient",
                                "id": f"patient-{i}",
                                "name": [{"family": f"Doe{i}", "given": ["John"]}],
                                "identifier": [{"value": f"id-{i}"}],
                            }
                        }
                    ],
                }
                import json

                zf.writestr(f"patient_{i}.json", json.dumps(patient_data))

        yield tmp.name

        # Cleanup
        import os

        os.unlink(tmp.name)


class TestPatientGeneratorPerformance:
    """Performance tests for PatientGenerator."""

    def test_patient_generator_initialization_speed(self, sample_zip_archive, benchmark):
        """Benchmark patient generator initialization."""

        def init_generator():
            with patch("src.utils.patient_generator.Config"):
                generator = PatientGenerator(sample_zip_archive)
                return generator

        result = benchmark(init_generator)
        assert result is not None
        assert len(result) == 10

    def test_patient_iteration_speed(self, sample_zip_archive, benchmark):
        """Benchmark patient iteration performance."""
        with patch("src.utils.patient_generator.Config"):
            generator = PatientGenerator(sample_zip_archive)

            def iterate_patients():
                patients = []
                for patient in generator:
                    patients.append(patient)
                return patients

            result = benchmark(iterate_patients)
            assert len(result) == 10

    def test_get_random_patient_speed(self, sample_zip_archive, benchmark):
        """Benchmark random patient retrieval."""
        with patch("src.utils.patient_generator.Config"):
            generator = PatientGenerator(sample_zip_archive)

            def get_random():
                return generator.get_random_patient()

            result = benchmark(get_random)
            assert result is not None
            assert result["resourceType"] == "Bundle"

    def test_get_patients_with_limit_speed(self, sample_zip_archive, benchmark):
        """Benchmark getting patients with limit."""
        with patch("src.utils.patient_generator.Config"):
            generator = PatientGenerator(sample_zip_archive)

            def get_limited_patients():
                return generator.get_patients(limit=5)

            result = benchmark(get_limited_patients)
            assert len(result) == 5

    def test_patient_id_extraction_speed(self, benchmark):
        """Benchmark patient ID extraction."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "test-patient-123",
                        "identifier": [{"value": "alt-id-456"}],
                    }
                }
            ]
        }

        from src.utils.patient_generator import extract_patient_id

        def extract_id():
            return extract_patient_id(bundle)

        result = benchmark(extract_id)
        assert result == "test-patient-123"

    def test_normalize_to_bundle_speed(self, benchmark):
        """Benchmark bundle normalization."""
        # Test with single patient
        single_patient = {"resourceType": "Patient", "id": "test-patient"}

        with patch("src.utils.patient_generator.Config"):
            generator = PatientGenerator.__new__(PatientGenerator)

            def normalize():
                return generator._normalize_to_bundle(single_patient)

            result = benchmark(normalize)
            assert result is not None
            assert result["resourceType"] == "Bundle"

    def test_matches_patient_id_speed(self, benchmark):
        """Benchmark patient ID matching."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "test-patient-123",
                        "identifier": [{"value": "alt-id-456"}],
                    }
                }
            ]
        }

        with patch("src.utils.patient_generator.Config"):
            generator = PatientGenerator.__new__(PatientGenerator)

            def match_id():
                return generator._matches_patient_id(bundle, "test-patient-123")

            result = benchmark(match_id)
            assert result is True

    @pytest.mark.parametrize("num_patients", [1, 5, 10])
    def test_scaling_with_patient_count(self, benchmark, num_patients):
        """Test performance scaling with different patient counts."""
        # Create archive with specified number of patients
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                for i in range(num_patients):
                    patient_data = {
                        "resourceType": "Bundle",
                        "type": "collection",
                        "entry": [
                            {
                                "resource": {
                                    "resourceType": "Patient",
                                    "id": f"patient-{i}",
                                }
                            }
                        ],
                    }
                    import json

                    zf.writestr(f"patient_{i}.json", json.dumps(patient_data))

            try:
                with patch("src.utils.patient_generator.Config"):
                    generator = PatientGenerator(tmp.name)

                    def count_patients():
                        return len(generator)

                    result = benchmark(count_patients)
                    assert result == num_patients

            finally:
                import os

                os.unlink(tmp.name)


class TestMemoryUsage:
    """Test memory usage patterns."""

    def test_memory_efficiency_large_archive(self):
        """Test memory efficiency with simulated large archive."""
        # This test would be more comprehensive with actual memory profiling
        # For now, just ensure basic functionality works
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, "w") as zf:
                # Create many small files to simulate large archive
                for i in range(100):
                    patient_data = {
                        "resourceType": "Bundle",
                        "entry": [{"resource": {"resourceType": "Patient", "id": f"p{i}"}}],
                    }
                    import json

                    zf.writestr(f"patient_{i}.json", json.dumps(patient_data))

            try:
                with patch("src.utils.patient_generator.Config"):
                    generator = PatientGenerator(tmp.name)

                    # Test that we can iterate without loading everything into memory
                    count = 0
                    for patient in generator:
                        count += 1
                        if count >= 10:  # Just test first 10
                            break

                    assert count == 10

            finally:
                import os

                os.unlink(tmp.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
