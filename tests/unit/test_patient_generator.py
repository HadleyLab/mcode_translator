"""
Unit tests for patient_generator module.
"""
import json
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pytest
from src.utils.patient_generator import (
    PatientGenerator,
    create_patient_generator,
    extract_patient_id,
    _extract_patient_id_from_bundle,
    PatientNotFoundError,
    ArchiveLoadError,
)


class TestExtractPatientId:
    """Test patient ID extraction functions."""

    def test_extract_patient_id_from_bundle_with_id(self):
        """Test extracting patient ID when ID is present."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-123"
                    }
                }
            ]
        }

        result = _extract_patient_id_from_bundle(bundle)
        assert result == "patient-123"

    def test_extract_patient_id_from_bundle_with_identifier(self):
        """Test extracting patient ID from identifier."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "identifier": [
                            {"value": "identifier-456"}
                        ]
                    }
                }
            ]
        }

        result = _extract_patient_id_from_bundle(bundle)
        assert result == "identifier-456"

    def test_extract_patient_id_from_bundle_with_name_fallback(self):
        """Test extracting patient ID using name fallback."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "name": [
                            {
                                "family": "Smith",
                                "given": ["John"]
                            }
                        ]
                    }
                }
            ]
        }

        result = _extract_patient_id_from_bundle(bundle)
        assert result == "Smith_John"

    def test_extract_patient_id_from_bundle_no_patient(self):
        """Test extracting patient ID when no patient resource exists."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation"
                    }
                }
            ]
        }

        result = _extract_patient_id_from_bundle(bundle)
        assert result is None

    def test_extract_patient_id_from_bundle_empty(self):
        """Test extracting patient ID from empty bundle."""
        bundle = {}

        result = _extract_patient_id_from_bundle(bundle)
        assert result is None

    def test_extract_patient_id_wrapper(self):
        """Test the public extract_patient_id wrapper."""
        bundle = {
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-789"
                    }
                }
            ]
        }

        result = extract_patient_id(bundle)
        assert result == "patient-789"


class TestPatientGeneratorInit:
    """Test PatientGenerator initialization."""

    @patch('src.utils.patient_generator.Config')
    @patch('src.utils.patient_generator.os.path.exists')
    def test_init_with_existing_path(self, mock_exists, mock_config):
        """Test initialization with existing archive path."""
        mock_exists.return_value = True
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with patch.object(PatientGenerator, '_load_file_list'):
            generator = PatientGenerator("test.zip")

            assert generator.archive_path == "test.zip"
            assert generator.shuffle is False
            assert generator.seed is None

    @patch('src.utils.patient_generator.Config')
    @patch('src.utils.patient_generator.os.path.exists')
    def test_init_with_named_archive(self, mock_exists, mock_config):
        """Test initialization with named archive resolution."""
        mock_exists.side_effect = lambda p: p == "/data/synthetic_patients/cancer/type1/archive.zip"
        mock_config_instance = Mock()
        mock_config_instance.synthetic_data_config = {
            "synthetic_data": {
                "archives": {
                    "cancer": {
                        "type1": {}
                    }
                },
                "base_directory": "/data/synthetic_patients"
            }
        }
        mock_config.return_value = mock_config_instance

        with patch.object(PatientGenerator, '_load_file_list'):
            generator = PatientGenerator("cancer_type1")

            assert generator.archive_path == "/data/synthetic_patients/cancer/type1/cancer_type1.zip"

    @patch('src.utils.patient_generator.Config')
    @patch('src.utils.patient_generator.os.path.exists')
    def test_init_archive_not_found(self, mock_exists, mock_config):
        """Test initialization when archive is not found."""
        mock_exists.return_value = False
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        with pytest.raises(ArchiveLoadError):
            PatientGenerator("nonexistent.zip")


class TestPatientGeneratorFileLoading:
    """Test file loading functionality."""

    def test_load_file_list_success(self):
        """Test successful file list loading."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            # Create a test ZIP file
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('patient1.json', '{"resourceType": "Bundle"}')
                zf.writestr('patient2.ndjson', '{"id": "1"}\n{"id": "2"}')

            try:
                with patch('src.utils.patient_generator.Config'):
                    generator = PatientGenerator(tmp.name)
                    generator._load_file_list()

                    assert len(generator._patient_files) == 2
                    assert 'patient1.json' in generator._patient_files
                    assert 'patient2.ndjson' in generator._patient_files
            finally:
                os.unlink(tmp.name)

    def test_load_file_list_invalid_zip(self):
        """Test loading file list from invalid ZIP."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"not a zip file")
            tmp.flush()

            try:
                with patch('src.utils.patient_generator.Config'):
                    generator = PatientGenerator(tmp.name)

                    with pytest.raises(ArchiveLoadError):
                        generator._load_file_list()
            finally:
                os.unlink(tmp.name)

    def test_load_file_list_no_patients(self):
        """Test loading file list with no patient files."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('readme.txt', 'This is not patient data')

            try:
                with patch('src.utils.patient_generator.Config'):
                    generator = PatientGenerator(tmp.name)
                    generator._load_file_list()

                    assert len(generator._patient_files) == 0
            finally:
                os.unlink(tmp.name)


class TestPatientGeneratorIteration:
    """Test iteration functionality."""

    def test_len(self):
        """Test length method."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._patient_files = ['file1.json', 'file2.json', 'file3.json']

            assert len(generator) == 3

    def test_iter(self):
        """Test iterator initialization."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._patient_files = ['file1.json', 'file2.json']
            generator._current_index = 5  # Some non-zero value

            iterator = iter(generator)

            assert generator._current_index == 0
            assert iterator is generator

    def test_reset(self):
        """Test reset method."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._current_index = 10

            generator.reset()

            assert generator._current_index == 0

    def test_close(self):
        """Test close method."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._patient_files = ['file1.json', 'file2.json']
            generator._current_index = 5
            generator._loaded = True

            generator.close()

            assert generator._patient_files == []
            assert generator._current_index == 0
            assert generator._loaded is False


class TestPatientGeneratorPatientOperations:
    """Test patient retrieval operations."""

    def test_get_random_patient_no_files(self):
        """Test get_random_patient with no patient files."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._patient_files = []

            with pytest.raises(ArchiveLoadError):
                generator.get_random_patient()

    def test_get_patient_by_id_not_found(self):
        """Test get_patient_by_id when patient is not found."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._patient_files = ['file1.json']
            generator._load_patient_from_file = Mock(side_effect=Exception("Load failed"))

            with pytest.raises(PatientNotFoundError):
                generator.get_patient_by_id("nonexistent")

    def test_get_patients_with_limit(self):
        """Test get_patients with limit."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._patient_files = ['file1.json', 'file2.json', 'file3.json']
            generator._load_patient_from_file = Mock(return_value={"bundle": "data"})

            result = generator.get_patients(limit=2, start=1)

            assert len(result) == 2
            assert generator._load_patient_from_file.call_count == 2

    def test_get_patient_ids(self):
        """Test get_patient_ids method."""
        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)
            generator._patient_files = ['file1.json']
            generator._current_index = 0
            generator._load_patient_from_file = Mock(return_value={
                "entry": [{"resource": {"resourceType": "Patient", "id": "patient-123"}}]
            })
            generator.extract_patient_id = Mock(return_value="patient-123")

            result = generator.get_patient_ids()

            assert result == ["patient-123"]


class TestCreatePatientGenerator:
    """Test create_patient_generator function."""

    @patch('src.utils.patient_generator.Config')
    @patch('src.utils.patient_generator.PatientGenerator')
    def test_create_patient_generator_with_config(self, mock_generator, mock_config):
        """Test create_patient_generator with provided config."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        result = create_patient_generator("test.zip", config=mock_config_instance, shuffle=True, seed=42)

        mock_generator.assert_called_once_with("test.zip", mock_config_instance, True, 42)

    @patch('src.utils.patient_generator.Config')
    @patch('src.utils.patient_generator.PatientGenerator')
    def test_create_patient_generator_no_config(self, mock_generator, mock_config):
        """Test create_patient_generator without config."""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        result = create_patient_generator("test.zip")

        mock_generator.assert_called_once_with("test.zip", mock_config_instance, False, None)


class TestPatientGeneratorLoadPatientFromFile:
    """Test _load_patient_from_file method."""

    def test_load_patient_from_file_json_success(self):
        """Test loading patient from JSON file successfully."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('patient.json', '{"resourceType": "Bundle", "entry": []}')

            try:
                with patch('src.utils.patient_generator.Config'):
                    generator = PatientGenerator.__new__(PatientGenerator)
                    generator.archive_path = tmp.name

                    result = generator._load_patient_from_file('patient.json')

                    assert result["resourceType"] == "Bundle"
            finally:
                os.unlink(tmp.name)

    def test_load_patient_from_file_ndjson_success(self):
        """Test loading patient from NDJSON file successfully."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('patients.ndjson', '{"resourceType": "Bundle", "entry": []}\n{"invalid": "json"}')

            try:
                with patch('src.utils.patient_generator.Config'):
                    generator = PatientGenerator.__new__(PatientGenerator)
                    generator.archive_path = tmp.name

                    result = generator._load_patient_from_file('patients.ndjson')

                    assert result["resourceType"] == "Bundle"
            finally:
                os.unlink(tmp.name)

    def test_load_patient_from_file_invalid_json(self):
        """Test loading patient from invalid JSON file."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('invalid.json', 'not json')

            try:
                with patch('src.utils.patient_generator.Config'):
                    generator = PatientGenerator.__new__(PatientGenerator)
                    generator.archive_path = tmp.name

                    with pytest.raises(ValueError):
                        generator._load_patient_from_file('invalid.json')
            finally:
                os.unlink(tmp.name)


class TestPatientGeneratorNormalizeToBundle:
    """Test _normalize_to_bundle method."""

    def test_normalize_bundle_already_bundle(self):
        """Test normalizing data that is already a bundle."""
        data = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }

        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)

            result = generator._normalize_to_bundle(data)

            assert result == data

    def test_normalize_single_patient(self):
        """Test normalizing single patient resource."""
        data = {
            "resourceType": "Patient",
            "id": "patient-123"
        }

        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)

            result = generator._normalize_to_bundle(data)

            assert result["resourceType"] == "Bundle"
            assert result["type"] == "collection"
            assert len(result["entry"]) == 1
            assert result["entry"][0]["resource"] == data

    def test_normalize_bundle_like_structure(self):
        """Test normalizing bundle-like structure."""
        data = {
            "entry": [{"resource": {"resourceType": "Patient"}}]
        }

        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)

            result = generator._normalize_to_bundle(data)

            assert result["resourceType"] == "Bundle"
            assert result["type"] == "collection"
            assert result["entry"] == data["entry"]

    def test_normalize_invalid_data(self):
        """Test normalizing invalid data."""
        data = "invalid data"

        with patch('src.utils.patient_generator.Config'):
            generator = PatientGenerator.__new__(PatientGenerator)

            result = generator._normalize_to_bundle(data)

            assert result is None


if __name__ == "__main__":
    pytest.main([__file__])
