"""
Unit tests for FHIRResourceExtractors class.
"""

import pytest
from unittest.mock import patch

from src.services.fhir_extractors import FHIRResourceExtractors


class TestFHIRResourceExtractors:
    """Test suite for FHIRResourceExtractors."""

    @pytest.fixture
    def extractor(self):
        """Create a FHIRResourceExtractors instance."""
        return FHIRResourceExtractors()

    def test_init(self, extractor):
        """Test initialization."""
        assert extractor.logger is not None

    def test_extract_condition_mcode_cancer(self, extractor):
        """Test condition extraction for cancer conditions."""
        condition = {
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "254837009",
                        "display": "Malignant neoplasm of breast"
                    }
                ]
            }
        }
        result = extractor.extract_condition_mcode(condition)
        assert result == {
            "system": "http://snomed.info/sct",
            "code": "254837009",
            "display": "Malignant neoplasm of breast",
            "interpretation": "Confirmed"
        }

    def test_extract_condition_mcode_no_cancer(self, extractor):
        """Test condition extraction for non-cancer conditions."""
        condition = {
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "123456",
                        "display": "Hypertension"
                    }
                ]
            }
        }
        result = extractor.extract_condition_mcode(condition)
        assert result is None

    def test_extract_condition_mcode_exception(self, extractor):
        """Test condition extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            condition = None  # This will cause an exception
            result = extractor.extract_condition_mcode(condition)
            assert result is None
            mock_error.assert_called_once()

    def test_extract_observation_mcode_er_positive(self, extractor):
        """Test observation extraction for ER receptor status."""
        observation = {
            "code": {
                "coding": [{
                    "display": "Estrogen receptor status"
                }]
            },
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "10828004",
                    "display": "Positive"
                }]
            }
        }
        result = extractor.extract_observation_mcode(observation)
        assert "ERReceptorStatus" in result
        assert result["ERReceptorStatus"]["interpretation"] == "Positive"

    def test_extract_observation_mcode_her2(self, extractor):
        """Test observation extraction for HER2 status."""
        observation = {
            "code": {
                "coding": [{
                    "display": "HER2 receptor status"
                }]
            },
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "10828004",
                    "display": "Positive"
                }]
            }
        }
        result = extractor.extract_observation_mcode(observation)
        assert "HER2ReceptorStatus" in result
        assert result["HER2ReceptorStatus"]["interpretation"] == "Positive"

    def test_extract_observation_mcode_stage(self, extractor):
        """Test observation extraction for TNM stage."""
        observation = {
            "code": {
                "coding": [{
                    "display": "TNM Stage"
                }]
            },
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "261650005",
                    "display": "T2N1M0"
                }]
            }
        }
        result = extractor.extract_observation_mcode(observation)
        assert "TNMStage" in result
        assert result["TNMStage"]["interpretation"] == "T2N1M0"

    def test_extract_observation_mcode_unknown(self, extractor):
        """Test observation extraction for unknown observation."""
        observation = {
            "code": {
                "coding": [{
                    "display": "Unknown observation"
                }]
            }
        }
        result = extractor.extract_observation_mcode(observation)
        assert result == {}

    def test_extract_observation_mcode_exception(self, extractor):
        """Test observation extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            observation = None
            result = extractor.extract_observation_mcode(observation)
            assert result == {}
            mock_error.assert_called_once()

    def test_extract_observation_mcode_comprehensive_ecog(self, extractor):
        """Test comprehensive observation extraction for ECOG performance status."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "ECOG Performance Status"
                }]
            },
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "425389002",
                    "display": "0 - Fully active"
                }]
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "ECOGPerformanceStatus" in result
        assert result["ECOGPerformanceStatus"]["display"] == "0 - Fully active"

    def test_extract_observation_mcode_comprehensive_karnofsky(self, extractor):
        """Test comprehensive observation extraction for Karnofsky performance status."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Karnofsky Performance Status"
                }]
            },
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "426927009",
                    "display": "90%"
                }]
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "KarnofskyPerformanceStatus" in result
        assert result["KarnofskyPerformanceStatus"]["display"] == "90%"

    def test_extract_observation_mcode_comprehensive_weight(self, extractor):
        """Test comprehensive observation extraction for body weight."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Body weight"
                }]
            },
            "valueQuantity": {
                "value": 70.5,
                "unit": "kg"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "BodyWeight" in result
        assert result["BodyWeight"]["value"] == 70.5
        assert result["BodyWeight"]["unit"] == "kg"

    def test_extract_observation_mcode_comprehensive_height(self, extractor):
        """Test comprehensive observation extraction for body height."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Body height"
                }]
            },
            "valueQuantity": {
                "value": 170,
                "unit": "cm"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "BodyHeight" in result
        assert result["BodyHeight"]["value"] == 170
        assert result["BodyHeight"]["unit"] == "cm"

    def test_extract_observation_mcode_comprehensive_bmi(self, extractor):
        """Test comprehensive observation extraction for BMI."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Body mass index"
                }]
            },
            "valueQuantity": {
                "value": 24.2,
                "unit": "kg/m2"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "BodyMassIndex" in result
        assert result["BodyMassIndex"]["value"] == 24.2

    def test_extract_observation_mcode_comprehensive_blood_pressure(self, extractor):
        """Test comprehensive observation extraction for blood pressure."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Blood pressure"
                }]
            },
            "component": [
                {
                    "code": {"coding": [{"display": "Systolic blood pressure"}]},
                    "valueQuantity": {"value": 120}
                },
                {
                    "code": {"coding": [{"display": "Diastolic blood pressure"}]},
                    "valueQuantity": {"value": 80}
                }
            ]
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "BloodPressure" in result
        assert result["BloodPressure"]["systolic"] == 120
        assert result["BloodPressure"]["diastolic"] == 80

    def test_extract_observation_mcode_comprehensive_hemoglobin(self, extractor):
        """Test comprehensive observation extraction for hemoglobin."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Hemoglobin"
                }]
            },
            "valueQuantity": {
                "value": 12.5,
                "unit": "g/dL"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "Hemoglobin" in result
        assert result["Hemoglobin"]["value"] == 12.5

    def test_extract_observation_mcode_comprehensive_wbc(self, extractor):
        """Test comprehensive observation extraction for WBC."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "White blood cell count"
                }]
            },
            "valueQuantity": {
                "value": 8.2,
                "unit": "10^9/L"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "WhiteBloodCellCount" in result
        assert result["WhiteBloodCellCount"]["value"] == 8.2

    def test_extract_observation_mcode_comprehensive_platelets(self, extractor):
        """Test comprehensive observation extraction for platelets."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Platelet count"
                }]
            },
            "valueQuantity": {
                "value": 250,
                "unit": "10^9/L"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "PlateletCount" in result
        assert result["PlateletCount"]["value"] == 250

    def test_extract_observation_mcode_comprehensive_creatinine(self, extractor):
        """Test comprehensive observation extraction for creatinine."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Creatinine"
                }]
            },
            "valueQuantity": {
                "value": 0.8,
                "unit": "mg/dL"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "Creatinine" in result
        assert result["Creatinine"]["value"] == 0.8

    def test_extract_observation_mcode_comprehensive_bilirubin(self, extractor):
        """Test comprehensive observation extraction for bilirubin."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "Total bilirubin"
                }]
            },
            "valueQuantity": {
                "value": 0.5,
                "unit": "mg/dL"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "TotalBilirubin" in result
        assert result["TotalBilirubin"]["value"] == 0.5

    def test_extract_observation_mcode_comprehensive_alt(self, extractor):
        """Test comprehensive observation extraction for ALT."""
        observation = {
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "display": "ALT"
                }]
            },
            "valueQuantity": {
                "value": 25,
                "unit": "U/L"
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert "AlanineAminotransferase" in result
        assert result["AlanineAminotransferase"]["value"] == 25

    def test_extract_observation_mcode_comprehensive_unknown(self, extractor):
        """Test comprehensive observation extraction for unknown observation."""
        observation = {
            "code": {
                "coding": [{
                    "display": "Unknown observation"
                }]
            }
        }
        result = extractor.extract_observation_mcode_comprehensive(observation)
        assert result == {}

    def test_extract_observation_mcode_comprehensive_exception(self, extractor):
        """Test comprehensive observation extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            observation = None
            result = extractor.extract_observation_mcode_comprehensive(observation)
            assert result == {}
            mock_error.assert_called_once()

    def test_extract_procedure_mcode_surgery(self, extractor):
        """Test procedure extraction for surgical procedures."""
        procedure = {
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "387713003",
                    "display": "Mastectomy"
                }]
            },
            "performedDateTime": "2023-07-01"
        }
        result = extractor.extract_procedure_mcode(procedure)
        assert result == {
            "system": "http://snomed.info/sct",
            "code": "387713003",
            "display": "Mastectomy",
            "date": "2023-07-01"
        }

    def test_extract_procedure_mcode_non_surgical(self, extractor):
        """Test procedure extraction for non-surgical procedures."""
        procedure = {
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "123456",
                    "display": "Physical therapy"
                }]
            }
        }
        result = extractor.extract_procedure_mcode(procedure)
        assert result is None

    def test_extract_procedure_mcode_exception(self, extractor):
        """Test procedure extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            procedure = None
            result = extractor.extract_procedure_mcode(procedure)
            assert result is None
            mock_error.assert_called_once()

    def test_extract_allergy_mcode(self, extractor):
        """Test allergy extraction."""
        allergy = {
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "91936005",
                    "display": "Penicillin allergy"
                }]
            },
            "criticality": "high",
            "recordedDate": "2023-01-15"
        }
        result = extractor.extract_allergy_mcode(allergy)
        assert result == {
            "system": "http://snomed.info/sct",
            "code": "91936005",
            "display": "Penicillin allergy",
            "criticality": "high",
            "recordedDate": "2023-01-15"
        }

    def test_extract_allergy_mcode_exception(self, extractor):
        """Test allergy extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            allergy = None
            result = extractor.extract_allergy_mcode(allergy)
            assert result is None
            mock_error.assert_called_once()

    def test_extract_immunization_mcode(self, extractor):
        """Test immunization extraction."""
        immunization = {
            "vaccineCode": {
                "coding": [{
                    "system": "http://hl7.org/fhir/sid/cvx",
                    "code": "207",
                    "display": "COVID-19 vaccine"
                }]
            },
            "occurrenceDateTime": "2023-03-01",
            "status": "completed"
        }
        result = extractor.extract_immunization_mcode(immunization)
        assert result == {
            "system": "http://hl7.org/fhir/sid/cvx",
            "code": "207",
            "display": "COVID-19 vaccine",
            "occurrenceDateTime": "2023-03-01",
            "status": "completed"
        }

    def test_extract_immunization_mcode_exception(self, extractor):
        """Test immunization extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            immunization = None
            result = extractor.extract_immunization_mcode(immunization)
            assert result is None
            mock_error.assert_called_once()

    def test_extract_family_history_mcode(self, extractor):
        """Test family history extraction."""
        family_history = {
            "relationship": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "72705000",
                    "display": "Mother"
                }]
            },
            "condition": [
                {
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": "254837009",
                            "display": "Breast Cancer"
                        }]
                    }
                }
            ],
            "born": "1950"
        }
        result = extractor.extract_family_history_mcode(family_history)
        assert result["relationship"]["display"] == "Mother"
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["display"] == "Breast Cancer"
        assert result["born"] == "1950"

    def test_extract_family_history_mcode_exception(self, extractor):
        """Test family history extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            family_history = None
            result = extractor.extract_family_history_mcode(family_history)
            assert result is None
            mock_error.assert_called_once()

    def test_extract_receptor_status(self, extractor):
        """Test receptor status extraction."""
        observation = {
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "10828004",
                    "display": "Positive"
                }]
            }
        }
        result = extractor._extract_receptor_status(observation)
        assert result["interpretation"] == "Positive"

    def test_extract_receptor_status_exception(self, extractor):
        """Test receptor status extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            observation = None
            result = extractor._extract_receptor_status(observation)
            assert result == {"interpretation": "Unknown"}
            mock_error.assert_called_once()

    def test_extract_stage_info(self, extractor):
        """Test stage info extraction."""
        observation = {
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "261650005",
                    "display": "T2N1M0"
                }]
            }
        }
        result = extractor._extract_stage_info(observation)
        assert result["interpretation"] == "T2N1M0"

    def test_extract_stage_info_exception(self, extractor):
        """Test stage info extraction with exception."""
        with patch.object(extractor.logger, 'error') as mock_error:
            observation = None
            result = extractor._extract_stage_info(observation)
            assert result == {"interpretation": "Unknown"}
            mock_error.assert_called_once()