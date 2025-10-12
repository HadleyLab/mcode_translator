"""
Unit tests for ClinicalNoteGenerator class.
"""

from unittest.mock import patch

import pytest

from src.services.clinical_note_generator import ClinicalNoteGenerator


class TestClinicalNoteGenerator:
    """Test suite for ClinicalNoteGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a ClinicalNoteGenerator instance."""
        return ClinicalNoteGenerator()

    def test_init(self, generator):
        """Test initialization."""
        assert generator.logger is not None

    def test_generate_patient_header_basic(self, generator):
        """Test patient header generation with basic demographics."""
        demographics = {"name": "John Doe", "age": "45", "gender": "Male"}
        result = generator._generate_patient_header("P001", demographics)
        assert "John Doe is a 45 year old Male Patient (ID: P001)." in result

    def test_generate_patient_header_unknown_name(self, generator):
        """Test patient header with unknown name."""
        demographics = {"name": "Unknown Patient", "age": "Unknown", "gender": "Unknown"}
        result = generator._generate_patient_header("P001", demographics)
        assert "Unknown Patient is a age unknown Unknown Patient (ID: P001)." in result

    def test_generate_patient_header_name_parsing(self, generator):
        """Test patient header with complex name parsing."""
        demographics = {"name": "Jane Mary Smith", "age": "30", "gender": "Female"}
        result = generator._generate_patient_header("P002", demographics)
        assert "Jane Mary Smith is a 30 year old Female Patient (ID: P002)." in result

    def test_generate_demographics_section_full(self, generator):
        """Test demographics section with all fields."""
        demographics = {
            "birthDate": "1980-01-01",
            "gender": "Female",
            "birthSex": "F",
            "maritalStatus": "M",
            "language": "English",
        }
        result = generator._generate_demographics_section(demographics)
        assert "Patient date of birth is 1980-01-01 (mCODE: BirthDate)" in result
        assert "Patient administrative gender is Female (mCODE: AdministrativeGender)" in result
        assert "Patient birth sex is Female (mCODE: BirthSexExtension)" in result
        assert "Patient marital status is Married (mCODE: MaritalStatus)" in result
        assert "Patient preferred language is English (mCODE: Communication)" in result

    def test_generate_demographics_section_minimal(self, generator):
        """Test demographics section with minimal fields."""
        demographics = {"gender": "Male"}
        result = generator._generate_demographics_section(demographics)
        assert "Patient administrative gender is Male (mCODE: AdministrativeGender)" in result
        assert "Patient race is White" in result
        assert "Patient ethnicity is Not Hispanic or Latino" in result

    def test_generate_demographics_section_empty(self, generator):
        """Test demographics section with no data."""
        demographics = {}
        result = generator._generate_demographics_section(demographics)
        # Even with empty demographics, race and ethnicity are always included
        assert "Patient race is White" in result
        assert "Patient ethnicity is Not Hispanic or Latino" in result

    def test_generate_cancer_diagnosis_section(self, generator):
        """Test cancer diagnosis section."""
        mcode_elements = {
            "CancerCondition": {
                "display": "Invasive ductal carcinoma (morphology)",
                "code": "444714004",
                "system": "http://snomed.info/sct",
                "onsetDateTime": "2023-06-15",
            }
        }
        result = generator._generate_cancer_diagnosis_section(mcode_elements)
        assert (
            "Invasive ductal carcinoma diagnosed on 2023-06-15 (mCODE: CancerCondition; SNOMED:444714004)"
            in result
        )

    def test_generate_cancer_diagnosis_section_no_date(self, generator):
        """Test cancer diagnosis section without date."""
        mcode_elements = {
            "CancerCondition": {
                "display": "Breast Cancer",
                "code": "254837009",
                "system": "http://snomed.info/sct",
            }
        }
        result = generator._generate_cancer_diagnosis_section(mcode_elements)
        assert "Breast Cancer (mCODE: CancerCondition; SNOMED:254837009)" in result

    def test_generate_biomarker_section(self, generator):
        """Test biomarker section with multiple markers."""
        mcode_elements = {
            "HER2ReceptorStatus": {
                "display": "Positive",
                "code": "10828004",
                "system": "http://snomed.info/sct",
            },
            "ERReceptorStatus": {
                "display": "Positive",
                "code": "373572006",
                "system": "http://snomed.info/sct",
            },
            "PRReceptorStatus": {
                "display": "Negative",
                "code": "260385009",
                "system": "http://snomed.info/sct",
            },
        }
        result = generator._generate_biomarker_section(mcode_elements)
        assert "HER2 receptor status is Positive" in result
        assert "ER receptor status is Positive" in result
        assert "PR receptor status is Negative" in result

    def test_generate_staging_section(self, generator):
        """Test staging section."""
        mcode_elements = {
            "TNMStage": {
                "display": "T2N1M0",
                "code": "261650005",
                "system": "http://snomed.info/sct",
            },
            "CancerStage": {
                "display": "Stage IIB",
                "code": "261650005",
                "system": "http://snomed.info/sct",
            },
        }
        result = generator._generate_staging_section(mcode_elements)
        assert "T2N1M0" in result
        assert "Stage IIB" in result

    def test_generate_treatments_section(self, generator):
        """Test treatments section with procedures and medications."""
        mcode_elements = {
            "CancerRelatedSurgicalProcedure": [
                {
                    "display": "Mastectomy",
                    "code": "387713003",
                    "system": "http://snomed.info/sct",
                    "performedDateTime": "2023-07-01",
                }
            ],
            "CancerRelatedMedicationStatement": [
                {
                    "display": "Tamoxifen",
                    "code": "10324",
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                }
            ],
            "CancerRelatedRadiationProcedure": [
                {
                    "display": "Radiation therapy",
                    "code": "108290001",
                    "system": "http://snomed.info/sct",
                }
            ],
        }
        result = generator._generate_treatments_section(mcode_elements)
        assert "Mastectomy performed on 2023-07-01" in result
        assert "Tamoxifen" in result
        assert "Radiation therapy" in result

    def test_generate_genetics_section(self, generator):
        """Test genetics section."""
        mcode_elements = {
            "CancerGeneticVariant": [
                {
                    "display": "BRCA1 gene mutation",
                    "code": "412734009",
                    "system": "http://snomed.info/sct",
                }
            ]
        }
        result = generator._generate_genetics_section(mcode_elements)
        assert "BRCA1 gene mutation" in result

    def test_generate_performance_section(self, generator):
        """Test performance status section."""
        mcode_elements = {
            "ECOGPerformanceStatus": {
                "display": "0 - Fully active",
                "code": "425389002",
                "system": "http://snomed.info/sct",
            },
            "KarnofskyPerformanceStatus": {
                "display": "90%",
                "code": "426927009",
                "system": "http://snomed.info/sct",
            },
        }
        result = generator._generate_performance_section(mcode_elements)
        assert "ECOG performance status is 0 - Fully active" in result
        assert "Karnofsky performance status is 90%" in result

    def test_generate_vitals_section(self, generator):
        """Test vital signs section."""
        mcode_elements = {
            "BodyWeight": {"value": "70", "unit": "kg"},
            "BodyHeight": {"value": "170", "unit": "cm"},
            "BodyMassIndex": {"value": "24.2"},
            "BloodPressure": {"systolic": "120", "diastolic": "80"},
        }
        result = generator._generate_vitals_section(mcode_elements)
        assert "Body weight is 70 kg" in result
        assert "Body height is 170 cm" in result
        assert "Body mass index is 24.2" in result
        assert "Blood pressure is 120/80 mmHg" in result

    def test_generate_lab_section(self, generator):
        """Test laboratory results section."""
        mcode_elements = {
            "Hemoglobin": {"value": "12.5", "unit": "g/dL"},
            "WhiteBloodCellCount": {"value": "8.2", "unit": "10^9/L"},
            "PlateletCount": {"value": "250", "unit": "10^9/L"},
            "Creatinine": {"value": "0.8", "unit": "mg/dL"},
            "TotalBilirubin": {"value": "0.5", "unit": "mg/dL"},
            "AlanineAminotransferase": {"value": "25", "unit": "U/L"},
        }
        result = generator._generate_lab_section(mcode_elements)
        assert "Hemoglobin is 12.5 g/dL" in result
        assert "White blood cell count is 8.2 10^9/L" in result
        assert "Platelet count is 250 10^9/L" in result
        assert "Creatinine is 0.8 mg/dL" in result
        assert "Total bilirubin is 0.5 mg/dL" in result
        assert "ALT is 25 U/L" in result

    def test_generate_comorbidities_section(self, generator):
        """Test comorbidities section."""
        mcode_elements = {
            "ComorbidCondition": [
                {"display": "Hypertension", "code": "38341003", "system": "http://snomed.info/sct"}
            ]
        }
        result = generator._generate_comorbidities_section(mcode_elements)
        assert "Hypertension" in result

    def test_generate_allergies_section(self, generator):
        """Test allergies section."""
        mcode_elements = {
            "AllergyIntolerance": [
                {
                    "display": "Penicillin allergy",
                    "code": "91936005",
                    "system": "http://snomed.info/sct",
                    "criticality": "high",
                    "recordedDate": "2023-01-15",
                }
            ]
        }
        result = generator._generate_allergies_section(mcode_elements)
        assert "Penicillin allergy recorded on 2023-01-15" in result
        assert "criticality: high" in result

    def test_generate_immunization_section(self, generator):
        """Test immunization section."""
        mcode_elements = {
            "Immunization": [
                {
                    "display": "COVID-19 vaccine",
                    "code": "207",
                    "system": "http://hl7.org/fhir/sid/cvx",
                    "occurrenceDateTime": "2023-03-01",
                    "status": "completed",
                }
            ]
        }
        result = generator._generate_immunization_section(mcode_elements)
        assert "COVID-19 vaccine administered on 2023-03-01" in result
        assert "status: completed" in result

    def test_generate_family_history_section(self, generator):
        """Test family history section."""
        mcode_elements = {
            "FamilyMemberHistory": [
                {
                    "relationship": {"display": "Mother"},
                    "conditions": [
                        {
                            "display": "Breast Cancer",
                            "code": "254837009",
                            "system": "http://snomed.info/sct",
                        }
                    ],
                    "born": "1950",
                }
            ]
        }
        result = generator._generate_family_history_section(mcode_elements)
        assert "Mother born 1950 with Breast Cancer" in result

    def test_generate_summary_success(self, generator):
        """Test successful summary generation."""
        with patch.object(generator.logger, "info") as mock_info:
            patient_id = "P001"
            mcode_elements = {
                "CancerCondition": {
                    "display": "Breast Cancer",
                    "code": "254837009",
                    "system": "http://snomed.info/sct",
                }
            }
            demographics = {"name": "Jane Doe", "age": "50", "gender": "Female"}

            result = generator.generate_summary(patient_id, mcode_elements, demographics)

            assert "Jane Doe is a 50 year old Female Patient" in result
            assert "Breast Cancer" in result
            mock_info.assert_called_once()

    def test_generate_summary_exception(self, generator):
        """Test summary generation with exception."""
        with patch.object(generator.logger, "error") as mock_error:
            # Force an exception by passing invalid data that causes issues
            with patch.object(
                generator, "_generate_patient_header", side_effect=Exception("Test error")
            ):
                result = generator.generate_summary("P001", {}, {})

            assert "Error generating clinical note" in result
            mock_error.assert_called_once()

    def test_decode_birth_sex(self, generator):
        """Test birth sex decoding."""
        assert generator._decode_birth_sex("F") == "Female"
        assert generator._decode_birth_sex("M") == "Male"
        assert generator._decode_birth_sex("UNK") == "Unknown"
        assert generator._decode_birth_sex("OTH") == "Other"
        assert generator._decode_birth_sex("X") == "X"

    def test_decode_marital_status(self, generator):
        """Test marital status decoding."""
        assert generator._decode_marital_status("M") == "Married"
        assert generator._decode_marital_status("S") == "Single"
        assert generator._decode_marital_status("D") == "Divorced"
        assert generator._decode_marital_status("W") == "Widowed"
        assert generator._decode_marital_status("UNK") == "Unknown"
        assert generator._decode_marital_status("X") == "X"

    def test_format_mcode_element(self, generator):
        """Test mCODE element formatting."""
        # SNOMED
        result = generator._format_mcode_element("TestElement", "http://snomed.info/sct", "12345")
        assert result == "(mCODE: TestElement; SNOMED:12345)"

        # LOINC
        result = generator._format_mcode_element("TestElement", "http://loinc.org", "67890")
        assert result == "(mCODE: TestElement; LOINC:67890)"

        # RxNorm
        result = generator._format_mcode_element(
            "TestElement", "http://www.nlm.nih.gov/research/umls/rxnorm", "11111"
        )
        assert result == "(mCODE: TestElement; RxNorm:11111)"

        # ICD
        result = generator._format_mcode_element(
            "TestElement", "http://hl7.org/fhir/sid/icd-10", "C50"
        )
        assert result == "(mCODE: TestElement; ICD:C50)"

        # Unknown system
        result = generator._format_mcode_element("TestElement", "http://example.com/system", "999")
        assert result == "(mCODE: TestElement; SYSTEM:999)"

    def test_generate_summary_comprehensive(self, generator):
        """Test comprehensive summary generation with all sections."""
        patient_id = "P001"
        mcode_elements = {
            "CancerCondition": {
                "display": "Breast Cancer",
                "code": "254837009",
                "system": "http://snomed.info/sct",
                "onsetDateTime": "2023-01-01",
            },
            "HER2ReceptorStatus": {
                "display": "Positive",
                "code": "10828004",
                "system": "http://snomed.info/sct",
            },
            "TNMStage": {
                "display": "T2N0M0",
                "code": "261650005",
                "system": "http://snomed.info/sct",
            },
            "BodyWeight": {"value": "65", "unit": "kg"},
            "Hemoglobin": {"value": "13.2", "unit": "g/dL"},
        }
        demographics = {
            "name": "Alice Johnson",
            "age": "45",
            "gender": "Female",
            "birthDate": "1978-03-15",
            "language": "English",
        }

        result = generator.generate_summary(patient_id, mcode_elements, demographics)

        # Check that all sections are included
        assert "Alice Johnson is a 45 year old Female Patient" in result
        assert "Patient date of birth is 1978-03-15" in result
        assert "Breast Cancer diagnosed on 2023-01-01" in result
        assert "HER2 receptor status is Positive" in result
        assert "T2N0M0" in result
        assert "Body weight is 65 kg" in result
        assert "Hemoglobin is 13.2 g/dL" in result
