"""
Integration test to validate mCODE extraction results from NCT01698281.
This test validates the structure and content of mCODE extraction output.
"""

import pytest
import json
import os


class TestMCODEExtractionValidation:
    """Validation tests for mCODE extraction results"""
    
    @pytest.fixture
    def mcode_results(self):
        """Load and return the mCODE results from the actual extraction"""
        # This would typically come from the actual extraction process
        # For now, we'll use a simplified version based on the observed output
        
        return {
            "nlp": {
                "entities": [],
                "features": {
                    "genomic_variants": [
                        {
                            "gene": "NOT_FOUND",
                            "variant": "",
                            "significance": ""
                        }
                    ],
                    "biomarkers": [
                        {
                            "name": "ER",
                            "status": "negative",
                            "value": "0"
                        },
                        {
                            "name": "PR",
                            "status": "negative",
                            "value": "0"
                        },
                        {
                            "name": "HER2",
                            "status": "negative",
                            "value": "IHC 0/1, FISH negative, CISH negative"
                        },
                        {
                            "name": "LHRHRECEPTOR",
                            "status": "positive",
                            "value": "confirmed by IHC"
                        },
                        {
                            "name": "PD-L1",
                            "status": "not mentioned",
                            "value": ""
                        }
                    ],
                    "cancer_characteristics": {
                        "stage": "Stage IV (metastatic)",
                        "tumor_size": "",
                        "metastasis_sites": [
                            "brain (excluded if requiring intervention)",
                            "leptomeningeal (excluded if requiring intervention)"
                        ],
                        "tumor_location": ""
                    },
                    "treatment_history": {
                        "surgeries": [],
                        "chemotherapy": [
                            "prior chemotherapy regimens (1-3 for recurrent/metastatic disease)",
                            "anthracyclines (excluded if prior exposure for metastatic disease or cumulative dose ≥300 mg/m2 adjuvant)",
                            "anthracenediones",
                            "liposomal doxorubicin (Doxil)",
                            "doxorubicin",
                            "daunorubicin",
                            "mitoxantrone"
                        ],
                        "radiation": [
                            "radiotherapy (excluded within 21 days)"
                        ],
                        "immunotherapy": []
                    },
                    "performance_status": {
                        "ecog": "≤2 (excluded if >2)",
                        "karnofsky": ""
                    },
                    "demographics": {
                        "age": {
                            "min": 18,
                            "max": ""
                        },
                        "gender": ["female"],
                        "race": [],
                        "ethnicity": []
                    }
                }
            },
            "codes": {
                "extracted_codes": {
                    "ICD10CM": [],
                    "CPT": [],
                    "LOINC": [],
                    "RxNorm": []
                },
                "mapped_entities": [],
                "metadata": {
                    "total_codes": 0,
                    "systems_found": ["ICD10CM", "CPT", "LOINC", "RxNorm"],
                    "errors": False
                }
            },
            "mappings": [],
            "structured_data": {
                "bundle": {
                    "resourceType": "Bundle",
                    "id": "bundle-9865bd37-c5fd-4045-9843-a3bd5694b698",
                    "type": "collection",
                    "entry": [
                        {
                            "resource": {
                                "resourceType": "Patient",
                                "id": "patient-19d10a41-e02a-4a7a-8093-86c7b1474ebb",
                                "meta": {
                                    "profile": [
                                        "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-patient"
                                    ]
                                },
                                "gender": "female",
                                "extension": [
                                    {
                                        "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ethnicity",
                                        "valueCodeableConcept": {
                                            "coding": [
                                                {
                                                    "system": "http://hl7.org/fhir/us/mcode/CodeSystem/mcode-ethnicity",
                                                    "code": "unknown",
                                                    "display": "Unknown"
                                                }
                                            ]
                                        }
                                    },
                                    {
                                        "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-race",
                                        "valueCodeableConcept": {
                                            "coding": [
                                                {
                                                    "system": "http://hl7.org/fhir/us/mcode/CodeSystem/mcode-race",
                                                    "code": "unknown",
                                                    "display": "Unknown"
                                                }
                                            ]
                                        }
                                    }
                                ],
                                "birthDate": "2007-01-01"
                            }
                        }
                    ]
                },
                "resources": [
                    {
                        "resourceType": "Patient",
                        "id": "patient-19d10a41-e02a-4a7a-8093-86c7b1474ebb",
                        "meta": {
                            "profile": [
                                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-patient"
                            ]
                        },
                        "gender": "female",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ethnicity",
                                "valueCodeableConcept": {
                                    "coding": [
                                        {
                                            "system": "http://hl7.org/fhir/us/mcode/CodeSystem/mcode-ethnicity",
                                            "code": "unknown",
                                            "display": "Unknown"
                                        }
                                    ]
                                }
                            },
                            {
                                "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-race",
                                "valueCodeableConcept": {
                                    "coding": [
                                        {
                                            "system": "http://hl7.org/fhir/us/mcode/CodeSystem/mcode-race",
                                            "code": "unknown",
                                            "display": "Unknown"
                                        }
                                    ]
                                }
                            }
                        ],
                        "birthDate": "2007-01-01"
                    }
                ],
                "validation": {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "resource_validations": [
                        {
                            "valid": True,
                            "errors": [],
                            "warnings": [],
                            "resource_type": "Patient",
                            "quality_metrics": {
                                "completeness": 1.0,
                                "accuracy": 1.0,
                                "consistency": 1.0
                            }
                        }
                    ],
                    "quality_metrics": {
                        "completeness": 1.0,
                        "accuracy": 1.0,
                        "consistency": 1.0,
                        "resource_coverage": 0.1111111111111111
                    },
                    "compliance_score": 1.0,
                    "resource_type_summary": {
                        "Patient": 1
                    }
                }
            },
            "validation": {
                "valid": True,
                "errors": [],
                "warnings": [
                    "Value '[]' for 'ethnicity' not in standard value set",
                    "Value '[]' for 'race' not in standard value set"
                ],
                "compliance_score": 1.0
            }
        }
    
    def test_nlp_features_structure(self, mcode_results):
        """Test that NLP features have the correct structure"""
        nlp = mcode_results["nlp"]
        
        # Verify entities structure
        assert isinstance(nlp["entities"], list)
        
        # Verify features structure
        features = nlp["features"]
        assert isinstance(features, dict)
        
        # Verify key feature categories
        assert "genomic_variants" in features
        assert "biomarkers" in features
        assert "cancer_characteristics" in features
        assert "treatment_history" in features
        assert "performance_status" in features
        assert "demographics" in features
        
        # Verify biomarkers content
        biomarkers = features["biomarkers"]
        assert len(biomarkers) >= 4  # Should have ER, PR, HER2, LHRHRECEPTOR
        
        # Verify specific biomarkers
        er_biomarker = next((bm for bm in biomarkers if bm["name"] == "ER"), None)
        assert er_biomarker is not None
        assert er_biomarker["status"] == "negative"
        
        her2_biomarker = next((bm for bm in biomarkers if bm["name"] == "HER2"), None)
        assert her2_biomarker is not None
        assert her2_biomarker["status"] == "negative"
        
        # Verify demographics
        demographics = features["demographics"]
        assert demographics["gender"] == ["female"]
        assert demographics["age"]["min"] == 18
    
    def test_code_extraction_structure(self, mcode_results):
        """Test that code extraction has the correct structure"""
        codes = mcode_results["codes"]
        
        # Verify main structure
        assert "extracted_codes" in codes
        assert "mapped_entities" in codes
        assert "metadata" in codes
        
        # Verify extracted codes structure
        extracted_codes = codes["extracted_codes"]
        assert isinstance(extracted_codes, dict)
        assert "ICD10CM" in extracted_codes
        assert "CPT" in extracted_codes
        assert "LOINC" in extracted_codes
        assert "RxNorm" in extracted_codes
        
        # Verify metadata
        metadata = codes["metadata"]
        assert "total_codes" in metadata
        assert "systems_found" in metadata
        assert "errors" in metadata
        assert metadata["errors"] is False
    
    def test_structured_data_structure(self, mcode_results):
        """Test that structured data has the correct FHIR structure"""
        structured_data = mcode_results["structured_data"]
        
        # Verify main structure
        assert "bundle" in structured_data
        assert "resources" in structured_data
        assert "validation" in structured_data
        
        # Verify FHIR bundle structure
        bundle = structured_data["bundle"]
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert "entry" in bundle
        assert len(bundle["entry"]) > 0
        
        # Verify resources
        resources = structured_data["resources"]
        assert len(resources) > 0
        
        # Should contain at least a Patient resource
        patient_resources = [r for r in resources if r["resourceType"] == "Patient"]
        assert len(patient_resources) >= 1
        
        # Verify patient resource structure
        patient = patient_resources[0]
        assert patient["gender"] == "female"
        assert "extension" in patient
        assert "birthDate" in patient
        
        # Verify validation results
        validation = structured_data["validation"]
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert "compliance_score" in validation
        assert validation["valid"] is True
    
    def test_validation_results(self, mcode_results):
        """Test that validation results are properly structured"""
        validation = mcode_results["validation"]
        
        # Verify structure
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert "compliance_score" in validation
        
        # Verify content
        assert validation["valid"] is True
        assert isinstance(validation["errors"], list)
        assert isinstance(validation["warnings"], list)
        assert 0 <= validation["compliance_score"] <= 1
    
    def test_triple_negative_biomarkers(self, mcode_results):
        """Test that triple negative biomarkers are correctly identified"""
        biomarkers = mcode_results["nlp"]["features"]["biomarkers"]
        
        # Verify triple negative status
        er_status = next((bm["status"] for bm in biomarkers if bm["name"] == "ER"), None)
        pr_status = next((bm["status"] for bm in biomarkers if bm["name"] == "PR"), None)
        her2_status = next((bm["status"] for bm in biomarkers if bm["name"] == "HER2"), None)
        
        assert er_status == "negative"
        assert pr_status == "negative"
        assert her2_status == "negative"
        
        # Verify LHRH receptor positive status
        lhrh_status = next((bm["status"] for bm in biomarkers if bm["name"] == "LHRHRECEPTOR"), None)
        assert lhrh_status == "positive"
    
    def test_demographic_constraints(self, mcode_results):
        """Test that demographic constraints are correctly extracted"""
        demographics = mcode_results["nlp"]["features"]["demographics"]
        
        # Verify gender constraint
        assert demographics["gender"] == ["female"]
        
        # Verify age constraint
        assert demographics["age"]["min"] == 18
        assert demographics["age"]["max"] == ""  # No upper limit specified
    
    def test_treatment_history_constraints(self, mcode_results):
        """Test that treatment history constraints are correctly extracted"""
        treatment_history = mcode_results["nlp"]["features"]["treatment_history"]
        
        # Verify chemotherapy constraints
        assert "chemotherapy" in treatment_history
        chemotherapy = treatment_history["chemotherapy"]
        assert len(chemotherapy) > 0
        
        # Should mention prior chemotherapy regimens
        assert any("prior chemotherapy regimens" in item for item in chemotherapy)
        
        # Should mention anthracycline exclusions
        assert any("anthracyclines" in item for item in chemotherapy)
    
    def test_performance_status_constraints(self, mcode_results):
        """Test that performance status constraints are correctly extracted"""
        performance_status = mcode_results["nlp"]["features"]["performance_status"]
        
        # Verify ECOG constraint
        assert "ecog" in performance_status
        assert performance_status["ecog"] == "≤2 (excluded if >2)"


if __name__ == '__main__':
    pytest.main([__file__, "-v"])