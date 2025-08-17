# Modular Test Components for mCODE Translator

This document outlines the modular test components that will be used across the mCODE Translator project to eliminate redundancy and promote code reuse through a component-based approach.

## Overview

Modular test components provide reusable functionality for common testing scenarios. These components encapsulate specific testing logic that can be reused across different test modules, promoting consistency and reducing duplication.

## Available Test Components

### 1. MCODE Test Components

#### `MCODETestComponents`
Reusable components for testing mCODE-specific functionality.

```python
class MCODETestComponents:
    """Reusable test components for mCODE testing scenarios"""
    
    @staticmethod
    def create_mcode_bundle(biomarkers=None, conditions=None, treatments=None):
        """Create a standardized mCODE bundle for testing"""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }
        
        # Add patient resource
        bundle["entry"].append({
            "resource": {
                "resourceType": "Patient",
                "gender": "female",
                "birthDate": "1970-01-01"
            }
        })
        
        # Add biomarker observations if provided
        if biomarkers:
            for biomarker in biomarkers:
                bundle["entry"].append({
                    "resource": {
                        "resourceType": "Observation",
                        "code": {
                            "coding": [{"code": biomarker["code"]}]
                        },
                        "valueCodeableConcept": {
                            "coding": [{"code": biomarker["value"]}]
                        }
                    }
                })
        
        # Add conditions if provided
        if conditions:
            for condition in conditions:
                bundle["entry"].append({
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [condition]
                        }
                    }
                })
        
        # Add treatments if provided
        if treatments:
            for treatment in treatments:
                bundle["entry"].append({
                    "resource": {
                        "resourceType": "Procedure" if treatment["type"] == "procedure" else "MedicationStatement",
                        "code": {
                            "coding": [treatment["code"]]
                        }
                    }
                })
        
        return bundle
    
    @staticmethod
    def create_biomarker_observation(biomarker_type, value):
        """Create a biomarker observation resource"""
        biomarker_codes = {
            "ER": "LP417347-6",
            "PR": "LP417348-4",
            "HER2": "LP417351-8"
        }
        
        value_codes = {
            "positive": "LA6576-8",
            "negative": "LA6577-6"
        }
        
        return {
            "resource": {
                "resourceType": "Observation",
                "code": {
                    "coding": [{"code": biomarker_codes.get(biomarker_type, "")}]
                },
                "valueCodeableConcept": {
                    "coding": [{"code": value_codes.get(value.lower(), "")}]
                }
            }
        }
    
    @staticmethod
    def create_condition_resource(condition_code, body_site=None):
        """Create a condition resource"""
        condition = {
            "resource": {
                "resourceType": "Condition",
                "code": {
                    "coding": [{
                        "system": "http://hl7.org/fhir/sid/icd-10-cm",
                        "code": condition_code
                    }]
                }
            }
        }
        
        if body_site:
            condition["resource"]["bodySite"] = {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": body_site
                }]
            }
        
        return condition
    
    @staticmethod
    def create_treatment_resource(treatment_code, treatment_type="medication", date=None):
        """Create a treatment resource"""
        resource_type = "MedicationStatement" if treatment_type == "medication" else "Procedure"
        
        treatment = {
            "resource": {
                "resourceType": resource_type,
                "code": {
                    "coding": [{
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm" if treatment_type == "medication" else "http://www.ama-assn.org/go/cpt",
                        "code": treatment_code
                    }]
                }
            }
        }
        
        if date:
            if treatment_type == "medication":
                treatment["resource"]["effectiveDateTime"] = date
            else:
                treatment["resource"]["performedDateTime"] = date
        
        return treatment
```

### 2. NLP Test Components

#### `NLPTestComponents`
Reusable components for testing NLP engine functionality.

```python
class NLPTestComponents:
    """Reusable test components for NLP testing scenarios"""
    
    @staticmethod
    def create_nlp_result(entities=None, features=None, demographics=None):
        """Create a standardized NLP result for testing"""
        return {
            "entities": entities or [],
            "features": features or {},
            "demographics": demographics or {},
            "confidence": 0.95
        }
    
    @staticmethod
    def create_entity(text, entity_type, confidence=0.9):
        """Create an entity for NLP testing"""
        return {
            "text": text,
            "type": entity_type,
            "confidence": confidence,
            "start": 0,
            "end": len(text)
        }
    
    @staticmethod
    def create_feature(feature_type, values):
        """Create a feature for NLP testing"""
        return {
            feature_type: values
        }
```

### 3. Data Processing Test Components

#### `DataProcessingTestComponents`
Reusable components for testing data processing functionality.

```python
class DataProcessingTestComponents:
    """Reusable test components for data processing testing scenarios"""
    
    @staticmethod
    def create_code_extraction_result(codes=None, metadata=None):
        """Create a standardized code extraction result for testing"""
        return {
            "extracted_codes": codes or {},
            "metadata": metadata or {
                "total_codes": 0,
                "systems_found": [],
                "processing_time": 0.0
            }
        }
    
    @staticmethod
    def create_mcode_mapping_result(mapped_elements=None, mcode_structure=None, validation=None):
        """Create a standardized mCODE mapping result for testing"""
        return {
            "mapped_elements": mapped_elements or [],
            "mcode_structure": mcode_structure or {},
            "validation": validation or {
                "valid": True,
                "errors": [],
                "compliance_score": 1.0
            }
        }
    
    @staticmethod
    def create_structured_data_result(resources=None, validation=None):
        """Create a standardized structured data result for testing"""
        return {
            "resources": resources or [],
            "validation": validation or {
                "valid": True,
                "errors": [],
                "compliance_score": 1.0
            }
        }
```

## Usage Examples

### Using MCODE Test Components

```python
def test_biomarker_matching():
    """Test biomarker matching with MCODE test components"""
    # Create a test bundle with positive ER biomarker
    bundle = MCODETestComponents.create_mcode_bundle(
        biomarkers=[{"code": "LP417347-6", "value": "LA6576-8"}]
    )
    
    # Test matching logic
    profile = BreastCancerProfile()
    matches = profile.matches_er_positive(bundle)
    assert matches == True
```

### Using NLP Test Components

```python
def test_nlp_entity_extraction():
    """Test NLP entity extraction with NLP test components"""
    # Create a test entity
    entity = NLPTestComponents.create_entity("breast cancer", "CONDITION", 0.95)
    
    # Test NLP processing
    nlp_engine = RegexNLPEngine()
    result = nlp_engine.process_entities([entity["text"]])
    assert len(result) > 0
```

### Using Data Processing Test Components

```python
def test_code_extraction():
    """Test code extraction with data processing test components"""
    # Create a test result
    result = DataProcessingTestComponents.create_code_extraction_result(
        codes={"ICD10CM": [{"code": "C50.911", "system": "ICD-10-CM"}]},
        metadata={"total_codes": 1, "systems_found": ["ICD10CM"]}
    )
    
    # Test processing logic
    assert result["metadata"]["total_codes"] == 1
```

## Benefits of Modular Test Components

1. **Reduced Code Duplication**: Common testing logic is encapsulated in reusable components
2. **Consistent Test Implementation**: All tests use the same standardized approaches
3. **Easier Maintenance**: Changes to components propagate to all tests that use them
4. **Improved Test Reliability**: Standardized components reduce implementation errors
5. **Faster Test Development**: Pre-built components accelerate test creation
6. **Enhanced Test Coverage**: Comprehensive components ensure thorough testing