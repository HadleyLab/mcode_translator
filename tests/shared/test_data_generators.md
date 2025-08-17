# Test Data Generators for mCODE Translator

This document outlines the test data generators and mock objects that will be used across the mCODE Translator project to eliminate redundancy and promote code reuse.

## Overview

Test data generators provide utilities for creating consistent, realistic test data for various testing scenarios. Mock objects provide controlled implementations of external dependencies for isolated testing.

## Test Data Generators

### 1. Clinical Trial Data Generators

#### `ClinicalTrialDataGenerator`
Generates realistic clinical trial data for testing.

```python
class ClinicalTrialDataGenerator:
    """Generates clinical trial data for testing"""
    
    @staticmethod
    def generate_eligibility_criteria(complexity="simple", cancer_type="breast"):
        """Generate sample eligibility criteria text"""
        if complexity == "simple":
            return f"""
            INCLUSION CRITERIA:
            - Histologically confirmed diagnosis of {cancer_type} cancer
            - Age 18 years or older
            - Eastern Cooperative Oncology Group (ECOG) performance status 0-1
            
            EXCLUSION CRITERIA:
            - Pregnant or nursing women
            - Active infection requiring systemic therapy
            """
        elif complexity == "medium":
            return f"""
            INCLUSION CRITERIA:
            - Histologically confirmed diagnosis of {cancer_type} cancer (ICD-10-CM: C50.911)
            - Female patients aged 18-75 years
            - Eastern Cooperative Oncology Group (ECOG) performance status of 0 or 1
            - Adequate organ function as defined by laboratory values
            - Measurable disease per RECIST 1.1 criteria
            
            EXCLUSION CRITERIA:
            - Pregnant or nursing women
            - History of other malignancies within 5 years
            - Active infection requiring systemic therapy
            - Known hypersensitivity to study treatment components
            """
        else:  # complex
            return f"""
            INCLUSION CRITERIA:
            - Histologically confirmed diagnosis of {cancer_type} cancer (ICD-10-CM: C50.911)
            - Female patients aged 18-75 years
            - Eastern Cooperative Oncology Group (ECOG) performance status of 0 or 1
            - Adequate organ function as defined by laboratory values:
              * Absolute neutrophil count ≥ 1,500/mm³
              * Platelet count ≥ 100,000/mm³
              * Hemoglobin ≥ 9 g/dL
              * Total bilirubin ≤ 1.5 × upper limit of normal (ULN)
              * AST/ALT ≤ 2.5 × ULN (≤ 5 × ULN for patients with liver metastases)
              * Creatinine clearance ≥ 60 mL/min
            - Measurable disease per RECIST 1.1 criteria
            - No prior systemic anticancer therapy within 21 days
            - Recovery from all prior therapy-related toxicities to ≤ Grade 1
            
            EXCLUSION CRITERIA:
            - Pregnant or nursing women
            - History of other malignancies within 5 years (except adequately treated basal cell or squamous cell skin cancer, in situ cervical cancer)
            - Active infection requiring systemic therapy
            - Known hypersensitivity to study treatment components
            - Uncontrolled intercurrent illness including, but not limited to:
              * Ongoing or active infection
              * Symptomatic congestive heart failure
              * Unstable angina pectoris
              * Cardiac arrhythmia
              * Psychiatric illness/social situations that would limit compliance
            - Known brain metastases
            - History of allergic reactions attributed to compounds of similar chemical or biologic composition
            """
    
    @staticmethod
    def generate_clinical_trial_data(cancer_type="breast", phase="II", nct_id=None):
        """Generate sample clinical trial data"""
        cancer_conditions = {
            "breast": {
                "condition": "Breast Cancer",
                "icd10_code": "C50.911",
                "snomed_code": "254837009"
            },
            "lung": {
                "condition": "Lung Cancer",
                "icd10_code": "C34.90",
                "snomed_code": "254637007"
            },
            "colorectal": {
                "condition": "Colorectal Cancer",
                "icd10_code": "C18.9",
                "snomed_code": "363406005"
            }
        }
        
        condition_info = cancer_conditions.get(cancer_type, cancer_conditions["breast"])
        
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id or f"NCT{random.randint(10000000, 99999999)}",
                    "briefTitle": f"{cancer_type.capitalize()} Cancer Treatment Trial",
                    "officialTitle": f"A Phase {phase} Study of Novel Treatment for {condition_info['condition']}"
                },
                "eligibilityModule": {
                    "eligibilityCriteria": ClinicalTrialDataGenerator.generate_eligibility_criteria("medium", cancer_type),
                    "healthyVolunteers": False
                },
                "descriptionModule": {
                    "briefSummary": f"This phase {phase} trial studies how well a novel treatment works in treating patients with {condition_info['condition']}."
                }
            }
        }
    
    @staticmethod
    def generate_multiple_trials(count=5, cancer_types=None):
        """Generate multiple clinical trials"""
        if cancer_types is None:
            cancer_types = ["breast", "lung", "colorectal"]
        
        trials = []
        for i in range(count):
            cancer_type = random.choice(cancer_types)
            trial = ClinicalTrialDataGenerator.generate_clinical_trial_data(
                cancer_type=cancer_type,
                nct_id=f"NCT{i+1:08d}"
            )
            trials.append(trial)
        
        return trials
```

### 2. Patient Data Generators

#### `PatientDataGenerator`
Generates realistic patient data for testing.

```python
class PatientDataGenerator:
    """Generates patient data for testing"""
    
    @staticmethod
    def generate_patient_demographics(age_range=(18, 75), gender_distribution=None):
        """Generate sample patient demographics"""
        if gender_distribution is None:
            gender_distribution = {"male": 0.4, "female": 0.6}
        
        # Randomly select gender based on distribution
        rand_val = random.random()
        cumulative = 0
        selected_gender = "female"  # default
        for gender, probability in gender_distribution.items():
            cumulative += probability
            if rand_val <= cumulative:
                selected_gender = gender
                break
        
        # Generate age within range
        age = random.randint(age_range[0], age_range[1])
        birth_year = 2023 - age
        birth_date = f"{birth_year}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        
        return {
            "gender": selected_gender,
            "age": str(age),
            "birthDate": birth_date
        }
    
    @staticmethod
    def generate_biomarker_profile(biomarker_types=None):
        """Generate a biomarker profile"""
        if biomarker_types is None:
            biomarker_types = ["ER", "PR", "HER2"]
        
        profile = {}
        for biomarker in biomarker_types:
            # Randomly assign positive or negative status
            status = random.choice(["positive", "negative"])
            profile[biomarker] = status
        
        return profile
    
    @staticmethod
    def generate_medical_history(condition_count=3, treatment_count=2):
        """Generate sample medical history"""
        conditions = [
            {"code": "E11.9", "description": "Type 2 Diabetes Mellitus"},
            {"code": "I10", "description": "Essential Hypertension"},
            {"code": "J45.909", "description": "Unspecified Asthma"},
            {"code": "F32.9", "description": "Major Depressive Disorder"},
            {"code": "K70.30", "description": "Alcoholic Cirrhosis of Liver"}
        ]
        
        treatments = [
            {"code": "705040", "description": "Lisinopril"},
            {"code": "860975", "description": "Metformin"},
            {"code": "1156605", "description": "Albuterol"},
            {"code": "197527", "description": "Sertraline"},
            {"code": "198039", "description": "Furosemide"}
        ]
        
        selected_conditions = random.sample(conditions, min(condition_count, len(conditions)))
        selected_treatments = random.sample(treatments, min(treatment_count, len(treatments)))
        
        return {
            "conditions": selected_conditions,
            "treatments": selected_treatments
        }
```

## Mock Objects

### 1. ClinicalTrials API Mock

#### `MockClinicalTrialsAPI`
Mock implementation of the ClinicalTrials.gov API client.

```python
class MockClinicalTrialsAPI:
    """Mock ClinicalTrials.gov API client for testing"""
    
    def __init__(self):
        self.call_count = 0
        self.rate_limit_exceeded = False
    
    def get_study_fields(self, search_expr, fields, max_studies=100, fmt="json"):
        """Mock implementation of get_study_fields"""
        self.call_count += 1
        
        # Simulate rate limiting
        if self.call_count > 10:
            self.rate_limit_exceeded = True
            raise Exception("API rate limit exceeded")
        
        # Generate mock response based on search expression
        if "NCTId =" in search_expr:
            # Specific NCT ID search
            nct_id = search_expr.split('"')[1] if '"' in search_expr else "NCT00000000"
            return {
                "StudyFields": [
                    {
                        "NCTId": [nct_id],
                        "BriefTitle": [f"Mock Study for {nct_id}"]
                    }
                ]
            }
        else:
            # General search
            study_count = min(max_studies, 5)  # Limit to 5 for testing
            studies = []
            for i in range(study_count):
                studies.append({
                    "NCTId": [f"NCT{i+1:08d}"],
                    "BriefTitle": [f"Mock Study {i+1}"],
                    "Condition": [search_expr],
                    "OverallStatus": ["Recruiting"]
                })
            
            return {"StudyFields": studies}
    
    def get_full_study(self, nct_id):
        """Mock implementation of get_full_study"""
        return {
            "protocolSection": {
                "identificationModule": {
                    "nctId": nct_id,
                    "briefTitle": f"Full Mock Study for {nct_id}"
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Mock eligibility criteria"
                }
            }
        }
```

### 2. Cache Manager Mock

#### `MockCacheManager`
Mock implementation of the cache manager.

```python
class MockCacheManager:
    """Mock cache manager for testing"""
    
    def __init__(self, config=None):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get(self, key):
        """Mock implementation of get"""
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None
    
    def set(self, key, value, ttl=None):
        """Mock implementation of set"""
        self.cache[key] = value
    
    def clear(self):
        """Mock implementation of clear"""
        self.cache.clear()
    
    def get_stats(self):
        """Get cache statistics"""
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "cache_size": len(self.cache)
        }
```

### 3. NLP Engine Mocks

#### `MockNLPEngine`
Base mock NLP engine with configurable behavior.

```python
class MockNLPEngine:
    """Base mock NLP engine for testing"""
    
    def __init__(self, entities=None, features=None, confidence=0.9):
        self.entities = entities or []
        self.features = features or {}
        self.confidence = confidence
        self.call_count = 0
    
    def process_criteria(self, text):
        """Mock implementation of process_criteria"""
        self.call_count += 1
        return {
            "entities": self.entities,
            "features": self.features,
            "confidence": self.confidence
        }
    
    def process_entities(self, texts):
        """Mock implementation of process_entities"""
        self.call_count += 1
        return self.entities

class MockRegexNLPEngine(MockNLPEngine):
    """Mock regex NLP engine"""
    pass

class MockSpacyNLPEngine(MockNLPEngine):
    """Mock SpaCy NLP engine"""
    pass

class MockLLMNLPEngine(MockNLPEngine):
    """Mock LLM NLP engine"""
    pass
```

## Usage Examples

### Using Clinical Trial Data Generators

```python
def test_trial_processing():
    """Test trial processing with generated data"""
    # Generate a clinical trial
    trial_data = ClinicalTrialDataGenerator.generate_clinical_trial_data(
        cancer_type="lung",
        phase="III"
    )
    
    # Process the trial
    assert "protocolSection" in trial_data
    assert trial_data["protocolSection"]["identificationModule"]["nctId"] is not None
```

### Using Patient Data Generators

```python
def test_patient_matching():
    """Test patient matching with generated data"""
    # Generate patient demographics
    demographics = PatientDataGenerator.generate_patient_demographics(
        age_range=(30, 65),
        gender_distribution={"male": 0.3, "female": 0.7}
    )
    
    # Generate biomarker profile
    biomarkers = PatientDataGenerator.generate_biomarker_profile(
        biomarker_types=["ER", "PR", "HER2"]
    )
    
    # Test matching logic
    assert demographics["age"] is not None
    assert len(biomarkers) == 3
```

### Using Mock Objects

```python
def test_api_with_mock():
    """Test API functionality with mock objects"""
    # Create mock API client
    mock_api = MockClinicalTrialsAPI()
    
    # Test search functionality
    result = mock_api.get_study_fields("cancer", ["NCTId", "BriefTitle"], 2)
    
    # Verify results
    assert "StudyFields" in result
    assert len(result["StudyFields"]) == 2
    assert mock_api.call_count == 1
```

## Benefits of Test Data Generators and Mock Objects

1. **Consistent Test Data**: All tests use standardized, realistic data
2. **Reduced Maintenance**: Changes to data generation logic propagate to all tests
3. **Controlled Testing**: Mock objects provide predictable behavior for isolated testing
4. **Improved Test Coverage**: Generators can create diverse test scenarios
5. **Faster Test Development**: Pre-built generators accelerate test creation
6. **Enhanced Reliability**: Mock objects eliminate external dependencies during testing