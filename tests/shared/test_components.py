"""
Shared test components for the mCODE Translator project.
This module provides reusable components for testing across all modules.
"""

import random
from typing import Dict, List, Any, Optional


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
                resource_type = "Procedure" if treatment["type"] == "procedure" else "MedicationStatement"
                bundle["entry"].append({
                    "resource": {
                        "resourceType": resource_type,
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
        from src.nlp_engine.nlp_engine import ProcessingResult
        return ProcessingResult(
            features=self.features,
            mcode_mappings={},
            metadata={'confidence': self.confidence},
            entities=self.entities
        )
    
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
    
    def extract_mcode_features(self, text):
        """Mock implementation of extract_mcode_features"""
        self.call_count += 1
        from src.nlp_engine.nlp_engine import ProcessingResult
        return ProcessingResult(
            features=self.features,
            mcode_mappings={},
            metadata={'confidence': self.confidence},
            entities=self.entities
        )