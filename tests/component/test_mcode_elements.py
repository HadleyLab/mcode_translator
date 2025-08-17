"""
Component-based tests for mCODE elements using the refactored approach.
"""

import pytest
from tests.shared.test_components import MCODETestComponents


@pytest.mark.component
class TestPrimaryCancerCondition:
    """Test the Primary Cancer Condition mCODE element"""
    
    def test_condition_creation(self, mcode_mapper):
        """Test creating a primary cancer condition"""
        condition = mcode_mapper._create_mcode_resource({
            'mcode_element': 'Condition',
            'primary_code': {'system': 'ICD10CM', 'code': 'C50.911'},
            'mapped_codes': {'SNOMEDCT': '254837009'}
        })
        
        assert condition['resourceType'] == 'Condition'
        assert condition['code']['coding'][0]['code'] == 'C50.911'
        assert 'bodySite' in condition
    
    def test_condition_validation(self, mcode_mapper):
        """Test validation of primary cancer condition"""
        valid_condition = {
            'resourceType': 'Condition',
            'meta': {
                'profile': ['http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-primary-cancer-condition']
            },
            'code': {
                'coding': [{
                    'system': 'http://hl7.org/fhir/sid/icd-10-cm',
                    'code': 'C50.911'
                }]
            },
            'bodySite': {
                'coding': [{
                    'system': 'http://snomed.info/sct',
                    'code': '76752008'
                }]
            }
        }
        
        validation = mcode_mapper.validate_mcode_compliance({'Condition': valid_condition})
        assert validation['valid'] == True
        assert validation['compliance_score'] >= 0.9


@pytest.mark.component
class TestTumorMarkers:
    """Test tumor marker mCODE elements"""
    
    @pytest.mark.parametrize("biomarker_type,code,expected_positive", [
        ("ER", "LP417347-6", True),
        ("PR", "LP417348-4", True),
        ("HER2", "LP417351-8", True),
    ])
    def test_biomarker_observation_creation(self, mcode_mapper, biomarker_type, code, expected_positive):
        """Test creating biomarker observations"""
        observation = mcode_mapper._create_mcode_resource({
            'mcode_element': 'Observation',
            'primary_code': {'system': 'LOINC', 'code': code},
            'value': {'system': 'SNOMEDCT', 'code': 'LA6576-8' if expected_positive else 'LA6577-6'}
        })
        
        assert observation['resourceType'] == 'Observation'
        assert observation['resourceType'] == 'Observation'
        assert 'resourceType' in observation
    
    def test_biomarker_panel_creation(self):
        """Test creating a biomarker panel"""
        panel = MCODETestComponents.create_mcode_bundle(
            biomarkers=[
                {"code": "LP417347-6", "value": "LA6576-8"},  # ER+
                {"code": "LP417348-4", "value": "LA6577-6"},  # PR-
                {"code": "LP417351-8", "value": "LA6576-8"}   # HER2+
            ]
        )
        
        assert len(panel['entry']) == 4  # Patient + 3 biomarkers
        assert panel['entry'][1]['resource']['resourceType'] == 'Observation'


@pytest.mark.component
class TestCancerRelatedSurgicalProcedures:
    """Test cancer-related surgical procedure mCODE elements"""
    
    def test_surgical_procedure_creation(self, mcode_mapper):
        """Test creating surgical procedure resources"""
        procedure = mcode_mapper._create_mcode_resource({
            'mcode_element': 'Procedure',
            'primary_code': {'system': 'CPT', 'code': '19303'},
            'bodySite': {'system': 'SNOMEDCT', 'code': '76752008'}
        })
        
        assert procedure['resourceType'] == 'Procedure'
        assert procedure['code']['coding'][0]['code'] == '19303'
    
    def test_mastectomy_procedure(self):
        """Test mastectomy procedure specifically"""
        mastectomy = MCODETestComponents.create_treatment_resource(
            treatment_code="19303",
            treatment_type="procedure",
            date="2023-06-15"
        )
        
        assert mastectomy['resource']['resourceType'] == 'Procedure'
        assert mastectomy['resource']['code']['coding'][0]['code'] == '19303'
        assert mastectomy['resource']['performedDateTime'] == '2023-06-15'


@pytest.mark.component
class TestMedicationStatements:
    """Test medication statement mCODE elements"""
    
    def test_chemotherapy_medication(self, mcode_mapper):
        """Test creating chemotherapy medication statements"""
        medication = mcode_mapper._create_mcode_resource({
            'mcode_element': 'MedicationStatement',
            'primary_code': {'system': 'RxNorm', 'code': '57359'},
            'status': 'active'
        })
        
        assert medication['resourceType'] == 'MedicationStatement'
        assert medication['medicationCodeableConcept']['coding'][0]['code'] == '57359'
    
    def test_hormone_therapy_medication(self):
        """Test hormone therapy medication specifically"""
        hormone_therapy = MCODETestComponents.create_treatment_resource(
            treatment_code="25964",
            treatment_type="medication",
            date="2023-06-15"
        )
        
        assert hormone_therapy['resource']['resourceType'] == 'MedicationStatement'
        assert hormone_therapy['resource']['resourceType'] == 'MedicationStatement'
        assert 'effectiveDateTime' in hormone_therapy['resource']


@pytest.mark.component
class TestMCODEElementIntegration:
    """Test integration between different mCODE elements"""
    
    def test_complete_patient_profile(self):
        """Test creating a complete patient profile with multiple elements"""
        # Create a complete mCODE bundle
        bundle = MCODETestComponents.create_mcode_bundle(
            biomarkers=[
                {"code": "LP417347-6", "value": "LA6576-8"},  # ER+
                {"code": "LP417348-4", "value": "LA6577-6"},  # PR-
                {"code": "LP417351-8", "value": "LA6576-8"}   # HER2+
            ],
            conditions=[
                {
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "code": "C50.911"
                }
            ],
            treatments=[
                {
                    "type": "medication",
                    "code": {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "25964"
                    }
                },
                {
                    "type": "procedure",
                    "code": {
                        "system": "http://www.ama-assn.org/go/cpt",
                        "code": "19303"
                    }
                }
            ]
        )
        
        # Verify bundle structure
        assert bundle['resourceType'] == 'Bundle'
        assert bundle['type'] == 'collection'
        assert len(bundle['entry']) >= 5  # Patient + 3 biomarkers + condition + 2 treatments
        
        # Verify resource types
        resource_types = [entry['resource']['resourceType'] for entry in bundle['entry']]
        assert 'Patient' in resource_types
        assert resource_types.count('Observation') == 3
        assert 'Condition' in resource_types
        assert 'MedicationStatement' in resource_types
        assert 'Procedure' in resource_types
    
    def test_breast_cancer_profile_matching(self, sample_breast_cancer_profile):
        """Test matching a complete profile against breast cancer trial criteria"""
        # Create a patient bundle with positive ER and HER2
        patient_bundle = MCODETestComponents.create_mcode_bundle(
            biomarkers=[
                {"code": "LP417347-6", "value": "LA6576-8"},  # ER+
                {"code": "LP417351-8", "value": "LA6576-8"}   # HER2+
            ]
        )
        
        # Test ER+ matching
        matches_er = sample_breast_cancer_profile.matches_er_positive(patient_bundle)
        assert matches_er == True
        
        # Test HER2+ matching
        matches_her2 = sample_breast_cancer_profile.matches_her2_positive(patient_bundle)
        assert matches_her2 == True
        
        # Test TNBC matching (should be False for this patient)
        matches_tnbc = sample_breast_cancer_profile.matches_triple_negative(patient_bundle)
        assert matches_tnbc == False


@pytest.mark.component
class TestMCODEElementParameterized:
    """Parameterized tests for mCODE elements"""
    
    @pytest.mark.parametrize("cancer_type,icd10_code,snomed_code", [
        ("breast", "C50.911", "254837009"),
        ("lung", "C34.90", "254637007"),
        ("colorectal", "C18.9", "363406005"),
    ])
    def test_cancer_condition_creation(self, mcode_mapper, cancer_type, icd10_code, snomed_code):
        """Test creating cancer conditions for different cancer types"""
        condition = mcode_mapper._create_mcode_resource({
            'mcode_element': 'Condition',
            'primary_code': {'system': 'ICD10CM', 'code': icd10_code},
            'mapped_codes': {'SNOMEDCT': snomed_code}
        })
        
        assert condition['resourceType'] == 'Condition'
        assert condition['code']['coding'][0]['code'] == icd10_code
        assert condition['code']['coding'][0]['system'] == 'http://hl7.org/fhir/sid/icd-10-cm'
    
    @pytest.mark.parametrize("treatment_type,code,resource_type", [
        ("chemotherapy", "57359", "MedicationStatement"),
        ("hormone_therapy", "25964", "MedicationStatement"),
        ("surgery", "19303", "Procedure"),
        ("radiation", "77800", "Procedure"),
    ])
    def test_treatment_creation(self, mcode_mapper, treatment_type, code, resource_type):
        """Test creating different types of treatments"""
        if resource_type == "MedicationStatement":
            treatment = mcode_mapper._create_mcode_resource({
                'mcode_element': 'MedicationStatement',
                'primary_code': {'system': 'RxNorm', 'code': code},
                'status': 'active'
            })
        else:  # Procedure
            treatment = mcode_mapper._create_mcode_resource({
                'mcode_element': 'Procedure',
                'primary_code': {'system': 'CPT', 'code': code}
            })
        
        assert treatment['resourceType'] == resource_type
        if resource_type == "MedicationStatement":
            assert treatment['medicationCodeableConcept']['coding'][0]['code'] == code
        else:
            assert treatment['code']['coding'][0]['code'] == code