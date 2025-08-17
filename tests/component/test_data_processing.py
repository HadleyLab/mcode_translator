"""
Component-based tests for data processing using the refactored approach.
"""

import pytest
from tests.shared.test_components import DataProcessingTestComponents, ClinicalTrialDataGenerator


class TestEligibilityCriteriaParsing:
    """Test eligibility criteria parsing components"""
    
    def test_inclusion_criteria_extraction(self, criteria_parser):
        """Test extraction of inclusion criteria"""
        text = """
        INCLUSION CRITERIA:
        - Histologically confirmed diagnosis of breast cancer
        - Female patients aged 18-75 years
        - ECOG performance status 0-1
        
        EXCLUSION CRITERIA:
        - Pregnant or nursing women
        - Active infection
        """
        result = criteria_parser.parse(text)
        
        assert 'inclusion' in result
        assert len(result['inclusion']) >= 3
        assert any('breast cancer' in criterion.lower() for criterion in result['inclusion'])
        assert any('female' in criterion.lower() for criterion in result['inclusion'])
        assert any('ecog' in criterion.lower() for criterion in result['inclusion'])
    
    def test_exclusion_criteria_extraction(self, criteria_parser):
        """Test extraction of exclusion criteria"""
        text = """
        INCLUSION CRITERIA:
        - Diagnosis of lung cancer
        
        EXCLUSION CRITERIA:
        - Pregnant or nursing women
        - History of other malignancies
        - Active infection requiring systemic therapy
        """
        result = criteria_parser.parse(text)
        
        assert 'exclusion' in result
        assert len(result['exclusion']) >= 3
        assert any('pregnant' in criterion.lower() for criterion in result['exclusion'])
        assert any('malignancies' in criterion.lower() for criterion in result['exclusion'])
        assert any('infection' in criterion.lower() for criterion in result['exclusion'])
    
    def test_age_restriction_parsing(self, criteria_parser):
        """Test parsing of age restrictions"""
        text = "Female patients aged 18-75 years"
        result = criteria_parser.parse(text)
        
        assert 'demographics' in result
        demographics = result['demographics']
        assert 'age' in demographics
        assert 'min' in demographics['age']
        assert 'max' in demographics['age']
        assert demographics['age']['min'] == 18
        assert demographics['age']['max'] == 75
    
    def test_performance_status_parsing(self, criteria_parser):
        """Test parsing of performance status criteria"""
        text = "ECOG performance status 0-1"
        result = criteria_parser.parse(text)
        
        assert 'performance_status' in result
        performance = result['performance_status']
        assert 'scale' in performance
        assert 'values' in performance
        assert performance['scale'] == 'ECOG'
        assert performance['values'] == [0, 1]


class TestCodeExtraction:
    """Test code extraction components"""
    
    def test_icd10cm_extraction(self, code_extractor):
        """Test extraction of ICD-10-CM codes"""
        text = "Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)"
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'extracted_codes' in codes
        extracted = codes['extracted_codes']
        assert 'ICD10CM' in extracted
        assert len(extracted['ICD10CM']) >= 1
        assert extracted['ICD10CM'][0]['code'] == 'C50.911'
        assert extracted['ICD10CM'][0]['system'] == 'ICD-10-CM'
    
    def test_rxnorm_extraction(self, code_extractor):
        """Test extraction of RxNorm codes"""
        text = "Currently taking paclitaxel (RxNorm: 57359)"
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'extracted_codes' in codes
        extracted = codes['extracted_codes']
        assert 'RxNorm' in extracted
        assert len(extracted['RxNorm']) >= 1
        assert extracted['RxNorm'][0]['code'] == '57359'
        assert extracted['RxNorm'][0]['system'] == 'RxNorm'
    
    def test_loinc_extraction(self, code_extractor):
        """Test extraction of LOINC codes"""
        text = "Laboratory values within normal limits (LOINC: 12345-6)"
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'extracted_codes' in codes
        extracted = codes['extracted_codes']
        assert 'LOINC' in extracted
        assert len(extracted['LOINC']) >= 1
        assert extracted['LOINC'][0]['code'] == '12345-6'
        assert extracted['LOINC'][0]['system'] == 'LOINC'
    
    def test_cpt_extraction(self, code_extractor):
        """Test extraction of CPT codes"""
        text = "Previous mastectomy (CPT: 19303)"
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'extracted_codes' in codes
        extracted = codes['extracted_codes']
        assert 'CPT' in extracted
        assert len(extracted['CPT']) >= 1
        assert extracted['CPT'][0]['code'] == '19303'
        assert extracted['CPT'][0]['system'] == 'CPT'
    
    def test_multiple_code_extraction(self, code_extractor):
        """Test extraction of multiple code types from text"""
        text = """
        Patient with:
        - Breast cancer (ICD-10-CM: C50.911)
        - Taking paclitaxel (RxNorm: 57359)
        - Laboratory tests (LOINC: 12345-6)
        - Previous surgery (CPT: 19303)
        """
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'extracted_codes' in codes
        extracted = codes['extracted_codes']
        assert 'ICD10CM' in extracted
        assert 'RxNorm' in extracted
        assert 'LOINC' in extracted
        assert 'CPT' in extracted
        assert len(extracted['ICD10CM']) >= 1
        assert len(extracted['RxNorm']) >= 1
        assert len(extracted['LOINC']) >= 1
        assert len(extracted['CPT']) >= 1


class TestMCODEMapping:
    """Test mCODE mapping components"""
    
    def test_condition_mapping(self, mcode_mapper):
        """Test mapping of conditions to mCODE elements"""
        entity = {'text': 'breast cancer', 'confidence': 0.9}
        result = mcode_mapper.map_concept_to_mcode(entity['text'], entity['confidence'])
        
        assert result is not None
        assert 'mcode_element' in result
        assert result['mcode_element'] == 'Condition'
        assert 'primary_code' in result
        assert result['primary_code']['code'] == 'C50.911'
        assert result['confidence'] == 0.9
    
    def test_medication_mapping(self, mcode_mapper):
        """Test mapping of medications to mCODE elements"""
        entity = {'text': 'paclitaxel', 'confidence': 0.85}
        result = mcode_mapper.map_concept_to_mcode(entity['text'], entity['confidence'])
        
        assert result is not None
        assert 'mcode_element' in result
        assert result['mcode_element'] == 'MedicationStatement'
        assert 'primary_code' in result
        assert result['primary_code']['code'] == '57359'
        assert result['confidence'] == 0.85
    
    def test_procedure_mapping(self, mcode_mapper):
        """Test mapping of procedures to mCODE elements"""
        entity = {'text': 'mastectomy', 'confidence': 0.95}
        result = mcode_mapper.map_concept_to_mcode(entity['text'], entity['confidence'])
        
        assert result is not None
        assert 'mcode_element' in result
        assert result['mcode_element'] == 'Procedure'
        assert 'primary_code' in result
        assert result['primary_code']['code'] == '19303'
        assert result['confidence'] == 0.95
    
    def test_biomarker_mapping(self, mcode_mapper):
        """Test mapping of biomarkers to mCODE elements"""
        # Test ER+ mapping
        entity = {'text': 'estrogen receptor positive', 'confidence': 0.9}
        result = mcode_mapper.map_concept_to_mcode(entity['text'], entity['confidence'])
        
        assert result is not None
        assert 'mcode_element' in result
        assert result['mcode_element'] == 'Observation'
        assert 'primary_code' in result
        assert result['primary_code']['code'] == 'LP417347-6'  # ER LOINC code
        assert result['confidence'] == 0.9
    
    def test_cross_system_mapping(self, mcode_mapper):
        """Test mapping between coding systems"""
        # Map ICD-10-CM to SNOMED CT
        result = mcode_mapper.map_code_to_mcode('C50.911', 'ICD10CM')
        
        assert result is not None
        assert 'mapped_codes' in result
        mapped = result['mapped_codes']
        assert 'SNOMEDCT' in mapped
        assert mapped['SNOMEDCT'] == '254837009'  # SNOMED CT code for breast cancer


class TestStructuredDataGeneration:
    """Test structured data generation components"""
    
    def test_patient_resource_generation(self, structured_data_generator):
        """Test generation of Patient resources"""
        demographics = {
            'gender': 'female',
            'age': '55',
            'birthDate': '1970-01-01'
        }
        
        patient = structured_data_generator.generate_patient_resource(demographics)
        
        assert patient['resourceType'] == 'Patient'
        assert patient['gender'] == 'female'
        assert patient['birthDate'] == '1970-01-01'
    
    def test_condition_resource_generation(self, structured_data_generator):
        """Test generation of Condition resources"""
        mapped_element = {
            'mcode_element': 'Condition',
            'primary_code': {'system': 'ICD10CM', 'code': 'C50.911'},
            'mapped_codes': {'SNOMEDCT': '254837009'}
        }
        
        condition = structured_data_generator.generate_condition_resource(mapped_element)
        
        assert condition['resourceType'] == 'Condition'
        assert 'meta' in condition
        assert 'profile' in condition['meta']
        assert 'mcode-primary-cancer-condition' in condition['meta']['profile'][0]
        assert condition['code']['coding'][0]['code'] == 'C50.911'
    
    def test_observation_resource_generation(self, structured_data_generator):
        """Test generation of Observation resources"""
        mapped_element = {
            'mcode_element': 'Observation',
            'primary_code': {'system': 'LOINC', 'code': 'LP417347-6'},
            'value': {'system': 'SNOMEDCT', 'code': 'LA6576-8'}
        }
        
        observation = structured_data_generator.generate_observation_resource(mapped_element)
        
        assert observation['resourceType'] == 'Observation'
        assert observation['code']['coding'][0]['code'] == 'LP417347-6'
        assert 'valueCodeableConcept' in observation
        assert observation['valueCodeableConcept']['coding'][0]['code'] == 'LA6576-8'
    
    def test_bundle_generation(self, structured_data_generator):
        """Test generation of complete FHIR bundles"""
        mapped_elements = [
            {
                'mcode_element': 'Condition',
                'primary_code': {'system': 'ICD10CM', 'code': 'C50.911'},
                'mapped_codes': {'SNOMEDCT': '254837009'}
            },
            {
                'mcode_element': 'Observation',
                'primary_code': {'system': 'LOINC', 'code': 'LP417347-6'},
                'value': {'system': 'SNOMEDCT', 'code': 'LA6576-8'}
            }
        ]
        
        demographics = {
            'gender': 'female',
            'age': '55'
        }
        
        bundle = structured_data_generator.generate_mcode_resources(mapped_elements, demographics)
        
        assert bundle['bundle']['resourceType'] == 'Bundle'
        assert bundle['bundle']['type'] == 'collection'
        assert len(bundle['bundle']['entry']) >= 2  # Patient + at least 2 resources
        
        # Verify resource types
        resource_types = [entry['resource']['resourceType'] for entry in bundle['bundle']['entry']]
        assert 'Patient' in resource_types
        assert 'Condition' in resource_types
        assert 'Observation' in resource_types


class TestDataProcessingPipeline:
    """Test the complete data processing pipeline"""
    
    def test_end_to_end_processing(self, sample_eligibility_criteria):
        """Test end-to-end processing of eligibility criteria"""
        # Step 1: Parse criteria
        from src.criteria_parser import CriteriaParser
        criteria_parser = CriteriaParser()
        parsed_criteria = criteria_parser.parse(sample_eligibility_criteria)
        
        # Step 2: Extract codes
        from src.code_extraction import CodeExtractionModule
        code_extractor = CodeExtractionModule()
        extracted_codes = code_extractor.process_criteria_for_codes(
            sample_eligibility_criteria, 
            parsed_criteria.get('entities', [])
        )
        
        # Step 3: Map to mCODE
        from src.mcode_mapping_engine import MCODEMappingEngine
        mcode_mapper = MCODEMappingEngine()
        mapped_elements = mcode_mapper.map_entities_to_mcode(parsed_criteria.get('entities', []))
        
        # Step 4: Generate structured data
        from src.structured_data_generator import StructuredDataGenerator
        structured_generator = StructuredDataGenerator()
        demographics = parsed_criteria.get('demographics', {})
        structured_data = structured_generator.generate_mcode_resources(mapped_elements, demographics)
        
        # Verify complete pipeline
        assert 'entities' in parsed_criteria
        assert 'extracted_codes' in extracted_codes
        assert len(mapped_elements) >= 1
        assert structured_data['bundle']['resourceType'] == 'Bundle'
        assert len(structured_data['bundle']['entry']) >= 2  # Patient + at least 1 other resource
    
    def test_pipeline_error_handling(self):
        """Test error handling in the processing pipeline"""
        # Test with invalid input
        invalid_text = ""
        
        # Step 1: Parse criteria (should handle gracefully)
        from src.criteria_parser import CriteriaParser
        criteria_parser = CriteriaParser()
        parsed_criteria = criteria_parser.parse(invalid_text)
        
        # Step 2: Extract codes (should handle gracefully)
        from src.code_extraction import CodeExtractionModule
        code_extractor = CodeExtractionModule()
        extracted_codes = code_extractor.process_criteria_for_codes(invalid_text)
        
        # Verify graceful handling
        assert 'entities' in parsed_criteria  # Should still have structure
        assert 'extracted_codes' in extracted_codes
        assert isinstance(extracted_codes['extracted_codes'], dict)
    
    def test_pipeline_with_complex_criteria(self, clinical_trial_generator):
        """Test pipeline with complex eligibility criteria"""
        complex_criteria = clinical_trial_generator.generate_eligibility_criteria("complex")
        
        # Process through pipeline
        from src.criteria_parser import CriteriaParser
        from src.code_extraction import CodeExtractionModule
        from src.mcode_mapping_engine import MCODEMappingEngine
        
        criteria_parser = CriteriaParser()
        parsed_criteria = criteria_parser.parse(complex_criteria)
        
        code_extractor = CodeExtractionModule()
        extracted_codes = code_extractor.process_criteria_for_codes(complex_criteria)
        
        mcode_mapper = MCODEMappingEngine()
        mapped_elements = mcode_mapper.map_entities_to_mcode(parsed_criteria.get('entities', []))
        
        # Verify processing of complex criteria
        assert len(parsed_criteria['inclusion']) > 5  # Should have many inclusion criteria
        assert len(parsed_criteria['exclusion']) > 3  # Should have several exclusion criteria
        assert len(extracted_codes['extracted_codes']) > 0  # Should extract codes
        assert len(mapped_elements) > 0  # Should map elements


class TestDataProcessingParameterized:
    """Parameterized tests for data processing components"""
    
    @pytest.mark.parametrize("cancer_type,expected_code", [
        ("breast", "C50.911"),
        ("lung", "C34.90"),
        ("colorectal", "C18.9"),
    ])
    def test_cancer_code_extraction(self, code_extractor, cancer_type, expected_code):
        """Test extraction of cancer codes for different cancer types"""
        text = f"Patient diagnosed with {cancer_type} cancer (ICD-10-CM: {expected_code})"
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'ICD10CM' in codes['extracted_codes']
        extracted_codes = codes['extracted_codes']['ICD10CM']
        assert len(extracted_codes) >= 1
        assert extracted_codes[0]['code'] == expected_code
    
    @pytest.mark.parametrize("biomarker_text,expected_code", [
        ("estrogen receptor positive", "LP417347-6"),
        ("progesterone receptor negative", "LP417348-4"),
        ("HER2 positive", "LP417351-8"),
    ])
    def test_biomarker_code_extraction(self, code_extractor, biomarker_text, expected_code):
        """Test extraction of biomarker codes"""
        text = f"Patient with {biomarker_text} (LOINC: {expected_code})"
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'LOINC' in codes['extracted_codes']
        extracted_codes = codes['extracted_codes']['LOINC']
        assert len(extracted_codes) >= 1
        assert extracted_codes[0]['code'] == expected_code
    
    @pytest.mark.parametrize("age_text,min_age,max_age", [
        ("aged 18-75 years", 18, 75),
        ("between 18 and 65 years old", 18, 65),
        ("18 years of age or older", 18, None),
    ])
    def test_age_parsing_variations(self, criteria_parser, age_text, min_age, max_age):
        """Test parsing of different age restriction formats"""
        result = criteria_parser.parse(age_text)
        
        assert 'demographics' in result
        demographics = result['demographics']
        assert 'age' in demographics
        assert demographics['age']['min'] == min_age
        if max_age is not None:
            assert demographics['age']['max'] == max_age
    
    @pytest.mark.parametrize("performance_text,scale,values", [
        ("ECOG performance status 0-1", "ECOG", [0, 1]),
        ("Karnofsky performance status >70", "Karnofsky", [70]),
        ("performance status 0-2", "ECOG", [0, 1, 2]),
    ])
    def test_performance_status_parsing_variations(self, criteria_parser, performance_text, scale, values):
        """Test parsing of different performance status formats"""
        result = criteria_parser.parse(performance_text)
        
        assert 'performance_status' in result
        performance = result['performance_status']
        assert performance['scale'] == scale
        if scale == "ECOG":
            assert all(v in performance['values'] for v in values)


if __name__ == '__main__':
    pytest.main([__file__])