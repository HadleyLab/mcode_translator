"""
Integration test for mCODE extraction pipeline on NCT01698281.
This test validates the complete mCODE extraction and transformation process.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import json
import os
from src.data_fetcher.fetcher import get_full_study, ClinicalTrialsAPIError
from src.nlp_engine.llm_nlp_engine import LLMNLPEngine
from src.code_extraction.code_extraction import CodeExtractionModule
from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from src.structured_data_generator.structured_data_generator import StructuredDataGenerator


class TestNCT01698281MCODEIntegration:
    """Integration tests for mCODE extraction on NCT01698281"""
    
    @pytest.fixture
    def nct_id(self):
        """Return the NCT ID for testing"""
        return "NCT01698281"
    
    @pytest.fixture
    def study_data(self, nct_id):
        """Fetch and return the study data"""
        return get_full_study(nct_id)
    
    def test_full_study_retrieval(self, study_data, nct_id):
        """Test that the full study data is retrieved correctly"""
        assert isinstance(study_data, dict)
        assert "protocolSection" in study_data
        
        protocol_section = study_data["protocolSection"]
        identification_module = protocol_section["identificationModule"]
        
        # Verify basic study information
        assert identification_module["nctId"] == nct_id
        assert "briefTitle" in identification_module
        assert "Phase 2 Trial of AEZS-108" in identification_module["briefTitle"]
        
        # Verify eligibility criteria exists
        eligibility_module = protocol_section["eligibilityModule"]
        assert "eligibilityCriteria" in eligibility_module
        criteria = eligibility_module["eligibilityCriteria"]
        assert isinstance(criteria, str)
        assert len(criteria) > 0
    
    def test_nlp_processing_integration(self, study_data):
        """Test NLP processing of eligibility criteria"""
        protocol_section = study_data["protocolSection"]
        eligibility_module = protocol_section["eligibilityModule"]
        criteria_text = eligibility_module["eligibilityCriteria"]
        
        # Initialize NLP engine
        nlp_engine = LLMNLPEngine()
        nlp_result = nlp_engine.process_text(criteria_text)
        
        # Verify NLP results
        assert hasattr(nlp_result, 'entities')
        assert hasattr(nlp_result, 'features')
        assert not nlp_result.error
        
        # Verify key features were extracted
        features = nlp_result.features
        assert "demographics" in features
        assert "biomarkers" in features
        assert "treatment_history" in features
        
        # Verify specific biomarker extraction
        biomarkers = features["biomarkers"]
        assert any(bm["name"] == "ER" and bm["status"] == "negative" for bm in biomarkers)
        assert any(bm["name"] == "PR" and bm["status"] == "negative" for bm in biomarkers)
        assert any(bm["name"] == "HER2" and bm["status"] == "negative" for bm in biomarkers)
        assert any(bm["name"] == "LHRHRECEPTOR" and bm["status"] == "positive" for bm in biomarkers)
        
        # Verify demographics
        demographics = features["demographics"]
        assert demographics["gender"] == ["female"]
        assert demographics["age"]["min"] == 18
    
    def test_code_extraction_integration(self, study_data):
        """Test code extraction from eligibility criteria"""
        protocol_section = study_data["protocolSection"]
        eligibility_module = protocol_section["eligibilityModule"]
        criteria_text = eligibility_module["eligibilityCriteria"]
        
        # Initialize code extractor
        code_extractor = CodeExtractionModule()
        code_result = code_extractor.process_criteria_for_codes(criteria_text, [])
        
        # Verify code extraction results structure
        assert isinstance(code_result, dict)
        assert "extracted_codes" in code_result
        assert "mapped_entities" in code_result
        assert "metadata" in code_result
        
        # Verify metadata
        metadata = code_result["metadata"]
        assert "total_codes" in metadata
        assert "systems_found" in metadata
        assert "errors" in metadata
    
    def test_mcode_mapping_integration(self, study_data):
        """Test mCODE mapping from extracted entities"""
        protocol_section = study_data["protocolSection"]
        eligibility_module = protocol_section["eligibilityModule"]
        criteria_text = eligibility_module["eligibilityCriteria"]
        
        # Process through NLP first
        nlp_engine = LLMNLPEngine()
        nlp_result = nlp_engine.process_text(criteria_text)
        
        # Initialize mCODE mapper
        mcode_mapper = MCODEMappingEngine()
        
        # Verify NLP output contains entities
        assert hasattr(nlp_result, 'entities'), "NLP result missing entities"
        assert len(nlp_result.entities) > 0, "NLP result has empty entities list"
        
        # Map entities to mCODE
        mapped_mcode = mcode_mapper.map_entities_to_mcode(nlp_result.entities)
        
        # Verify mCODE mapping results
        assert isinstance(mapped_mcode, list)
        
        # Should contain at least the biomarkers
        assert len(mapped_mcode) >= 4  # ER, PR, HER2, LHRH
        assert any(m['element_name'] == 'ER' and m['status'] == 'negative' for m in mapped_mcode)
        assert any(m['element_name'] == 'PR' and m['status'] == 'negative' for m in mapped_mcode)
        assert any(m['element_name'] == 'HER2' and m['status'] == 'negative' for m in mapped_mcode)
        assert any('lhrh' in m['element_name'].lower() and m['status'] == 'positive' for m in mapped_mcode)
        
        # Verify validation
        validation_result = mcode_mapper.validate_mcode_compliance({
            'mapped_elements': mapped_mcode,
            'demographics': nlp_result.features.get('demographics', {})
        })
        
        assert "valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
    
    def test_structured_data_generation_integration(self, study_data):
        """Test structured data generation from mCODE mappings"""
        protocol_section = study_data["protocolSection"]
        eligibility_module = protocol_section["eligibilityModule"]
        criteria_text = eligibility_module["eligibilityCriteria"]
        
        # Process through NLP first
        nlp_engine = LLMNLPEngine()
        nlp_result = nlp_engine.process_text(criteria_text)
        
        # Initialize mCODE mapper and generator
        mcode_mapper = MCODEMappingEngine()
        generator = StructuredDataGenerator()
        
        # Map entities to mCODE
        mapped_mcode = mcode_mapper.map_entities_to_mcode(nlp_result.entities)
        
        # Generate structured data
        structured_result = generator.generate_mcode_resources(
            mapped_mcode, 
            nlp_result.features.get('demographics', {})
        )
        
        # Verify structured data results
        assert isinstance(structured_result, dict)
        assert "bundle" in structured_result
        assert "resources" in structured_result
        assert "validation" in structured_result
        
        # Verify FHIR bundle structure
        bundle = structured_result["bundle"]
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert "entry" in bundle
        
        # Should contain at least a Patient resource
        resources = structured_result["resources"]
        assert len(resources) > 0
        assert any(res["resourceType"] == "Patient" for res in resources)
        
        # Verify validation results
        validation = structured_result["validation"]
        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
    
    def test_complete_pipeline_integration(self, study_data):
        """Test the complete mCODE extraction pipeline"""
        protocol_section = study_data["protocolSection"]
        eligibility_module = protocol_section["eligibilityModule"]
        criteria_text = eligibility_module["eligibilityCriteria"]
        
        # Step 1: NLP processing
        nlp_engine = LLMNLPEngine()
        nlp_result = nlp_engine.process_text(criteria_text)
        assert not nlp_result.error
        
        # Step 2: Code extraction
        code_extractor = CodeExtractionModule()
        code_result = code_extractor.process_criteria_for_codes(
            criteria_text, 
            nlp_result.entities if nlp_result and not nlp_result.error else None
        )
        assert not code_result.get("metadata", {}).get("errors", True)
        
        # Step 3: Combine entities for mapping
        all_entities = []
        if nlp_result and not nlp_result.error and hasattr(nlp_result, 'entities'):
            all_entities.extend(nlp_result.entities)
        
        # Add codes as entities for mapping
        if code_result and 'extracted_codes' in code_result:
            for system, codes in code_result['extracted_codes'].items():
                for code_info in codes:
                    all_entities.append({
                        'text': code_info.get('text', ''),
                        'confidence': code_info.get('confidence', 0.8),
                        'codes': {system: code_info.get('code', '')}
                    })
        
        # Step 4: mCODE mapping
        mcode_mapper = MCODEMappingEngine()
        mapped_mcode = mcode_mapper.map_entities_to_mcode(all_entities)
        assert isinstance(mapped_mcode, list)
        
        # Step 5: Generate structured data
        demographics = {}
        if nlp_result and not nlp_result.error and hasattr(nlp_result, 'features'):
            demographics = nlp_result.features.get('demographics', {})
        
        generator = StructuredDataGenerator()
        structured_result = generator.generate_mcode_resources(mapped_mcode, demographics)
        
        # Step 6: Validate mCODE compliance
        validation_result = mcode_mapper.validate_mcode_compliance({
            'mapped_elements': mapped_mcode,
            'demographics': demographics
        })
        
        # Verify final results
        assert structured_result["validation"]["valid"] is True
        assert validation_result["valid"] is True
        
        # Should have at least one resource
        assert len(structured_result["resources"]) >= 1
        
        # Should have a compliance score
        assert 0 <= validation_result.get("compliance_score", 0) <= 1


if __name__ == '__main__':
    pytest.main([__file__, "-v"])