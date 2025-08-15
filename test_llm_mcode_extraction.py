import json
from src.llm_interface import LLMInterface
from src.mcode_mapping_engine import MCODEMappingEngine
from src.extraction_pipeline import ExtractionPipeline

# Sample clinical trial criteria
SAMPLE_CRITERIA = """
INCLUSION CRITERIA:
- Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
- ER positive (>90%), PR negative (<1%), HER2 equivocal (IHC 2+) (LOINC: 16112-5, 16113-3, 48676-1)
- Must have BRCA2 c.5946delT pathogenic mutation (HGNC: 1101)
- Must have received prior chemotherapy (doxorubicin, cyclophosphamide) and radiation therapy
- Stage T2N1M0 with metastasis to bone
- ECOG performance status 1
- Age between 35-70 years
"""

def test_llm_extraction():
    """Test LLM extraction independently"""
    print("\n=== Testing LLM Extraction ===")
    llm = LLMInterface()
    features = llm.extract_mcode_features(SAMPLE_CRITERIA)
    print("LLM Extraction Results:")
    print(json.dumps(features, indent=2))

def test_mcode_mapping():
    """Test mCODE mapping with sample LLM output"""
    print("\n=== Testing mCODE Mapping ===")
    mapper = MCODEMappingEngine()
    
    # Sample LLM output structure (matching actual LLM output)
    sample_llm_output = {
        "genomic_variants": [
            {"gene": "BRCA2", "variant": "c.5946delT", "significance": "pathogenic"}
        ],
        "biomarkers": [
            {"name": "ER", "status": "positive", "value": ">90%"},
            {"name": "PR", "status": "negative", "value": "<1%"},
            {"name": "HER2", "status": "equivocal", "value": "IHC 2+"}
        ],
        "cancer_characteristics": {
            "stage": "T2N1M0",
            "tumor_size": "",
            "metastasis_sites": ["bone"]
        },
        "treatment_history": {
            "surgeries": [],
            "chemotherapy": ["doxorubicin", "cyclophosphamide"],
            "radiation": ["radiation therapy"],
            "immunotherapy": []
        },
        "performance_status": {
            "ecog": "1",
            "karnofsky": ""
        },
        "demographics": {
            "age": {"min": 35, "max": 70},
            "gender": [],
            "race": [],
            "ethnicity": []
        }
    }
    
    # Process through mapper
    result = mapper.process_nlp_output({
        "genomic_features": sample_llm_output,
        "entities": [],
        "codes": {
            "extracted_codes": {
                "ICD10CM": [{"code": "C50.911", "system": "ICD10CM"}],
                "LOINC": [
                    {"code": "16112-5", "system": "LOINC"},
                    {"code": "16113-3", "system": "LOINC"},
                    {"code": "48676-1", "system": "LOINC"}
                ],
                "CPT": [{"code": "12345", "system": "CPT"}]
            }
        }
    })
    
    print("mCODE Mapping Results:")
    print(json.dumps(result, indent=2))

def test_full_pipeline():
    """Test the full extraction pipeline"""
    print("\n=== Testing Full Pipeline ===")
    pipeline = ExtractionPipeline(use_llm=True)
    result = pipeline.process_criteria(SAMPLE_CRITERIA)
    
    print("Pipeline Results:")
    print(json.dumps(result, indent=2))
    
    # Check if mCODE mappings were generated
    if result['mcode_mappings']['mapped_elements']:
        print("\nSuccessfully generated mCODE mappings!")
        print(f"Mapped {len(result['mcode_mappings']['mapped_elements'])} elements")
        
        # Validate breast cancer features
        biomarkers = result['features'].get('biomarkers', [])
        genomic_variants = result['features'].get('genomic_variants', [])
        stage = result['features'].get('cancer_characteristics', {}).get('stage', '')
        
        print("\nBreast Cancer Feature Validation:")
        print(f"- Found {len(biomarkers)} biomarkers: {[b['name'] for b in biomarkers]}")
        print(f"- Found {len(genomic_variants)} genomic variants: {[v['gene'] for v in genomic_variants]}")
        print(f"- Stage: {stage}")
        
        # Basic assertions
        assert any(b['name'] == 'ER' for b in biomarkers), "ER biomarker missing"
        assert any(v['gene'] == 'BRCA2' for v in genomic_variants), "BRCA2 variant missing"
        assert 'T2N1M0' in stage, "Stage not extracted correctly"
        print("\nAll breast cancer features validated!")
    else:
        print("\nFailed to generate mCODE mappings")

# NCT04963608 Eligibility Criteria
NCT04963608_CRITERIA = """
Inclusion Criteria:

1. Patients with stage IV breast cancer
2. Patients with HER2 positive status
3. Patients that received Inetetamab
4. Patients that began Inetetamab therapy prior to June 30, 2021.

Exclusion Criteria:

Patients treated with an investigational anticancer agent Inetetamab
"""

def test_nct04963608_extraction():
    """Test extraction for real NCT04963608 trial"""
    print("\n=== Testing NCT04963608 Extraction ===")
    pipeline = ExtractionPipeline(use_llm=True)
    result = pipeline.process_criteria(NCT04963608_CRITERIA)
    
    print("NCT04963608 Extraction Results:")
    print(json.dumps(result, indent=2))
    
    # Check key mCODE elements
    features = result['features']
    biomarkers = features.get('biomarkers', [])
    cancer_condition = features.get('cancer_characteristics', {})
    treatment = features.get('treatment_history', {})
    
    # Verify HER2 biomarker was extracted
    assert any(b.get('name') == 'HER2' and b.get('status') == 'positive' for b in biomarkers), \
        "HER2 positive biomarker missing"
    
    # Verify cancer stage
    assert cancer_condition.get('stage') == 'IV', "Stage IV not extracted"
    
    # Verify treatment history contains Inetetamab
    assert any('Inetetamab' in drug for drug in treatment.get('chemotherapy', [])), \
        "Inetetamab treatment missing"
    
    print("\nNCT04963608 extraction validated successfully!")

if __name__ == "__main__":
    test_llm_extraction()
    test_mcode_mapping()
    test_full_pipeline()
    test_nct04963608_extraction()