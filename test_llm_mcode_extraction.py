import json
from src.llm_interface import LLMInterface
from src.mcode_mapping_engine import MCODEMappingEngine
from src.extraction_pipeline import ExtractionPipeline

# Sample clinical trial criteria
SAMPLE_CRITERIA = """
INCLUSION CRITERIA:
- Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
- ER/PR positive (LOINC: 16112-5, 16113-3)
- HER2 negative (LOINC: 48676-1)
- Must have PIK3CA mutation (HGNC: 8985)
- Must have received prior chemotherapy treatment (CPT: 12345)
"""

def test_llm_extraction():
    """Test LLM extraction independently"""
    print("\n=== Testing LLM Extraction ===")
    llm = LLMInterface()
    features = llm.extract_genomic_features(SAMPLE_CRITERIA)
    print("LLM Extraction Results:")
    print(json.dumps(features, indent=2))

def test_mcode_mapping():
    """Test mCODE mapping with sample LLM output"""
    print("\n=== Testing mCODE Mapping ===")
    mapper = MCODEMappingEngine()
    
    # Sample LLM output structure as strings (matching actual LLM output)
    sample_llm_output = {
        "genomic_variants": [
            "PIK3CA mutation",
            "TP53 R175H"
        ],
        "biomarkers": [
            "ER positive",
            "PR positive",
            "HER2 negative"
        ],
        "functional_characteristics": ["HR+"]
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
    else:
        print("\nFailed to generate mCODE mappings")

if __name__ == "__main__":
    test_llm_extraction()
    test_mcode_mapping() 
    test_full_pipeline()