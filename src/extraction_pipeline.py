import logging
from typing import Dict, Any, List
from src.code_extraction import CodeExtractionModule
from src.mcode_mapping_engine import MCODEMappingEngine

class ExtractionPipeline:
    """
    mCODE Extraction Pipeline
    Orchestrates code extraction and mCODE mapping from clinical trial eligibility criteria
    """
    
    def __init__(self, use_llm: bool = True, model: str = "deepseek-coder"):
        """
        Initialize the extraction pipeline with required components
        
        Args:
            use_llm: Whether to enable LLM-based feature extraction
            model: LLM model to use for extraction
        """
        self.logger = logging.getLogger(__name__)
        self.code_extractor = CodeExtractionModule()
        self.mcode_mapper = MCODEMappingEngine()
        self.use_llm = use_llm
        self.model = model
        if use_llm:
            from src.llm_interface import LLMInterface
            self.llm_extractor = LLMInterface()
            self.llm_extractor.model = model
        
    def process_criteria(self, criteria_text: str) -> Dict[str, Any]:
        """
        Process eligibility criteria text through full extraction pipeline
        
        Args:
            criteria_text: Text containing eligibility criteria
            
        Returns:
            Dictionary with extracted codes and mCODE mappings
        """
        # Step 1: Extract codes from text
        extracted_codes = self.code_extractor.process_criteria_for_codes(criteria_text)
        
        # Step 2: Extract genomic features if LLM enabled
        genomic_features = []
        if self.use_llm:
            genomic_features = self.llm_extractor.extract_genomic_features(criteria_text)
        
        # Step 3: Map to mCODE elements
        # DEBUG: Log genomic_features type and content
        self.logger.debug(f"genomic_features type: {type(genomic_features)}")
        self.logger.debug(f"genomic_features value: {genomic_features}")
        
        mapping_result = self.mcode_mapper.process_nlp_output({
            'codes': extracted_codes,
            'entities': [],
            'genomic_features': genomic_features.get('genomic_variants', []) if isinstance(genomic_features, dict) else genomic_features
        })
        
        # Add genomic feature counts to metadata
        if isinstance(genomic_features, dict):
            variant_count = len(genomic_features.get('genomic_variants', []))
            biomarker_count = len(genomic_features.get('biomarkers', []))
            mapping_result['metadata'].update({
                'genomic_variants_count': variant_count,
                'biomarkers_count': biomarker_count
            })

        return {
            'extracted_codes': extracted_codes,
            'mcode_mappings': mapping_result,
            'genomic_features': genomic_features,  # Include raw features
            'original_criteria': criteria_text,
            'model_used': self.model if self.use_llm else None
        }
        
    def process_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process clinical trial search results through extraction pipeline
        
        Args:
            search_results: List of clinical trial results from search
            
        Returns:
            List of enriched results with extracted mCODE data
        """
        enriched_results = []
        
        for trial in search_results:
            try:
                # Handle both dict and string trial representations
                if isinstance(trial, str):
                    criteria = trial
                    trial_data = {'raw': trial}
                else:
                    # Extract eligibility criteria from nested structure
                    protocol_section = trial.get('protocolSection', {})
                    eligibility_module = protocol_section.get('eligibilityModule', {})
                    criteria = eligibility_module.get('eligibilityCriteria', '')
                    
                    self.logger.info(f"Processing trial {trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')}")
                    self.logger.debug(f"Criteria found: {bool(criteria)}")
                    
                    trial_data = trial.copy()
                
                if not criteria:
                    continue
                    
                # Process criteria through pipeline
                processed = self.process_criteria(str(criteria))
                
                # Add to enriched result
                trial_data['mcode_data'] = {
                    **processed,
                    'original_criteria': criteria  # Include original criteria
                }
                enriched_results.append(trial_data)
                
            except Exception as e:
                self.logger.error(f"Error processing trial {trial.get('id')}: {str(e)}")
                
        return enriched_results

# Example usage
if __name__ == "__main__":
    from pprint import pprint
    
    pipeline = ExtractionPipeline()
    
    # Sample criteria text
    sample_text = """
    INCLUSION CRITERIA:
    - Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
    - Must have received prior chemotherapy treatment (CPT: 12345)
    """
    
    # Process through pipeline
    result = pipeline.process_criteria(sample_text)
    print("Extraction pipeline results:")
    pprint(result)