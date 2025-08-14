import logging
from typing import Dict, Any, List
from .code_extraction import CodeExtractionModule
from .mcode_mapping_engine import MCODEMappingEngine

class ExtractionPipeline:
    """
    mCODE Extraction Pipeline
    Orchestrates code extraction and mCODE mapping from clinical trial eligibility criteria
    """
    
    def __init__(self):
        """
        Initialize the extraction pipeline with required components
        """
        self.logger = logging.getLogger(__name__)
        self.code_extractor = CodeExtractionModule()
        self.mcode_mapper = MCODEMappingEngine()
        
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
        
        # Step 2: Map to mCODE elements
        mapping_result = self.mcode_mapper.process_nlp_output({
            'codes': extracted_codes,
            'entities': []
        })
        
        return {
            'extracted_codes': extracted_codes,
            'mcode_mappings': mapping_result
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
                    criteria = trial.get('protocolSection', {}).get('eligibilityModule', {}).get('eligibilityCriteria', '')
                    trial_data = trial.copy()
                
                if not criteria:
                    continue
                    
                # Process criteria through pipeline
                processed = self.process_criteria(str(criteria))
                
                # Add to enriched result
                trial_data['mcode_data'] = processed
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