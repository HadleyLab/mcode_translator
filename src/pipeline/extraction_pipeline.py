import logging
from typing import Dict, Any, List
from src.code_extraction.code_extraction import CodeExtractionModule
from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from src.pipeline.breast_cancer_profile import BreastCancerProfile
from src.matching_engine.matching_engine import MatchingEngine

class ExtractionPipeline:
    """
    mCODE Extraction Pipeline
    Orchestrates code extraction and mCODE mapping from clinical trial eligibility criteria
    """
    
    def __init__(self, engine_type: str = "LLM", model: str = "deepseek-coder"):
        """
        Initialize the extraction pipeline with required components
        
        Args:
            engine_type: Type of NLP engine to use (Regex, SpaCy, LLM)
            model: LLM model to use for extraction (if engine_type is LLM)
        """
        self.logger = logging.getLogger(__name__)
        self.code_extractor = CodeExtractionModule()
        if hasattr(self, 'cache_manager'):
            self.code_extractor.cache_manager = self.cache_manager
        self.mcode_mapper = MCODEMappingEngine()
        self.breast_cancer_profile = BreastCancerProfile()
        self.matching_engine = MatchingEngine()
        self.engine_type = engine_type
        self.model = model
        if engine_type == "LLM":
            from src.nlp_engine.llm_nlp_engine import LLMNLPEngine
            self.llm_extractor = LLMNLPEngine()
            self.llm_extractor.model = model
        
    def process_criteria(self, criteria_text: str) -> Dict[str, Any]:
        """
        Process eligibility criteria text through full extraction pipeline
        
        Args:
            criteria_text: Text containing eligibility criteria
            
        Returns:
            Dictionary with extracted codes and mCODE mappings
            
        Raises:
            ValueError: If invalid engine configuration is detected
        """
        self.logger.info(f"Starting criteria processing with engine: {self.engine_type}/{self.model}")
        self.logger.debug(f"Input criteria text: {criteria_text[:200]}...")
        # Step 1: Extract codes from text (with engine-specific caching)
        cache_key = f"{self.engine_type}_{hash(criteria_text)}"
        self.logger.debug(f"Using cache key: {cache_key}")
        extracted_codes = self.code_extractor.process_criteria_for_codes(criteria_text, cache_key=cache_key)
        self.logger.info(f"Extracted {len(extracted_codes)} codes")
        
        # Step 2: Extract mCODE features using configured engine
        # - LLM: Uses DeepSeek API for comprehensive extraction
        # - Regex/SpaCy: Uses pattern matching and medical NLP
        mcode_features = {}
        self.logger.debug(f"Starting mCODE feature extraction with engine: {self.engine_type}")
        if self.engine_type == "LLM":
            if not hasattr(self, 'llm_extractor'):
                raise ValueError("LLM engine requested but not initialized")
            self.logger.info(f"Using LLM engine: {self.llm_extractor.model}")
            try:
                result = self.llm_extractor.extract_mcode_features(criteria_text)
                mcode_features = result.features if hasattr(result, 'features') else {}
                self.logger.info(f"Extracted {len(mcode_features.get('genomic_variants', []))} genomic variants")
                self.logger.info(f"Extracted {len(mcode_features.get('biomarkers', []))} biomarkers")
            except Exception as e:
                self.logger.exception(f"LLM extraction failed: {str(e)}")
                # Return minimal structure to prevent pipeline failure
                mcode_features = {
                    'genomic_variants': [{"gene": "EXTRACTION_ERROR", "variant": str(e), "significance": ""}],
                    'biomarkers': [{"name": "EXTRACTION_ERROR", "status": str(e), "value": ""}],
                    'cancer_characteristics': {},
                    'treatment_history': {},
                    'performance_status': {},
                    'demographics': {}
                }
        else:
            self.logger.info("Using non-LLM extraction")
        
        # Step 3: Map extracted features to mCODE elements
        # - Validates breast cancer specific features
        # - Converts to standardized mCODE format
        self.logger.debug(f"mcode_features type: {type(mcode_features)}")
        self.logger.debug(f"mcode_features value: {mcode_features}")
        
        # Prepare full feature set for mapping with validation
        features = {
            'codes': extracted_codes,
            'entities': [],
            'genomic_features': mcode_features.get('genomic_variants', []),
            'biomarkers': mcode_features.get('biomarkers', []),
            'cancer_characteristics': mcode_features.get('cancer_characteristics', {}),
            'treatment_history': mcode_features.get('treatment_history', {}),
            'performance_status': mcode_features.get('performance_status', {}),
            'demographics': mcode_features.get('demographics', {})
        }
        
        # Validate and process breast cancer-specific features
        try:
            validated_result = self.breast_cancer_profile.validate_features(features)
            features = validated_result['validated_features']
        except ValueError as e:
            self.logger.warning(f"Breast cancer validation failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error validating breast cancer profile: {str(e)}")
            # Add detailed debug info
            self.logger.debug(f"Features structure: {features.keys()}")
            self.logger.debug(f"Features content: {features}")
        
        self.logger.debug(f"Features being sent to mapper: {features.keys()}")
        mapping_result = self.mcode_mapper.process_nlp_output(features)
        self.logger.debug(f"Received mapping result with keys: {mapping_result.keys()}")
        
        # Add genomic feature counts to metadata
        if isinstance(mcode_features, dict):
            variant_count = len(mcode_features.get('genomic_variants', []))
            biomarker_count = len(mcode_features.get('biomarkers', []))
            if 'original_mappings' in mapping_result and 'metadata' in mapping_result['original_mappings']:
                mapping_result['original_mappings']['metadata']['genomic_variants_count'] = variant_count
                mapping_result['original_mappings']['metadata']['biomarkers_count'] = biomarker_count

        # Prepare final output structure with:
        # - Extracted codes
        # - Mapped mCODE elements
        # - Original criteria for reference
        # - Engine metadata
        features_output = {
            'genomic_variants': mcode_features.get('genomic_variants', []),
            'biomarkers': mcode_features.get('biomarkers', []),
            'cancer_characteristics': mcode_features.get('cancer_characteristics', {}),
            'treatment_history': mcode_features.get('treatment_history', {}),
            'performance_status': mcode_features.get('performance_status', {}),
            'demographics': mcode_features.get('demographics', {})
        }
        
        return {
            'extracted_codes': extracted_codes,
            'mcode_mappings': mapping_result,
            'features': features_output,
            'metadata': {
                'genomic_variants_count': variant_count if self.engine_type == "LLM" else mapping_result['metadata'].get('genomic_variants_count', 0),
                'biomarkers_count': biomarker_count if self.engine_type == "LLM" else mapping_result['metadata'].get('biomarkers_count', 0)
            },
            'original_criteria': criteria_text,
            'model_used': self.model if self.engine_type == "LLM" else None
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
                    
                    trial_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', 'unknown')
                    self.logger.info(f"Processing trial {trial_id}")
                    self.logger.debug(f"Criteria found: {bool(criteria)}")
                    
                    trial_data = trial.copy()
                    
                    # Validate genomic features structure if present
                    if 'genomic_features' in trial_data:
                        if not isinstance(trial_data['genomic_features'], (list, dict)):
                            self.logger.error(f"Invalid genomic_features type in trial {trial_id}\n"
                                            f"Expected list or dict, got: {type(trial_data['genomic_features'])}\n"
                                            f"Raw value: {str(trial_data['genomic_features'])[:200]}")
                            trial_data['genomic_features'] = []
                
                if not criteria:
                    continue
                    
                try:
                    # Process criteria through pipeline with additional validation
                    criteria_text = str(criteria)
                    if not criteria_text.strip():
                        continue
                        
                    processed = self.process_criteria(criteria_text)
                    
                    # Validate genomic variants structure
                    if 'genomic_features' in processed:
                        if not isinstance(processed['genomic_features'], (list, dict)):
                            self.logger.error(f"Invalid genomic_features in processed result for trial {trial_id}\n"
                                            f"Input criteria: {criteria_text[:200]}...\n"
                                            f"Genomic features: {str(processed.get('genomic_features'))[:500]}")
                            processed['genomic_features'] = []
                            
                except Exception as e:
                    self.logger.error(f"Error processing criteria for trial {trial_id}: {str(e)}")
                    continue
                
                # Add to enriched result
                trial_data['mcode_data'] = {
                    **processed,
                    'original_criteria': criteria  # Include original criteria
                }
                enriched_results.append(trial_data)
                
            except Exception as e:
                self.logger.error(f"Error processing trial {trial.get('id')}: {str(e)}")
                
        return enriched_results
        
    def match_patient_to_trials(self, patient_profile: Dict, trials: List[Dict]) -> List[Dict]:
        """
        Match a patient profile to enriched clinical trials
        
        Args:
            patient_profile: Dictionary with patient's mCODE features
            trials: List of trials enriched with mCODE data
            
        Returns:
            List of matching trials with scores and reasons
        """
        try:
            # Validate patient profile before matching
            validated_result = self.breast_cancer_profile.validate_features(patient_profile.copy())
            validated_profile = validated_result['validated_features']
            return self.matching_engine.match_trials(validated_profile, trials)
        except Exception as e:
            self.logger.error(f"Patient-trial matching failed: {str(e)}")
            return []

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