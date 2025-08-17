import os
import json
import requests
from typing import Dict, List, Optional
from src.utils.config import Config
from .nlp_engine import ProcessingResult, NLPEngine
import logging
import openai
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class LLMNLPEngine(NLPEngine):
    """NLP Engine using LLMs for extracting mCODE features from clinical text.
    
    Implements NLPEngine abstract base class with LLM-specific processing.
    
    Specialized for breast cancer genomic feature extraction from clinical trial
    eligibility criteria. Uses prompt engineering to guide LLM responses.
    
    Attributes:
        config (Config): Engine configuration
        logger (logging.Logger): Configured logger instance
        api_key (str): API key for LLM service
        BREAST_CANCER_PROMPT_TEMPLATE (str): Template for structured LLM prompts
    """
    
    BREAST_CANCER_PROMPT_TEMPLATE = """
    **FOCUS: BREAST CANCER PATIENTS - EXTRACT ALL MENTIONED FEATURES**
    Extract ALL mCODE elements from clinical trial eligibility criteria text.
    REQUIRED OUTPUT FORMAT: Pure JSON with EXACTLY these fields:
    - genomic_variants (array)
    - biomarkers (array)
    - cancer_characteristics (object)
    - treatment_history (object)
    - performance_status (object)
    - demographics (object)

    For breast cancer, ALWAYS extract:
    - Biomarkers: ER, PR, HER2 (status: positive/negative/equivocal), PD-L1
    - Genomic variants: BRCA1, BRCA2, PIK3CA, TP53, HER2 amplification
    - Treatment history: chemotherapy drugs, radiation, surgery types
    - Cancer characteristics:
        - Stage (TNM staging preferred)
        - Tumor location (left/right breast, quadrant)
        - Metastasis sites (bone, liver, lung, brain)
    - Treatment history specific to breast cancer:
        - Surgeries: lumpectomy, mastectomy, breast-conserving surgery
        - Chemotherapy: anthracyclines, taxanes, platinum-based
        - Radiation: whole breast, partial breast, boost
        - Immunotherapy: checkpoint inhibitors
    - Performance status (ECOG 0-4, Karnofsky %)
    - Demographic criteria
    
    **CRITICAL INSTRUCTIONS:**
    1. NEVER return empty arrays/objects - if no values found, use:
       - For biomarkers/genomic variants: include with status "not mentioned"
       - For other fields: include empty strings/arrays but document why
    2. For treatment history: Extract ANY mentioned treatments
    3. For cancer characteristics: Extract stage, size, grade if mentioned
    4. If no values found in text, return:
       "genomic_variants": [{{"gene": "NOT_FOUND", "variant": "", "significance": ""}}]
       "biomarkers": [{{"name": "NOT_FOUND", "status": "", "value": ""}}]
    
    **EXAMPLE RESPONSES:**
    
    Example 1 (Simple):
    {{
        "genomic_variants": [
            {{{{"gene": "BRCA1", "variant": "c.68_69delAG", "significance": "pathogenic"}}}},
            {{{{"gene": "PIK3CA", "variant": "H1047R", "significance": "pathogenic"}}}}
        ],
        "biomarkers": [
            {{{{"name": "ER", "status": "positive", "value": ">90%"}}}},
            {{{{"name": "HER2", "status": "negative", "value": "IHC 0"}}}}
        ],
        "cancer_characteristics": {{
            "stage": "T2N1M0",
            "tumor_size": "3.2 cm",
            "metastasis_sites": ["bone"]
        }},
        "treatment_history": {{
            "surgeries": ["lumpectomy"],
            "chemotherapy": ["doxorubicin", "cyclophosphamide"],
            "radiation": ["whole breast radiation"],
            "immunotherapy": []
        }},
        "performance_status": {{
            "ecog": "1",
            "karnofsky": "90%"
        }},
        "demographics": {{
            "age": {{"min": 40, "max": 75}},
            "gender": ["female"],
            "race": [],
            "ethnicity": []
        }}
    }}
    
    Example 2 (Advanced):
    {{
        "genomic_variants": [
            {{{{"gene": "BRCA2", "variant": "c.5946delT", "significance": "pathogenic"}}}},
            {{{{"gene": "TP53", "variant": "R175H", "significance": "pathogenic"}}}},
            {{{{"gene": "HER2", "variant": "amplification", "significance": "pathogenic"}}}}
        ],
        "biomarkers": [
            {{{{"name": "ER", "status": "negative", "value": "<1%"}}}},
            {{{{"name": "PR", "status": "negative", "value": "<1%"}}}},
            {{{{"name": "HER2", "status": "positive", "value": "IHC 3+"}}}},
            {{{{"name": "PD-L1", "status": "positive", "value": "CPS >=10"}}}}
        ],
        "cancer_characteristics": {{
            "stage": "T3N2M1",
            "tumor_size": "5.1 cm",
            "metastasis_sites": ["bone", "liver"]
        }},
        "treatment_history": {{
            "surgeries": ["mastectomy"],
            "chemotherapy": ["paclitaxel", "carboplatin"],
            "radiation": ["chest wall radiation"],
            "immunotherapy": ["pembrolizumab"]
        }},
        "performance_status": {{
            "ecog": "0",
            "karnofsky": "100%"
        }},
        "demographics": {{
            "age": {{"min": 35, "max": 80}},
            "gender": ["female", "male"],
            "race": ["White", "Black or African American"],
            "ethnicity": ["Hispanic or Latino"]
        }}
    }}
    
    **CRITERIA TEXT:**
    {criteria_text}
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            self.logger.error("DEEPSEEK_API_KEY environment variable not set")
        
    def extract_mcode_features(self, criteria_text: str) -> ProcessingResult:
        """Extract mCODE features from criteria text using LLM.
        
        Args:
            criteria_text (str): Clinical trial eligibility criteria text.
                               Must be non-empty string.
            
        Returns:
            ProcessingResult: Contains:
                - features (Dict): Extracted mCODE features in standardized format
                - mcode_mappings (Dict): Placeholder for future FHIR mappings
                - metadata (Dict): Processing metadata including LLM response info
                - entities (List): Placeholder for raw entities
                - error (Optional[str]): Error message if processing failed
            
        Raises:
            ValueError: If input text is empty or invalid type
            
        Example:
            >>> engine = LLMNLPEngine()
            >>> result = engine.extract_mcode_features("ER+ HER2- breast cancer")
            >>> "ER" in result.features["biomarkers"][0]["name"]
            True
        """
        if not criteria_text or not isinstance(criteria_text, str):
            raise ValueError("Criteria text must be non-empty string")
            
        prompt = self.BREAST_CANCER_PROMPT_TEMPLATE.format(
            criteria_text=criteria_text
        )
        
        if not self.api_key:
            self.logger.warning("No API key available for LLM calls")
            return self._create_error_result("Missing API key")
        
        try:
            # For testing, return mock data if no API key
            if not self.api_key or os.getenv('TEST_MODE'):
                features = {
                    "genomic_variants": [{"gene": "BRCA1", "variant": "", "significance": ""}],
                    "biomarkers": [
                        {"name": "ER", "status": "positive", "value": ""},
                        {"name": "HER2", "status": "negative", "value": ""}
                    ],
                    "cancer_characteristics": {},
                    "treatment_history": {},
                    "performance_status": {},
                    "demographics": {}
                }
                return ProcessingResult(
                    features=features,
                    mcode_mappings={},
                    metadata={'test_data': True},
                    entities=[],
                    error=None
                )
                
            client = openai.OpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=self.api_key
            )
            
            response = client.chat.completions.create(
                model="deepseek-coder",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            if not response.choices or not response.choices[0].message.content:
                self.logger.error("Empty LLM response received")
                return self._get_empty_response()
                
            # Add debug logging of raw response
            self.logger.debug(f"Raw LLM response: {response.choices[0].message.content}")
            
            content = response.choices[0].message.content
            self.logger.debug(f"Full LLM response structure: {response}")
            self.logger.debug(f"LLM response content: {content}")
            
            try:
                self.logger.debug(f"Raw LLM response before parsing:\n{content}")
                
                # Parse and validate response
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response must be a JSON object")
                    
                # Ensure required fields exist
                required_fields = ['genomic_variants', 'biomarkers']
                for field in required_fields:
                    if field not in parsed:
                        parsed[field] = [{"name": "NOT_FOUND", "status": ""}]
                        
                # Convert to ProcessingResult
                return ProcessingResult(
                    features=parsed,
                    mcode_mappings={},
                    metadata={'source': 'LLM'},
                    entities=[],
                    error=None
                )
            except Exception as e:
                self.logger.error(f"Parsing failed: {str(e)}\nContent: {content[:200]}")
                # Return structure with NOT_FOUND markers
                return ProcessingResult(
                    features={
                        "genomic_variants": [{"gene": "PARSE_ERROR", "variant": str(e), "significance": ""}],
                        "biomarkers": [{"name": "PARSE_ERROR", "status": str(e), "value": ""}],
                        "cancer_characteristics": {},
                        "treatment_history": {},
                        "performance_status": {},
                        "demographics": {}
                    },
                    mcode_mappings={},
                    metadata={'error': True},
                    entities=[],
                    error=f"Parsing failed: {str(e)}"
                )
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {str(e)}")
            return self._create_error_result(f"LLM extraction failed: {str(e)}")
    def _get_empty_response(self) -> ProcessingResult:
        """Return an empty response with the expected structure.
        
        Returns:
            ProcessingResult with empty feature structure
        """
        return ProcessingResult(
            features={
                "genomic_variants": [],
                "biomarkers": [],
                "cancer_characteristics": {
                    "stage": "",
                    "tumor_size": "",
                    "metastasis_sites": []
                },
                "treatment_history": {
                    "surgeries": [],
                    "chemotherapy": [],
                    "radiation": [],
                    "immunotherapy": []
                },
                "performance_status": {
                    "ecog": "",
                    "karnofsky": ""
                },
                "demographics": {
                    "age": {},
                    "gender": [],
                    "race": [],
                    "ethnicity": []
                }
            },
            mcode_mappings={},
            metadata={'empty_response': True},
            entities=[]
        )
    
    def _parse_response(self, response_text: str, prompt: str) -> ProcessingResult:
        """
        Parse LLM response into structured format with validation.
        
        Args:
            response_text: Raw LLM response from LLM API
            prompt: Original prompt used (for error context)
            
        Returns:
            ProcessingResult containing parsed features or error information
            
        Raises:
            ValueError: If response cannot be parsed as valid JSON
            KeyError: If required fields are missing
        """
        try:
            import json
            import re
            
            # Clean response text
            cleaned_text = re.sub(r'^.*?(\{.*\}).*?$', r'\1', response_text, flags=re.DOTALL)
            cleaned_text = cleaned_text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON with detailed error handling
            try:
                parsed = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                # Try extracting inner JSON if wrapped in text
                match = re.search(r'\{.*\}', cleaned_text)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON response: {str(e)}")
                else:
                    raise ValueError(f"Invalid JSON response: {str(e)}")
            
            if not isinstance(parsed, dict):
                raise ValueError("LLM response must be a JSON object")
                    
            # Validate required fields
            required_fields = {
                'genomic_variants': list,
                'biomarkers': list,
                'cancer_characteristics': dict,
                'treatment_history': dict,
                'performance_status': dict,
                'demographics': dict
            }
            
            missing_fields = []
            invalid_types = []
            
            for field, expected_type in required_fields.items():
                if field not in parsed:
                    missing_fields.append(field)
                elif not isinstance(parsed[field], expected_type):
                    invalid_types.append(field)
            
            if missing_fields or invalid_types:
                error_msg = []
                if missing_fields:
                    error_msg.append(f"Missing fields: {missing_fields}")
                if invalid_types:
                    error_msg.append(f"Invalid types for: {invalid_types}")
                raise ValueError(", ".join(error_msg))
                    
            # Validate genomic variants structure with fallback
            valid_genomic_variants = []
            breast_cancer_genes = ['BRCA1', 'BRCA2', 'PIK3CA', 'TP53', 'HER2']
            
            # Handle missing 'genomic_variants' field
            if 'genomic_variants' not in parsed:
                self.logger.warning("Missing 'genomic_variants' field in LLM response")
                parsed['genomic_variants'] = []
            
            # Process variants if field exists
            if isinstance(parsed['genomic_variants'], list):
                for variant in parsed['genomic_variants']:
                    if not isinstance(variant, dict):
                        self.logger.warning(f"Invalid genomic variant type: {type(variant)}")
                        continue
                    
                    # Create a safe variant object with default values
                    safe_variant = {
                        'gene': variant.get('gene', 'Unknown'),
                        'variant': variant.get('variant', ''),
                        'significance': variant.get('significance', '')
                    }
                    
                    # If both gene and variant are missing, skip this variant
                    if safe_variant['gene'] == 'Unknown' and not safe_variant['variant']:
                        self.logger.warning(f"Skipping invalid variant with no gene or variant: {variant}")
                        continue
                        
                    # Check if gene is breast cancer relevant
                    if safe_variant['gene'] not in breast_cancer_genes:
                        self.logger.info(f"Non-breast cancer gene detected: {safe_variant['gene']}")
                        
                    valid_genomic_variants.append(safe_variant)
                parsed['genomic_variants'] = valid_genomic_variants
            else:
                self.logger.error(f"Invalid genomic_variants type: {type(parsed['genomic_variants'])}")
                parsed['genomic_variants'] = []
            
            # Validate biomarkers structure - focus on breast cancer biomarkers
            valid_biomarkers = []
            breast_cancer_biomarkers = ['ER', 'PR', 'HER2', 'PD-L1']
            
            # Handle missing 'biomarkers' field
            if 'biomarkers' not in parsed:
                self.logger.warning("Missing 'biomarkers' field in LLM response")
                parsed['biomarkers'] = []
            
            # Process biomarkers if field exists
            if isinstance(parsed['biomarkers'], list):
                for biomarker in parsed['biomarkers']:
                    if not isinstance(biomarker, dict):
                        self.logger.warning(f"Invalid biomarker type: {type(biomarker)}")
                        continue
                    if 'name' not in biomarker or 'status' not in biomarker:
                        self.logger.warning(f"Biomarker missing required fields: {biomarker}")
                        continue
                        
                    # Standardize biomarker names
                    name = biomarker['name'].upper().replace(' ', '')
                    if name == 'HER2/NEU':
                        name = 'HER2'
                        
                    # Check if biomarker is breast cancer relevant
                    if name not in breast_cancer_biomarkers:
                        self.logger.info(f"Non-breast cancer biomarker detected: {name}")
                        
                    valid_biomarkers.append({
                        'name': name,
                        'status': biomarker['status'],
                        'value': biomarker.get('value', '')
                    })
                parsed['biomarkers'] = valid_biomarkers
            else:
                self.logger.error(f"Invalid biomarkers type: {type(parsed['biomarkers'])}")
                parsed['biomarkers'] = []
            
            # Validate breast cancer characteristics
            if 'stage' in parsed['cancer_characteristics']:
                stage = parsed['cancer_characteristics'].get('stage', '')
                if stage and not any(x in stage for x in ['T', 'N', 'M']):
                    self.logger.warning(f"Non-TNM stage format detected: {stage}")
            
            return ProcessingResult(
                features=parsed,
                mcode_mappings={},
                metadata={'source': 'LLM'},
                entities=[]
            )
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Validation error in LLM response: {str(e)}")
            return self._create_error_result(f"Validation error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {str(e)}")
            return self._create_error_result(f"Unexpected error: {str(e)}")
    
    def process_text(self, text: str) -> ProcessingResult:
        """Process clinical text and extract mCODE features.
        
        Implements abstract method from NLPEngine base class.
        
        Args:
            text: Clinical text to process
            
        Returns:
            ProcessingResult containing extracted features or error information
        """
        try:
            return self.extract_mcode_features(text)
        except Exception as e:
            return self._create_error_result(f"Error processing text: {str(e)}")
    
    def _create_error_result(self, error_msg: str) -> ProcessingResult:
        """Create a ProcessingResult with error information.
        
        Args:
            error_msg: Description of the error
            
        Returns:
            ProcessingResult with error details
        """
        return ProcessingResult(
            features=self._get_empty_response().features,
            mcode_mappings={},
            metadata={'error': True},
            entities=[],
            error=error_msg
        )