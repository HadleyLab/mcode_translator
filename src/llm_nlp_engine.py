import os
import requests
from typing import Dict, List, Optional
from .config import Config
import logging
import openai
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class LLMNLPEngine:
    """
    NLP Engine for LLM-based extraction of breast cancer genomic features
    from clinical trial eligibility criteria
    """
    
    BREAST_CANCER_PROMPT_TEMPLATE = """
    **FOCUS: BREAST CANCER PATIENTS**
    Extract mCODE elements specific to breast cancer from clinical trial eligibility criteria.
    Pay special attention to:
    - Biomarkers: ER, PR, HER2 (status: positive/negative/equivocal), PD-L1
    - Genomic variants: BRCA1, BRCA2, PIK3CA, TP53, HER2 amplification
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
    
    **OUTPUT REQUIREMENTS:**
    - Return PURE JSON ONLY - no additional text or markdown
    - Use the exact JSON structure below
    - For biomarkers, use standardized names: 'ER', 'PR', 'HER2', 'PD-L1'
    - For genomic variants, use HGNC gene symbols
    - Extract ALL mentioned features regardless of cancer type
    
    **EXAMPLE RESPONSES:**
    
    Example 1 (Simple):
    {{
        "genomic_variants": [
            {{"gene": "BRCA1", "variant": "c.68_69delAG", "significance": "pathogenic"}},
            {{"gene": "PIK3CA", "variant": "H1047R", "significance": "pathogenic"}}
        ],
        "biomarkers": [
            {{"name": "ER", "status": "positive", "value": ">90%"}},
            {{"name": "HER2", "status": "negative", "value": "IHC 0"}}
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
            {{"gene": "BRCA2", "variant": "c.5946delT", "significance": "pathogenic"}},
            {{"gene": "TP53", "variant": "R175H", "significance": "pathogenic"}},
            {{"gene": "HER2", "variant": "amplification", "significance": "pathogenic"}}
        ],
        "biomarkers": [
            {{"name": "ER", "status": "negative", "value": "<1%"}},
            {{"name": "PR", "status": "negative", "value": "<1%"}},
            {{"name": "HER2", "status": "positive", "value": "IHC 3+"}},
            {{"name": "PD-L1", "status": "positive", "value": "CPS >=10"}}
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
        
    def extract_mcode_features(self, criteria_text: str) -> Dict:
        """
        Extract all mCODE features using LLM
        
        Args:
            criteria_text: Clinical trial eligibility criteria text
            
        Returns:
            Dictionary with extracted mCODE features across categories
        """
        if not criteria_text or not isinstance(criteria_text, str):
            return self._get_empty_response()
            
        prompt = self.BREAST_CANCER_PROMPT_TEMPLATE.format(
            criteria_text=criteria_text
        )
        
        if not self.api_key:
            self.logger.warning("No API key available for LLM calls")
            return self._get_empty_response()
        
        try:
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
            self.logger.info(f"Full LLM response structure: {response}")
            self.logger.debug(f"LLM response content: {content}")
            
            try:
                # First try parsing normally
                parsed = self._parse_response(content, prompt)
                return parsed
            except Exception as e:
                self.logger.error(f"Parsing failed: {str(e)}\nContent: {content[:200]}")
                # Return test data if parsing fails
                return self._get_empty_response()
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {str(e)}")
            return self._get_empty_response()
    def _get_empty_response(self) -> Dict:
        """Return an empty response with the expected structure"""
        return {
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
        }
    
    def _parse_response(self, response_text: str, prompt: str) -> Dict:
        """
        Parse LLM response into structured format with breast cancer validation
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed dictionary of genomic features
        """
        try:
            import json
            # Clean response text by removing markdown code blocks if present
            cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(cleaned_text)
            
            if not isinstance(parsed, dict):
                self.logger.error(f"LLM response is not a dictionary\n"
                                f"Response: {response_text[:500]}\n"
                                f"Parsed: {parsed}")
                return self._get_empty_response()
                
            # Validate required fields exist
            required_fields = [
                'genomic_variants',
                'biomarkers',
                'cancer_characteristics',
                'treatment_history',
                'performance_status',
                'demographics'
            ]
            if not all(field in parsed for field in required_fields):
                self.logger.error(f"Missing required fields in LLM response\n"
                                f"Response: {response_text[:500]}\n"
                                f"Missing fields: {[f for f in required_fields if f not in parsed]}")
                return self._get_empty_response()
                
            # Validate genomic variants structure with fallback
            valid_genomic_variants = []
            breast_cancer_genes = ['BRCA1', 'BRCA2', 'PIK3CA', 'TP53', 'HER2']
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
            
            # Validate biomarkers structure - focus on breast cancer biomarkers
            valid_biomarkers = []
            breast_cancer_biomarkers = ['ER', 'PR', 'HER2', 'PD-L1']
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
            
            # Validate breast cancer characteristics
            if 'stage' in parsed['cancer_characteristics']:
                stage = parsed['cancer_characteristics'].get('stage', '')
                if stage and not any(x in stage for x in ['T', 'N', 'M']):
                    self.logger.warning(f"Non-TNM stage format detected: {stage}")
            
            return parsed
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}\n"
                            f"Response: {response_text[:500]}")
            return self._get_empty_response()