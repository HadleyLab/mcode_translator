import openai
from typing import Dict, List, Optional
from .config import Config
import logging

class LLMInterface:
    """
    Interface for LLM-based extraction of breast cancer genomic features
    from clinical trial eligibility criteria
    """
    
    BREAST_CANCER_PROMPT_TEMPLATE = """
    Analyze the following clinical trial eligibility criteria and extract:
    1. Genomic variants (e.g., PIK3CA, TP53, ESR1 mutations)
    2. Biomarkers (ER, PR, HER2 status)
    3. Functional characteristics (e.g., HR+, HER2+)
    
    Return JSON format with:
    - genomic_variants: list of {gene: str, variant: str}
    - biomarkers: list of {name: str, status: str}
    - functional_characteristics: list of str
    
    Criteria text:
    {criteria_text}
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
    def extract_genomic_features(self, criteria_text: str) -> Dict:
        """
        Extract breast cancer genomic features using LLM
        
        Args:
            criteria_text: Clinical trial eligibility criteria text
            
        Returns:
            Dictionary with extracted genomic features
        """
        prompt = self.BREAST_CANCER_PROMPT_TEMPLATE.format(
            criteria_text=criteria_text
        )
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {str(e)}")
            return {
                "genomic_variants": [],
                "biomarkers": [],
                "functional_characteristics": []
            }
    
    def _parse_response(self, response_text: str) -> Dict:
        """
        Parse LLM response into structured format
        
        Args:
            response_text: Raw LLM response text
            
        Returns:
            Parsed dictionary of genomic features
        """
        try:
            import json
            return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse LLM response as JSON")
            return {
                "genomic_variants": [],
                "biomarkers": [],
                "functional_characteristics": []
            }