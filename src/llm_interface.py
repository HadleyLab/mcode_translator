import os
import requests
from typing import Dict, List, Optional
from .config import Config
import logging
import openai
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

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
    - genomic_variants: list of objects with "gene" and "variant" fields
    - biomarkers: list of objects with "name" and "status" fields
    - functional_characteristics: list of strings
    
    Criteria text:
    {criteria_text}
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            self.logger.error("DEEPSEEK_API_KEY environment variable not set")
        
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
        
        if not self.api_key:
            return {
                "genomic_variants": [],
                "biomarkers": [],
                "functional_characteristics": []
            }
        
        try:
            client = openai.OpenAI(
                base_url="https://api.deepseek.com/v1",
                api_key=self.api_key
            )
            
            response = client.chat.completions.create(
                model="deepseek-coder",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {str(e)}")
    def _get_empty_response(self) -> Dict:
        """Return an empty response with the expected structure"""
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
            # Clean response text by removing markdown code blocks if present
            cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
            parsed = json.loads(cleaned_text)
            
            if not isinstance(parsed, dict):
                self.logger.warning(f"LLM response is not a dictionary: {parsed}")
                return self._get_empty_response()
                
            # Validate required fields exist
            required_fields = ['genomic_variants', 'biomarkers', 'functional_characteristics']
            if not all(field in parsed for field in required_fields):
                self.logger.warning(f"Missing required fields in response: {parsed}")
                return self._get_empty_response()
                
            return parsed
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {str(e)}. Response: {response_text[:200]}")
            return self._get_empty_response()