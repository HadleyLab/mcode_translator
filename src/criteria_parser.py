import re
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CriteriaParser:
    """
    Component for parsing and structuring eligibility criteria from clinical trial records
    """
    
    def __init__(self):
        """
        Initialize the Criteria Parser
        """
        logger.info("Criteria Parser initialized")
        
        # Define patterns for criteria section identification
        self.section_patterns = {
            'inclusion': [
                r'inclusion criteria',
                r'eligible subjects',
                r'selection criteria',
                r'patient selection',
                r'entry criteria'
            ],
            'exclusion': [
                r'exclusion criteria',
                r'ineligible subjects',
                r'non[-\s]inclusion criteria',
                r'exclusionary criteria',
                r'contraindications'
            ]
        }
        
        # Define patterns for structured elements
        self.structured_patterns = {
            'age': [
                r'(?:age|aged?)\s*(?:of\s*)?(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?',
                r'(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?\s*(?:of\s+)?age',
                r'(?:between|from)\s+(\d+)\s*(?:and|to)\s*(\d+)'
            ],
            'gender': [
                r'\b(male|men)\b',
                r'\b(female|women)\b',
                r'\b(pregnant|nursing|breast[-\s]?feeding)\b'
            ],
            'performance_status': [
                r'ECOG\s*[-\s]?\s*(\d)',
                r'WHO\s*[-\s]?\s*(\d)',
                r'Karnofsky\s*(\d{2})'
            ],
            'lab_values': [
                r'(?:WBC|Hemoglobin|Platelets|ANC|Creatinine|ALT|AST|Bilirubin|LDH|Albumin|INR|PTT|aPTT|CrCl|eGFR|TSH|T3|T4|PSA|CA-125|CEA|AFP|HCG|HbA1c|Glucose|Potassium|Sodium|Calcium|Magnesium|Phosphorus|BUN|Creatinine|pH|pCO2|pO2|HCO3|O2 saturation|O2 sat|SpO2|ECMO|CPAP|BiPAP|NIPPV|NIV|IMV|MV|Ventilator|ECMO|CPB|CABG|PCI|PTCA|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|CABG|C