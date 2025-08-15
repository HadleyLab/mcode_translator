import re
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNLPEngine:
    """
    A simplified NLP engine for parsing clinical trial eligibility criteria
    """
    
    def __init__(self):
        """
        Initialize the simple NLP engine with mCODE extraction patterns
        """
        logger.info("Simple NLP engine initialized")
        
        # Biomarker patterns
        self.biomarker_patterns = {
            'ER': re.compile(r'ER\s*(?:status)?\s*[:=]?\s*(positive|negative)', re.IGNORECASE),
            'PR': re.compile(r'PR\s*(?:status)?\s*[:=]?\s*(positive|negative)', re.IGNORECASE),
            'HER2': re.compile(r'HER2\s*(?:status)?\s*[:=]?\s*(positive|negative)', re.IGNORECASE),
            'PD-L1': re.compile(r'PD-?L1\s*(?:status)?\s*[:=]?\s*(positive|negative)', re.IGNORECASE),
            'Ki-67': re.compile(r'Ki-?67\s*(?:status)?\s*[:=]?\s*(positive|negative)', re.IGNORECASE)
        }
        
        # Genomic variant patterns
        self.gene_pattern = re.compile(r'\b(BRCA1|BRCA2|TP53|PIK3CA|PTEN|AKT1|ERBB2|HER2)\b', re.IGNORECASE)
        
        # Cancer condition patterns
        self.stage_pattern = re.compile(r'stage\s+(I{1,3}V?|IV)', re.IGNORECASE)
        self.cancer_type_pattern = re.compile(r'\b(breast|lung|colorectal)\s+cancer\b', re.IGNORECASE)
        
        # Treatment patterns
        self.medication_pattern = re.compile(r'\b(trastuzumab|inetetamab|pembrolizumab|doxorubicin)\b', re.IGNORECASE)
        
        # Performance status patterns
        self.ecog_pattern = re.compile(r'ECOG\s+status?\s*[:=]?\s*([0-4])', re.IGNORECASE)
        
        # Demographic patterns
        self.gender_pattern = re.compile(r'\b(male|female)\b', re.IGNORECASE)
        self.age_pattern = re.compile(r'age\s+([0-9]+)\s+to\s+([0-9]+)', re.IGNORECASE)
    
    def extract_mcode_features(self, criteria_text: str) -> Dict:
        """
        Extract mCODE features from eligibility criteria text
        
        Args:
            criteria_text: Text containing eligibility criteria
            
        Returns:
            Dictionary with mCODE features across categories
        """
        return {
            'demographics': self._extract_demographics(criteria_text),
            'cancer_condition': self._extract_cancer_condition(criteria_text),
            'genomics': {
                'genomic_variants': self._extract_genomic_variants(criteria_text)
            },
            'biomarkers': self._extract_biomarkers(criteria_text),
            'treatment': self._extract_treatment(criteria_text),
            'performance_status': self._extract_performance_status(criteria_text)
        }

    def _extract_demographics(self, text: str) -> Dict:
        demographics = {}
        
        # Extract gender
        gender_match = self.gender_pattern.search(text)
        if gender_match:
            demographics['gender'] = gender_match.group(1).capitalize()
            
        # Extract age range
        age_match = self.age_pattern.search(text)
        if age_match:
            demographics['age_range'] = f"{age_match.group(1)}-{age_match.group(2)}"
            
        return demographics

    def _extract_cancer_condition(self, text: str) -> Dict:
        condition = {}
        
        # Extract cancer type
        cancer_match = self.cancer_type_pattern.search(text)
        if cancer_match:
            condition['cancer_type'] = cancer_match.group(1).capitalize() + " cancer"
            
        # Extract stage
        stage_match = self.stage_pattern.search(text)
        if stage_match:
            condition['stage'] = stage_match.group(1).upper()
            
        return condition

    def _extract_biomarkers(self, text: str) -> List[Dict]:
        biomarkers = []
        for name, pattern in self.biomarker_patterns.items():
            match = pattern.search(text)
            if match:
                status = match.group(1).capitalize()
                biomarkers.append({
                    'name': name,
                    'status': status
                })
        return biomarkers

    def _extract_genomic_variants(self, text: str) -> List[Dict]:
        variants = []
        for match in self.gene_pattern.finditer(text):
            variants.append({
                'gene': match.group(1).upper(),
                'variant': ''
            })
        return variants

    def _extract_treatment(self, text: str) -> Dict:
        treatment = {'medications': []}
        
        # Extract medications
        for match in self.medication_pattern.finditer(text):
            treatment['medications'].append(match.group(1).capitalize())
            
        return treatment

    def _extract_performance_status(self, text: str) -> Dict:
        status = {}
        
        # Extract ECOG score
        ecog_match = self.ecog_pattern.search(text)
        if ecog_match:
            status['ecog_score'] = ecog_match.group(1)
            
        return status

    # Keep existing methods for backward compatibility
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        return text.strip()
    
    def identify_sections(self, text: str) -> Dict[str, str]:
        """
        Identify inclusion and exclusion sections in criteria text
        
        Args:
            text: Input criteria text
            
        Returns:
            Dictionary with section names and content
        """
        sections = {}
        
        # Common inclusion section headers
        inclusion_patterns = [
            r'inclusion criteria',
            r'eligible subjects',
            r'selection criteria'
        ]
        
        # Common exclusion section headers
        exclusion_patterns = [
            r'exclusion criteria',
            r'ineligible subjects',
            r'non[-\s]inclusion criteria'
        ]
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # Find inclusion section
        inclusion_start = -1
        for pattern in inclusion_patterns:
            match = re.search(pattern, text_lower)
            if match:
                inclusion_start = match.start()
                break
        
        # Find exclusion section
        exclusion_start = -1
        for pattern in exclusion_patterns:
            match = re.search(pattern, text_lower)
            if match:
                exclusion_start = match.start()
                break
        
        # Split text into sections
        if inclusion_start >= 0 and exclusion_start >= 0:
            # Both sections exist
            if inclusion_start < exclusion_start:
                # Inclusion section comes first
                sections['inclusion'] = text[inclusion_start:exclusion_start].strip()
                sections['exclusion'] = text[exclusion_start:].strip()
            else:
                # Exclusion section comes first
                sections['exclusion'] = text[exclusion_start:inclusion_start].strip()
                sections['inclusion'] = text[inclusion_start:].strip()
        elif inclusion_start >= 0:
            # Only inclusion section exists
            sections['inclusion'] = text[inclusion_start:].strip()
        elif exclusion_start >= 0:
            # Only exclusion section exists
            sections['exclusion'] = text[exclusion_start:].strip()
        else:
            # No clear sections, treat entire text as inclusion
            sections['inclusion'] = text.strip()
            
        return sections
    
    def classify_criteria(self, criteria_text: str) -> str:
        """
        Classify criteria as inclusion or exclusion
        
        Args:
            criteria_text: Criteria text to classify
            
        Returns:
            Classification: 'inclusion', 'exclusion', or 'unclear'
        """
        inclusion_indicators = [
            'must have', 'required', 'diagnosed with', 'history of',
            'able to', 'capable of', 'willing to'
        ]
        
        exclusion_indicators = [
            'must not have', 'excluded', 'ineligible', 'unwilling',
            'unable to', 'contraindicated', 'allergy to', 'intolerance to'
        ]
        
        classification = 'unclear'
        
        # Check for inclusion indicators
        if any(indicator in criteria_text.lower() for indicator in inclusion_indicators):
            classification = 'inclusion'
        
        # Check for exclusion indicators
        elif any(indicator in criteria_text.lower() for indicator in exclusion_indicators):
            classification = 'exclusion'
        
        return classification
    
    def extract_age_criteria(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract age criteria from text
        
        Args:
            text: Input text to process
            
        Returns:
            List of age criteria
        """
        age_patterns = [
            r'(age|aged?)\s*(?:of\s*)?(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?',
            r'(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?\s*(?:of\s+)?age',
            r'(?:between|from)\s+(\d+)\s*(?:and|to)\s*(\d+)'
        ]
        
        ages = []
        for pattern in age_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ages.append({
                    'text': match.group(),
                    'min_age': match.group(1) if len(match.groups()) >= 1 else None,
                    'max_age': match.group(2) if len(match.groups()) >= 2 else None,
                    'unit': 'years'
                })
        
        return ages
    
    def extract_gender_criteria(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract gender criteria from text
        
        Args:
            text: Input text to process
            
        Returns:
            List of gender criteria
        """
        gender_patterns = {
            'male': r'\b(male|men)\b',
            'female': r'\b(female|women)\b',
            'pregnant': r'\b(pregnant|nursing|breast[-\s]?feeding)\b'
        }
        
        genders = []
        for gender, pattern in gender_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                genders.append({
                    'text': match.group(),
                    'gender': gender,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return genders
    
    def extract_conditions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical conditions from text
        
        Args:
            text: Input text to process
            
        Returns:
            List of medical conditions
        """
        # Patterns for condition extraction
        condition_patterns = [
            r'(diagnosis|history|presence)\s+of\s+([^,;.]+)',
            r'(?:diagnosed|suffering|afflicted)\s+(?:with|from)\s+([^,;.]+)',
            r'([^,;.]+)\s+(?:cancer|tumor|carcinoma|malignancy|disease|disorder)'
        ]
        
        conditions = []
        for pattern in condition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                conditions.append({
                    'text': match.group(),
                    'condition': match.group(2) if len(match.groups()) >= 2 else match.group(1),
                    'confidence': 0.8  # Placeholder confidence score
                })
        
        return conditions
    
    def extract_procedures(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract procedures from text
        
        Args:
            text: Input text to process
            
        Returns:
            List of procedures
        """
        procedure_patterns = [
            r'(underwent|received|history of)\s+([^,;.]+(?:surgery|therapy|treatment|procedure))',
            r'(radiation|radiograph|ct|pet|mri|scan)\s+(?:scan|imaging|therapy)',
            r'(chemo|radio|immuno|targeted|combination)\s+therapy'
        ]
        
        procedures = []
        for pattern in procedure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                procedures.append({
                    'text': match.group(),
                    'procedure': match.group(2) if len(match.groups()) >= 2 else match.group(1),
                    'type': 'procedure'
                })
        
        return procedures
    
    def process_criteria(self, criteria_text: str) -> Dict[str, Any]:
        """
        Process eligibility criteria text and extract structured information
        
        Args:
            criteria_text: Eligibility criteria text to process
            
        Returns:
            Dictionary containing extracted information
        """
        if not criteria_text:
            return {}
        
        # Clean the text
        cleaned_text = self.clean_text(criteria_text)
        
        # Identify sections
        sections = self.identify_sections(cleaned_text)
        
        # Process each section
        result = {
            'entities': [],
            'demographics': {},
            'conditions': [],
            'procedures': [],
            'sections': sections
        }
        
        # Extract demographic information
        result['demographics'] = {
            'age': self.extract_age_criteria(cleaned_text),
            'gender': self.extract_gender_criteria(cleaned_text)
        }
        
        # Extract conditions
        result['conditions'] = self.extract_conditions(cleaned_text)
        
        # Extract procedures
        result['procedures'] = self.extract_procedures(cleaned_text)
        
        # Add metadata
        result['metadata'] = {
            'text_length': len(cleaned_text),
            'condition_count': len(result['conditions']),
            'procedure_count': len(result['procedures'])
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # This is just for testing purposes
    nlp_engine = SimpleNLPEngine()
    
    # Sample criteria text
    sample_text = """
    Inclusion Criteria:
    - Male or female patients aged 18 years or older
    - Histologically confirmed diagnosis of breast cancer
    - Must have received prior chemotherapy treatment
    - Currently receiving radiation therapy
    
    Exclusion Criteria:
    - Pregnant or nursing women
    - History of other malignancies within the past 5 years
    - Allergy to contrast agents
    """
    
    # Process the sample text
    result = nlp_engine.process_criteria(sample_text)
    print("Processing complete. Found:")
    print(f"- {result['metadata']['condition_count']} conditions")
    print(f"- {result['metadata']['procedure_count']} procedures")
    print(f"- {len(result['demographics']['age'])} age criteria")
    print(f"- {len(result['demographics']['gender'])} gender criteria")