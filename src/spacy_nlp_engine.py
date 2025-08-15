import re
import spacy
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Add the src directory to the path so we can import the code extraction module
sys.path.insert(0, os.path.dirname(__file__))

from code_extraction import CodeExtractionModule
from mcode_mapping_engine import MCODEMappingEngine
from structured_data_generator import StructuredDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpacyNLPEngine:
    """
    Natural Language Processing Engine for parsing clinical trial eligibility criteria
    """
    
    def __init__(self):
        """
        Initialize the NLP engine with medical models
        """
        try:
            # Load the medical NLP model
            self.nlp = spacy.load("en_core_sci_md")
            logger.info("Medical NLP model loaded successfully")
        except OSError:
            logger.error("Medical NLP model not found. Please install with: python -m spacy download en_core_sci_md")
            raise
        except Exception as e:
            logger.error(f"Error loading medical NLP model: {str(e)}")
            raise
            
        # Initialize the code extraction module
        self.code_extractor = CodeExtractionModule()
        
        # Initialize the mCODE mapping engine
        self.mcode_mapper = MCODEMappingEngine()
        
        # Initialize the Structured Data Generator
        self.structured_data_generator = StructuredDataGenerator()
            
        # Define custom patterns for clinical trial specific entities
        self._define_custom_patterns()
    
    def _define_custom_patterns(self):
        """
        Define custom patterns for clinical trial specific entities
        """
        from spacy.matcher import PhraseMatcher
        
        # Create matcher for custom patterns
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Define patterns for age expressions
        age_patterns = [
            "age", "aged", "older", "younger", "years old", "year old"
        ]
        self.age_phrases = [self.nlp.make_doc(text) for text in age_patterns]
        self.matcher.add("AGE_EXPRESSION", self.age_phrases)
        
        # Define patterns for gender references
        gender_patterns = [
            "male", "female", "men", "women", "pregnant", "nursing", "breast-feeding"
        ]
        self.gender_phrases = [self.nlp.make_doc(text) for text in gender_patterns]
        self.matcher.add("GENDER_REFERENCE", self.gender_phrases)
        
        # Define inclusion indicators
        inclusion_patterns = [
            "must have", "required", "diagnosed with", "history of",
            "able to", "capable of", "willing to"
        ]
        self.inclusion_phrases = [self.nlp.make_doc(text) for text in inclusion_patterns]
        self.matcher.add("INCLUSION_INDICATOR", self.inclusion_phrases)
        
        # Define exclusion indicators
        exclusion_patterns = [
            "must not have", "excluded", "ineligible", "unwilling",
            "unable to", "contraindicated", "allergy to", "intolerance to"
        ]
        self.exclusion_phrases = [self.nlp.make_doc(text) for text in exclusion_patterns]
        self.matcher.add("EXCLUSION_INDICATOR", self.exclusion_phrases)
    
    def clean_text(self, text) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Handle list inputs
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
            
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
    
    def extract_medical_entities(self, text) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text using NLP
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted entities
        """
        # Handle list inputs
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
            
        if not text:
            return []
            
        # Process text with NLP pipeline
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent._, 'confidence', 0.8) if hasattr(ent._, 'confidence') else 0.8
            })
        
        # Match custom patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            rule_id = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            entities.append({
                'text': span.text,
                'label': rule_id,
                'start': span.start_char,
                'end': span.end_char,
                'confidence': 0.9
            })
        
        return entities
    
    def classify_criteria(self, criteria_text) -> str:
        """
        Classify criteria as inclusion or exclusion
        
        Args:
            criteria_text: Criteria text to classify
            
        Returns:
            Classification: 'inclusion', 'exclusion', or 'unclear'
        """
        # Handle list inputs
        if isinstance(criteria_text, list):
            criteria_text = ' '.join(str(item) for item in criteria_text)
        elif not isinstance(criteria_text, str):
            criteria_text = str(criteria_text)
            
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
    
    def extract_temporal_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract temporal expressions from text
        
        Args:
            text: Input text to process
            
        Returns:
            List of temporal expressions
        """
        # Patterns for temporal expressions
        temporal_patterns = [
            r'(current|currently|present|recent|previous|past)\s+',
            r'within\s+(the\s+)?(last|past|previous)\s+(\d+\s*(day|week|month|year))',
            r'for\s+at\s+least\s+(\d+\s*(day|week|month|year))',
            r'no\s+.*\s+(for|during|over)\s+the\s+last\s+(\d+\s*(day|week|month|year))'
        ]
        
        expressions = []
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                expressions.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'temporal'
                })
        
        return expressions
    
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
    
    def extract_conditions_with_umls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical conditions from text (simplified implementation)
        
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
    
    def calculate_confidence(self, entity: Dict[str, Any], context = "") -> float:
        """
        Calculate confidence score for an entity
        
        Args:
            entity: Entity dictionary
            context: Context text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Handle list inputs for context
        if isinstance(context, list):
            context = ' '.join(str(item) for item in context)
        elif not isinstance(context, str):
            context = str(context)
            
        confidence = 0.5  # Base confidence
        
        # Increase confidence for longer entities
        entity_text = entity.get('text', '')
        # Handle list inputs for entity text
        if isinstance(entity_text, list):
            entity_text = ' '.join(str(item) for item in entity_text)
        elif not isinstance(entity_text, str):
            entity_text = str(entity_text)
            
        if len(entity_text) > 10:
            confidence += 0.1
        
        # Increase confidence for context matches
        if context and entity_text and isinstance(entity_text, str) and isinstance(context, str):
            if entity_text.lower() in context.lower():
                confidence += 0.2
        
        # Decrease confidence for potentially ambiguous terms
        ambiguous_terms = ['patient', 'subject', 'participant', 'individual']
        if isinstance(entity_text, str) and entity_text.lower() in ambiguous_terms:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    
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
        
        # Ensure criteria_text is a string
        if isinstance(criteria_text, list):
            criteria_text = ' '.join(str(item) for item in criteria_text)
        elif not isinstance(criteria_text, str):
            criteria_text = str(criteria_text)
        
        # Clean the text
        cleaned_text = self.clean_text(criteria_text)
        
        # Identify sections
        sections = self.identify_sections(cleaned_text)
        
        # Process each section
        result = {
            'entities': [],
            'demographics': {},
            'temporal_expressions': [],
            'conditions': [],
            'procedures': [],
            'sections': sections
        }
        
        # Process all text for entities
        all_entities = self.extract_medical_entities(cleaned_text)
        result['entities'] = all_entities
        
        # Extract demographic information
        result['demographics'] = {
            'age': self.extract_age_criteria(cleaned_text),
            'gender': self.extract_gender_criteria(cleaned_text)
        }
        
        # Extract temporal expressions
        result['temporal_expressions'] = self.extract_temporal_expressions(cleaned_text)
        
        # Extract conditions
        result['conditions'] = self.extract_conditions_with_umls(cleaned_text)
        
        # Extract procedures
        result['procedures'] = self.extract_procedures(cleaned_text)
        
        # Add confidence scores to entities
        for entity in result['entities']:
            entity['confidence'] = self.calculate_confidence(entity, cleaned_text)
        
        # Extract codes from the criteria text
        result['codes'] = self.code_extractor.process_criteria_for_codes(cleaned_text, result['entities'])
        
        # Process with mCODE mapping engine
        result['mcode_mappings'] = self.mcode_mapper.process_nlp_output(result)
        
        # Generate structured mCODE FHIR resources
        mapped_elements = result['mcode_mappings'].get('mapped_elements', [])
        # Format demographics data for structured data generator
        original_demographics = result.get('demographics', {})
        formatted_demographics = {}
        
        # Extract gender from gender criteria
        if 'gender' in original_demographics:
            gender_criteria = original_demographics['gender']
            if isinstance(gender_criteria, list) and gender_criteria:
                # Extract gender from the first gender criterion
                first_criterion = gender_criteria[0]
                if isinstance(first_criterion, dict) and 'gender' in first_criterion:
                    formatted_demographics['gender'] = first_criterion['gender']
                # If we have text, try to extract gender from it
                elif isinstance(first_criterion, dict) and 'text' in first_criterion:
                    text = first_criterion['text'].lower()
                    if 'male' in text or 'men' in text:
                        formatted_demographics['gender'] = 'male'
                    elif 'female' in text or 'women' in text:
                        formatted_demographics['gender'] = 'female'
            elif isinstance(gender_criteria, str):
                # Direct gender string
                formatted_demographics['gender'] = gender_criteria.lower()
        
        # Extract age from age criteria
        if 'age' in original_demographics:
            age_criteria = original_demographics['age']
            if isinstance(age_criteria, list) and age_criteria:
                # Extract age from the first age criterion
                first_criterion = age_criteria[0]
                if isinstance(first_criterion, dict) and 'min_age' in first_criterion:
                    formatted_demographics['age'] = first_criterion['min_age']
                elif isinstance(first_criterion, dict) and 'text' in first_criterion:
                    # Try to extract age from text
                    import re
                    text = first_criterion['text']
                    age_match = re.search(r'(\d+)\s*(?:years?\s*)?(?:or\s+(older|younger|greater|less))?', text, re.IGNORECASE)
                    if age_match:
                        formatted_demographics['age'] = age_match.group(1)
        
        result['structured_data'] = self.structured_data_generator.generate_mcode_resources(
            mapped_elements,
            formatted_demographics
        )
        
        # Add metadata
        result['metadata'] = {
            'text_length': len(cleaned_text),
            'entity_count': len(result['entities']),
            'code_count': result['codes']['metadata']['total_codes'] if 'metadata' in result['codes'] and 'total_codes' in result['codes']['metadata'] else 0,
            'confidence_average': sum(e.get('confidence', 0) for e in result['entities']) / len(result['entities']) if result['entities'] else 0
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # This is just for testing purposes
    nlp_engine = SpacyNLPEngine()
    
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
    print(f"- {len(result['entities'])} entities")
    print(f"- {len(result['demographics']['age'])} age criteria")
    print(f"- {len(result['demographics']['gender'])} gender criteria")
    print(f"- {len(result['conditions'])} conditions")
    print(f"- {len(result['procedures'])} procedures")
    print(f"- {result['mcode_mappings']['metadata']['total_mapped_elements']} mCODE elements mapped")
    print(f"- mCODE validation: {'Passed' if result['mcode_mappings']['validation']['valid'] else 'Failed'}")
    print(f"- Structured data validation: {'Passed' if result['structured_data']['validation']['valid'] else 'Failed'}")
    print(f"- Generated {len(result['structured_data']['resources'])} FHIR resources")