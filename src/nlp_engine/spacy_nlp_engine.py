import re
import spacy
from typing import List, Dict, Any, Optional
import logging
import sys
import os
from .nlp_engine import NLPEngine, ProcessingResult
import time

# Add the src directory to the path so we can import the code extraction module
sys.path.insert(0, os.path.dirname(__file__))

from src.code_extraction.code_extraction import CodeExtractionModule
from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from src.structured_data_generator.structured_data_generator import StructuredDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpacyNLPEngine(NLPEngine):
    """NLP Engine using spaCy for clinical text processing.
    
    Specialized for extracting mCODE features from clinical trial eligibility criteria
    using medical NLP models and custom pattern matching.
    
    Attributes:
        nlp (spacy.Language): Loaded medical NLP model
        code_extractor (CodeExtractionModule): For extracting medical codes
        mcode_mapper (MCODEMappingEngine): For mapping to mCODE standards
        structured_data_generator (StructuredDataGenerator): For FHIR output
        matcher (PhraseMatcher): Custom pattern matcher
        logger (logging.Logger): Configured logger instance
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
            logger.warning("Medical NLP model not found. Falling back to en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Fallback NLP model loaded successfully")
            except OSError:
                logger.error("Fallback NLP model not found. Please install with: python -m spacy download en_core_web_sm")
                raise
            except Exception as e:
                logger.error(f"Error loading fallback NLP model: {str(e)}")
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
        
        # Define biomarker patterns
        biomarker_patterns = [
            "er+", "er-", "er positive", "er negative", "estrogen receptor positive", "estrogen receptor negative",
            "pr+", "pr-", "pr positive", "pr negative", "progesterone receptor positive", "progesterone receptor negative",
            "her2+", "her2-", "her2 positive", "her2 negative", "human epidermal growth factor receptor 2 positive", "human epidermal growth factor receptor 2 negative",
            "hr+", "hr-", "hr positive", "hr negative", "hormone receptor positive", "hormone receptor negative"
        ]
        self.biomarker_phrases = [self.nlp.make_doc(text) for text in biomarker_patterns]
        self.matcher.add("BIOMARKER", self.biomarker_phrases)
    
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
        
    def extract_biomarkers(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract biomarker information from text
        
        Args:
            text: Input text to process
            
        Returns:
            List of biomarker information
        """
        biomarkers = []
        
        # Patterns for biomarker extraction
        # ER/PR/HER2/HR with +/-
        biomarker_patterns = [
            r'(ER|PR|HER2|HR)[\s\-\+]*(positive|negative|\+|\-)',
            r'(estrogen receptor|progesterone receptor|human epidermal growth factor receptor 2|hormone receptor)[\s\-\+]*(positive|negative|\+|\-)'
        ]
        
        for pattern in biomarker_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Determine name and status
                if len(match.groups()) >= 2:
                    name_match = match.group(1)
                    status_match = match.group(2)
                    
                    # Normalize name
                    name_map = {
                        'estrogen receptor': 'ER',
                        'progesterone receptor': 'PR',
                        'human epidermal growth factor receptor 2': 'HER2',
                        'hormone receptor': 'HR'
                    }
                    name = name_map.get(name_match.lower(), name_match.upper())
                    
                    # Normalize status
                    if status_match.lower() in ['+', 'positive']:
                        status = 'positive'
                    elif status_match.lower() in ['-', 'negative']:
                        status = 'negative'
                    else:
                        status = status_match.lower()
                    
                    biomarkers.append({
                        'text': match.group(),
                        'name': name,
                        'status': status,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9
                    })
        
        return biomarkers
    
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
    
    def process_text(self, text: str) -> ProcessingResult:
        """Process clinical text and extract mCODE features.
        
        Args:
            text (str): Input clinical text to process. Must be non-empty.
            
        Returns:
            ProcessingResult: Contains:
                - features (Dict): Extracted mCODE features in standardized format
                - mcode_mappings (Dict): FHIR mappings for extracted features
                - metadata (Dict): Processing metadata including counts
                - entities (List): Raw extracted entities with confidence scores
                - error (Optional[str]): None if successful, error message if failed
            
        Raises:
            ValueError: If input text is empty or invalid type
            
        Example:
            >>> engine = SpacyNLPEngine()
            >>> result = engine.process_text("ER+ HER2- breast cancer")
            >>> "cancer_type" in result.features["cancer_characteristics"]
            True
        """
        if not text or not isinstance(text, str):
            error_msg = "Input text must be a non-empty string"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)
            
        try:
            return self.process_criteria(text)
        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return self._create_error_result(error_msg)

    def process_criteria(self, criteria_text: str) -> ProcessingResult:
        """
        Process eligibility criteria text and extract structured information
        
        Args:
            criteria_text: Eligibility criteria text to process
            
        Returns:
            ProcessingResult containing extracted information
        """
        start_time = time.time()
        if not criteria_text:
            return ProcessingResult(
                features={},
                mcode_mappings={},
                metadata={},
                entities=[],
                error="Empty criteria text"
            )
        
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
        features = {
            'demographics': {},
            'cancer_characteristics': {},
            'biomarkers': [],
            'genomic_variants': [],
            'treatment_history': {},
            'performance_status': {}
        }
        
        # Process all text for entities
        entities = self.extract_medical_entities(cleaned_text)
        
        # Extract demographic information
        age_criteria = self.extract_age_criteria(cleaned_text)
        gender_criteria = self.extract_gender_criteria(cleaned_text)
        
        # Map to standardized features format
        features['demographics'] = {
            'age_range': age_criteria[0]['text'] if age_criteria else None,
            'gender': gender_criteria[0]['gender'] if gender_criteria else None
        }
        
        # Extract conditions and map to cancer characteristics
        conditions = self.extract_conditions_with_umls(cleaned_text)
        if conditions:
            features['cancer_characteristics']['cancer_type'] = conditions[0]['condition']
        
        # Extract biomarkers
        biomarkers = self.extract_biomarkers(cleaned_text)
        features['biomarkers'] = biomarkers
        
        # Extract procedures and map to treatment history
        procedures = self.extract_procedures(cleaned_text)
        if procedures:
            features['treatment_history']['procedures'] = [p['procedure'] for p in procedures]
        
        # Add confidence scores to entities
        for entity in entities:
            entity['confidence'] = self.calculate_confidence(entity, cleaned_text)
        
        # Extract codes from the criteria text
        codes = self.code_extractor.process_criteria_for_codes(cleaned_text, entities)
        
        # Process with mCODE mapping engine
        mcode_mappings = self.mcode_mapper.process_nlp_output({
            'entities': entities,
            'features': features,
            'codes': codes
        })
        
        # Generate structured mCODE FHIR resources
        mapped_elements = mcode_mappings.get('mapped_elements', [])
        structured_data = self.structured_data_generator.generate_mcode_resources(
            mapped_elements,
            features['demographics']
        )
        
        # Ensure features structure matches ProcessingResult expectations
        standardized_features = {
            'demographics': {
                'gender': features.get('demographics', {}).get('gender', '').capitalize() if features.get('demographics', {}).get('gender') else '',
                'age': features.get('demographics', {}).get('age', {})
            },
            'cancer_characteristics': features.get('cancer_characteristics', {}),
            'biomarkers': features.get('biomarkers', []),
            'genomic_variants': features.get('genomic_variants', []),
            'treatment_history': features.get('treatment_history', {}),
            'performance_status': features.get('performance_status', {})
        }
        
        # Prepare metadata with actual processing time
        metadata = {
            'processing_time': time.time() - start_time,  # Calculate elapsed time
            'engine': 'spacy',
            'biomarkers_count': len(standardized_features['biomarkers']),
            'genomic_variants_count': len(standardized_features['genomic_variants'])
        }
        
        # Explicitly create ProcessingResult instance
        result = ProcessingResult(
            features=standardized_features,
            mcode_mappings=mcode_mappings,
            metadata=metadata,
            entities=entities,
            error=None
        )
        
        # Verify type before returning
        if not isinstance(result, ProcessingResult):
            raise TypeError(f"Expected ProcessingResult but got {type(result)}")
            
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