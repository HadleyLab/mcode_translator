# Natural Language Processing for Criteria Parsing Design

## Overview
This document outlines the design and implementation approach for the Natural Language Processing (NLP) engine that will parse clinical trial eligibility criteria. The NLP engine will extract medical concepts, patient characteristics, and other relevant information from unstructured text.

## NLP Engine Architecture

### Core Components

#### 1. Text Preprocessing Module
- Text cleaning and normalization
- Section identification and segmentation
- Sentence boundary detection
- Tokenization and lemmatization

#### 2. Medical Concept Recognition Module
- Named Entity Recognition (NER) for medical terms
- Medical abbreviation expansion
- Context-aware entity disambiguation
- Confidence scoring for recognized entities

#### 3. Criteria Classification Module
- Inclusion/exclusion criteria classification
- Temporal expression identification
- Logical relationship extraction
- Quantitative value extraction

#### 4. Structured Information Extraction Module
- Demographic characteristic extraction
- Medical condition identification
- Procedure and intervention recognition
- Lab value and measurement extraction

## Text Preprocessing

### Normalization Steps

#### Text Cleaning
```python
def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n+', '\n', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char == '\n')
    
    return text.strip()
```

#### Section Identification
```python
def identify_sections(text):
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
    
    # Split text into sections
    sections = {}
    # Implementation details...
    
    return sections
```

### Tokenization Strategy
- Use medical-specific tokenization rules
- Handle medical abbreviations appropriately
- Preserve important punctuation (e.g., "<", ">", "=")
- Maintain context for numerical values

## Medical Concept Recognition

### Named Entity Recognition (NER)

#### Medical NER Libraries
- **spaCy**: Pre-trained medical models (en_core_sci_md)
- **BioBERT**: Biomedical language model for entity recognition
- **ScispaCy**: Specialized for biomedical text processing

#### Entity Types to Recognize
1. **Medical Conditions**
   - Diseases and disorders
   - Signs and symptoms
   - Anatomical locations

2. **Medical Procedures**
   - Surgical procedures
   - Diagnostic procedures
   - Therapeutic interventions

3. **Medications**
   - Drug names
   - Chemical compounds
   - Treatment categories

4. **Demographic Characteristics**
   - Age expressions
   - Gender references
   - Ethnicity indicators

5. **Lab Values and Measurements**
   - Vital signs
   - Laboratory test results
   - Body measurements

### Implementation Approach

#### Using spaCy with Medical Models
```python
import spacy

# Load medical NER model
nlp = spacy.load("en_core_sci_md")

def extract_medical_entities(text):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'confidence': ent._.confidence if hasattr(ent._, 'confidence') else None
        })
    
    return entities
```

#### Custom Entity Patterns
```python
# Define custom patterns for clinical trial specific entities
patterns = [
    {
        "label": "AGE_EXPRESSION",
        "pattern": [
            {"LOWER": {"REGEX": "age|aged|older|younger"}},
            {"OP": "*"},
            {"LOWER": {"REGEX": "year|yr|month|week|day"}},
            {"OP": "?"}
        ]
    },
    {
        "label": "GENDER_REFERENCE",
        "pattern": [
            {"LOWER": {"REGEX": "male|female|men|women|pregnant|nursing"}}
        ]
    }
]
```

## Criteria Classification

### Inclusion/Exclusion Classification

#### Rule-Based Approach
```python
def classify_criteria(criteria_text):
    inclusion_indicators = [
        'must have', 'required', 'diagnosed with', 'history of',
        'able to', 'capable of', 'willing to'
    ]
    
    exclusion_indicators = [
        'must not have', 'excluded', 'ineligible', 'unwilling',
        'unable to', 'contraindicated', 'allergy to'
    ]
    
    classification = 'unclear'
    
    # Check for inclusion indicators
    if any(indicator in criteria_text.lower() for indicator in inclusion_indicators):
        classification = 'inclusion'
    
    # Check for exclusion indicators
    elif any(indicator in criteria_text.lower() for indicator in exclusion_indicators):
        classification = 'exclusion'
    
    return classification
```

#### Machine Learning Approach
- Train classifier on annotated criteria samples
- Use features like keywords, sentence structure, context
- Apply to new criteria for automatic classification

### Temporal Expression Recognition

#### Common Temporal Patterns
- "within X months/years"
- "history of X within Y timeframe"
- "no X for at least Y"
- "currently receiving X"

#### Implementation
```python
def extract_temporal_expressions(text):
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
```

## Structured Information Extraction

### Demographic Characteristic Extraction

#### Age Extraction
```python
def extract_age_criteria(text):
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
```

#### Gender Extraction
```python
def extract_gender_criteria(text):
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
```

### Medical Condition Extraction

#### Using UMLS Terminology
```python
def extract_conditions_with_umls(text):
    # This would interface with UMLS API or local installation
    # For prototype, use pattern matching
    
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
```

### Procedure and Intervention Recognition

#### Procedure Patterns
```python
def extract_procedures(text):
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
```

## Confidence Scoring and Quality Metrics

### Entity Confidence Scoring
```python
def calculate_confidence(entity, context):
    confidence = 0.5  # Base confidence
    
    # Increase confidence for longer entities
    if len(entity['text']) > 10:
        confidence += 0.1
    
    # Increase confidence for context matches
    if context_matches(entity, context):
        confidence += 0.2
    
    # Decrease confidence for ambiguous terms
    if is_ambiguous(entity['text']):
        confidence -= 0.3
    
    return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
```

### Quality Assurance Checks
- Cross-reference extracted entities with medical dictionaries
- Validate numerical values (e.g., age ranges)
- Check for contradictory information
- Ensure completeness of extracted information

## Integration with mCODE Mapping Engine

### Data Flow
1. Raw eligibility criteria text → NLP Engine
2. NLP Engine → Extracted entities with confidence scores
3. Extracted entities → mCODE Mapping Engine
4. mCODE Mapping Engine → Standardized mCODE elements

### Data Exchange Format
```json
{
  "entities": [
    {
      "text": "breast cancer",
      "type": "condition",
      "start": 10,
      "end": 22,
      "confidence": 0.95,
      "mapped_codes": {
        "ICD10CM": "C50.911",
        "SNOMEDCT": "254837009"
      }
    }
  ],
  "demographics": {
    "age": {
      "min": 18,
      "max": null,
      "unit": "years"
    },
    "gender": ["female"]
  },
  "metadata": {
    "processing_time": "0.05s",
    "confidence_average": 0.87
  }
}
```

## Performance Optimization

### Batch Processing
- Process multiple criteria texts simultaneously
- Use parallel processing where possible
- Implement streaming for large texts

### Caching Strategy
- Cache results for identical text inputs
- Store intermediate processing results
- Implement LRU eviction for memory management

### Memory Management
- Use efficient data structures
- Process texts in chunks for large documents
- Release resources after processing

## Error Handling and Fallbacks

### Unrecognized Entities
- Flag low-confidence entities for manual review
- Provide suggestions for similar recognized entities
- Log unrecognized terms for future model improvement

### Processing Failures
- Graceful degradation to rule-based approaches
- Fallback to simpler pattern matching
- Clear error messages for troubleshooting

## Testing and Validation

### Test Cases
1. Standard inclusion criteria
2. Complex exclusion criteria
3. Age and gender restrictions
4. Medical condition references
5. Procedure and medication mentions
6. Temporal expressions
7. Edge cases and ambiguous text

### Validation Metrics
- Precision and recall for entity recognition
- Accuracy of demographic extraction
- Correctness of criteria classification
- Overall processing time and resource usage

## Future Enhancements

### Deep Learning Approaches
- Fine-tune BERT models on clinical trial criteria
- Implement transformer-based entity recognition
- Use attention mechanisms for context understanding

### Active Learning
- Identify uncertain predictions for expert review
- Continuously improve model with new annotations
- Implement feedback loops for model refinement

### Multilingual Support
- Process criteria in multiple languages
- Use translation services for non-English text
- Maintain accuracy across language boundaries