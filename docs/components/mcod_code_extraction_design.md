# mCODE Code Extraction Functionality Design

## Overview
This document outlines the design and implementation approach for the mCODE code extraction functionality. This component is responsible for identifying and extracting specific mCODE codes and related standard codes (ICD-10-CM, CPT, LOINC, RxNorm) from parsed clinical trial eligibility criteria.

## Component Architecture

### Core Modules

#### 1. Code Identification Module
- Recognizes code references in parsed text
- Identifies code patterns and formats
- Extracts codes from structured data elements
- Handles code variations and synonyms

#### 2. Code Validation Module
- Validates extracted codes against standard terminologies
- Checks code format and structure compliance
- Verifies code existence in reference databases
- Handles deprecated and updated codes

#### 3. Code Mapping Module
- Maps between different coding systems
- Handles cross-walks between ICD-10-CM, CPT, LOINC, RxNorm
- Applies mCODE-specific mapping rules
- Manages code hierarchies and relationships

#### 4. Code Confidence Scoring Module
- Assigns confidence scores to extracted codes
- Evaluates mapping quality and accuracy
- Identifies potential ambiguities
- Flags codes requiring manual review

## Code Identification Strategies

### Pattern-Based Recognition

#### ICD-10-CM Pattern Recognition
```python
def identify_icd10cm_codes(text):
    # ICD-10-CM code patterns
    icd10cm_patterns = [
        r'\b[A-TV-Z][0-9][A-Z0-9]{0,5}(\.[A-Z0-9]{1,4})?\b',  # Standard format
        r'\bICD[-\s]?10(?:[-\s]?CM)?[-\s]?([A-TV-Z][0-9][A-Z0-9]{0,5}(\.[A-Z0-9]{1,4})?)\b'
    ]
    
    codes = []
    for pattern in icd10cm_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            codes.append({
                'text': match.group(),
                'code': match.group(1) if match.groups() else match.group(),
                'system': 'ICD-10-CM',
                'start': match.start(),
                'end': match.end()
            })
    
    return codes
```

#### CPT Code Pattern Recognition
```python
def identify_cpt_codes(text):
    # CPT code patterns (5 digits)
    cpt_pattern = r'\b\d{5}\b'
    
    codes = []
    matches = re.finditer(cpt_pattern, text)
    for match in matches:
        # Additional validation for CPT codes
        code = match.group()
        if is_valid_cpt_code(code):  # Function to check against CPT database
            codes.append({
                'text': match.group(),
                'code': code,
                'system': 'CPT',
                'start': match.start(),
                'end': match.end()
            })
    
    return codes
```

#### LOINC Code Pattern Recognition
```python
def identify_loinc_codes(text):
    # LOINC code patterns (various formats)
    loinc_patterns = [
        r'\b\d+-\d+\b',  # Basic LOINC format
        r'\bLOINC[-\s]?(\d+-\d+)\b'
    ]
    
    codes = []
    for pattern in loinc_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            codes.append({
                'text': match.group(),
                'code': match.group(1) if match.groups() else match.group(),
                'system': 'LOINC',
                'start': match.start(),
                'end': match.end()
            })
    
    return codes
```

#### RxNorm Code Pattern Recognition
```python
def identify_rxnorm_codes(text):
    # RxNorm code patterns
    rxnorm_patterns = [
        r'\bR[x]?[-\s]?(\d+)\b',
        r'\bRxNorm[-\s]?(\d+)\b'
    ]
    
    codes = []
    for pattern in rxnorm_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            codes.append({
                'text': match.group(),
                'code': match.group(1) if match.groups() else match.group(),
                'system': 'RxNorm',
                'start': match.start(),
                'end': match.end()
            })
    
    return codes
```

### Context-Aware Recognition

#### Medical Term to Code Mapping
```python
def map_terms_to_codes(entities):
    # Map recognized medical terms to standard codes
    term_to_code_map = {
        'breast cancer': {
            'ICD10CM': 'C50.911',
            'SNOMEDCT': '254837009'
        },
        'lung cancer': {
            'ICD10CM': 'C34.90',
            'SNOMEDCT': '254837009'
        }
        # More mappings...
    }
    
    mapped_codes = []
    for entity in entities:
        term = entity['text'].lower()
        if term in term_to_code_map:
            mapped_codes.append({
                'entity': entity,
                'codes': term_to_code_map[term],
                'confidence': entity.get('confidence', 0.8)
            })
    
    return mapped_codes
```

## Code Validation Process

### Validation Steps

#### 1. Format Validation
```python
def validate_code_format(code, system):
    format_rules = {
        'ICD10CM': r'^[A-TV-Z][0-9][A-Z0-9]{0,5}(\.[A-Z0-9]{1,4})?$',
        'CPT': r'^\d{5}$',
        'LOINC': r'^\d+-\d+$',
        'RxNorm': r'^\d+$'
    }
    
    if system in format_rules:
        pattern = format_rules[system]
        return bool(re.match(pattern, code))
    
    return False
```

#### 2. Existence Validation
```python
def validate_code_existence(code, system):
    # This would interface with terminology servers or local databases
    # For prototype, use simplified validation
    
    valid_codes = {
        'ICD10CM': ['C50.911', 'C34.90', 'C18.9'],  # Sample codes
        'CPT': ['12345', '67890', '54321'],         # Sample codes
        'LOINC': ['12345-6', '78901-2', '34567-8'], # Sample codes
        'RxNorm': ['123456', '789012', '345678']    # Sample codes
    }
    
    if system in valid_codes:
        return code in valid_codes[system]
    
    return True  # Default to valid if not in sample set
```

#### 3. mCODE Compliance Validation
```python
def validate_mc ode_compliance(code, system):
    # Check if code is part of mCODE required or recommended codes
    mc ode_required_codes = {
        'ICD10CM': ['C50.911', 'C34.90'],  # Example required codes
        'CPT': ['12345', '67890'],          # Example required codes
        # Add more as needed
    }
    
    if system in mc ode_required_codes:
        return code in mc ode_required_codes[system]
    
    return True  # Default to compliant if not specifically required
```

## Code Mapping Engine

### Cross-System Mapping
```python
def map_between_systems(source_code, source_system, target_system):
    # This would use terminology services for actual mapping
    # For prototype, use sample mappings
    
    cross_walks = {
        ('C50.911', 'ICD10CM', 'SNOMEDCT'): '254837009',
        ('C50.911', 'ICD10CM', 'LOINC'): 'LP12345-6',
        # More mappings...
    }
    
    key = (source_code, source_system, target_system)
    if key in cross_walks:
        return cross_walks[key]
    
    return None  # No mapping found
```

### Hierarchical Mapping
```python
def get_code_hierarchy(code, system):
    # Get parent/child relationships for codes
    hierarchies = {
        'ICD10CM': {
            'C50.911': {
                'parent': 'C50',
                'children': ['C50.9111', 'C50.9112']
            }
        }
    }
    
    if system in hierarchies and code in hierarchies[system]:
        return hierarchies[system][code]
    
    return {}
```

## Confidence Scoring System

### Scoring Factors
```python
def calculate_code_confidence(code_info):
    confidence = 0.5  # Base confidence
    
    # Increase for direct code references
    if code_info.get('direct_reference'):
        confidence += 0.3
    
    # Increase for context matches
    if code_info.get('context_match'):
        confidence += 0.2
    
    # Decrease for potential ambiguities
    if code_info.get('ambiguous'):
        confidence -= 0.3
    
    # Decrease for deprecated codes
    if code_info.get('deprecated'):
        confidence -= 0.2
    
    return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
```

## Integration with NLP Engine

### Data Flow
1. NLP Engine → Extracted entities and parsed criteria
2. Code Extraction → Identify and validate codes
3. Code Extraction → Map codes between systems
4. Code Extraction → Assign confidence scores
5. mCODE Mapper → Use extracted codes for mCODE generation

### Data Exchange Format
```json
{
  "extracted_codes": [
    {
      "text": "C50.911",
      "code": "C50.911",
      "system": "ICD-10-CM",
      "start": 45,
      "end": 52,
      "confidence": 0.95,
      "mapped_codes": {
        "SNOMEDCT": "254837009",
        "LOINC": "LP12345-6"
      },
      "validation": {
        "format_valid": true,
        "exists": true,
        "mc ode_compliant": true
      }
    }
  ],
  "mapped_entities": [
    {
      "entity": {
        "text": "breast cancer",
        "type": "condition",
        "confidence": 0.85
      },
      "mapped_codes": {
        "ICD10CM": "C50.911",
        "SNOMEDCT": "254837009"
      },
      "confidence": 0.90
    }
  ]
}
```

## Error Handling and Edge Cases

### Common Issues

#### 1. Ambiguous Codes
```python
def handle_ambiguous_codes(codes):
    # Handle cases where multiple codes could apply
    # Implement disambiguation logic based on context
    disambiguated = []
    
    for code in codes:
        if is_ambiguous(code):
            # Apply context-based disambiguation
            resolved_code = disambiguate_code(code)
            if resolved_code:
                disambiguated.append(resolved_code)
            else:
                # Flag for manual review
                code['requires_review'] = True
                disambiguated.append(code)
        else:
            disambiguated.append(code)
    
    return disambiguated
```

#### 2. Invalid or Deprecated Codes
```python
def handle_invalid_codes(codes):
    valid_codes = []
    invalid_codes = []
    
    for code in codes:
        if validate_code(code):
            valid_codes.append(code)
        else:
            # Check if it's a deprecated code with replacement
            replacement = get_code_replacement(code)
            if replacement:
                valid_codes.append(replacement)
            else:
                invalid_codes.append(code)
    
    return valid_codes, invalid_codes
```

## Performance Considerations

### Caching Strategy
```python
class CodeCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to front of access order
            self.access_order.remove(key)
            self.access_order.insert(0, key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = self.access_order.pop()
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.insert(0, key)
```

### Batch Processing
```python
def extract_codes_batch(criteria_list):
    # Process multiple criteria texts in batch
    results = []
    
    for criteria in criteria_list:
        codes = extract_codes(criteria)
        results.append({
            'criteria': criteria,
            'codes': codes
        })
    
    return results
```

## Testing Strategy

### Unit Tests
```python
def test_icd10cm_extraction():
    text = "Patients with diagnosis of C50.911 (breast cancer)"
    codes = identify_icd10cm_codes(text)
    assert len(codes) == 1
    assert codes[0]['code'] == 'C50.911'
    assert codes[0]['system'] == 'ICD-10-CM'

def test_code_validation():
    assert validate_code_format('C50.911', 'ICD10CM') == True
    assert validate_code_format('INVALID', 'ICD10CM') == False
```

### Integration Tests
```python
def test_full_extraction_pipeline():
    criteria = "INCLUSION: Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)"
    
    # Parse criteria
    parsed = parse_criteria(criteria)
    
    # Extract codes
    codes = extract_codes(parsed)
    
    # Validate codes
    validated = validate_codes(codes)
    
    # Check results
    assert len(validated) > 0
    assert any(code['code'] == 'C50.911' for code in validated)
```

## Future Enhancements

### Terminology Service Integration
- Connect to UMLS, LOINC, and other terminology servers
- Implement real-time code validation and mapping
- Access comprehensive code hierarchies and relationships

### Machine Learning Approaches
- Train models to recognize code references in context
- Implement neural networks for code disambiguation
- Use transformer models for context-aware mapping

### Active Learning
- Identify uncertain code mappings for expert review
- Continuously improve extraction accuracy with feedback
- Implement confidence-based routing to human reviewers