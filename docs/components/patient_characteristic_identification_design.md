# Patient Characteristic Identification Module Design

## Overview
This document outlines the design for the Patient Characteristic Identification Module, which extracts patient demographic and clinical characteristics from clinical trial eligibility criteria.

## Core Components

### 1. Demographic Extraction
- Age requirement identification
- Gender restriction detection
- Ethnicity and race references
- Pregnancy and lactation status

### 2. Clinical History
- Medical condition history
- Previous treatment history
- Surgical history
- Medication history

### 3. Current Status
- Performance status assessment
- Current treatment status
- Disease progression status
- Laboratory value requirements

### 4. Lifestyle Factors
- Smoking history
- Alcohol consumption
- Drug use history
- Occupational exposures

## Implementation Approach

### Age Extraction
```python
# Patterns for age requirements
age_patterns = [
    r'(?:between|from)\s+(\d+)\s*(?:and|to)\s*(\d+)',
    r'(?:at least|minimum)\s+(\d+)',
    r'(?:at most|maximum)\s+(\d+)',
    r'(\d+)\s*(?:years?|yrs?)\s*old'
]
```

### Gender Extraction
```python
# Patterns for gender restrictions
gender_patterns = {
    'male': r'\b(male|men)\b',
    'female': r'\b(female|women)\b',
    'pregnant': r'\b(pregnant|pregnancy)\b'
}
```

### Medical History
```python
# Patterns for medical history
history_patterns = [
    r'(?:history|previous)\s+(?:of|with)\s+([^,;.]+)',
    r'(?:diagnosed|diagnosis)\s+(?:with|of)\s+([^,;.]+)'
]
```

## Data Output Format
```json
{
  "demographics": {
    "age": {
      "min": 18,
      "max": 75,
      "unit": "years"
    },
    "gender": ["female"],
    "pregnancy": "excluded"
  },
  "medical_history": [
    {
      "condition": "breast cancer",
      "confidence": 0.95
    }
  ],
  "treatment_history": [
    {
      "treatment": "chemotherapy",
      "confidence": 0.90
    }
  ]
}
```

## Integration Points
- Receives parsed eligibility criteria from NLP engine
- Provides structured patient data to mCODE mapper
- Uses terminology services for code validation
- Implements confidence scoring for extracted characteristics