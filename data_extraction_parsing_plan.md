# Data Extraction and Parsing Components Plan

## Overview
This document details the components responsible for extracting clinical trial data from clinicaltrials.gov API and parsing the eligibility criteria for further processing by the mCODE translator.

## Data Extraction Component

### Responsibilities
- Fetch clinical trial data from clinicaltrials.gov API
- Handle API rate limiting and pagination
- Parse API responses into structured data
- Cache data to minimize API calls
- Handle error conditions and retries

### API Interaction Design

#### Request Structure
```
GET https://clinicaltrials.gov/api/query/study_fields?
    expr=<search_expression>&
    fields=<comma_separated_fields>&
    min_rnk=<start_rank>&
    max_rnk=<end_rank>&
    fmt=json
```

#### Key Fields for Extraction
- `NCTId` - Unique trial identifier
- `BriefTitle` - Short title of the trial
- `OfficialTitle` - Official title of the trial
- `EligibilityCriteria` - Full eligibility criteria text
- `Gender` - Gender restrictions
- `MinimumAge` - Minimum age requirement
- `MaximumAge` - Maximum age requirement
- `HealthyVolunteers` - Accepts healthy volunteers
- `Condition` - Medical conditions being studied
- `InterventionName` - Names of interventions being tested

### Data Structure
```json
{
  "StudyFields": {
    "MatchedStudies": [
      {
        "StudyId": "NCT00000000",
        "BriefTitle": "Sample Clinical Trial",
        "EligibilityCriteria": "INCLUSION CRITERIA:\n- Age ≥ 18 years\n- Histologically confirmed diagnosis\n\nEXCLUSION CRITERIA:\n- Pregnant or nursing women\n- Known allergy to study drug",
        "Gender": "All",
        "MinimumAge": "18 Years",
        "MaximumAge": "N/A",
        "HealthyVolunteers": "No",
        "Condition": ["Breast Cancer", "Metastatic Breast Cancer"],
        "InterventionName": ["Chemotherapy", "Radiation Therapy"]
      }
    ]
  }
}
```

## Criteria Parsing Component

### Responsibilities
- Parse structured elements from API response
- Extract unstructured eligibility criteria text
- Identify inclusion and exclusion criteria sections
- Preprocess text for NLP analysis
- Handle various text formatting conventions

### Text Preprocessing Steps

1. **Section Identification**
   - Split inclusion and exclusion criteria
   - Identify numbered or bulleted lists
   - Recognize common section headers

2. **Text Cleaning**
   - Remove extra whitespace and line breaks
   - Normalize punctuation
   - Handle special characters and symbols

3. **Structured Data Extraction**
   - Extract age limits from text
   - Identify gender restrictions
   - Parse medical conditions
   - Recognize lab value requirements

### Parsing Logic

#### Age Parsing
```python
# Example patterns to recognize
patterns = [
    r"age\s*[<>]?\s*(\d+)",
    r"(\d+)\s*years?\s*old",
    r"between\s*(\d+)\s*and\s*(\d+)"
]
```

#### Gender Parsing
```python
# Common patterns
gender_patterns = {
    "male": r"\b(male|men)\b",
    "female": r"\b(female|women)\b",
    "all": r"\b(all\s+genders?|both\s+sexes)\b"
}
```

#### Section Splitting
```python
# Common section headers
inclusion_indicators = [
    "inclusion criteria", 
    "eligible subjects", 
    "selection criteria"
]

exclusion_indicators = [
    "exclusion criteria", 
    "ineligible subjects", 
    "non-inclusion criteria"
]
```

## Data Validation

### Input Validation
- Verify required fields are present
- Check data types and formats
- Validate age range consistency
- Ensure criteria text is not empty

### Error Handling
- Handle API rate limiting with backoff
- Manage network timeouts gracefully
- Log parsing errors for troubleshooting
- Provide meaningful error messages to users

## Caching Strategy

### Cache Structure
```json
{
  "NCT00000000": {
    "timestamp": "2023-01-01T00:00:00Z",
    "data": { /* full trial data */ }
  }
}
```

### Cache Management
- Time-based expiration (24 hours)
- Memory-limited cache size
- LRU eviction policy
- Persistent storage option

## Performance Considerations

### Batch Processing
- Process multiple trials in batches
- Optimize API calls with field selection
- Parallel processing where possible
- Progress tracking for long-running operations

### Memory Management
- Stream processing for large datasets
- Efficient data structures
- Garbage collection optimization
- Memory usage monitoring

## Integration Points

### With NLP Engine
- Provide clean, preprocessed text
- Pass structured elements separately
- Handle text segmentation
- Manage processing order

### With mCODE Mapper
- Supply parsed criteria elements
- Provide trial metadata
- Handle mapping feedback
- Manage data flow coordination