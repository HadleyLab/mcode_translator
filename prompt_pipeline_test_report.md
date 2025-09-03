# Prompt Pipeline Test Report

## Summary

Total tests: 48
Successful tests: 7
Failed tests: 41
Success rate: 14.6%

NLP to mCODE tests: 42
Direct to mCODE tests: 6

## Failed Tests

- **NLP Pipeline**: Extraction='generic_extraction', Mapping='comprehensive_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='generic_extraction', Mapping='detailed_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='generic_extraction', Mapping='error_robust_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='generic_extraction', Mapping='simple_mapping'
  - Error: Unexpected error during mapping: '\n    "mcode_mappings"'

- **NLP Pipeline**: Extraction='generic_extraction', Mapping='standard_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='generic_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='comprehensive_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='detailed_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='error_robust_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='minimal_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='simple_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='standard_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='generic_mapping'
  - Error: Unexpected error during extraction: '\n    "entities"'

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='comprehensive_mapping'
  - Error: Unexpected error during extraction: '\n    "entities"'

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='detailed_mapping'
  - Error: Unexpected error during extraction: '\n    "entities"'

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='error_robust_mapping'
  - Error: Unexpected error during extraction: '\n    "entities"'

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='minimal_mapping'
  - Error: Unexpected error during extraction: '\n    "entities"'

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='simple_mapping'
  - Error: Unexpected error during extraction: '\n    "entities"'

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='standard_mapping'
  - Error: Unexpected error during extraction: '\n    "entities"'

- **NLP Pipeline**: Extraction='minimal_extraction', Mapping='comprehensive_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='minimal_extraction', Mapping='detailed_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='minimal_extraction', Mapping='error_robust_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='minimal_extraction', Mapping='simple_mapping'
  - Error: Unexpected error during mapping: '\n    "mcode_mappings"'

- **NLP Pipeline**: Extraction='minimal_extraction', Mapping='standard_mapping'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='generic_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: DISEASE']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='comprehensive_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: DISEASE']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='detailed_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: DISEASE']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='error_robust_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: DISEASE']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='minimal_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: DISEASE']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='simple_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: DISEASE']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='standard_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: DISEASE']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='generic_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='comprehensive_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='detailed_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='error_robust_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='minimal_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='simple_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='standard_mapping'
  - Error: Unexpected error during extraction: '\n  "entities"'

- **Direct Pipeline**: Prompt='direct_mcode_comprehensive'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **Direct Pipeline**: Prompt='direct_mcode_structured'
  - Error: Unexpected error during mapping: '\n  "mcode_mappings"'

- **Direct Pipeline**: Prompt='direct_mcode_optimization'
  - Error: LLM-based mapping failed: Mapping response parsing failed: Each mapping must contain 'mcode_element' field

## Successful Tests Performance

| Pipeline Type | Prompt(s) | F1 Score | Precision | Recall | Execution Time (s) |
|---------------|-----------|----------|-----------|--------|-------------------|
| NLP to mCODE | Ext:generic_extraction / Map:generic_mapping | 0.000 | 0.000 | 0.000 | 0.04 |
| NLP to mCODE | Ext:generic_extraction / Map:minimal_mapping | 0.000 | 0.000 | 0.000 | 0.03 |
| NLP to mCODE | Ext:minimal_extraction / Map:generic_mapping | 0.000 | 0.000 | 0.000 | 0.03 |
| NLP to mCODE | Ext:minimal_extraction / Map:minimal_mapping | 0.000 | 0.000 | 0.000 | 0.02 |
| Direct to mCODE | Direct:direct_mcode | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to mCODE | Direct:direct_mcode_simple | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to mCODE | Direct:direct_mcode_minimal | 0.000 | 0.000 | 0.000 | 0.01 |

## Best Performers (by F1 Score)

1. **NLP Pipeline**: Extraction='generic_extraction', Mapping='generic_mapping' (F1: 0.000)
2. **NLP Pipeline**: Extraction='generic_extraction', Mapping='minimal_mapping' (F1: 0.000)
3. **NLP Pipeline**: Extraction='minimal_extraction', Mapping='generic_mapping' (F1: 0.000)
4. **NLP Pipeline**: Extraction='minimal_extraction', Mapping='minimal_mapping' (F1: 0.000)
5. **Direct Pipeline**: Prompt='direct_mcode' (F1: 0.000)