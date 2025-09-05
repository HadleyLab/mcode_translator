# Prompt Pipeline Test Report

## Summary

Total tests: 49
Successful tests: 25
Failed tests: 24
Success rate: 51.0%

NLP to Mcode tests: 42
Direct to Mcode tests: 7

## Failed Tests

- **NLP Pipeline**: Extraction='generic_extraction', Mapping='detailed_mapping'
  - Error: LLM-based mapping failed: Mapping response parsing failed: JSON response appears truncated due to max_tokens limit (4000). Increase max_tokens parameter to allow complete JSON responses. Error: Expecting value: line 537 column 25 (char 21084)

- **NLP Pipeline**: Extraction='generic_extraction', Mapping='error_robust_mapping'
  - Error: LLM-based mapping failed: Mapping response parsing failed: JSON response appears truncated due to max_tokens limit (4000). Increase max_tokens parameter to allow complete JSON responses. Error: Unterminated string starting at: line 551 column 40 (char 21313)

- **NLP Pipeline**: Extraction='comprehensive_extraction', Mapping='detailed_mapping'
  - Error: LLM-based mapping failed: Mapping response parsing failed: JSON response appears truncated due to max_tokens limit (4000). Increase max_tokens parameter to allow complete JSON responses. Error: Unterminated string starting at: line 544 column 13 (char 21486)

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='generic_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: condition_status']

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='comprehensive_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: condition_status']

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='detailed_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: condition_status']

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='error_robust_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: condition_status']

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='minimal_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: condition_status']

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='simple_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: condition_status']

- **NLP Pipeline**: Extraction='basic_extraction', Mapping='standard_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: condition_status']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='generic_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: Disease']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='comprehensive_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: Disease']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='detailed_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: Disease']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='error_robust_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: Disease']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='minimal_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: Disease']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='simple_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: Disease']

- **NLP Pipeline**: Extraction='minimal_extraction_optimization', Mapping='standard_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: Disease']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='generic_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: assessment']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='comprehensive_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: assessment']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='detailed_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: assessment']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='error_robust_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: assessment']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='minimal_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: assessment']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='simple_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: assessment']

- **NLP Pipeline**: Extraction='structured_extraction', Mapping='standard_mapping'
  - Error: Unexpected error during extraction: Invalid entity: ['Invalid entity type: assessment']

## Successful Tests Performance

| Pipeline Type | Prompt(s) | F1 Score | Precision | Recall | Execution Time (s) |
|---------------|-----------|----------|-----------|--------|-------------------|
| NLP to Mcode | Ext:generic_extraction / Map:generic_mapping | 0.000 | 0.000 | 0.000 | 0.05 |
| NLP to Mcode | Ext:generic_extraction / Map:comprehensive_mapping | 0.000 | 0.000 | 0.000 | 0.02 |
| NLP to Mcode | Ext:generic_extraction / Map:minimal_mapping | 0.000 | 0.000 | 0.000 | 0.02 |
| NLP to Mcode | Ext:generic_extraction / Map:simple_mapping | 0.000 | 0.000 | 0.000 | 196.63 |
| NLP to Mcode | Ext:generic_extraction / Map:standard_mapping | 0.000 | 0.000 | 0.000 | 383.20 |
| NLP to Mcode | Ext:comprehensive_extraction / Map:generic_mapping | 0.000 | 0.000 | 0.000 | 357.92 |
| NLP to Mcode | Ext:comprehensive_extraction / Map:comprehensive_mapping | 0.000 | 0.000 | 0.000 | 201.64 |
| NLP to Mcode | Ext:comprehensive_extraction / Map:error_robust_mapping | 0.000 | 0.000 | 0.000 | 432.84 |
| NLP to Mcode | Ext:comprehensive_extraction / Map:minimal_mapping | 0.000 | 0.000 | 0.000 | 138.55 |
| NLP to Mcode | Ext:comprehensive_extraction / Map:simple_mapping | 0.000 | 0.000 | 0.000 | 185.55 |
| NLP to Mcode | Ext:comprehensive_extraction / Map:standard_mapping | 0.000 | 0.000 | 0.000 | 277.53 |
| NLP to Mcode | Ext:minimal_extraction / Map:generic_mapping | 0.000 | 0.000 | 0.000 | 0.02 |
| NLP to Mcode | Ext:minimal_extraction / Map:comprehensive_mapping | 0.000 | 0.000 | 0.000 | 146.61 |
| NLP to Mcode | Ext:minimal_extraction / Map:detailed_mapping | 0.000 | 0.000 | 0.000 | 318.54 |
| NLP to Mcode | Ext:minimal_extraction / Map:error_robust_mapping | 0.000 | 0.000 | 0.000 | 300.33 |
| NLP to Mcode | Ext:minimal_extraction / Map:minimal_mapping | 0.000 | 0.000 | 0.000 | 0.02 |
| NLP to Mcode | Ext:minimal_extraction / Map:simple_mapping | 0.000 | 0.000 | 0.000 | 162.68 |
| NLP to Mcode | Ext:minimal_extraction / Map:standard_mapping | 0.000 | 0.000 | 0.000 | 290.41 |
| Direct to Mcode | Direct:direct_mcode | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to Mcode | Direct:direct_mcode_simple | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to Mcode | Direct:direct_mcode_comprehensive | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to Mcode | Direct:direct_mcode_minimal | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to Mcode | Direct:direct_mcode_structured | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to Mcode | Direct:direct_mcode_optimization | 0.000 | 0.000 | 0.000 | 0.01 |
| Direct to Mcode | Direct:direct_mcode_improved | 0.000 | 0.000 | 0.000 | 0.01 |

## Best Performers (by F1 Score)

1. **NLP Pipeline**: Extraction='generic_extraction', Mapping='generic_mapping' (F1: 0.000)
2. **NLP Pipeline**: Extraction='generic_extraction', Mapping='comprehensive_mapping' (F1: 0.000)
3. **NLP Pipeline**: Extraction='generic_extraction', Mapping='minimal_mapping' (F1: 0.000)
4. **NLP Pipeline**: Extraction='generic_extraction', Mapping='simple_mapping' (F1: 0.000)
5. **NLP Pipeline**: Extraction='generic_extraction', Mapping='standard_mapping' (F1: 0.000)