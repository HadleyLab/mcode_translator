# Natural Language Processing for Criteria Parsing Design

## Overview
This document outlines the design and implementation approaches for the Natural Language Processing (NLP) engines that parse clinical trial eligibility criteria. The system supports multiple NLP approaches to handle different use cases and requirements.

## NLP Engine Architecture

### Multi-Engine Approach
The system implements three distinct NLP engines:

1. **LLM-based Engine** (`llm_nlp_engine.py`)
   - Uses DeepSeek API for advanced natural language understanding
   - Specialized for breast cancer criteria extraction
   - Handles complex criteria with high accuracy
   - Outputs structured JSON with validation

2. **spaCy-based Engine** (`spacy_nlp_engine.py`) 
   - Uses medical NLP models (en_core_sci_md)
   - Modular pipeline architecture
   - Supports general clinical trial criteria
   - Includes custom pattern matching

3. **Regex-based Engine** (`regex_nlp_engine.py`)
   - Simple pattern matching approach
   - Fast processing with minimal dependencies
   - Good for well-structured criteria
   - Easy to extend with new patterns

### Engine Selection Guidelines

| Engine Type | Best For | Performance | Accuracy | Dependencies |
|-------------|----------|-------------|----------|--------------|
| LLM | Complex, unstructured text | Moderate | High | DeepSeek API |
| spaCy | General clinical criteria | Fast | Medium | spaCy models |
| Regex | Structured, predictable criteria | Very Fast | Low-Medium | None |

## Implementation Details

### LLM Engine (DeepSeek API)
```python
class LLMNLPEngine:
    """
    NLP Engine for LLM-based extraction of breast cancer genomic features
    Features:
    - Specialized prompt engineering
    - JSON output validation
    - Breast cancer-specific validation
    """
```

Key Features:
- Breast cancer-specific prompt template
- Structured JSON output
- Confidence scoring
- Error handling and fallbacks

### spaCy Engine
```python 
class SpacyNLPEngine:
    """
    NLP Engine using spaCy medical models
    Features:
    - Medical concept recognition
    - Criteria classification  
    - Temporal expression extraction
    """
```

Key Features:
- Pre-trained medical NER
- Custom pattern matching
- Section identification
- Confidence scoring

### Regex Engine  
```python
class RegexNLPEngine:
    """
    Simple NLP engine using regex patterns
    Features:
    - Fast pattern matching
    - Minimal dependencies
    - Easy to extend
    """
```

Key Features:
- Biomarker pattern matching
- Demographic extraction
- Condition identification
- Procedure recognition

## Performance Optimization

### Batch Processing
- All engines support batch processing
- LLM engine benefits most from parallel processing

### Caching Strategy  
- LLM: Cache API responses
- spaCy: Cache processed documents
- Regex: No caching needed

## Testing Strategy

### Unit Tests
- Each engine has dedicated unit tests
- Test coverage:
  - LLM: Basic functionality
  - spaCy: Full pipeline
  - Regex: Pattern matching

### Validation Metrics
- Precision/recall for each engine
- Processing time benchmarks
- Accuracy on sample criteria

## Example Usage

### Multi-Engine Processing
```python
# Initialize engines
llm_engine = LLMNLPEngine()
spacy_engine = SpacyNLPEngine() 
regex_engine = RegexNLPEngine()

# Process criteria with all engines
criteria = "Inclusion: ER+ breast cancer, HER2-negative"
llm_result = llm_engine.extract_mcode_features(criteria)
spacy_result = spacy_engine.process_criteria(criteria)
regex_result = regex_engine.process_criteria(criteria)
```

## Future Enhancements

### Engine Harmonization
- Standardize output formats
- Implement engine voting system
- Add fallback mechanisms

### Active Learning
- Identify uncertain predictions
- Improve models with new annotations