# System Documentation

## NLP Engine Configuration

### Default Engine
- LLM is now the default NLP engine (using deepseek-coder model)
- Provides most accurate extraction but requires API access
- Fallback to Regex/SpaCy if LLM unavailable

### Regex Engine Improvements
- Enhanced genomic variant detection:
  - 50+ cancer-related genes
  - Protein-level changes (e.g., p.Val600Glu)
  - Variant types (mutations, fusions, deletions)
- Improved biomarker status detection:
  - Quantitative values (e.g., "ER 80%")
  - IHC scores (e.g., "HER2 3+")
  - MSI/TMB status
- Deduplication of extracted entities
- Enhanced biomarker extraction patterns for ER/PR/HER2

### SpaCy Engine Improvements
- Added fallback model loading (en_core_web_sm) when medical model unavailable
- Enhanced biomarker extraction patterns for ER/PR/HER2
- Improved error handling and logging

### Benchmarking
- All engines remain available for benchmarking
- Configure via ExtractionPipeline(engine_type="LLM|Regex|SpaCy")

## Overview
The mCODE Translator system processes clinical trial eligibility criteria into structured mCODE elements using multiple NLP approaches with benchmark capabilities.

Key Features:
- Multi-engine processing (Regex/SpaCy/LLM)
- Benchmark mode for accuracy/speed comparison
- Visual feedback on extraction results
- Optimized pipeline architecture
- Robust error handling and fallback mechanisms

## Key Components

### NLP Engines
- **LLM Engine**: DeepSeek API for complex criteria
- **spaCy Engine**: Medical NLP for general criteria with fallback support
- **Regex Engine**: Fast pattern matching

### Core Modules
- **Criteria Parser**: Identifies inclusion/exclusion sections
- **Code Extractor**: Maps text to standard codes (ICD-10-CM, CPT, LOINC, RxNorm)
- **Mapping Engine**: Converts to mCODE format
- **Structured Data Generator**: Creates FHIR resources

## Data Flow
1. Input criteria text → NLP Engine
2. Extracted entities → Mapping Engine
3. Mapped elements → Structured Data Generator
4. Output: mCODE FHIR resources

## Integration Points
- Clinical Trials API
- EHR Systems (via FHIR)
- Matching Engine

## Performance
- Processing Times:
  - Regex: 5-50ms
  - SpaCy: 50-200ms
  - LLM: 200-1000ms
- Accuracy Benchmarks:
  - Regex: 75-85% (fastest)
  - SpaCy: 85-90% (balanced)
  - LLM: 90-95% (most accurate)

## Error Handling and Fallbacks
- SpaCy engine automatically falls back to en_core_web_sm if en_core_sci_md is not available
- All engines include comprehensive error handling and logging
- Fallback mechanisms ensure system reliability