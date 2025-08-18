# System Documentation

## NLP Engine Configuration

### Default Engine
- LLM (deepseek-coder) is the default NLP engine
- Provides most accurate extraction but requires API access
- Fallback to Regex/SpaCy if LLM unavailable
- Specialized for breast cancer with:
  - Focused biomarker extraction (ER/PR/HER2/PD-L1)
  - Genomic variant detection (BRCA1/2, PIK3CA, TP53, HER2 amp)
  - Treatment history parsing (chemo drugs, radiation, surgery types)
  - Cancer characteristics (TNM staging, tumor location)

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
- **Code Extractor**: Maps text to standard codes (ICD-10-CM, CPT, LOINC, RxNorm)
- **mCODE Mapping Engine**: Converts to mCODE format with:
  - Breast cancer-specific mappings (ER/PR/HER2 biomarkers)
  - Genomic variant handling (BRCA1/2, PIK3CA, TP53)
  - Treatment history processing (chemo, radiation, surgery)
  - Code cross-walks between systems (ICD10CM ↔ SNOMEDCT)
  - mCODE compliance validation
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

## Data Fetcher Dashboard

### Key Features
- Interactive search interface with real-time results
- Multiple view modes (cards, list, table)
- Visual analytics:
  - Study status distribution (pie chart)
  - Top conditions (bar chart)
- Pagination support for large result sets
- Detailed study view with tabs for overview/full data

### Technical Implementation
- Built with NiceGUI framework
- Responsive design for desktop/tablet use
- Client-side caching for performance
- Integrated with Data Fetcher API
- Supports both CLI and web interfaces

### Performance Metrics
- Initial search: 500-1500ms
- Page navigation: 200-800ms
- Study details fetch: 300-1000ms
- Visualization rendering: <100ms

## Error Handling and Fallbacks
- SpaCy engine automatically falls back to en_core_web_sm if en_core_sci_md is not available
- All engines include comprehensive error handling and logging
- Fallback mechanisms ensure system reliability