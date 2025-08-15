# System Documentation

## Overview
The mCODE Translator system processes clinical trial eligibility criteria into structured mCODE elements using multiple NLP approaches.

## Key Components

### NLP Engines
- **LLM Engine**: DeepSeek API for complex criteria
- **spaCy Engine**: Medical NLP for general criteria
- **Regex Engine**: Fast pattern matching

### Core Modules
- **Criteria Parser**: Identifies inclusion/exclusion sections
- **Code Extractor**: Maps text to standard codes
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
- Processing time: 50-500ms per criteria
- Accuracy: 85-95% depending on criteria complexity