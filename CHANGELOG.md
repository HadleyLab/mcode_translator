# Changelog

## [Unreleased]
### Added
- **Gold Standard Validation System**: Comprehensive validation framework for pipeline results
  - Precision, recall, and F1-score metrics calculation
  - Fuzzy text matching with fuzzywuzzy (85% threshold) and difflib SequenceMatcher (0.8 threshold)
  - Tuple comparison for Mcode mapping validation
  - Color-coded validation badges in UI (green/red/yellow)
- **Benchmarking Metrics Collection**: Performance monitoring system
  - Processing time tracking (total, per-task, per-engine)
  - Token usage monitoring across LLM providers
  - CPU and memory usage monitoring
  - Real-time metrics display in pipeline task tracker UI
- Unified token tracking system for standardized token usage reporting across all LLM providers
- File-based model library for centralized LLM model configuration management
- Made LLM the default NLP engine in ExtractionPipeline
- Engine-specific caching system to prevent conflicts between different NLP engines
- Enhanced RegexNlpBase with:
  - More comprehensive gene variant patterns (BRCA1/2, TP53, PIK3CA, etc.)
  - Protein-level variant detection (e.g., p.Val600Glu)
  - Improved biomarker status detection (quantitative values, IHC scores)
  - Deduplication of extracted variants and biomarkers
- Data Fetcher Dashboard with NiceGUI interface
- Pagination support using nextPageToken instead of min_rank
- Enhanced documentation for system architecture and components
- Comprehensive test suite refactoring with pytest
- Sample data files for API responses

## [Unreleased]
### Added
- Engine selection system (Regex/SpaCy/LLM)
- Benchmark mode for comparing NLP engines
- Visual feedback for extracted Mcode elements

### Changed
- Updated ExtractionPipeline to properly handle engine types
- Improved UI with switches instead of toggles
- Optimized processing pipeline
- Refactored ClinicalTrials API integration to use nextPageToken for pagination
- Updated documentation to reflect current architecture
- Enhanced error handling and validation in data fetcher
- Archived legacy code_extraction and criteria_parser modules

### Fixed
- Regex engine selection issues
- AttributeError in process_criteria()
- Deprecated code cleanup
- Test suite refactoring to use pytest consistently
- Mock API responses to match actual ClinicalTrials.gov API structure
- Pagination logic and data structure alignment

## [0.1.0] - 2025-07-01
### Initial Release
- Basic NLP processing capabilities
- Mcode mapping functionality
- Clinical Trials API integration