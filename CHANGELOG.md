# Changelog

## [Unreleased]
### Added
- Made LLM the default NLP engine in ExtractionPipeline
- Engine-specific caching system to prevent conflicts between different NLP engines
- Enhanced RegexNLPEngine with:
  - More comprehensive gene variant patterns (BRCA1/2, TP53, PIK3CA, etc.)
  - Protein-level variant detection (e.g., p.Val600Glu)
  - Improved biomarker status detection (quantitative values, IHC scores)
  - Deduplication of extracted variants and biomarkers

## [Unreleased]
### Added
- Engine selection system (Regex/SpaCy/LLM)
- Benchmark mode for comparing NLP engines
- Visual feedback for extracted mCODE elements

### Changed
- Updated ExtractionPipeline to properly handle engine types
- Improved UI with switches instead of toggles
- Optimized processing pipeline

### Fixed
- Regex engine selection issues
- AttributeError in process_criteria()
- Deprecated code cleanup

## [0.1.0] - 2025-07-01
### Initial Release
- Basic NLP processing capabilities
- mCODE mapping functionality
- Clinical Trials API integration