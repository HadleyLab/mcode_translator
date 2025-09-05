# Mcode Translator

A comprehensive clinical trial data processing pipeline that extracts and maps eligibility criteria to standardized Mcode elements using LLM-based natural language processing.

## Overview

The Mcode Translator is a sophisticated system that processes clinical trial eligibility criteria from ClinicalTrials.gov and maps them to standardized Mcode (Minimal Common Oncology Data Elements) format. The system features:

- **NLP Extraction to Mcode Mapping Pipeline**: LLM-based entity extraction with no fallbacks
- **Patient-Trial Matching**: Interactive web application for matching patient profiles to clinical trials
- **Prompt Optimization Framework**: Advanced prompt engineering for improved extraction accuracy
- **Source Provenance Tracking**: Comprehensive tracking of extraction sources and confidence scores
- **API Response Caching**: Disk-based caching for improved performance and reduced API calls
- **Gold Standard Validation**: Automated validation against expert-annotated gold standard data
- **Benchmarking System**: Comprehensive performance metrics collection and analysis

## Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended) or virtual environment
- ClinicalTrials.gov API access

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mcode_translator
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n mcode_translator python=3.8
   conda activate mcode_translator
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Running the Mcode Translator CLI

The project now uses a unified CLI interface for all functionality:

```bash
# Show CLI help
python Mcode-cli.py --help

# Run optimization demo
python Mcode-cli.py optimization demo

# Run comprehensive benchmark
python Mcode-cli.py benchmark comprehensive

# Run prompt library demo
python Mcode-cli.py prompt demo

# Run patient-trial matcher web application
python Mcode-cli.py webapp start

# Run pipeline task tracker with gold standard validation
python Mcode-cli.py pipeline-tracker start
```

### Running the Patient-Trial Matcher Web Application

```bash
python Mcode-cli.py webapp start
```

The application will be accessible at:
- http://localhost:8080
- http://127.0.0.1:8080

### Running the Pipeline Task Tracker with Gold Standard Validation

```bash
python Mcode-cli.py pipeline-tracker start
```

The pipeline task tracker provides:
- Real-time validation against gold standard data
- Precision, recall, and F1-score metrics
- Benchmarking of processing time and token usage
- Concurrent task processing with adjustable concurrency
- Accessible at: http://localhost:8090

## Project Architecture

### Core Components

#### 1. Strict LLM Base Engine ([`src/pipeline/llm_base.py`](src/pipeline/llm_base.py:52))
- **Purpose**: Foundation for all LLM-based components with strict error handling
- **Features**: Cache isolation, token tracking, strict JSON parsing, no fallbacks
- **Key Classes**: `LlmBase`, `LLMCallMetrics`, `LlmConfigurationError`

#### 2. Strict NLP Extractor ([`src/pipeline/strict_nlp_extractor.py`](src/pipeline/strict_nlp_extractor.py))
- **Purpose**: Entity extraction from clinical text using LLMs
- **Features**: Pattern-based extraction, confidence scoring, strict validation
- **Key Classes**: `NlpLlm`

#### 3. NLP Extraction to Mcode Mapping Pipeline ([`src/pipeline/nlp_mcode_pipeline.py`](src/pipeline/nlp_mcode_pipeline.py:33))
- **Purpose**: Main processing pipeline for clinical trial data
- **Features**: LLM-based entity extraction, Mcode mapping, source provenance tracking
- **Key Classes**: `NlpMcodePipeline`, `PipelineResult`

#### 4. Mcode Mapper ([`src/pipeline/Mcode_mapper.py`](src/pipeline/Mcode_mapper.py))
- **Purpose**: Map extracted entities to standardized Mcode elements
- **Features**: Rule-based mapping, validation, compliance scoring

#### 5. Document Ingestor ([`src/pipeline/document_ingestor.py`](src/pipeline/document_ingestor.py))
- **Purpose**: Extract and structure clinical trial document sections
- **Features**: Section identification, content extraction, metadata enrichment

#### 6. Patient-Trial Matcher ([`src/patient_matcher.py`](src/patient_matcher.py:31))
- **Purpose**: Interactive web application for matching patients to trials
- **Features**: Real-time search, visualization, matching analysis
- **Technologies**: NiceGUI, asynchronous processing

#### 7. Pipeline Task Tracker ([`src/optimization/pipeline_task_tracker.py`](src/optimization/pipeline_task_tracker.py))
- **Purpose**: Track individual pipeline tasks with gold standard validation
- **Features**: Concurrent task processing, validation metrics (precision/recall/F1), benchmarking
- **Technologies**: NiceGUI, asyncio queue-based concurrency

### Strict Framework Implementation

The project uses a **strict framework** approach with no fallbacks:

- **No Legacy Code**: All components use the latest LLM-based extraction
- **Direct Integration**: Tight coupling between NLP engine and Mcode mapper
- **Hard Failures**: Invalid configurations result in immediate failures
- **Source Provenance**: Comprehensive tracking of extraction sources
- **Cache Isolation**: Proper cache separation between different model configurations
- **Strict JSON Parsing**: No fallbacks or repair mechanisms - valid JSON or exception
- **Gold Standard Validation**: Automated validation against expert-annotated datasets
- **Benchmarking Metrics**: Comprehensive performance tracking and analysis

## Usage Examples

### Programmatic Usage

```python
from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline

# Initialize the strict pipeline
pipeline = StrictDynamicExtractionPipeline()

# Process clinical trial data
result = pipeline.process_clinical_trial(trial_data)

# Access results
entities = result.extracted_entities
mappings = result.mcode_mappings
sources = result.source_references
```

### Web Application Features

- **Patient Profile Management**: Edit cancer type, stage, biomarkers
- **Clinical Trial Search**: Advanced filtering and pagination
- **NLP to Mcode Visualization**: Interactive mapping visualization
- **Matching Analysis**: Biomarker-based compatibility scoring
- **Source Provenance**: Detailed extraction source tracking
- **Gold Standard Validation**: Precision, recall, and F1-score metrics
- **Benchmarking**: Processing time, token usage, and performance metrics

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
CLINICAL_TRIALS_API_KEY=your_clinical_trials_api_key
CACHE_ENABLED=true
CACHE_EXPIRY=3600
LOG_LEVEL=INFO
```

### Prompt Library

The system uses a file-based prompt library located in [`prompts/`](prompts/):

- **Extraction Prompts**: [`prompts/txt/nlp_extraction/`](prompts/txt/nlp_extraction/)
- **Mapping Prompts**: [`prompts/txt/Mcode_mapping/`](prompts/txt/Mcode_mapping/)

## Testing

### Running Tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run end-to-end tests
python -m pytest tests/e2e/ -v

# Run complete test suite
python -m pytest tests/ -v
```

### Test Structure

- **Unit Tests**: [`tests/unit/`](tests/unit/) - Individual component testing
- **Integration Tests**: [`tests/integration/`](tests/integration/) - Component interaction testing
- **End-to-End Tests**: [`tests/e2e/`](tests/e2e/) - Full pipeline testing
- **Test Data**: [`tests/data/`](tests/data/) - Gold standard test cases

## Development

### Code Standards

- **PEP 8 Compliance**: All code follows Python style guidelines
- **Type Hints**: Comprehensive type annotations throughout
- **Docstrings**: Google-style docstrings for all public methods
- **Logging**: Structured logging with configurable levels

### Project Structure

```
mcode_translator/
├── src/                    # Source code
│   ├── pipeline/          # Core processing pipeline
│   ├── optimization/      # Prompt optimization and benchmarking
│   ├── utils/            # Utility functions and helpers
│   └── validation/       # Validation components
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   ├── e2e/            # End-to-end tests
│   └── data/           # Test data
├── prompts/             # Prompt library
│   ├── txt/
│   │   ├── nlp_extraction/
│   │   └── Mcode_mapping/
│   └── prompts_config.json
├── examples/            # Usage examples and configurations
├── docs/               # Documentation
├── archive/            # Archived components (deprecated)
└── optimization_results/ # Benchmark results (gitignored)

## Performance

### Benchmark Results

The strict pipeline demonstrates:

- **High Accuracy**: >90% entity extraction accuracy
- **Fast Processing**: ~2-3 seconds per clinical trial
- **Scalable**: Handles large volumes of clinical data
- **Reliable**: Consistent results across diverse trial types
- **Validated Performance**: Gold standard validation with precision, recall, and F1-score metrics

### Gold Standard Validation

The system includes comprehensive gold standard validation capabilities:

- **Precision/Recall/F1 Metrics**: Quantitative validation against expert-annotated datasets
- **Fuzzy Text Matching**: Advanced text similarity algorithms for robust comparison
- **Real-time Validation**: Immediate feedback on extraction and mapping accuracy
- **Color-coded Results**: Visual indicators of validation quality (excellent/good/poor)

### Token Usage Tracking

The system includes a unified token tracking system that monitors and reports token consumption across all LLM calls:

- **Cross-Provider Compatibility**: Tracks token usage for all supported LLM providers
- **Detailed Metrics**: Monitors prompt tokens, completion tokens, and total consumption
- **Aggregated Reporting**: Provides both per-call and aggregate token usage statistics
- **Cost Optimization**: Enables detailed cost analysis and optimization opportunities

### Benchmarking System

The system includes comprehensive benchmarking capabilities:

- **Processing Time Tracking**: End-to-end and per-component timing metrics
- **Resource Consumption**: Memory and CPU usage monitoring
- **Performance Analysis**: Comparative analysis across different configurations
- **Validation Integration**: Combined accuracy and performance metrics

### Model Library

The system includes a file-based model library ([`models/models_config.json`](models/models_config.json)) that centralizes all LLM model configurations:

- **Centralized Management**: All model configurations in one location
- **Version Control**: Track model configuration changes through git
- **Experimentation**: Easy A/B testing of model configurations
- **Reusability**: Model configurations can be shared across components

### Optimization

The system includes an advanced prompt optimization framework ([`src/optimization/prompt_optimization_framework.py`](src/optimization/prompt_optimization_framework.py)) and prompt library integration that:

- Automatically tunes prompt templates
- Optimizes extraction patterns
- Improves mapping accuracy
- Reduces false positives
- Validates improvements against gold standard data

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

## Support

For support and questions:

1. Check the [`docs/`](docs/) directory for detailed documentation
2. Review [`TEST_RUNNER_GUIDE.md`](TEST_RUNNER_GUIDE.md) for testing guidance
3. Examine [`examples/`](examples/) for usage patterns

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for recent changes and updates.

## Acknowledgments

- ClinicalTrials.gov for providing the clinical trial data API
- Mcode initiative for standardized oncology data elements
- DeepSeek for LLM API services
- NiceGUI for the web application framework

## Caching System

The project now includes a custom disk-based caching system for API responses and LLM calls to improve performance and reduce API calls.

### Cache Implementation

The caching system uses a custom `@cache_api_response` decorator and direct cache access for LLM calls:

- **Custom Cache Decorator**: [`src/utils/cache_decorator.py`](src/utils/cache_decorator.py) - Disk-based caching with TTL support
- **Automatic Caching**: Applied to key API functions in [`src/pipeline/fetcher.py`](src/pipeline/fetcher.py)
- **LLM Caching**: Built-in caching for all LLM API calls in [`src/pipeline/llm_base.py`](src/pipeline/llm_base.py)
- **Cache Management**: Built-in cache statistics and clearing functionality

### Cached Functions

The following functions in [`src/pipeline/fetcher.py`](src/pipeline/fetcher.py) are automatically cached:

- `search_trials()` - Cached for 1 hour (3600 seconds)
- `get_full_study()` - Cached for 24 hours (86400 seconds)
- `calculate_total_studies()` - Cached for 1 hour (3600 seconds)

### LLM Caching

All LLM API calls through the [`LlmBase`](src/pipeline/llm_base.py:52) class are automatically cached:

- **Cache Key Generation**: Deterministic cache keys based on prompt content and model parameters
- **Cache Storage**: Disk-based storage in `./.llm_cache/` directory
- **Cache TTL**: 24 hours by default
- **Cache Isolation**: Separate cache instances for different model configurations

### Cache Locations

- **API Cache**: `./.api_cache/` - Caches ClinicalTrials.gov API responses with JSON serialization
- **LLM Cache**: `./.llm_cache/` - Caches LLM API responses with JSON serialization

### Cache Management

The system provides utilities for cache management:

```python
from src.utils.cache_decorator import get_cache_stats, clear_api_cache

# Get cache statistics
stats = get_cache_stats()
print(f"Cached items: {stats['cached_items']}")

# Clear all cached data
clear_api_cache()
```

### Cache Demo

A demonstration script is available to show the caching functionality in action:

```bash
# Run cache functionality demo
python demo_cache_functionality.py

# Run LLM cache test
python test_llm_caching.py
```