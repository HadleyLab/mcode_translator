# Module Reference

## src.cli

Command-line interface components.

### __init__.py

Main CLI application with global state management and command registration.

**Key Classes:**
- `GlobalState`: Manages global CLI state (API keys, configuration)
- CLI callback functions for authentication and configuration

**Key Functions:**
- `cli_callback()`: Global CLI callback for authentication setup
- `version()`: Show version information
- `status()`: Check system status and connectivity
- `doctor()`: Run comprehensive system diagnostics

### commands/

#### config.py

Configuration management commands.

**Commands:**
- `check`: Validate configuration files
- `validate`: Comprehensive configuration validation

#### memory.py

CORE Memory management commands.

**Commands:**
- `status`: Check memory system status
- `clear`: Clear memory spaces
- `stats`: Show memory usage statistics

#### patients.py

Patient data processing commands.

**Commands:**
- `process`: Process patient data with mCODE mapping
- `fetch`: Fetch patient data from APIs
- `summarize`: Generate patient summaries

#### trials.py

Clinical trial processing commands.

**Commands:**
- `process`: Process trial data with mCODE mapping
- `fetch`: Fetch trial data from ClinicalTrials.gov
- `optimize`: Optimize processing parameters
- `summarize`: Generate trial summaries

---

## src.core

Core infrastructure components.

### batch_processor.py

Batch processing utilities for concurrent operations.

### data_flow_coordinator.py

Coordinates data flow between components.

### dependency_container.py

Dependency injection container for component management.

**Key Classes:**
- `DependencyContainer`: Manages component creation and dependencies

**Key Functions:**
- `get_container()`: Get global container instance
- `create_trial_pipeline()`: Create trial processing pipeline
- `create_patient_pipeline()`: Create patient processing pipeline

### flow_summary_generator.py

Generates summaries of data processing flows.

---

## src.optimization

Performance optimization and analysis components.

### biological_analyzer.py

Biological data analysis and validation.

### cross_validation.py

Cross-validation for model performance assessment.

### execution_manager.py

Manages execution of optimization tasks.

### inter_rater_reliability.py

Calculates inter-rater reliability metrics.

### pairwise_cross_validation.py

Pairwise cross-validation implementation.

### performance_analyzer.py

Analyzes system performance metrics.

### report_generator.py

Generates optimization reports.

### result_aggregator.py

Aggregates optimization results.

---

## src.pipeline

Core processing pipeline components.

### __init__.py

Pipeline module exports.

### pipeline.py

Main mCODE processing pipeline.

**Key Classes:**
- `McodePipeline`: Ultra-lean pipeline with LLM and regex engines

**Key Methods:**
- `process()`: Process single trial data
- `process_batch()`: Process multiple trials concurrently

### document_ingestor.py

Document processing and section extraction.

**Key Classes:**
- `DocumentIngestor`: Extracts and cleans clinical trial text

---

## src.services

Specialized service components.

### __init__.py

Services module initialization.

### clinical_note_generator.py

Generates clinical notes from processed data.

### demographics_extractor.py

Extracts demographic information from patient data.

### fhir_extractors.py

FHIR resource extraction utilities.

### heysol_client.py

HeySol API client integration.

### summarizer.py

Natural language summarization service.

**Key Classes:**
- `McodeSummarizer`: Generates summaries from mCODE elements

**Key Methods:**
- `create_patient_summary()`: Generate patient summaries
- `create_trial_summary()`: Generate trial summaries

### llm/

#### api_caller.py

LLM API calling utilities.

#### engine.py

LLM processing engine.

#### response_parser.py

LLM response parsing and validation.

#### service.py

LLM service for mCODE mapping.

**Key Classes:**
- `LLMService`: Handles AI-powered mCODE mapping

**Key Methods:**
- `map_to_mcode()`: Map clinical text to mCODE elements

### regex/

#### service.py

Regex-based processing service.

---

## src.shared

Shared data models and utilities.

### __init__.py

Shared module initialization.

### cli_utils.py

CLI utility functions.

### extractors.py

Data extraction utilities.

### models.py

Pydantic data models for the application.

**Key Classes:**
- `ClinicalTrialData`: Clinical trial data structure
- `PipelineResult`: Processing result structure
- `McodeElement`: Individual mCODE element
- `ValidationResult`: Validation results
- `ProcessingMetadata`: Processing metadata
- `WorkflowResult`: Workflow execution results

### types.py

Type definitions and annotations.

---

## src.storage

Data persistence components.

### __init__.py

Storage module initialization.

### mcode_memory_storage.py

CORE Memory storage integration.

**Key Classes:**
- `OncoCoreMemory`: HeySol memory integration

**Key Methods:**
- `store_patient_data()`: Store patient data
- `store_trial_data()`: Store trial data
- `get_memory_stats()`: Get memory statistics

---

## src.utils

Utility functions and helpers.

### __init__.py

Utils module initialization.

### api_cache.py

API response caching.

### api_manager.py

API management and coordination.

**Key Classes:**
- `APIManager`: Manages API caching and requests

### async_api_cache.py

Asynchronous API caching.

### concurrency.py

Concurrency utilities.

### config.py

Configuration management.

**Key Classes:**
- `Config`: Application configuration

### data_downloader.py

Data downloading utilities.

### data_loader.py

Data loading and parsing.

### error_handler.py

Error handling utilities.

### feature_utils.py

Feature extraction utilities.

### fetcher.py

Data fetching utilities.

### llm_loader.py

LLM configuration loading.

### logging_config.py

Logging configuration.

### metrics.py

Metrics collection and reporting.

### patient_generator.py

Synthetic patient data generation.

### prompt_loader.py

Prompt template loading.

### token_tracker.py

Token usage tracking.

**Key Classes:**
- `TokenTracker`: Tracks LLM token usage

---

## src.workflows

High-level workflow orchestrators.

### __init__.py

Workflows module initialization.

### base_summarizer.py

Base summarization workflow.

### base_workflow.py

Base workflow class with common functionality.

### patients_fetcher.py

Patient data fetching workflow.

### patients_processor.py

Patient data processing workflow.

### patients_summarizer.py

Patient data summarization workflow.

### trial_extractor.py

Trial data extraction utilities.

### trial_summarizer.py

Trial data summarization utilities.

### trials_fetcher.py

Clinical trial fetching workflow.

### trials_optimizer.py

Trial processing optimization workflow.

### trials_processor.py

Clinical trial processing workflow.

**Key Classes:**
- `TrialsProcessor`: Main trial processing workflow

### trials_summarizer.py

Trial data summarization workflow.

---

## Configuration Modules

### src.config

Configuration files and settings.

#### __init__.py

Configuration module initialization.

#### apis_config.json

API endpoint configurations.

#### cache_config.json

Caching configurations.

#### core_memory_config.json

CORE Memory settings.

#### heysol_config.py

HeySol API configuration.

#### llms_config.json

LLM provider configurations.

#### logging_config.json

Logging settings.

#### patterns_config.json

Processing pattern configurations.

#### prompts_config.json

Prompt template configurations.

#### py.typed

Type checking marker.

#### README.md

Configuration documentation.

#### synthetic_data_config.json

Synthetic data generation settings.

#### validation_config.json

Validation rule configurations.

---

## Test Modules

### tests/

Comprehensive test suite.

#### __init__.py

Test module initialization.

#### conftest.py

Pytest configuration and fixtures.

#### README.md

Testing documentation.

#### data/

Test data files.

#### e2e/

End-to-end tests.

#### integration/

Integration tests.

#### performance/

Performance tests.

#### unit/

Unit tests.

---

## Example Modules

### examples/

Usage examples and demonstrations.

#### __init__.py

Examples module initialization.

#### engine_demo.py

Engine demonstration.

#### quick_start.py

Quick start example.

#### README.md

Examples documentation.