# mCODE Translator Documentation

## Overview

The mCODE Translator is a comprehensive Python framework for transforming clinical trial data into standardized mCODE (Minimal Common Oncology Data Elements) format using advanced AI-powered processing.

## Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

- **Pipeline**: Ultra-lean processing pipeline with LLM and regex engines
- **Services**: Specialized services for LLM processing, summarization, and data extraction
- **Workflows**: High-level workflows for trials and patients processing
- **Storage**: Memory storage integration with HeySol API
- **CLI**: Command-line interface for all operations

### Data Flow

```
Raw Data → Pipeline → mCODE Elements → Summarization → Storage
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.pipeline import McodePipeline

# Process clinical trial data
pipeline = McodePipeline()
result = pipeline.process(trial_data)
```

### CLI Usage

```bash
# Process trials
python mcode_cli.py trials process --input trials.ndjson

# Process patients
python mcode_cli.py patients process --input patients.ndjson
```

## API Reference

### Core Classes

#### McodePipeline

Main processing pipeline for mCODE translation.

**Methods:**
- `process(trial_data: Dict[str, Any]) -> PipelineResult`: Process single trial
- `process_batch(trials_data: List[Dict[str, Any]]) -> List[PipelineResult]`: Process multiple trials

#### LLMService

Handles AI-powered mCODE mapping using various LLM providers.

**Methods:**
- `map_to_mcode(clinical_text: str) -> List[McodeElement]`: Map text to mCODE elements

#### McodeSummarizer

Generates natural language summaries from mCODE elements.

**Methods:**
- `create_patient_summary(patient_data: Dict[str, Any]) -> str`: Generate patient summary
- `create_trial_summary(trial_data: Dict[str, Any]) -> str`: Generate trial summary

### Data Models

#### PipelineResult

Standardized result from processing operations.

**Fields:**
- `extracted_entities`: List of extracted entities
- `mcode_mappings`: List of mCODE element mappings
- `validation_results`: Validation results
- `metadata`: Processing metadata
- `original_data`: Original input data

#### McodeElement

Individual mCODE element mapping.

**Fields:**
- `element_type`: Type of mCODE element
- `code`: Element code
- `display`: Human-readable display name
- `system`: Coding system
- `confidence_score`: Confidence score

## Configuration

### Environment Variables

- `HEYSOL_API_KEY`: API key for HeySol memory storage
- `OPENAI_API_KEY`: API key for OpenAI models
- `DEEPSEEK_API_KEY`: API key for DeepSeek models

### Configuration Files

- `src/config/llms_config.json`: LLM provider configurations
- `src/config/apis_config.json`: API endpoint configurations
- `src/config/core_memory_config.json`: Memory storage settings

## Usage Examples

### Processing Clinical Trials

```python
from src.workflows.trials_processor import TrialsProcessor

processor = TrialsProcessor()
result = processor.execute(trials_data=[trial_data])
```

### Processing Patient Data

```python
from src.workflows.patients_processor import PatientsProcessor

processor = PatientsProcessor()
result = processor.execute(patients_data=[patient_data])
```

### Custom Pipeline Configuration

```python
from src.pipeline import McodePipeline

# Use specific model and prompt
pipeline = McodePipeline(
    model_name="gpt-4",
    prompt_name="direct_mcode_evidence_based_concise"
)
```

## Module Reference

### src.pipeline

Core processing pipeline components.

- `McodePipeline`: Main processing pipeline
- `DocumentIngestor`: Document processing and section extraction

### src.services

Specialized service components.

- `llm.service.LLMService`: LLM processing service
- `summarizer.McodeSummarizer`: Summary generation service

### src.workflows

High-level workflow orchestrators.

- `trials_processor.TrialsProcessor`: Clinical trials processing
- `patients_processor.PatientsProcessor`: Patient data processing

### src.storage

Data persistence components.

- `mcode_memory_storage.OncoCoreMemory`: HeySol memory integration

### src.cli

Command-line interface.

- `commands.trials`: Trial processing commands
- `commands.patients`: Patient processing commands
- `commands.memory`: Memory management commands

## Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Layer     │ -> │   Workflow Layer │ -> │  Service Layer  │
│                 │    │                  │    │                 │
│ • Commands      │    │ • Orchestration  │    │ • LLM Service   │
│ • Configuration │    │ • Validation     │    │ • Summarizer    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Storage Layer  │ <- │   Pipeline       │ <- │   Data Models   │
│                 │    │   Processing     │    │                 │
│ • HeySol API    │    │ • mCODE Mapping  │    │ • Pydantic       │
│ • Core Memory   │    │ • Validation     │    │ • Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Contributing

See the main README.md for contribution guidelines and development setup.