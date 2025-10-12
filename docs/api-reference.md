# API Reference

## Core Classes

### McodePipeline

The main processing pipeline for mCODE translation.

#### Constructor

```python
McodePipeline(
    model_name: Optional[str] = None,
    prompt_name: Any = _PROMPT_DEFAULT,
    config: Optional[Config] = None,
    engine: str = "llm"
)
```

**Parameters:**
- `model_name`: LLM model name (default: "deepseek-coder" for LLM, "regex" for regex)
- `prompt_name`: Prompt template name (default: "direct_mcode_evidence_based_concise")
- `config`: Application configuration instance
- `engine`: Processing engine ("llm" or "regex")

#### Methods

##### process(trial_data: Dict[str, Any]) -> PipelineResult

Process a single clinical trial data dictionary.

**Parameters:**
- `trial_data`: Raw clinical trial data

**Returns:** `PipelineResult` with processing results

##### process_batch(trials_data: List[Dict[str, Any]]) -> List[PipelineResult]

Process multiple clinical trials concurrently.

**Parameters:**
- `trials_data`: List of trial data dictionaries

**Returns:** List of `PipelineResult` instances

---

### LLMService

Handles AI-powered mCODE mapping using LLM providers.

#### Constructor

```python
LLMService(config: Config, model_name: str, prompt_name: str)
```

**Parameters:**
- `config`: Application configuration
- `model_name`: LLM model identifier
- `prompt_name`: Prompt template name

#### Methods

##### map_to_mcode(clinical_text: str) -> List[McodeElement]

Map clinical text to mCODE elements using LLM processing.

**Parameters:**
- `clinical_text`: Clinical trial text to process

**Returns:** List of `McodeElement` instances

---

### McodeSummarizer

Generates natural language summaries from mCODE elements.

#### Constructor

```python
McodeSummarizer(
    include_dates: bool = True,
    detail_level: str = "full",
    include_mcode: bool = True
)
```

**Parameters:**
- `include_dates`: Whether to include dates in summaries
- `detail_level`: Detail level ("minimal", "standard", "full")
- `include_mcode`: Whether to include mCODE annotations

#### Methods

##### create_patient_summary(patient_data: Dict[str, Any], include_dates: bool = None) -> str

Generate natural language summary for patient data.

**Parameters:**
- `patient_data`: FHIR patient bundle data
- `include_dates`: Override date inclusion setting

**Returns:** Formatted patient summary string

##### create_trial_summary(trial_data: Dict[str, Any]) -> str

Generate natural language summary for clinical trial data.

**Parameters:**
- `trial_data`: Clinical trial data dictionary

**Returns:** Formatted trial summary string

---

## Workflow Classes

### TrialsProcessor

High-level workflow for processing clinical trials.

#### Constructor

```python
TrialsProcessor(config: Any, memory_storage: Optional[OncoCoreMemory] = None)
```

#### Methods

##### execute(**kwargs: Any) -> WorkflowResult

Execute trial processing workflow.

**Parameters:**
- `trials_data`: List of trial data to process
- `engine`: Processing engine ("llm" or "regex")
- `model`: LLM model name
- `prompt`: Prompt template name
- `workers`: Number of concurrent workers
- `store_in_memory`: Whether to store results in memory

**Returns:** `WorkflowResult` with processing results

##### process_single_trial(trial: Dict[str, Any], **kwargs: Any) -> WorkflowResult

Process a single clinical trial.

---

### PatientsProcessor

High-level workflow for processing patient data.

#### Constructor

```python
PatientsProcessor(config: Any, memory_storage: Optional[OncoCoreMemory] = None)
```

#### Methods

##### execute(**kwargs: Any) -> WorkflowResult

Execute patient processing workflow.

**Parameters:**
- `patients_data`: List of patient data to process
- `engine`: Processing engine ("llm" or "regex")
- `model`: LLM model name
- `prompt`: Prompt template name
- `workers`: Number of concurrent workers
- `store_in_memory`: Whether to store results in memory

---

## Data Models

### PipelineResult

Standardized result from processing operations.

#### Fields

- `extracted_entities: List[Dict[str, Any]]` - Extracted entities
- `mcode_mappings: List[McodeElement]` - mCODE element mappings
- `source_references: List[SourceReference]` - Source text references
- `validation_results: ValidationResult` - Validation results
- `metadata: ProcessingMetadata` - Processing metadata
- `original_data: Dict[str, Any]` - Original input data
- `error: Optional[str]` - Error message if processing failed

### McodeElement

Individual mCODE element mapping.

#### Fields

- `element_type: str` - Type of mCODE element
- `code: Optional[str]` - Element code
- `display: Optional[str]` - Human-readable display name
- `system: Optional[str]` - Coding system
- `confidence_score: Optional[float]` - Confidence score (0.0-1.0)

### ValidationResult

Results of mCODE mapping validation.

#### Fields

- `compliance_score: float` - Overall compliance score (0.0-1.0)
- `validation_errors: List[str]` - List of validation errors
- `validation_warnings: List[str]` - List of validation warnings
- `required_elements_present: List[str]` - Required elements found
- `missing_elements: List[str]` - Required elements missing

### ProcessingMetadata

Metadata about processing operations.

#### Fields

- `engine_type: str` - Processing engine used
- `entities_count: int` - Number of entities extracted
- `mapped_count: int` - Number of elements mapped
- `processing_time_seconds: Optional[float]` - Processing time
- `model_used: Optional[str]` - LLM model used
- `prompt_used: Optional[str]` - Prompt template used
- `token_usage: Optional[TokenUsage]` - Token usage statistics

---

## Storage Classes

### OncoCoreMemory

Memory storage integration with HeySol API.

#### Constructor

```python
OncoCoreMemory()
```

#### Methods

##### store_patient_data(patient_id: str, data: Dict[str, Any]) -> bool

Store patient data in memory.

##### store_trial_data(trial_id: str, data: Dict[str, Any]) -> bool

Store trial data in memory.

##### get_memory_stats() -> MemoryStats

Get memory usage statistics.

---

## Utility Classes

### Config

Application configuration management.

#### Methods

##### get_api_key(model_name: str) -> Optional[str]

Get API key for specified model.

##### get_temperature(model_name: str) -> float

Get temperature setting for model.

##### get_max_tokens(model_name: str) -> int

Get max tokens setting for model.

---

### APIManager

API caching and management.

#### Methods

##### get_cache(cache_type: str) -> APICache

Get cache instance for specified type.

##### clear_cache(cache_type: str) -> None

Clear specified cache type.

---

## CLI Commands

### Trials Commands

```bash
# Process trials
python mcode_cli.py trials process --input trials.ndjson --output results.ndjson

# Fetch trials
python mcode_cli.py trials fetch --condition "breast cancer" --limit 10

# Optimize processing
python mcode_cli.py trials optimize --trials-file trials.ndjson
```

### Patients Commands

```bash
# Process patients
python mcode_cli.py patients process --input patients.ndjson --output results.ndjson

# Fetch patients
python mcode_cli.py patients fetch --limit 100
```

### Memory Commands

```bash
# Check memory status
python mcode_cli.py memory status

# Clear memory
python mcode_cli.py memory clear
```

### Configuration Commands

```bash
# Check configuration
python mcode_cli.py config check

# Validate setup
python mcode_cli.py config validate