# Usage Examples

## Basic Usage

### Processing a Single Clinical Trial

```python
from src.pipeline import McodePipeline

# Initialize pipeline
pipeline = McodePipeline()

# Process trial data
result = pipeline.process(trial_data)

# Access results
print(f"Extracted {len(result.mcode_mappings)} mCODE elements")
for element in result.mcode_mappings:
    print(f"- {element.element_type}: {element.display}")
```

### Processing Multiple Trials

```python
from src.pipeline import McodePipeline

# Initialize pipeline
pipeline = McodePipeline()

# Process batch of trials
results = pipeline.process_batch(trials_data)

# Process results
for i, result in enumerate(results):
    print(f"Trial {i+1}: {len(result.mcode_mappings)} elements")
```

## Advanced Configuration

### Custom Model and Prompt

```python
from src.pipeline import McodePipeline

# Use specific LLM model and prompt
pipeline = McodePipeline(
    model_name="gpt-4",
    prompt_name="direct_mcode_evidence_based_concise"
)

result = pipeline.process(trial_data)
```

### Regex Processing

```python
from src.pipeline import McodePipeline

# Use regex engine instead of LLM
pipeline = McodePipeline(engine="regex")

result = pipeline.process(trial_data)
```

## Workflow Usage

### Clinical Trials Processing

```python
from src.workflows.trials_processor import TrialsProcessor

# Initialize processor
processor = TrialsProcessor()

# Process trials with full workflow
result = processor.execute(
    trials_data=trials_data,
    engine="llm",
    model="deepseek-coder",
    workers=4,
    store_in_memory=True
)

print(f"Processed {result.metadata['successful']} trials successfully")
```

### Patient Data Processing

```python
from src.workflows.patients_processor import PatientsProcessor

# Initialize processor
processor = PatientsProcessor()

# Process patients
result = processor.execute(
    patients_data=patients_data,
    engine="llm",
    model="gpt-4",
    store_in_memory=True
)
```

## Summarization

### Generate Patient Summary

```python
from src.services.summarizer import McodeSummarizer

# Initialize summarizer
summarizer = McodeSummarizer(
    include_dates=True,
    detail_level="full",
    include_mcode=True
)

# Generate summary
summary = summarizer.create_patient_summary(patient_data)
print(summary)
```

### Generate Trial Summary

```python
from src.services.summarizer import McodeSummarizer

# Initialize summarizer
summarizer = McodeSummarizer()

# Generate summary
summary = summarizer.create_trial_summary(trial_data)
print(summary)
```

### Minimal Detail Level

```python
from src.services.summarizer import McodeSummarizer

# Minimal summary for quick overview
summarizer = McodeSummarizer(
    detail_level="minimal",
    include_dates=False,
    include_mcode=False
)

summary = summarizer.create_patient_summary(patient_data)
print(summary)  # Only most critical information
```

## CLI Usage

### Process Clinical Trials

```bash
# Process trials from file
python mcode_cli.py trials process --input trials.ndjson --output results.ndjson

# Process with specific model
python mcode_cli.py trials process --input trials.ndjson --model gpt-4

# Process with regex engine
python mcode_cli.py trials process --input trials.ndjson --engine regex
```

### Fetch Trial Data

```bash
# Fetch by NCT ID
python mcode_cli.py trials fetch --nct-ids NCT04348955

# Search by condition
python mcode_cli.py trials fetch --condition "breast cancer" --limit 5

# Save to file
python mcode_cli.py trials fetch --condition "lung cancer" --output trials.ndjson
```

### Process Patient Data

```bash
# Process patients from file
python mcode_cli.py patients process --input patients.ndjson --output results.ndjson

# Process with memory storage
python mcode_cli.py patients process --input patients.ndjson --store-in-memory
```

### Memory Operations

```bash
# Check memory status
python mcode_cli.py memory status

# Clear memory spaces
python mcode_cli.py memory clear

# Get memory statistics
python mcode_cli.py memory stats
```

### System Diagnostics

```bash
# Check system status
python mcode_cli.py status

# Run comprehensive diagnostics
python mcode_cli.py doctor

# Show version
python mcode_cli.py version
```

## Configuration Examples

### Environment Variables

```bash
# Set API keys
export HEYSOL_API_KEY="your-heysol-key"
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"

# Configure logging
export LOG_LEVEL="DEBUG"

# Enable live tests
export ENABLE_LIVE_TESTS="true"
```

### Custom Configuration

```python
from src.utils.config import Config

# Load custom configuration
config = Config()

# Override settings
config.set_api_key("gpt-4", "your-api-key")
config.set_temperature("gpt-4", 0.3)
config.set_max_tokens("gpt-4", 2000)
```

## Data Formats

### Clinical Trial Data

```json
{
  "protocolSection": {
    "identificationModule": {
      "nctId": "NCT04348955",
      "briefTitle": "Study of Treatment for Cancer",
      "officialTitle": "Official Study Title"
    },
    "eligibilityModule": {
      "eligibilityCriteria": "Inclusion criteria: Patients with...",
      "sex": "All",
      "minimumAge": "18 Years",
      "maximumAge": "75 Years"
    },
    "conditionsModule": {
      "conditions": [
        {"name": "Breast Cancer"}
      ]
    }
  }
}
```

### Patient Data (FHIR Bundle)

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "12345",
        "name": [{"family": "Smith", "given": ["John"]}],
        "gender": "male",
        "birthDate": "1980-01-01"
      }
    }
  ]
}
```

## Error Handling

### Handling Processing Errors

```python
from src.pipeline import McodePipeline

pipeline = McodePipeline()

try:
    result = pipeline.process(trial_data)
    if result.error:
        print(f"Processing error: {result.error}")
    else:
        print(f"Success: {len(result.mcode_mappings)} elements")
except Exception as e:
    print(f"Pipeline error: {e}")
```

### Workflow Error Handling

```python
from src.workflows.trials_processor import TrialsProcessor

processor = TrialsProcessor()

result = processor.execute(trials_data=trials_data)

if not result.success:
    print(f"Workflow failed: {result.error_message}")
else:
    print(f"Processed {result.metadata['successful']} trials")
```

## Performance Optimization

### Concurrent Processing

```python
from src.workflows.trials_processor import TrialsProcessor

# Use multiple workers for concurrent processing
processor = TrialsProcessor()
result = processor.execute(
    trials_data=trials_data,
    workers=8,  # Use 8 concurrent workers
    store_in_memory=True
)
```

### Batch Processing

```python
from src.pipeline import McodePipeline

# Process large batches efficiently
pipeline = McodePipeline()

# Split large dataset into chunks
chunk_size = 50
for i in range(0, len(all_trials), chunk_size):
    chunk = all_trials[i:i + chunk_size]
    results = pipeline.process_batch(chunk)
    # Process results...
```

## Integration Examples

### With External APIs

```python
import requests
from src.pipeline import McodePipeline

# Fetch data from external API
response = requests.get("https://api.clinicaltrials.gov/v2/studies/NCT04348955")
trial_data = response.json()

# Process with mCODE pipeline
pipeline = McodePipeline()
result = pipeline.process(trial_data)
```

### Custom Data Processing

```python
from src.shared.models import ClinicalTrialData
from src.pipeline import McodePipeline

# Validate and process custom data
validated_trial = ClinicalTrialData(**custom_trial_data)
pipeline = McodePipeline()
result = pipeline.process(validated_trial.model_dump())
```

### Memory Integration

```python
from src.storage.mcode_memory_storage import OncoCoreMemory
from src.workflows.trials_processor import TrialsProcessor

# Process and store in memory
processor = TrialsProcessor()
memory = OncoCoreMemory()

result = processor.execute(
    trials_data=trials_data,
    store_in_memory=True
)

# Query stored data
stats = memory.get_memory_stats()
print(f"Stored {stats.total_spaces} memory spaces")