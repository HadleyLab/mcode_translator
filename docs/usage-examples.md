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

## Expert Panel Assessment Examples

### Basic Expert Panel Assessment

```python
from src.matching.expert_panel_manager import ExpertPanelManager
from src.utils.config import Config

# Initialize expert panel with default configuration
panel = ExpertPanelManager(
    model_name="deepseek-coder",
    max_concurrent_experts=3,
    enable_diversity_selection=True
)

# Assess patient-trial match using expert panel
assessment = await panel.assess_with_expert_panel(
    patient_data={
        "id": "patient_123",
        "cancer_type": "breast_cancer",
        "stage": "II",
        "age": 45,
        "comorbidities": ["hypertension"],
        "current_medications": ["tamoxifen"]
    },
    trial_criteria={
        "eligibilityCriteria": "Patients with stage II breast cancer, age 18-65",
        "sex": "female",
        "conditions": ["Breast Cancer"]
    }
)

print(f"Match: {assessment['is_match']}")
print(f"Confidence: {assessment['confidence_score']:.3f}")
print(f"Consensus Level: {assessment['consensus_level']}")
print(f"Expert Assessments: {len(assessment['expert_assessments'])}")
```

### Custom Expert Selection

```python
from src.matching.expert_panel_manager import ExpertPanelManager

# Initialize panel
panel = ExpertPanelManager(model_name="gpt-4")

# Use specific experts for assessment
assessment = await panel.assess_with_expert_panel(
    patient_data=patient_data,
    trial_criteria=trial_criteria,
    expert_selection=["clinical_reasoning", "comprehensive_analyst"],
    diversity_threshold=0.8
)

# Access individual expert assessments
for assessment_data in assessment["expert_assessments"]:
    expert_type = assessment_data["expert_type"]
    confidence = assessment_data["assessment"]["confidence_score"]
    reasoning = assessment_data["assessment"]["reasoning"]
    print(f"{expert_type}: {confidence:.3f} - {reasoning[:100]}...")
```

### Expert Panel Status Monitoring

```python
from src.matching.expert_panel_manager import ExpertPanelManager

panel = ExpertPanelManager()

# Get comprehensive panel status
status = await panel.get_expert_panel_status()

print(f"Total Experts: {status['total_experts']}")
print(f"Expert Types: {status['expert_types']}")
print(f"Max Concurrent: {status['max_concurrent_experts']}")
print(f"Diversity Selection: {status['diversity_selection_enabled']}")

# Check individual expert status
for expert_type, expert_info in status["experts_status"].items():
    print(f"{expert_type}: {expert_info['model_name']} - Initialized: {expert_info['initialized']}")
```

## Ensemble Decision Configuration Examples

### Basic Ensemble Decision Engine

```python
from src.matching.ensemble_decision_engine import EnsembleDecisionEngine, ConsensusMethod

# Initialize ensemble engine with dynamic weighting
ensemble = EnsembleDecisionEngine(
    model_name="deepseek-coder",
    consensus_method=ConsensusMethod.DYNAMIC_WEIGHTING,
    min_experts=2,
    max_experts=3,
    enable_rule_based_integration=True,
    enable_dynamic_weighting=True
)

# Perform ensemble matching
is_match = await ensemble.match(patient_data, trial_criteria)

# Get detailed ensemble result
result = await ensemble._perform_ensemble_assessment(patient_data, trial_criteria)

print(f"Ensemble Match: {result.is_match}")
print(f"Confidence Score: {result.confidence_score:.3f}")
print(f"Consensus Method: {result.consensus_method}")
print(f"Expert Diversity: {result.diversity_score:.3f}")
print(f"Processing Time: {result.processing_metadata['processing_time']:.2f}s")
```

### Weighted Majority Vote Ensemble

```python
from src.matching.ensemble_decision_engine import EnsembleDecisionEngine, ConsensusMethod

# Configure for weighted majority voting
ensemble = EnsembleDecisionEngine(
    consensus_method=ConsensusMethod.WEIGHTED_MAJORITY_VOTE,
    confidence_calibration="isotonic_regression",
    enable_rule_based_integration=False
)

# Get ensemble assessment with detailed breakdown
result = await ensemble._perform_ensemble_assessment(patient_data, trial_criteria)

# Analyze individual expert contributions
for decision in result.individual_decisions:
    print(f"Expert: {decision['expert_type']}")
    print(f"  Match: {decision['is_match']}")
    print(f"  Confidence: {decision['confidence']:.3f}")
    print(f"  Weight: {decision['weight']:.3f}")
    print(f"  Weighted Confidence: {decision['weighted_confidence']:.3f}")
```

### Bayesian Ensemble Decision

```python
from src.matching.ensemble_decision_engine import EnsembleDecisionEngine, ConsensusMethod

# Configure for Bayesian ensemble (falls back to weighted majority)
ensemble = EnsembleDecisionEngine(
    consensus_method=ConsensusMethod.BAYESIAN_ENSEMBLE,
    enable_dynamic_weighting=True
)

# Perform assessment
result = await ensemble._perform_ensemble_assessment(patient_data, trial_criteria)

# Access ensemble metadata
metadata = result.processing_metadata
print(f"Consensus Method: {metadata['consensus_method']}")
print(f"Dynamic Weighting: {metadata['dynamic_weighting']}")
print(f"Rule-based Integrated: {metadata['rule_based_integrated']}")
```

### Expert Weight Management

```python
from src.matching.ensemble_decision_engine import EnsembleDecisionEngine

ensemble = EnsembleDecisionEngine()

# Update expert weights based on performance
performance_data = {
    "clinical_reasoning": {
        "accuracy": 0.85,
        "reliability": 0.90
    },
    "pattern_recognition": {
        "accuracy": 0.78,
        "reliability": 0.82
    },
    "comprehensive_analyst": {
        "accuracy": 0.88,
        "reliability": 0.95
    }
}

ensemble.update_expert_weights(performance_data)

# Get current ensemble status
status = ensemble.get_ensemble_status()
for expert_type, weights in status["expert_weights"].items():
    print(f"{expert_type}:")
    print(f"  Base Weight: {weights['base_weight']}")
    print(f"  Reliability: {weights['reliability_score']}")
    print(f"  Historical Accuracy: {weights['historical_accuracy']}")
```

## Advanced Matching Engine Usage Patterns

### LLM Engine with Expert Panel

```python
from src.matching.llm_engine import LLMMatchingEngine

# Initialize LLM engine with expert panel configuration
engine = LLMMatchingEngine(
    model_name="deepseek-coder",
    expert_panel_config={
        "max_concurrent_experts": 3,
        "enable_diversity_selection": True,
        "enable_caching": True
    }
)

# Perform matching with expert panel
result = await engine.match_with_experts(patient_data, trial_criteria)

print(f"Match Result: {result.is_match}")
print(f"Confidence: {result.confidence_score}")
print(f"Expert Panel Used: {len(result.expert_assessments) > 0}")
```

### Batch Matching with Ensemble Engine

```python
from src.matching.ensemble_decision_engine import EnsembleDecisionEngine
from src.matching.batch_matcher import BatchMatcher

# Initialize ensemble engine
ensemble_engine = EnsembleDecisionEngine(max_experts=3)

# Create batch matcher for concurrent processing
batch_matcher = BatchMatcher(ensemble_engine, max_concurrent=5)

# Prepare patient-trial pairs
pairs = [
    {"patient": patient1, "trial": trial1},
    {"patient": patient2, "trial": trial2},
    # ... more pairs
]

# Execute batch matching
results = await batch_matcher.match_batch(pairs)

# Process results
for i, result in enumerate(results):
    print(f"Pair {i+1}: Match={result.is_match}, Confidence={result.confidence_score:.3f}")
```

### Hybrid Gold Standard Generator

```python
from src.matching.hybrid_gold_standard_generator import HybridGoldStandardGenerator

# Initialize hybrid generator
generator = HybridGoldStandardGenerator(
    model_name="deepseek-coder",
    ensemble_config={
        "consensus_method": "dynamic_weighting",
        "min_experts": 2,
        "max_experts": 3
    }
)

# Generate gold standard labels
gold_standard = await generator.generate_gold_standard(
    patient_trial_pairs=pairs,
    validation_trials=trials_data
)

print(f"Gold Standard Generated: {len(gold_standard.labels)} labels")
print(f"Accuracy: {gold_standard.accuracy:.3f}")
print(f"Consensus Rate: {gold_standard.consensus_rate:.3f}")
```

## Performance Tuning Examples for Concurrent Expert Execution

### Optimizing Expert Panel Concurrency

```python
from src.matching.expert_panel_manager import ExpertPanelManager

# High-performance configuration for large-scale assessments
panel = ExpertPanelManager(
    model_name="deepseek-coder",
    max_concurrent_experts=5,  # Increase for better throughput
    enable_diversity_selection=True
)

# Process multiple assessments concurrently
assessments = []
patient_trial_pairs = [
    (patient1, trial1),
    (patient2, trial2),
    (patient3, trial3)
]

# Execute concurrently (panel handles internal concurrency)
for patient, trial in patient_trial_pairs:
    assessment = await panel.assess_with_expert_panel(patient, trial)
    assessments.append(assessment)

# Log performance metrics
cache_stats = panel._get_panel_cache_stats()
print(f"Cache Hit Rate: {cache_stats['panel_hit_rate']:.2%}")
print(f"Time Saved: {cache_stats['panel_total_time_saved_seconds']:.2f}s")
```

### Ensemble Engine Performance Tuning

```python
from src.matching.ensemble_decision_engine import EnsembleDecisionEngine

# Performance-optimized ensemble configuration
ensemble = EnsembleDecisionEngine(
    model_name="deepseek-coder",
    min_experts=2,
    max_experts=4,  # Balance speed vs. accuracy
    cache_enabled=True,
    max_retries=2  # Reduce retries for speed
)

# Batch processing with optimized concurrency
batch_size = 10
for i in range(0, len(pairs), batch_size):
    batch = pairs[i:i + batch_size]

    # Process batch concurrently
    tasks = [ensemble.match(p["patient"], p["trial"]) for p in batch]
    results = await asyncio.gather(*tasks)

    print(f"Processed batch {i//batch_size + 1}: {len(results)} matches")
```

### CLI Performance Tuning

```bash
# High-performance expert panel matching
python mcode_cli.py matching run --engine ensemble \
    --input patient_trial_pairs.ndjson \
    --output results.ndjson \
    --workers 8 \
    --model deepseek-coder \
    --expert-panel-config '{"max_concurrent_experts": 4, "enable_caching": true}'

# Batch processing with optimized concurrency
python mcode_cli.py matching batch --engine ensemble \
    --input pairs.ndjson \
    --output batch_results.ndjson \
    --max-concurrent 10 \
    --cache-enabled

# Memory-optimized processing
python mcode_cli.py matching run --engine ensemble \
    --input data.ndjson \
    --store-in-memory false \
    --workers 4 \
    --cache-ttl 3600
```

### Cache Performance Monitoring

```python
from src.matching.expert_panel_manager import ExpertPanelManager

panel = ExpertPanelManager(enable_caching=True)

# Perform multiple assessments to build cache
for patient, trial in patient_trial_pairs:
    await panel.assess_with_expert_panel(patient, trial)

# Get comprehensive cache performance
panel.log_cache_performance()

# Get optimization recommendations
recommendations = panel.get_cache_optimization_recommendations()
for rec in recommendations:
    print(f"💡 {rec}")

# Access detailed cache stats
cache_stats = panel._get_panel_cache_stats()
print(f"Panel Cache Stats:")
print(f"  Total Requests: {cache_stats['panel_total_requests']}")
print(f"  Hit Rate: {cache_stats['panel_hit_rate']:.2%}")
print(f"  Time Saved: {cache_stats['panel_total_time_saved_seconds']:.2f}s")

# Per-expert cache performance
for expert_type, stats in cache_stats['expert_cache_stats'].items():
    print(f"{expert_type}: {stats['hit_rate']:.2%} hit rate")
```

### Concurrent Workflow Integration

```python
import asyncio
from src.matching.ensemble_decision_engine import EnsembleDecisionEngine
from src.workflows.patients_processor import PatientsProcessor
from src.workflows.trials_processor import TrialsProcessor

async def concurrent_matching_workflow(patients_data, trials_data):
    """High-performance concurrent matching workflow."""

    # Initialize components
    ensemble = EnsembleDecisionEngine(max_experts=3)
    patients_processor = PatientsProcessor()
    trials_processor = TrialsProcessor()

    # Process patients and trials concurrently
    patients_task = patients_processor.execute(patients_data, workers=4)
    trials_task = trials_processor.execute(trials_data, workers=4, engine="llm")

    patients_result, trials_result = await asyncio.gather(
        patients_task, trials_task
    )

    # Generate patient-trial pairs
    pairs = []
    for patient in patients_result.processed_patients:
        for trial in trials_result.processed_trials:
            pairs.append({"patient": patient, "trial": trial})

    # Batch matching with ensemble engine
    batch_matcher = BatchMatcher(ensemble, max_concurrent=8)
    matching_results = await batch_matcher.match_batch(pairs)

    return {
        "patients_processed": len(patients_result.processed_patients),
        "trials_processed": len(trials_result.processed_trials),
        "matches_found": sum(1 for r in matching_results if r.is_match),
        "processing_time": patients_result.metadata["processing_time"] +
                          trials_result.metadata["processing_time"]
    }

# Execute concurrent workflow
results = await concurrent_matching_workflow(patients_data, trials_data)
print(f"Concurrent processing completed: {results}")
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