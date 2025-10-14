# API Reference

## mCODE Ontology API Reference

This section documents the mCODE (Minimal Common Oncology Data Elements) ontology implementation, providing standardized oncology data structures and validation.

### McodeVersion

Enumeration of supported mCODE STU versions.

```python
class McodeVersion(Enum):
    STU1 = "1.0.0"
    STU2 = "2.0.0"
    STU3 = "3.0.0"
    STU4 = "4.0.0"  # Current version
```

#### Methods

##### latest() -> McodeVersion
Get the latest supported mCODE version.

##### from_string(version_str: str) -> McodeVersion
Parse version string into McodeVersion enum.

**Parameters:**
- `version_str`: Version string (e.g., "4.0.0", "STU4")

### McodeProfile

Enumeration of all supported mCODE profiles organized by category.

```python
class McodeProfile(Enum):
    # Patient Information
    PATIENT = "Patient"
    BIRTH_SEX = "BirthSex"
    ADMINISTRATIVE_GENDER = "AdministrativeGender"
    US_CORE_RACE = "USCoreRaceExtension"
    US_CORE_ETHNICITY = "USCoreEthnicityExtension"

    # Disease Characterization
    CANCER_CONDITION = "CancerCondition"
    CANCER_STAGING = "CancerStaging"
    TNM_STAGE_GROUP = "TNMStageGroup"
    HISTOLOGY_MORPHOLOGY_BEHAVIOR = "HistologyMorphologyBehavior"

    # Assessment
    TUMOR_MARKER_TEST = "TumorMarkerTest"
    ECOG_PERFORMANCE_STATUS = "ECOGPerformanceStatus"

    # Treatments
    CANCER_RELATED_MEDICATION_STATEMENT = "CancerRelatedMedicationStatement"
    CANCER_RELATED_SURGICAL_PROCEDURE = "CancerRelatedSurgicalProcedure"
    CANCER_RELATED_RADIATION_PROCEDURE = "CancerRelatedRadiationProcedure"
```

### McodeVersionManager

Central manager for mCODE version operations and profile URL generation.

#### Constructor

```python
McodeVersionManager(default_version: Optional[McodeVersion] = None)
```

#### Methods

##### get_profile_url(profile: McodeProfile, version: Optional[McodeVersion] = None) -> str
Generate canonical URL for an mCODE profile.

**Parameters:**
- `profile`: mCODE profile to generate URL for
- `version`: Version to use (defaults to manager's default)

**Returns:** Canonical profile URL

##### check_compatibility(source_version: McodeVersion, target_version: McodeVersion) -> VersionCompatibility
Check compatibility between two mCODE versions.

##### get_supported_versions() -> List[McodeVersion]
Get list of all supported mCODE versions.

##### validate_version_string(version_str: str) -> bool
Validate if a version string represents a supported mCODE version.

### McodeValidator

Comprehensive validator for mCODE resources and business rules.

#### Constructor

```python
McodeValidator(version: Optional[McodeVersion] = None)
```

#### Methods

##### validate_patient_eligibility(patient: McodePatient, criteria: Dict[str, Any]) -> Dict[str, Any]
Validate patient eligibility against clinical trial criteria.

**Parameters:**
- `patient`: mCODE patient instance
- `criteria`: Eligibility criteria dictionary

**Returns:** Validation results with eligibility status and criteria analysis

##### validate_cancer_condition(condition: CancerCondition) -> Dict[str, Any]
Validate cancer condition data quality.

**Returns:** Validation results with quality score and issues

##### validate_resource_compatibility(resource: BaseModel, profile: McodeProfile) -> bool
Validate resource compatibility with mCODE version.

### mCODE Profile Models

#### McodePatient

mCODE Patient profile extending FHIR Patient with oncology requirements.

```python
class McodePatient(FHIRPatient):
    extension: Optional[List[Union[BirthSexExtension, USCoreRaceExtension, USCoreEthnicityExtension]]] = None
```

**Required Fields:**
- `resourceType`: "Patient"
- `gender`: Administrative gender

#### CancerCondition

mCODE Cancer Condition profile for oncology diagnoses.

```python
class CancerCondition(FHIRCondition):
    extension: Optional[List[Union[HistologyMorphologyBehaviorExtension, LateralityExtension, RelatedConditionExtension]]] = None
```

**Required Fields:**
- `resourceType`: "Condition"
- `subject`: Patient reference
- `category`: Must include "problem-list-item"
- `code`: Cancer diagnosis code

#### TNMStageGroup

mCODE TNM Stage Group observation.

```python
class TNMStageGroup(CancerStaging):
    # Inherits from CancerStaging with TNM-specific validation
    pass
```

#### TumorMarkerTest

mCODE Tumor Marker Test observation.

```python
class TumorMarkerTest(FHIRObservation):
    extension: Optional[List[ConditionRelatedExtension]] = None
    category: List[FHIRCodeableConcept]  # Must include "laboratory"
```

#### ECOGPerformanceStatusObservation

mCODE ECOG Performance Status observation.

```python
class ECOGPerformanceStatusObservation(FHIRObservation):
    valueCodeableConcept: FHIRCodeableConcept  # Must be from ECOG value set
```

#### CancerRelatedMedicationStatement

mCODE Cancer Related Medication Statement.

```python
class CancerRelatedMedicationStatement(FHIRMedicationStatement):
    category: FHIRCodeableConcept  # Must indicate cancer-related
```

#### CancerRelatedSurgicalProcedure

mCODE Cancer Related Surgical Procedure.

```python
class CancerRelatedSurgicalProcedure(FHIRProcedure):
    category: FHIRCodeableConcept  # Must indicate surgical procedure
```

#### CancerRelatedRadiationProcedure

mCODE Cancer Related Radiation Procedure.

```python
class CancerRelatedRadiationProcedure(FHIRProcedure):
    category: FHIRCodeableConcept  # Must indicate radiation procedure
```

### Utility Functions

#### create_mcode_patient()

Create a properly structured mCODE Patient resource.

```python
def create_mcode_patient(
    id: str,
    name: List[Dict[str, Any]],
    gender: AdministrativeGender,
    birth_date: str,
    birth_sex: Optional[BirthSex] = None,
    race: Optional[FHIRCodeableConcept] = None,
    ethnicity: Optional[FHIRCodeableConcept] = None
) -> McodePatient
```

#### create_cancer_condition()

Create a properly structured mCODE Cancer Condition.

```python
def create_cancer_condition(
    patient_id: str,
    cancer_code: CancerConditionCode,
    clinical_status: str = "active",
    histology_behavior: Optional[HistologyMorphologyBehavior] = None,
    laterality: Optional[FHIRCodeableConcept] = None
) -> CancerCondition
```

### McodeOntologyValidator

Validator for mCODE ontology completeness against sample data.

#### Constructor

```python
McodeOntologyValidator()
```

#### Methods

##### validate_ontology_completeness(patient_file: Path, trial_file: Path) -> CompletenessReport
Main method to validate ontology completeness against sample data.

**Parameters:**
- `patient_file`: Path to patient data file (NDJSON)
- `trial_file`: Path to trial data file (NDJSON)

**Returns:** Comprehensive completeness report

##### validate_patient_resource(resource: Dict[str, Any]) -> ValidationResult
Validate a patient resource against mCODE models.

##### validate_condition_resource(resource: Dict[str, Any]) -> ValidationResult
Validate a condition resource against mCODE CancerCondition model.

##### validate_trial_data(trial_data: Dict[str, Any]) -> ValidationResult
Validate trial data against mCODE trial elements.

##### print_report(report: CompletenessReport)
Print the completeness report in a readable format.

## Expert Multi-LLM Curator API Reference

This section documents the core classes for the Expert Multi-LLM Curator system, which provides advanced ensemble decision making for clinical trial matching using multiple specialized LLM experts.

### EnsembleDecisionEngine

Advanced ensemble decision engine that combines multiple expert opinions with sophisticated weighting and consensus mechanisms.

#### Constructor

```python
EnsembleDecisionEngine(
    model_name: str = "deepseek-coder",
    config: Optional[Config] = None,
    consensus_method: ConsensusMethod = ConsensusMethod.DYNAMIC_WEIGHTING,
    confidence_calibration: ConfidenceCalibration = ConfidenceCalibration.ISOTONIC_REGRESSION,
    enable_rule_based_integration: bool = True,
    enable_dynamic_weighting: bool = True,
    min_experts: int = 2,
    max_experts: int = 3,
    cache_enabled: bool = True,
    max_retries: int = 3
)
```

**Parameters:**
- `model_name`: LLM model to use for expert assessments (default: "deepseek-coder")
- `config`: Application configuration instance
- `consensus_method`: Method for combining expert opinions (WEIGHTED_MAJORITY_VOTE, CONFIDENCE_WEIGHTED, BAYESIAN_ENSEMBLE, DYNAMIC_WEIGHTING)
- `confidence_calibration`: Method for calibrating confidence scores (ISOTONIC_REGRESSION, PLATT_SCALING, HISTOGRAM_BINNING, NONE)
- `enable_rule_based_integration`: Whether to integrate rule-based scoring
- `enable_dynamic_weighting`: Whether to use dynamic expert weighting
- `min_experts`: Minimum number of experts to use
- `max_experts`: Maximum number of experts to use
- `cache_enabled`: Whether to enable caching
- `max_retries`: Maximum number of retries on failure

#### Methods

##### match(patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]) -> bool

Match patient data against trial criteria using ensemble decision making.

**Parameters:**
- `patient_data`: Patient information dictionary
- `trial_criteria`: Trial eligibility criteria dictionary

**Returns:** Boolean match result from ensemble decision

**Example:**
```python
engine = EnsembleDecisionEngine()
is_match = await engine.match(
    patient_data={"age": 45, "cancer_type": "breast_cancer", "stage": "II"},
    trial_criteria={"conditions": ["breast cancer"], "minimumAge": "18", "maximumAge": "65"}
)
```

##### update_expert_weights(performance_data: Dict[str, Dict[str, float]])

Update expert weights based on performance data.

**Parameters:**
- `performance_data`: Dictionary mapping expert types to performance metrics

**Example:**
```python
performance_data = {
    "clinical_reasoning": {"accuracy": 0.85, "reliability": 0.90},
    "pattern_recognition": {"accuracy": 0.78, "reliability": 0.82}
}
engine.update_expert_weights(performance_data)
```

##### get_ensemble_status() -> Dict[str, Any]

Get status of the ensemble decision engine.

**Returns:** Dictionary with ensemble configuration and status information

##### shutdown()

Shutdown the ensemble decision engine and cleanup resources.

---

### ExpertPanelManager

Manages a panel of specialized LLM experts for comprehensive clinical trial matching.

#### Constructor

```python
ExpertPanelManager(
    model_name: str = "deepseek-coder",
    config: Optional[Config] = None,
    max_concurrent_experts: int = 3,
    enable_diversity_selection: bool = True
)
```

**Parameters:**
- `model_name`: LLM model to use for all experts
- `config`: Configuration instance
- `max_concurrent_experts`: Maximum number of experts to run concurrently
- `enable_diversity_selection`: Whether to use diversity-aware expert selection

#### Methods

##### assess_with_expert_panel(patient_data: Dict[str, Any], trial_criteria: Dict[str, Any], expert_selection: Optional[List[str]] = None, diversity_threshold: float = 0.7) -> Dict[str, Any]

Assess patient-trial match using expert panel with caching.

**Parameters:**
- `patient_data`: Patient information dictionary
- `trial_criteria`: Trial eligibility criteria dictionary
- `expert_selection`: Specific experts to use (if None, uses diversity-aware selection)
- `diversity_threshold`: Threshold for diversity in expert selection

**Returns:** Comprehensive assessment with ensemble decision

**Example:**
```python
panel = ExpertPanelManager()
result = await panel.assess_with_expert_panel(
    patient_data={"age": 55, "cancer_type": "lung_cancer"},
    trial_criteria={"conditions": ["lung cancer"], "stages": ["III", "IV"]}
)
print(f"Match: {result['is_match']}, Confidence: {result['confidence_score']}")
```

##### get_expert_panel_status() -> Dict[str, Any]

Get status of all experts in the panel.

**Returns:** Status information for all experts

##### log_cache_performance()

Log comprehensive cache performance statistics.

##### get_cache_optimization_recommendations() -> List[str]

Get recommendations for cache optimization based on performance data.

**Returns:** List of optimization recommendations

##### shutdown()

Shutdown the expert panel manager and cleanup resources.

---

### ClinicalExpertAgent

Specialized clinical expert agent for patient-trial matching with different clinical reasoning styles.

#### Constructor

```python
ClinicalExpertAgent(
    model_name: str = "deepseek-coder",
    expert_type: str = "clinical_reasoning",
    config: Optional[Config] = None
)
```

**Parameters:**
- `model_name`: LLM model to use for analysis
- `expert_type`: Type of clinical expertise ("clinical_reasoning", "pattern_recognition", "comprehensive_analyst")
- `config`: Configuration instance

#### Methods

##### assess_match(patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]) -> Dict[str, Any]

Assess patient-trial match using specialized clinical expertise with caching.

**Parameters:**
- `patient_data`: Patient information dictionary
- `trial_criteria`: Trial eligibility criteria dictionary

**Returns:** Detailed match assessment with clinical rationale

**Example:**
```python
expert = ClinicalExpertAgent(expert_type="clinical_reasoning")
assessment = await expert.assess_match(
    patient_data={"age": 62, "cancer_type": "colon_cancer", "stage": "III"},
    trial_criteria={"conditions": ["colon cancer"], "minimumAge": "50"}
)
print(f"Match: {assessment['is_match']}, Reasoning: {assessment['reasoning']}")
```

---

### LLMMatchingEngine

A matching engine that uses an LLM to find matches between patient and trial data, enhanced with expert panel support.

#### Constructor

```python
LLMMatchingEngine(
    model_name: str,
    prompt_name: str,
    cache_enabled: bool = True,
    max_retries: int = 3,
    enable_expert_panel: bool = False,
    expert_panel_config: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `model_name`: The name of the LLM model to use
- `prompt_name`: The name of the prompt to use for matching
- `cache_enabled`: Whether to enable caching for API calls
- `max_retries`: Maximum number of retries on failure
- `enable_expert_panel`: Whether to use expert panel for ensemble decisions
- `expert_panel_config`: Configuration for expert panel (if enabled)

#### Methods

##### match(patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]) -> bool

Matches patient data against trial criteria using an LLM.

**Parameters:**
- `patient_data`: The patient's data
- `trial_criteria`: The trial's eligibility criteria

**Returns:** Boolean indicating if a match was found

**Example:**
```python
engine = LLMMatchingEngine(
    model_name="deepseek-coder",
    prompt_name="clinical_matching",
    enable_expert_panel=True
)
is_match = await engine.match(
    patient_data={"age": 45, "diagnosis": "breast cancer"},
    trial_criteria={"conditions": ["breast cancer"], "age_range": "18-65"}
)
```

##### get_detailed_assessment(patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]) -> Dict[str, Any]

Get detailed assessment with reasoning and confidence scores.

**Parameters:**
- `patient_data`: Patient data dictionary
- `trial_criteria`: Trial criteria dictionary

**Returns:** Detailed assessment dictionary with reasoning and confidence

**Example:**
```python
assessment = await engine.get_detailed_assessment(
    patient_data={"age": 50, "cancer_type": "lung_cancer"},
    trial_criteria={"conditions": ["lung cancer"], "stages": ["II", "III"]}
)
print(f"Confidence: {assessment['confidence_score']}")
print(f"Reasoning: {assessment['reasoning']}")
```

##### get_expert_panel_status() -> Dict[str, Any]

Get status of the expert panel if enabled.

**Returns:** Expert panel status information

##### shutdown()

Shutdown the LLM engine and cleanup resources.

---

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