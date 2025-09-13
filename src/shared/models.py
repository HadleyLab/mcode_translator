"""
Pydantic models for standardized data structures in mCODE Translator.

This module provides type-safe, validated data models for clinical trial data,
mCODE mappings, and pipeline results. These models ensure data integrity
and provide clear interfaces for data flow throughout the system.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class IdentificationModule(BaseModel):
    """Clinical trial identification information."""
    nctId: str = Field(..., description="ClinicalTrials.gov identifier")
    briefTitle: Optional[str] = Field(None, description="Brief trial title")
    officialTitle: Optional[str] = Field(None, description="Official trial title")
    organization: Optional[Dict[str, Any]] = Field(None, description="Organization details")


class EligibilityModule(BaseModel):
    """Trial eligibility criteria and patient information."""
    eligibilityCriteria: Optional[str] = Field(None, description="Eligibility criteria text")
    healthyVolunteers: Optional[bool] = Field(None, description="Accepts healthy volunteers")
    sex: Optional[str] = Field(None, description="Patient sex requirements")
    minimumAge: Optional[str] = Field(None, description="Minimum age")
    maximumAge: Optional[str] = Field(None, description="Maximum age")
    stdAges: Optional[List[str]] = Field(None, description="Standardized age groups")


class Condition(BaseModel):
    """Medical condition information."""
    name: str = Field(..., description="Condition name")
    code: Optional[str] = Field(None, description="Condition code")
    codeSystem: Optional[str] = Field(None, description="Coding system")


class Intervention(BaseModel):
    """Trial intervention information."""
    name: str = Field(..., description="Intervention name")
    type: Optional[str] = Field(None, description="Intervention type")
    description: Optional[str] = Field(None, description="Intervention description")


class ProtocolSection(BaseModel):
    """Clinical trial protocol section."""
    identificationModule: IdentificationModule
    eligibilityModule: Optional[EligibilityModule] = None
    conditionsModule: Optional[Dict[str, List[Condition]]] = Field(None, description="Trial conditions")
    armsInterventionsModule: Optional[Dict[str, List[Intervention]]] = Field(None, description="Trial interventions")


class ClinicalTrialData(BaseModel):
    """Complete clinical trial data structure."""
    protocolSection: ProtocolSection
    hasResults: Optional[bool] = Field(False, description="Whether trial has results")
    studyType: Optional[str] = Field(None, description="Study type")
    overallStatus: Optional[str] = Field(None, description="Trial status")
    phase: Optional[str] = Field(None, description="Trial phase")

    @property
    def nct_id(self) -> str:
        """Get the NCT ID for convenience."""
        return self.protocolSection.identificationModule.nctId

    @property
    def brief_title(self) -> Optional[str]:
        """Get the brief title for convenience."""
        return self.protocolSection.identificationModule.briefTitle


class McodeElement(BaseModel):
    """Individual mCODE element mapping."""
    element_type: str = Field(..., description="Type of mCODE element (e.g., 'CancerCondition', 'CancerTreatment')")
    code: Optional[str] = Field(None, description="Element code")
    display: Optional[str] = Field(None, description="Human-readable display name")
    system: Optional[str] = Field(None, description="Coding system")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score for the mapping")
    evidence_text: Optional[str] = Field(None, description="Supporting evidence text")


class SourceReference(BaseModel):
    """Reference to source text for mCODE mapping."""
    document_type: str = Field(..., description="Type of source document")
    section_name: Optional[str] = Field(None, description="Document section name")
    text_snippet: Optional[str] = Field(None, description="Relevant text snippet")
    position: Optional[int] = Field(None, description="Position in document")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in this reference")


class ValidationResult(BaseModel):
    """Results of mCODE mapping validation."""
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Overall compliance score")
    validation_errors: List[str] = Field(default_factory=list, description="List of validation errors")
    validation_warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    required_elements_present: List[str] = Field(default_factory=list, description="Required mCODE elements found")
    missing_elements: List[str] = Field(default_factory=list, description="Required mCODE elements missing")


class TokenUsage(BaseModel):
    """Token usage statistics for LLM operations."""
    prompt_tokens: int = Field(0, description="Tokens used in prompt")
    completion_tokens: int = Field(0, description="Tokens used in completion")
    total_tokens: int = Field(0, description="Total tokens used")

    @model_validator(mode='before')
    @classmethod
    def calculate_total(cls, values):
        """Calculate total tokens if not provided."""
        if isinstance(values, dict):
            prompt = values.get('prompt_tokens', 0)
            completion = values.get('completion_tokens', 0)
            if 'total_tokens' not in values:
                values['total_tokens'] = prompt + completion
        return values


class ProcessingMetadata(BaseModel):
    """Metadata about processing operations."""
    engine_type: str = Field(..., description="Processing engine used")
    entities_count: int = Field(0, description="Number of entities extracted")
    mapped_count: int = Field(0, description="Number of elements mapped")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken for processing")
    model_used: Optional[str] = Field(None, description="LLM model used")
    prompt_used: Optional[str] = Field(None, description="Prompt template used")
    token_usage: Optional[TokenUsage] = Field(None, description="Token usage statistics")


class PipelineResult(BaseModel):
    """Standardized result from processing pipelines."""
    extracted_entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    mcode_mappings: List[McodeElement] = Field(default_factory=list, description="mCODE element mappings")
    source_references: List[SourceReference] = Field(default_factory=list, description="Source references")
    validation_results: ValidationResult = Field(..., description="Validation results")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    original_data: Dict[str, Any] = Field(..., description="Original input data")
    error: Optional[str] = Field(None, description="Error message if processing failed")

    @field_validator('validation_results', mode='before')
    @classmethod
    def ensure_validation_results(cls, v):
        """Ensure validation_results is always present."""
        if v is None:
            return ValidationResult(compliance_score=0.0)
        return v

    @field_validator('metadata', mode='before')
    @classmethod
    def ensure_metadata(cls, v):
        """Ensure metadata is always present."""
        if v is None:
            return ProcessingMetadata(engine_type="unknown")
        return v


class WorkflowResult(BaseModel):
    """Standardized result from workflow operations."""
    success: bool = Field(..., description="Whether the workflow succeeded")
    data: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict, description="Result data (dict or list)")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When processing occurred")


# FHIR Resource Models for Patient Data
class FHIRIdentifier(BaseModel):
    """FHIR identifier structure."""
    use: Optional[str] = Field(None, description="Identifier use")
    system: Optional[str] = Field(None, description="Identifier system")
    value: Optional[str] = Field(None, description="Identifier value")


class FHIRHumanName(BaseModel):
    """FHIR human name structure."""
    use: Optional[str] = Field(None, description="Name use")
    family: Optional[str] = Field(None, description="Family name")
    given: Optional[List[str]] = Field(None, description="Given names")
    prefix: Optional[List[str]] = Field(None, description="Name prefixes")
    suffix: Optional[List[str]] = Field(None, description="Name suffixes")


class FHIRContactPoint(BaseModel):
    """FHIR contact point structure."""
    system: Optional[str] = Field(None, description="Contact system")
    value: Optional[str] = Field(None, description="Contact value")
    use: Optional[str] = Field(None, description="Contact use")


class FHIRAddress(BaseModel):
    """FHIR address structure."""
    use: Optional[str] = Field(None, description="Address use")
    line: Optional[List[str]] = Field(None, description="Address lines")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State")
    postalCode: Optional[str] = Field(None, description="Postal code")
    country: Optional[str] = Field(None, description="Country")


class FHIRPatient(BaseModel):
    """FHIR Patient resource."""
    resourceType: str = Field(..., description="Resource type", pattern="^Patient$")
    id: Optional[str] = Field(None, description="Patient ID")
    identifier: Optional[List[FHIRIdentifier]] = Field(None, description="Patient identifiers")
    active: Optional[bool] = Field(None, description="Whether patient is active")
    name: Optional[List[FHIRHumanName]] = Field(None, description="Patient names")
    telecom: Optional[List[FHIRContactPoint]] = Field(None, description="Contact information")
    gender: Optional[str] = Field(None, description="Patient gender")
    birthDate: Optional[str] = Field(None, description="Birth date")
    deceasedBoolean: Optional[bool] = Field(None, description="Deceased status")
    deceasedDateTime: Optional[str] = Field(None, description="Deceased date/time")
    address: Optional[List[FHIRAddress]] = Field(None, description="Patient addresses")
    maritalStatus: Optional[Dict[str, Any]] = Field(None, description="Marital status")
    multipleBirthBoolean: Optional[bool] = Field(None, description="Multiple birth status")
    multipleBirthInteger: Optional[int] = Field(None, description="Multiple birth integer")
    communication: Optional[List[Dict[str, Any]]] = Field(None, description="Communication preferences")


class FHIREntry(BaseModel):
    """FHIR Bundle entry."""
    resource: Dict[str, Any] = Field(..., description="The resource in this entry")


class FHIRBundle(BaseModel):
    """FHIR Bundle resource for patient data."""
    resourceType: str = Field(..., description="Resource type", pattern="^Bundle$")
    id: Optional[str] = Field(None, description="Bundle ID")
    type: str = Field(..., description="Bundle type")
    total: Optional[int] = Field(None, description="Total number of resources")
    link: Optional[List[Dict[str, Any]]] = Field(None, description="Bundle links")
    entry: List[FHIREntry] = Field(..., description="Bundle entries")

    @property
    def patient_resources(self) -> List[FHIRPatient]:
        """Extract all Patient resources from the bundle."""
        patients = []
        for entry in self.entry:
            resource = entry.resource
            if resource.get("resourceType") == "Patient":
                patients.append(FHIRPatient(**resource))
        return patients

    @property
    def patient_id(self) -> Optional[str]:
        """Get the primary patient ID from the bundle."""
        patients = self.patient_resources
        if patients:
            patient = patients[0]
            # Try ID first
            if patient.id:
                return patient.id
            # Try identifier
            if patient.identifier:
                for identifier in patient.identifier:
                    if identifier.use == "usual" or identifier.system:
                        return identifier.value
        return None


class PatientData(BaseModel):
    """Standardized patient data structure."""
    bundle: FHIRBundle = Field(..., description="FHIR bundle containing patient data")
    source_file: Optional[str] = Field(None, description="Source file path")
    archive_name: Optional[str] = Field(None, description="Archive name")
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When data was processed")

    @property
    def patient_id(self) -> Optional[str]:
        """Convenience property for patient ID."""
        return self.bundle.patient_id

    @property
    def patient(self) -> Optional[FHIRPatient]:
        """Convenience property for primary patient resource."""
        patients = self.bundle.patient_resources
        return patients[0] if patients else None


class BenchmarkResult(BaseModel):
    """Result from benchmark operations."""
    task_id: str = Field(..., description="Unique task identifier")
    trial_id: str = Field(..., description="Clinical trial identifier")
    pipeline_result: PipelineResult = Field(..., description="Pipeline processing result")
    execution_time_seconds: float = Field(..., description="Time taken to execute")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    status: str = Field(..., description="Task execution status")

    # Performance metrics
    entities_extracted: Optional[int] = Field(None, description="Number of entities extracted")
    entities_mapped: Optional[int] = Field(None, description="Number of entities mapped")
    extraction_completeness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Extraction completeness score")
    mapping_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Mapping accuracy score")
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Precision metric")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recall metric")
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")
    compliance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Compliance score")

    # Additional metadata
    prompt_variant_id: Optional[str] = Field(None, description="Prompt variant identifier")
    api_config_name: Optional[str] = Field(None, description="API configuration name")
    test_case_id: Optional[str] = Field(None, description="Test case identifier")
    pipeline_type: Optional[str] = Field(None, description="Pipeline type used")
    start_time: Optional[datetime] = Field(None, description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")
    success: bool = Field(True, description="Whether execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# Utility functions for data conversion
def clinical_trial_from_dict(data: Dict[str, Any]) -> ClinicalTrialData:
    """Convert dictionary to ClinicalTrialData model with validation."""
    return ClinicalTrialData(**data)


def pipeline_result_from_dict(data: Dict[str, Any]) -> PipelineResult:
    """Convert dictionary to PipelineResult model with validation."""
    return PipelineResult(**data)


def workflow_result_from_dict(data: Dict[str, Any]) -> WorkflowResult:
    """Convert dictionary to WorkflowResult model with validation."""
    return WorkflowResult(**data)


def create_mcode_results_structure(pipeline_result) -> Dict[str, Any]:
    """
    Create standardized McodeResults structure from pipeline result.

    This function consolidates the duplicate logic for creating McodeResults
    structures that was previously scattered across multiple files.

    Args:
        pipeline_result: PipelineResult object with mCODE processing results

    Returns:
        Dict containing standardized McodeResults structure
    """
    return {
        "extracted_entities": pipeline_result.extracted_entities,
        "mcode_mappings": pipeline_result.mcode_mappings,
        "source_references": pipeline_result.source_references,
        "validation_results": pipeline_result.validation_results,
        "metadata": pipeline_result.metadata,
        "token_usage": pipeline_result.metadata.token_usage if pipeline_result.metadata else None,
        "error": pipeline_result.error,
    }


def enhance_trial_with_mcode_results(trial_data: Dict[str, Any], pipeline_result) -> Dict[str, Any]:
    """
    Enhance trial data with mCODE processing results.

    Args:
        trial_data: Original trial data dictionary
        pipeline_result: PipelineResult object with mCODE processing results

    Returns:
        Enhanced trial data with McodeResults structure
    """
    enhanced_trial = trial_data.copy()
    enhanced_trial["McodeResults"] = create_mcode_results_structure(pipeline_result)
    return enhanced_trial


def patient_data_from_dict(data: Dict[str, Any]) -> PatientData:
    """Convert dictionary to PatientData model with validation."""
    return PatientData(**data)


def benchmark_result_from_dict(data: Dict[str, Any]) -> BenchmarkResult:
    """Convert dictionary to BenchmarkResult model with validation."""
    return BenchmarkResult(**data)


def benchmark_result_from_dataclass(old_result) -> BenchmarkResult:
    """Convert old BenchmarkResult dataclass to new Pydantic model."""
    return BenchmarkResult(
        task_id=old_result.run_id,
        trial_id=old_result.test_case_id,
        pipeline_result=pipeline_result_from_dict({
            "extracted_entities": old_result.extracted_entities,
            "mcode_mappings": old_result.mcode_mappings,
            "validation_results": old_result.validation_results,
            "metadata": {
                "engine_type": "unknown",
                "model_used": getattr(old_result, 'api_config_name', None),
                "prompt_used": getattr(old_result, 'prompt_variant_id', None),
            },
            "original_data": {},
        }),
        execution_time_seconds=old_result.duration_ms / 1000 if old_result.duration_ms else 0,
        status="success" if old_result.success else "failed",
        entities_extracted=old_result.entities_extracted,
        entities_mapped=old_result.entities_mapped,
        extraction_completeness=old_result.extraction_completeness,
        mapping_accuracy=old_result.mapping_accuracy,
        precision=old_result.precision,
        recall=old_result.recall,
        f1_score=old_result.f1_score,
        compliance_score=old_result.compliance_score,
        prompt_variant_id=old_result.prompt_variant_id,
        api_config_name=old_result.api_config_name,
        test_case_id=old_result.test_case_id,
        pipeline_type=old_result.pipeline_type,
        start_time=old_result.start_time,
        end_time=old_result.end_time,
        duration_ms=old_result.duration_ms,
        success=old_result.success,
        error_message=old_result.error_message,
    )


def patient_bundle_from_dict(data: Dict[str, Any]) -> FHIRBundle:
    """Convert dictionary to FHIRBundle model with validation."""
    return FHIRBundle(**data)


def validate_clinical_trial_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate clinical trial data structure.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        clinical_trial_from_dict(data)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_patient_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate patient data structure.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        patient_data_from_dict(data)
        return True, None
    except Exception as e:
        return False, str(e)