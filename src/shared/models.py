"""
Pydantic models for standardized data structures in mCODE Translator.

This module provides type-safe, validated data models for clinical trial data,
mCODE mappings, and pipeline results. These models ensure data integrity
and provide clear interfaces for data flow throughout the system.
"""

"""
Simplified models.py - consolidated to use mcode_models.py as single source of truth.

This module now serves as a clean import interface to mcode_models.py,
eliminating duplicate model definitions and reducing codebase complexity.
"""

# Import and re-export essential models from mcode_models for backward compatibility
from .mcode_models import (
    CancerCondition,
    McodePatient,
    TumorMarkerTest,
    ECOGPerformanceStatusObservation,
    CancerRelatedMedicationStatement,
    CancerRelatedSurgicalProcedure,
    CancerRelatedRadiationProcedure,
    TNMStageGroup,
    CancerStaging,
    McodeElement,
    McodeValidator,
    AdministrativeGender,
    BirthSex,
    CancerConditionCode,
    TNMStageGroupEnum,
    ECOGPerformanceStatus,
    ReceptorStatus,
    HistologyMorphologyBehavior,
    FHIRIdentifier,
    FHIRCodeableConcept,
    FHIRReference,
    FHIRQuantity,
    FHIRRange,
    FHIRRatio,
    FHIRPeriod,
    FHIRHumanName,
    FHIRContactPoint,
    FHIRAddress,
    FHIRPatient,
    FHIRCondition,
    FHIRObservation,
    FHIRProcedure,
    FHIRMedicationStatement,
    McodeExtension,
    BirthSexExtension,
    USCoreRaceExtension,
    USCoreEthnicityExtension,
    HistologyMorphologyBehaviorExtension,
    LateralityExtension,
    RelatedConditionExtension,
    ConditionRelatedExtension,
    VersionedMcodeResource,
    create_mcode_patient,
    create_cancer_condition,
)

# Additional models for LLM processing, pipeline results, and workflow management

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ParsedLLMResponse(BaseModel):
    """Response from LLM with validation tracking."""
    raw_content: str = Field(..., description="Raw LLM response content")
    parsed_json: Optional[Dict[str, Any]] = Field(None, description="Parsed JSON response")
    is_valid_json: bool = Field(False, description="Whether response is valid JSON")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    cleaned_content: Optional[str] = Field(None, description="Cleaned response content")


class McodeMappingResponse(BaseModel):
    """Response from mCODE mapping operation."""
    mcode_elements: List["McodeElement"] = Field(default_factory=list, description="Mapped mCODE elements")
    raw_response: ParsedLLMResponse = Field(..., description="Raw LLM response")
    processing_metadata: "ProcessingMetadata" = Field(..., description="Processing metadata")
    success: bool = Field(True, description="Whether mapping succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class PatientTrialMatchResponse(BaseModel):
    """Response from patient-trial matching operation."""
    is_match: bool = Field(..., description="Whether patient matches trial criteria")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in match decision")
    reasoning: str = Field(..., description="Clinical reasoning for match decision")
    matched_criteria: List[str] = Field(default_factory=list, description="Criteria that were matched")
    unmatched_criteria: List[str] = Field(default_factory=list, description="Criteria that were not matched")
    clinical_notes: str = Field(default="", description="Additional clinical notes")
    matched_elements: List[Dict[str, Any]] = Field(default_factory=list, description="Matched mCODE elements")
    raw_response: ParsedLLMResponse = Field(..., description="Raw LLM response")
    processing_metadata: "ProcessingMetadata" = Field(..., description="Processing metadata")
    success: bool = Field(True, description="Whether matching succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ProcessingMetadata(BaseModel):
    """Metadata about processing operations."""
    engine_type: str = Field(..., description="Type of processing engine used")
    entities_count: int = Field(0, description="Number of entities processed")
    mapped_count: int = Field(0, description="Number of entities successfully mapped")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    model_used: Optional[str] = Field(None, description="LLM model used")
    prompt_used: Optional[str] = Field(None, description="Prompt template used")




class WorkflowResult(BaseModel):
    """Result from workflow execution."""
    success: bool = Field(..., description="Whether workflow succeeded")
    data: Dict[str, Any] = Field(default_factory=dict, description="Workflow result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[Union[Dict[str, Any], ProcessingMetadata]] = Field(None, description="Processing metadata")


class PipelineResult(BaseModel):
    """Result from pipeline processing."""
    mcode_elements: List[McodeElement] = Field(default_factory=list, description="Extracted mCODE elements")
    validation_results: "ValidationResult" = Field(..., description="Validation results")
    metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    original_data: Dict[str, Any] = Field(default_factory=dict, description="Original input data")


class ValidationResult(BaseModel):
    """Result from validation operations."""
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Compliance score")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    required_elements_present: Optional[List[str]] = Field(None, description="Required elements present")
    missing_elements: Optional[List[str]] = Field(None, description="Missing elements")




class IdentificationModule(BaseModel):
    """Module for identifying mCODE elements."""
    module_name: str = Field(..., description="Module name")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Confidence threshold")
    supported_element_types: List[str] = Field(default_factory=list, description="Supported element types")


class StatusModule(BaseModel):
    """Module for trial status information."""
    overall_status: Optional[str] = Field(None, description="Overall trial status")
    status_verified_date: Optional[str] = Field(None, description="Date status was verified")
    why_stopped: Optional[str] = Field(None, description="Reason trial was stopped")
    start_date: Optional[str] = Field(None, description="Trial start date")
    completion_date: Optional[str] = Field(None, description="Trial completion date")
    primary_completion_date: Optional[str] = Field(None, description="Primary completion date")


class EligibilityModule(BaseModel):
    """Module for trial eligibility criteria."""
    eligibility_criteria: Optional[str] = Field(None, description="Detailed eligibility criteria text")
    minimum_age: Optional[str] = Field(None, description="Minimum age requirement")
    maximum_age: Optional[str] = Field(None, description="Maximum age requirement")
    sex: Optional[str] = Field(None, description="Sex eligibility requirement")
    accepts_healthy_volunteers: Optional[bool] = Field(None, description="Whether healthy volunteers are accepted")
    inclusion_criteria: List[str] = Field(default_factory=list, description="Inclusion criteria")
    exclusion_criteria: List[str] = Field(default_factory=list, description="Exclusion criteria")


class ProtocolSection(BaseModel):
    """Protocol section of a clinical trial."""
    identification_module: Optional[IdentificationModule] = Field(None, description="Trial identification information")
    status_module: Optional[StatusModule] = Field(None, description="Trial status information")
    eligibility_module: Optional[EligibilityModule] = Field(None, description="Trial eligibility criteria")


class ClinicalTrialData(BaseModel):
    """Clinical trial data structure."""
    trial_id: str = Field(..., description="Trial identifier")
    title: str = Field(..., description="Trial title")
    eligibility_criteria: Optional[str] = Field(None, description="Eligibility criteria text")
    conditions: List[str] = Field(default_factory=list, description="Medical conditions")
    interventions: List[str] = Field(default_factory=list, description="Trial interventions")
    phase: Optional[str] = Field(None, description="Trial phase")
    protocol_section: Optional[ProtocolSection] = Field(None, description="Protocol section with detailed trial information")
    has_results: Optional[bool] = Field(None, description="Whether trial has results")
    study_type: Optional[str] = Field(None, description="Study type")
    overall_status: Optional[str] = Field(None, description="Overall status")


class PatientData(BaseModel):
    """Patient data structure."""
    patient_id: str = Field(..., description="Patient identifier")
    bundle: Dict[str, Any] = Field(..., description="FHIR bundle containing patient data")
    source_file: Optional[str] = Field(None, description="Source file name")
    archive_name: Optional[str] = Field(None, description="Archive name")
    conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Patient conditions")
    demographics: Dict[str, Any] = Field(default_factory=dict, description="Patient demographics")
    observations: List[Dict[str, Any]] = Field(default_factory=list, description="Patient observations")


class TokenUsage(BaseModel):
    """Token usage tracking."""
    prompt_tokens: int = Field(0, description="Tokens used in prompt")
    completion_tokens: int = Field(0, description="Tokens used in completion")
    total_tokens: int = Field(0, description="Total tokens used")

    def __init__(self, **data):
        super().__init__(**data)
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            object.__setattr__(self, 'total_tokens', self.prompt_tokens + self.completion_tokens)


class BenchmarkResult(BaseModel):
    """Result from benchmarking operations."""
    benchmark_name: str = Field(..., description="Benchmark name")
    score: float = Field(..., description="Benchmark score")
    task_id: Optional[str] = Field(None, description="Task identifier")
    trial_id: Optional[str] = Field(None, description="Trial identifier")
    pipeline_result: Optional[PipelineResult] = Field(None, description="Pipeline result")
    execution_time_seconds: Optional[float] = Field(None, description="Execution time")
    status: Optional[str] = Field(None, description="Execution status")
    precision: Optional[float] = Field(None, description="Precision metric")
    recall: Optional[float] = Field(None, description="Recall metric")
    f1_score: Optional[float] = Field(None, description="F1 score metric")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Benchmark metrics")


class SearchResult(BaseModel):
    """Result from search operations."""
    query: str = Field(..., description="Search query")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results")
    total_count: int = Field(0, description="Total number of results")


class LLMRequest(BaseModel):
    """LLM API request structure."""
    model: str = Field(..., description="Model name")
    prompt: str = Field(..., description="Prompt text")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature parameter")
    max_tokens: int = Field(1000, gt=0, description="Maximum tokens")


class LLMResponse(BaseModel):
    """LLM API response structure."""
    content: str = Field(..., description="Response content")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage")
    finish_reason: Optional[str] = Field(None, description="Finish reason")


class LLMAPIError(BaseModel):
    """LLM API error structure."""
    error_type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class SourceReference(BaseModel):
    """Reference to source material."""
    source_type: str = Field(..., description="Type of source")
    source_id: str = Field(..., description="Source identifier")
    page_number: Optional[int] = Field(None, description="Page number")
    section: Optional[str] = Field(None, description="Section reference")


# All models now imported from mcode_models.py and defined above

# Re-export all imported models for backward compatibility
__all__ = [
    # mCODE models
    "CancerCondition",
    "McodePatient",
    "TumorMarkerTest",
    "ECOGPerformanceStatusObservation",
    "CancerRelatedMedicationStatement",
    "CancerRelatedSurgicalProcedure",
    "CancerRelatedRadiationProcedure",
    "TNMStageGroup",
    "CancerStaging",
    "McodeElement",
    "McodeValidator",
    "AdministrativeGender",
    "BirthSex",
    "CancerConditionCode",
    "TNMStageGroupEnum",
    "ECOGPerformanceStatus",
    "ReceptorStatus",
    "HistologyMorphologyBehavior",
    "FHIRIdentifier",
    "FHIRCodeableConcept",
    "FHIRReference",
    "FHIRQuantity",
    "FHIRRange",
    "FHIRRatio",
    "FHIRPeriod",
    "FHIRHumanName",
    "FHIRContactPoint",
    "FHIRAddress",
    "FHIRPatient",
    "FHIRCondition",
    "FHIRObservation",
    "FHIRProcedure",
    "FHIRMedicationStatement",
    "McodeExtension",
    "BirthSexExtension",
    "USCoreRaceExtension",
    "USCoreEthnicityExtension",
    "HistologyMorphologyBehaviorExtension",
    "LateralityExtension",
    "RelatedConditionExtension",
    "ConditionRelatedExtension",
    "VersionedMcodeResource",
    "create_mcode_patient",
    "create_cancer_condition",

    # Additional models defined in this file
    "ParsedLLMResponse",
    "McodeMappingResponse",
    "PatientTrialMatchResponse",
    "ProcessingMetadata",
    "WorkflowResult",
    "PipelineResult",
    "ValidationResult",
    "ClinicalTrialData",
    "PatientData",
    "IdentificationModule",
    "StatusModule",
    "EligibilityModule",
    "ProtocolSection",
    "TokenUsage",
    "BenchmarkResult",
    "SearchResult",
    "LLMRequest",
    "LLMResponse",
    "LLMAPIError",
    "SourceReference",
]

