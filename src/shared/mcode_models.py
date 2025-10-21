"""
Comprehensive Pydantic Models for mCODE Ontology

This module provides complete Pydantic models for the mCODE (Minimal Common Oncology Data Elements)
ontology with validation, type safety, and integration with the versioning system.

Models include:
- Base FHIR resource models (Patient, Condition, Observation, etc.)
- mCODE profile models (CancerCondition, CancerStageGroup, TumorMarkerTest, etc.)
- Extension models for mCODE-specific fields
- Value set enums for controlled terminologies
- Validation rules and business logic
- Integration with versioning system

All models follow mCODE STU4 (4.0.0) specification and provide comprehensive validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from .mcode_versioning import McodeProfile, McodeVersion, get_version_manager

# =============================================================================
# VALUE SET ENUMS - Controlled Terminologies
# =============================================================================

class AdministrativeGender(str, Enum):
    """Administrative gender value set."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class BirthSex(str, Enum):
    """Birth sex value set."""
    M = "M"
    F = "F"
    UNK = "UNK"


class CancerConditionCode(str, Enum):
    """Cancer condition codes from SNOMED CT."""
    BREAST_CANCER = "254837009"
    LUNG_CANCER = "363358000"
    COLORECTAL_CANCER = "363406005"
    PROSTATE_CANCER = "399068003"
    MELANOMA = "372244006"
    LEUKEMIA = "93143009"
    LYMPHOMA = "118600007"
    OVARIAN_CANCER = "363443007"
    PANCREATIC_CANCER = "363418001"
    LIVER_CANCER = "363349007"


class TNMStageGroupEnum(str, Enum):
    """TNM stage group value set."""
    STAGE_0 = "0"
    STAGE_I = "I"
    STAGE_II = "II"
    STAGE_III = "III"
    STAGE_IV = "IV"
    STAGE_IA = "IA"
    STAGE_IB = "IB"
    STAGE_IIA = "IIA"
    STAGE_IIB = "IIB"
    STAGE_IIIA = "IIIA"
    STAGE_IIIB = "IIIB"
    STAGE_IIIC = "IIIC"
    STAGE_IVA = "IVA"
    STAGE_IVB = "IVB"


class ECOGPerformanceStatus(str, Enum):
    """ECOG performance status value set."""
    FULLY_ACTIVE = "0"
    RESTRICTED_ACTIVITY = "1"
    UNABLE_HEAVY_WORK = "2"
    CAPABLE_ONLY_SELFCARE = "3"
    COMPLETELY_DISABLED = "4"
    DEAD = "5"


class ReceptorStatus(str, Enum):
    """Receptor status value set."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"
    NOT_PERFORMED = "not-performed"


class HistologyMorphologyBehavior(str, Enum):
    """Histology morphology behavior value set."""
    MALIGNANT = "3"
    BENIGN = "2"
    UNCERTAIN = "1"


# =============================================================================
# BASE FHIR RESOURCE MODELS
# =============================================================================

class FHIRIdentifier(BaseModel):
    """FHIR Identifier resource."""
    use: Optional[str] = Field(None, description="usual | official | temp | secondary | old")
    system: Optional[str] = Field(None, description="The namespace for the identifier value")
    value: str = Field(..., description="The value that is unique")
    period: Optional[Dict[str, Any]] = Field(None, description="Time period when id is/was valid")
    assigner: Optional[Dict[str, Any]] = Field(None, description="Organization that issued id")


class FHIRCodeableConcept(BaseModel):
    """FHIR CodeableConcept resource."""
    coding: List[Dict[str, Any]] = Field(default_factory=list, description="Code defined by a terminology system")
    text: Optional[str] = Field(None, description="Plain text representation of the concept")

    @field_validator('coding')
    @classmethod
    def validate_coding(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate coding entries have required fields."""
        for coding in v:
            if not isinstance(coding, dict):
                raise ValueError("Each coding must be a dictionary")
            if 'system' not in coding or 'code' not in coding:
                raise ValueError("Coding must have 'system' and 'code' fields")
        return v


class FHIRReference(BaseModel):
    """FHIR Reference resource."""
    reference: Optional[str] = Field(None, description="Literal reference, Relative, internal or absolute URL")
    type: Optional[str] = Field(None, description="Type the reference refers to")
    identifier: Optional[FHIRIdentifier] = Field(None, description="Logical reference, when literal reference is not known")
    display: Optional[str] = Field(None, description="Text alternative for the resource")


class FHIRQuantity(BaseModel):
    """FHIR Quantity resource."""
    value: Optional[float] = Field(None, description="Numerical value")
    comparator: Optional[str] = Field(None, description="< | <= | >= | > - how to understand the value")
    unit: Optional[str] = Field(None, description="Unit representation")
    system: Optional[str] = Field(None, description="System that defines coded unit form")
    code: Optional[str] = Field(None, description="Coded form of the unit")


class FHIRRange(BaseModel):
    """FHIR Range resource."""
    low: Optional[FHIRQuantity] = Field(None, description="Low limit")
    high: Optional[FHIRQuantity] = Field(None, description="High limit")


class FHIRRatio(BaseModel):
    """FHIR Ratio resource."""
    numerator: Optional[FHIRQuantity] = Field(None, description="Numerator value")
    denominator: Optional[FHIRQuantity] = Field(None, description="Denominator value")


class FHIRPeriod(BaseModel):
    """FHIR Period resource."""
    start: Optional[str] = Field(None, description="Starting time with inclusive boundary")
    end: Optional[str] = Field(None, description="End time with inclusive boundary")


class FHIRHumanName(BaseModel):
    """FHIR HumanName resource."""
    use: Optional[str] = Field(None, description="usual | official | temp | nickname | anonymous | old | maiden")
    text: Optional[str] = Field(None, description="Text representation of the full name")
    family: Optional[str] = Field(None, description="Family name (often called 'Surname')")
    given: Optional[List[str]] = Field(None, description="Given names (not always 'first'). Includes middle names")
    prefix: Optional[List[str]] = Field(None, description="Parts that come before the name")
    suffix: Optional[List[str]] = Field(None, description="Parts that come after the name")
    period: Optional[FHIRPeriod] = Field(None, description="Time period when name was/is in use")


class FHIRContactPoint(BaseModel):
    """FHIR ContactPoint resource."""
    system: Optional[str] = Field(None, description="phone | fax | email | pager | url | sms | other")
    value: Optional[str] = Field(None, description="The actual contact point details")
    use: Optional[str] = Field(None, description="home | work | temp | old | mobile")
    rank: Optional[int] = Field(None, description="Specify preferred order of use (1 = highest)")
    period: Optional[FHIRPeriod] = Field(None, description="Time period when the contact point was/is in use")


class FHIRAddress(BaseModel):
    """FHIR Address resource."""
    use: Optional[str] = Field(None, description="home | work | temp | old | billing")
    type: Optional[str] = Field(None, description="postal | physical | both")
    text: Optional[str] = Field(None, description="Text representation of the address")
    line: Optional[List[str]] = Field(None, description="Street name, number, direction & P.O. Box etc.")
    city: Optional[str] = Field(None, description="Name of city, town etc.")
    district: Optional[str] = Field(None, description="District name (aka county)")
    state: Optional[str] = Field(None, description="Sub-unit of country (abbreviations ok)")
    postalCode: Optional[str] = Field(None, description="Postal code for area")
    country: Optional[str] = Field(None, description="Country (e.g. can be ISO 3166 2 or 3 letter code)")


class FHIRPatient(BaseModel):
    """FHIR Patient resource."""
    resourceType: Literal["Patient"] = Field(..., description="Resource type")
    id: Optional[str] = Field(None, description="Logical id of this artifact")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the resource")
    implicitRules: Optional[str] = Field(None, description="A set of rules under which this content was created")
    language: Optional[str] = Field(None, description="Language of the resource content")

    identifier: Optional[List[FHIRIdentifier]] = Field(None, description="An identifier for this patient")
    active: Optional[bool] = Field(None, description="Whether this patient's record is in active use")
    name: Optional[List[FHIRHumanName]] = Field(None, description="A name associated with the patient")
    telecom: Optional[List[FHIRContactPoint]] = Field(None, description="A contact detail for the individual")
    gender: Optional[AdministrativeGender] = Field(None, description="Administrative Gender")
    birthDate: Optional[str] = Field(None, description="The date of birth for the individual")
    deceasedBoolean: Optional[bool] = Field(None, description="Indicates if the individual is deceased")
    deceasedDateTime: Optional[str] = Field(None, description="Indicates when the individual died")
    address: Optional[List[FHIRAddress]] = Field(None, description="An address for the individual")
    maritalStatus: Optional[FHIRCodeableConcept] = Field(None, description="Marital (civil) status of a patient")
    multipleBirthBoolean: Optional[bool] = Field(None, description="Whether patient is part of a multiple birth")
    multipleBirthInteger: Optional[int] = Field(None, description="If multiple birth, what number in sequence")
    photo: Optional[List[Dict[str, Any]]] = Field(None, description="Image of the patient")
    contact: Optional[List[Dict[str, Any]]] = Field(None, description="A contact party (e.g. guardian, partner, friend) for the patient")
    communication: Optional[List[Dict[str, Any]]] = Field(None, description="A language which may be used to communicate with the patient about his or her health")
    generalPractitioner: Optional[List[FHIRReference]] = Field(None, description="Patient's nominated primary care provider")
    managingOrganization: Optional[FHIRReference] = Field(None, description="Organization that is the custodian of the patient record")
    link: Optional[List[Dict[str, Any]]] = Field(None, description="Link to another patient resource that concerns the same actual patient")

    @computed_field
    @property
    def full_name(self) -> Optional[str]:
        """Get the patient's full name."""
        if not self.name:
            return None
        name = self.name[0]  # Use first name
        if name.text:
            return name.text
        parts = []
        if name.prefix:
            parts.extend(name.prefix)
        if name.given:
            parts.extend(name.given)
        if name.family:
            parts.append(name.family)
        if name.suffix:
            parts.extend(name.suffix)
        return " ".join(parts) if parts else None


class FHIRCondition(BaseModel):
    """FHIR Condition resource."""
    resourceType: Literal["Condition"] = Field(..., description="Resource type")
    id: Optional[str] = Field(None, description="Logical id of this artifact")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the resource")
    implicitRules: Optional[str] = Field(None, description="A set of rules under which this content was created")
    language: Optional[str] = Field(None, description="Language of the resource content")

    identifier: Optional[List[FHIRIdentifier]] = Field(None, description="External Ids for this condition")
    clinicalStatus: Optional[FHIRCodeableConcept] = Field(None, description="active | recurrence | relapse | inactive | remission | resolved")
    verificationStatus: Optional[FHIRCodeableConcept] = Field(None, description="unconfirmed | provisional | differential | confirmed | refuted | entered-in-error")
    category: Optional[List[FHIRCodeableConcept]] = Field(None, description="problem-list-item | encounter-diagnosis")
    severity: Optional[FHIRCodeableConcept] = Field(None, description="Subjective severity of condition")
    code: Optional[FHIRCodeableConcept] = Field(None, description="Identification of the condition, problem or diagnosis")
    bodySite: Optional[List[FHIRCodeableConcept]] = Field(None, description="Anatomical location, if relevant")
    subject: FHIRReference = Field(..., description="Who has the condition?")
    encounter: Optional[FHIRReference] = Field(None, description="Encounter created as part of")
    onsetDateTime: Optional[str] = Field(None, description="Estimated or actual date, date-time, or age")
    onsetAge: Optional[FHIRQuantity] = Field(None, description="Estimated or actual date, date-time, or age")
    onsetPeriod: Optional[FHIRPeriod] = Field(None, description="Estimated or actual date, date-time, or age")
    onsetRange: Optional[FHIRRange] = Field(None, description="Estimated or actual date, date-time, or age")
    onsetString: Optional[str] = Field(None, description="Estimated or actual date, date-time, or age")
    abatementDateTime: Optional[str] = Field(None, description="When in resolution/remission")
    abatementAge: Optional[FHIRQuantity] = Field(None, description="When in resolution/remission")
    abatementPeriod: Optional[FHIRPeriod] = Field(None, description="When in resolution/remission")
    abatementRange: Optional[FHIRRange] = Field(None, description="When in resolution/remission")
    abatementString: Optional[str] = Field(None, description="When in resolution/remission")
    recordedDate: Optional[str] = Field(None, description="Date record was first recorded")
    recorder: Optional[FHIRReference] = Field(None, description="Who recorded the condition")
    asserter: Optional[FHIRReference] = Field(None, description="Person who asserts this condition")
    stage: Optional[List[Dict[str, Any]]] = Field(None, description="Stage/grade, usually assessed formally")
    evidence: Optional[List[Dict[str, Any]]] = Field(None, description="Supporting evidence")
    note: Optional[List[Dict[str, Any]]] = Field(None, description="Additional information about the Condition")


class FHIRObservation(BaseModel):
    """FHIR Observation resource."""
    resourceType: Literal["Observation"] = Field(..., description="Resource type")
    id: Optional[str] = Field(None, description="Logical id of this artifact")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the resource")
    implicitRules: Optional[str] = Field(None, description="A set of rules under which this content was created")
    language: Optional[str] = Field(None, description="Language of the resource content")

    identifier: Optional[List[FHIRIdentifier]] = Field(None, description="Business Identifier for observation")
    basedOn: Optional[List[FHIRReference]] = Field(None, description="Fulfills plan, proposal or order")
    partOf: Optional[List[FHIRReference]] = Field(None, description="Part of referenced event")
    status: str = Field(..., description="registered | preliminary | final | amended +")
    category: Optional[List[FHIRCodeableConcept]] = Field(None, description="Classification of type of observation")
    code: FHIRCodeableConcept = Field(..., description="Type of observation (code / type)")
    subject: Optional[FHIRReference] = Field(None, description="Who and/or what the observation is about")
    focus: Optional[List[FHIRReference]] = Field(None, description="What the observation is about, when it is not about the subject of record")
    encounter: Optional[FHIRReference] = Field(None, description="Healthcare event during which this observation is made")
    effectiveDateTime: Optional[str] = Field(None, description="Clinically relevant time/time-period for observation")
    effectivePeriod: Optional[FHIRPeriod] = Field(None, description="Clinically relevant time/time-period for observation")
    effectiveTiming: Optional[Dict[str, Any]] = Field(None, description="Clinically relevant time/time-period for observation")
    effectiveInstant: Optional[str] = Field(None, description="Clinically relevant time/time-period for observation")
    issued: Optional[str] = Field(None, description="Date/Time this version was made available")
    performer: Optional[List[FHIRReference]] = Field(None, description="Who is responsible for the observation")
    valueQuantity: Optional[FHIRQuantity] = Field(None, description="Actual result")
    valueCodeableConcept: Optional[FHIRCodeableConcept] = Field(None, description="Actual result")
    valueString: Optional[str] = Field(None, description="Actual result")
    valueBoolean: Optional[bool] = Field(None, description="Actual result")
    valueInteger: Optional[int] = Field(None, description="Actual result")
    valueRange: Optional[FHIRRange] = Field(None, description="Actual result")
    valueRatio: Optional[FHIRRatio] = Field(None, description="Actual result")
    valueSampledData: Optional[Dict[str, Any]] = Field(None, description="Actual result")
    valueTime: Optional[str] = Field(None, description="Actual result")
    valueDateTime: Optional[str] = Field(None, description="Actual result")
    valuePeriod: Optional[FHIRPeriod] = Field(None, description="Actual result")
    dataAbsentReason: Optional[FHIRCodeableConcept] = Field(None, description="Why the result is missing")
    interpretation: Optional[List[FHIRCodeableConcept]] = Field(None, description="High, low, normal, etc.")
    note: Optional[List[Dict[str, Any]]] = Field(None, description="Comments about the observation")
    bodySite: Optional[FHIRCodeableConcept] = Field(None, description="Observed body part")
    method: Optional[FHIRCodeableConcept] = Field(None, description="How it was done")
    specimen: Optional[FHIRReference] = Field(None, description="Specimen used for this observation")
    device: Optional[FHIRReference] = Field(None, description="A reference to the device that generates the measurements or the device settings")
    referenceRange: Optional[List[Dict[str, Any]]] = Field(None, description="Provides guide for interpretation")
    hasMember: Optional[List[FHIRReference]] = Field(None, description="Related resource that belongs to the Observation group")
    derivedFrom: Optional[List[FHIRReference]] = Field(None, description="Related measurements the observation is made from")
    component: Optional[List[Dict[str, Any]]] = Field(None, description="Component results")


class FHIRProcedure(BaseModel):
    """FHIR Procedure resource."""
    resourceType: Literal["Procedure"] = Field(..., description="Resource type")
    id: Optional[str] = Field(None, description="Logical id of this artifact")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the resource")
    implicitRules: Optional[str] = Field(None, description="A set of rules under which this content was created")
    language: Optional[str] = Field(None, description="Language of the resource content")

    identifier: Optional[List[FHIRIdentifier]] = Field(None, description="External Identifiers for this procedure")
    instantiatesCanonical: Optional[List[str]] = Field(None, description="Instantiates FHIR protocol or definition")
    instantiatesUri: Optional[List[str]] = Field(None, description="Instantiates external protocol or definition")
    basedOn: Optional[List[FHIRReference]] = Field(None, description="A request for this procedure")
    partOf: Optional[List[FHIRReference]] = Field(None, description="Part of referenced event")
    status: str = Field(..., description="preparation | in-progress | not-done | suspended | aborted | completed | entered-in-error | unknown")
    statusReason: Optional[FHIRCodeableConcept] = Field(None, description="Reason for current status")
    category: Optional[FHIRCodeableConcept] = Field(None, description="Classification of the procedure")
    code: Optional[FHIRCodeableConcept] = Field(None, description="Identification of the procedure")
    subject: FHIRReference = Field(..., description="Who the procedure was performed on")
    encounter: Optional[FHIRReference] = Field(None, description="Encounter created as part of")
    performedDateTime: Optional[str] = Field(None, description="When the procedure was performed")
    performedPeriod: Optional[FHIRPeriod] = Field(None, description="When the procedure was performed")
    performedString: Optional[str] = Field(None, description="When the procedure was performed")
    performedAge: Optional[FHIRQuantity] = Field(None, description="When the procedure was performed")
    performedRange: Optional[FHIRRange] = Field(None, description="When the procedure was performed")
    recorder: Optional[FHIRReference] = Field(None, description="Who recorded the procedure")
    asserter: Optional[FHIRReference] = Field(None, description="Person who asserts this procedure")
    performer: Optional[List[Dict[str, Any]]] = Field(None, description="The people who performed the procedure")
    location: Optional[FHIRReference] = Field(None, description="Where the procedure happened")
    reasonCode: Optional[List[FHIRCodeableConcept]] = Field(None, description="Coded reason procedure performed")
    reasonReference: Optional[List[FHIRReference]] = Field(None, description="The justification that the procedure was performed")
    bodySite: Optional[List[FHIRCodeableConcept]] = Field(None, description="Target body sites")
    outcome: Optional[FHIRCodeableConcept] = Field(None, description="The result of procedure")
    report: Optional[List[FHIRReference]] = Field(None, description="Any report resulting from the procedure")
    complication: Optional[List[FHIRCodeableConcept]] = Field(None, description="Complication following the procedure")
    complicationDetail: Optional[List[FHIRReference]] = Field(None, description="Complication following the procedure")
    followUp: Optional[List[FHIRCodeableConcept]] = Field(None, description="Instructions for follow up")
    note: Optional[List[Dict[str, Any]]] = Field(None, description="Additional information about the procedure")
    focalDevice: Optional[List[Dict[str, Any]]] = Field(None, description="Manipulated, implanted, or removed device")
    usedReference: Optional[List[FHIRReference]] = Field(None, description="Items used during procedure")
    usedCode: Optional[List[FHIRCodeableConcept]] = Field(None, description="Coded items used during the procedure")


class FHIRMedicationStatement(BaseModel):
    """FHIR MedicationStatement resource."""
    resourceType: Literal["MedicationStatement"] = Field(..., description="Resource type")
    id: Optional[str] = Field(None, description="Logical id of this artifact")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the resource")
    implicitRules: Optional[str] = Field(None, description="A set of rules under which this content was created")
    language: Optional[str] = Field(None, description="Language of the resource content")

    identifier: Optional[List[FHIRIdentifier]] = Field(None, description="External identifier")
    basedOn: Optional[List[FHIRReference]] = Field(None, description="Fulfills plan, proposal or order")
    partOf: Optional[List[FHIRReference]] = Field(None, description="Part of referenced event")
    status: str = Field(..., description="active | completed | entered-in-error | intended | stopped | on-hold | unknown | not-taken")
    statusReason: Optional[List[FHIRCodeableConcept]] = Field(None, description="Reason for current status")
    category: Optional[FHIRCodeableConcept] = Field(None, description="Type of medication usage")
    medicationCodeableConcept: Optional[FHIRCodeableConcept] = Field(None, description="What medication was taken")
    medicationReference: Optional[FHIRReference] = Field(None, description="What medication was taken")
    subject: FHIRReference = Field(..., description="Who is taking the medication")
    context: Optional[FHIRReference] = Field(None, description="Encounter / Episode associated with MedicationStatement")
    effectiveDateTime: Optional[str] = Field(None, description="The date/time or interval when the medication is/was/will be taken")
    effectivePeriod: Optional[FHIRPeriod] = Field(None, description="The date/time or interval when the medication is/was/will be taken")
    dateAsserted: Optional[str] = Field(None, description="When the statement was asserted?")
    informationSource: Optional[FHIRReference] = Field(None, description="Person or organization that provided the information about the taking of this medication")
    derivedFrom: Optional[List[FHIRReference]] = Field(None, description="Additional supporting information")
    reasonCode: Optional[List[FHIRCodeableConcept]] = Field(None, description="Reason for why the medication is being/was taken")
    reasonReference: Optional[List[FHIRReference]] = Field(None, description="Condition or observation that supports why the medication is being/was taken")
    note: Optional[List[Dict[str, Any]]] = Field(None, description="Further information about the statement")
    dosage: Optional[List[Dict[str, Any]]] = Field(None, description="Details of how medication is/was taken or should be taken")


# =============================================================================
# EXTENSION MODELS - mCODE-Specific Fields
# =============================================================================

class McodeExtension(BaseModel):
    """Base class for mCODE extensions."""
    url: str = Field(..., description="Extension URL")
    valueCode: Optional[str] = Field(None, description="Value as code")
    valueString: Optional[str] = Field(None, description="Value as string")
    valueBoolean: Optional[bool] = Field(None, description="Value as boolean")
    valueInteger: Optional[int] = Field(None, description="Value as integer")
    valueDecimal: Optional[float] = Field(None, description="Value as decimal")
    valueDateTime: Optional[str] = Field(None, description="Value as dateTime")
    valueCodeableConcept: Optional[FHIRCodeableConcept] = Field(None, description="Value as CodeableConcept")
    valueReference: Optional[FHIRReference] = Field(None, description="Value as Reference")

    @model_validator(mode="before")
    @classmethod
    def validate_extension_value(cls, values: Any) -> Any:
        """Ensure exactly one value field is present."""
        if isinstance(values, dict):
            value_fields = [k for k in values.keys() if k.startswith('value')]
            if len(value_fields) != 1:
                raise ValueError(f"Extension must have exactly one value field, got: {value_fields}")
        return values


class BirthSexExtension(McodeExtension):
    """mCODE Birth Sex Extension."""
    url: str = Field(default="http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-birth-sex", description="Extension URL")
    valueCode: BirthSex = Field(..., description="Birth sex value")


class USCoreRaceExtension(McodeExtension):
    """US Core Race Extension."""
    url: str = Field(default="http://hl7.org/fhir/us/core/StructureDefinition/us-core-race", description="Extension URL")
    valueCodeableConcept: FHIRCodeableConcept = Field(..., description="Race classification")


class USCoreEthnicityExtension(McodeExtension):
    """US Core Ethnicity Extension."""
    url: str = Field(default="http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity", description="Extension URL")
    valueCodeableConcept: FHIRCodeableConcept = Field(..., description="Ethnicity classification")


class HistologyMorphologyBehaviorExtension(McodeExtension):
    """mCODE Histology Morphology Behavior Extension."""
    url: str = Field(default="http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-histology-morphology-behavior", description="Extension URL")
    valueCodeableConcept: FHIRCodeableConcept = Field(..., description="Histology morphology behavior")


class LateralityExtension(McodeExtension):
    """mCODE Laterality Extension."""
    url: str = Field(default="http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-laterality", description="Extension URL")
    valueCodeableConcept: FHIRCodeableConcept = Field(..., description="Body side or laterality")


class RelatedConditionExtension(McodeExtension):
    """mCODE Related Condition Extension."""
    url: str = Field(default="http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-related-condition", description="Extension URL")
    valueReference: FHIRReference = Field(..., description="Reference to related condition")


class ConditionRelatedExtension(McodeExtension):
    """mCODE Condition Related Extension."""
    url: str = Field(default="http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-condition-related", description="Extension URL")
    valueReference: FHIRReference = Field(..., description="Reference to related condition")


# =============================================================================
# mCODE PROFILE MODELS
# =============================================================================

class McodePatient(FHIRPatient):
    """mCODE Patient profile - extends FHIR Patient with mCODE requirements."""

    # Add mCODE-specific extensions
    extension: Optional[List[Union[BirthSexExtension, USCoreRaceExtension, USCoreEthnicityExtension]]] = Field(
        None, description="mCODE extensions for patient"
    )

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: Optional[AdministrativeGender]) -> Optional[AdministrativeGender]:
        """Validate administrative gender is provided."""
        if v is None:
            raise ValueError("Administrative gender is required for mCODE Patient")
        return v

    @field_validator('birthDate')
    @classmethod
    def validate_birth_date(cls, v: Optional[str]) -> Optional[str]:
        """Validate birth date format."""
        if v is not None:
            # Basic ISO date validation
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Birth date must be in ISO format")
        return v


class CancerCondition(FHIRCondition):
    """mCODE Cancer Condition profile."""

    # Profile URL for this resource
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the resource")

    # mCODE-specific extensions
    extension: Optional[List[Union[
        HistologyMorphologyBehaviorExtension,
        LateralityExtension,
        RelatedConditionExtension
    ]]] = Field(None, description="mCODE extensions for cancer condition")

    # Override category to be required for cancer
    category: List[FHIRCodeableConcept] = Field(..., description="Category must include 'problem-list-item'")

    # Code must represent a cancer condition
    code: FHIRCodeableConcept = Field(..., description="Cancer diagnosis code")

    @model_validator(mode="after")
    def validate_cancer_condition(self) -> "CancerCondition":
        """Validate this is a valid cancer condition."""
        # Check that category includes problem-list-item
        if self.category:
            has_problem_list = any(
                coding.get('code') == 'problem-list-item'
                for cat in self.category
                for coding in cat.coding
            )
            if not has_problem_list:
                raise ValueError("Cancer condition must have category 'problem-list-item'")

        # Validate code represents cancer
        if self.code and self.code.coding:
            cancer_codes = {member.value for member in CancerConditionCode}
            has_cancer_code = any(
                coding.get('code') in cancer_codes
                for coding in self.code.coding
            )
            if not has_cancer_code:
                raise ValueError("Condition code must represent a cancer diagnosis")

        return self


class CancerStaging(BaseModel):
    """mCODE Cancer Staging profile - represents cancer staging information."""

    resourceType: Literal["Observation"] = Field(..., description="Resource type")
    id: Optional[str] = Field(None, description="Logical id of this artifact")
    meta: Dict[str, Any] = Field(..., description="Metadata about the resource")
    status: str = Field(..., description="Observation status")
    category: List[FHIRCodeableConcept] = Field(..., description="Observation category")
    code: FHIRCodeableConcept = Field(..., description="Staging observation code")
    subject: FHIRReference = Field(..., description="Patient reference")
    effectiveDateTime: Optional[str] = Field(None, description="When staging was determined")
    valueCodeableConcept: Optional[FHIRCodeableConcept] = Field(None, description="Staging value")

    # mCODE-specific components for TNM staging
    component: Optional[List[Dict[str, Any]]] = Field(None, description="TNM components")

    @model_validator(mode="after")
    def validate_staging_profile(self) -> "CancerStaging":
        """Validate staging profile requirements."""
        # Check profile URL in meta
        if self.meta and 'profile' in self.meta:
            profile_urls = self.meta['profile']
            expected_url = "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-stage"
            if expected_url not in profile_urls:
                raise ValueError(f"Meta.profile must include {expected_url}")

        # Validate category includes exam
        if self.category:
            has_exam = any(
                coding.get('code') == 'exam'
                for cat in self.category
                for coding in cat.coding
            )
            if not has_exam:
                raise ValueError("Staging observation must have category 'exam'")

        return self


class TNMStageGroup(CancerStaging):
    """mCODE TNM Stage Group profile."""

    @model_validator(mode="after")
    def validate_tnm_stage_group(self) -> "TNMStageGroup":
        """Validate TNM stage group specific requirements."""
        # Code must be TNM stage group
        if self.code and self.code.coding:
            has_tnm_code = any(
                coding.get('code') == '21908-9'  # LOINC code for TNM stage group
                for coding in self.code.coding
            )
            if not has_tnm_code:
                raise ValueError("TNM Stage Group must have LOINC code 21908-9")

        # Value must be from TNM stage group value set
        if self.valueCodeableConcept and self.valueCodeableConcept.coding:
            valid_stages = {stage.value for stage in TNMStageGroup}
            has_valid_stage = any(
                coding.get('code') in valid_stages
                for coding in self.valueCodeableConcept.coding
            )
            if not has_valid_stage:
                raise ValueError("Stage group must be from TNM stage group value set")

        return self


class TumorMarkerTest(FHIRObservation):
    """mCODE Tumor Marker Test profile."""

    # mCODE-specific extensions
    extension: Optional[List[ConditionRelatedExtension]] = Field(
        None, description="mCODE extensions for tumor marker test"
    )

    # Category must include laboratory
    category: List[FHIRCodeableConcept] = Field(..., description="Must include laboratory category")

    @model_validator(mode="after")
    def validate_tumor_marker_test(self) -> "TumorMarkerTest":
        """Validate tumor marker test requirements."""
        # Check profile URL in meta
        if self.meta and 'profile' in self.meta:
            profile_urls = self.meta['profile']
            expected_url = "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-tumor-marker-test"
            if expected_url not in profile_urls:
                raise ValueError(f"Meta.profile must include {expected_url}")

        # Validate category includes laboratory
        if self.category:
            has_laboratory = any(
                coding.get('code') == 'laboratory'
                for cat in self.category
                for coding in cat.coding
            )
            if not has_laboratory:
                raise ValueError("Tumor marker test must have category 'laboratory'")

        return self


class ECOGPerformanceStatusObservation(FHIRObservation):
    """mCODE ECOG Performance Status profile."""

    # Value must be from ECOG value set
    valueCodeableConcept: FHIRCodeableConcept = Field(..., description="ECOG performance status value")

    @model_validator(mode="after")
    def validate_ecog_performance_status(self) -> "ECOGPerformanceStatusObservation":
        """Validate ECOG performance status requirements."""
        # Check profile URL in meta
        if self.meta and 'profile' in self.meta:
            profile_urls = self.meta['profile']
            expected_url = "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ecog-performance-status"
            if expected_url not in profile_urls:
                raise ValueError(f"Meta.profile must include {expected_url}")

        # Validate value is from ECOG value set
        if self.valueCodeableConcept and self.valueCodeableConcept.coding:
            valid_ecog = {status.value for status in ECOGPerformanceStatus}
            has_valid_ecog = any(
                coding.get('code') in valid_ecog
                for coding in self.valueCodeableConcept.coding
            )
            if not has_valid_ecog:
                raise ValueError("ECOG performance status must be from ECOG value set")

        return self


class CancerRelatedMedicationStatement(FHIRMedicationStatement):
    """mCODE Cancer Related Medication Statement profile."""

    # Category must indicate cancer-related
    category: FHIRCodeableConcept = Field(..., description="Must indicate cancer-related medication")

    @model_validator(mode="after")
    def validate_cancer_medication(self) -> "CancerRelatedMedicationStatement":
        """Validate cancer medication requirements."""
        # Check profile URL in meta
        if self.meta and 'profile' in self.meta:
            profile_urls = self.meta['profile']
            expected_url = "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-related-medication-statement"
            if expected_url not in profile_urls:
                raise ValueError(f"Meta.profile must include {expected_url}")

        return self


class CancerRelatedSurgicalProcedure(FHIRProcedure):
    """mCODE Cancer Related Surgical Procedure profile."""

    # Category must indicate surgical procedure
    category: FHIRCodeableConcept = Field(..., description="Must indicate surgical procedure")

    @model_validator(mode="after")
    def validate_cancer_surgery(self) -> "CancerRelatedSurgicalProcedure":
        """Validate cancer surgery requirements."""
        # Check profile URL in meta
        if self.meta and 'profile' in self.meta:
            profile_urls = self.meta['profile']
            expected_url = "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-related-surgical-procedure"
            if expected_url not in profile_urls:
                raise ValueError(f"Meta.profile must include {expected_url}")

        return self


class CancerRelatedRadiationProcedure(FHIRProcedure):
    """mCODE Cancer Related Radiation Procedure profile."""

    # Category must indicate radiation procedure
    category: FHIRCodeableConcept = Field(..., description="Must indicate radiation procedure")

    @model_validator(mode="after")
    def validate_cancer_radiation(self) -> "CancerRelatedRadiationProcedure":
        """Validate cancer radiation requirements."""
        # Check profile URL in meta
        if self.meta and 'profile' in self.meta:
            profile_urls = self.meta['profile']
            expected_url = "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-related-radiation-procedure"
            if expected_url not in profile_urls:
                raise ValueError(f"Meta.profile must include {expected_url}")

        return self


# =============================================================================
# VERSIONING INTEGRATION
# =============================================================================

class VersionedMcodeResource(BaseModel):
    """Base class for versioned mCODE resources."""

    version: McodeVersion = Field(default_factory=lambda: McodeVersion.latest(), description="mCODE version")
    profile_url: Optional[str] = Field(None, description="Canonical profile URL")

    def __init__(self, **data):
        super().__init__(**data)
        # Set profile URL based on version if not provided
        if not self.profile_url:
            # This would be implemented based on the specific profile
            pass

    @property
    def version_manager(self):
        """Get the version manager instance."""
        return get_version_manager()

    def get_profile_url(self, profile: McodeProfile) -> str:
        """Get canonical URL for a profile in this resource's version."""
        return self.version_manager.get_profile_url(profile, self.version)


# =============================================================================
# VALIDATION RULES AND BUSINESS LOGIC
# =============================================================================

class McodeValidator:
    """Validator for mCODE resources and business rules."""

    def __init__(self, version: Optional[McodeVersion] = None):
        self.version = version or McodeVersion.latest()
        self.version_manager = get_version_manager()

    def validate_resource_compatibility(self, resource: BaseModel, profile: McodeProfile) -> bool:
        """Validate resource compatibility with mCODE version."""
        # Check if profile is supported in this version
        try:
            profile_url = self.version_manager.get_profile_url(profile, self.version)
            return True
        except ValueError:
            return False

    def validate_patient_eligibility(self, patient: McodePatient, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate patient eligibility against clinical trial criteria."""
        results = {
            "eligible": True,
            "criteria_met": [],
            "criteria_not_met": [],
            "warnings": []
        }

        # Age criteria
        if "min_age" in criteria or "max_age" in criteria:
            if not patient.birthDate:
                results["eligible"] = False
                results["criteria_not_met"].append("Age criteria: birth date not available")
            else:
                birth_date = datetime.fromisoformat(patient.birthDate.replace('Z', '+00:00'))
                age = (datetime.utcnow() - birth_date).days // 365

                if "min_age" in criteria and age < criteria["min_age"]:
                    results["eligible"] = False
                    results["criteria_not_met"].append(f"Age criteria: patient age {age} < minimum {criteria['min_age']}")
                elif "max_age" in criteria and age > criteria["max_age"]:
                    results["eligible"] = False
                    results["criteria_not_met"].append(f"Age criteria: patient age {age} > maximum {criteria['max_age']}")
                else:
                    results["criteria_met"].append(f"Age criteria: patient age {age} within range")

        # Gender criteria
        if "gender" in criteria:
            if not patient.gender:
                results["eligible"] = False
                results["criteria_not_met"].append("Gender criteria: patient gender not specified")
            elif patient.gender.value != criteria["gender"]:
                results["eligible"] = False
                results["criteria_not_met"].append(f"Gender criteria: patient gender {patient.gender.value} != required {criteria['gender']}")
            else:
                results["criteria_met"].append(f"Gender criteria: patient gender matches {criteria['gender']}")

        return results

    def validate_cancer_condition(self, condition: CancerCondition) -> Dict[str, Any]:
        """Validate cancer condition data quality."""
        issues = []

        # Check for required histology
        has_histology = any(
            isinstance(ext, HistologyMorphologyBehaviorExtension)
            for ext in (condition.extension or [])
        )
        if not has_histology:
            issues.append("Missing histology morphology behavior extension")

        # Check for staging information
        # This would typically involve checking related observations

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0, 1.0 - (len(issues) * 0.2))
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_mcode_patient(
    id: str,
    name: List[Dict[str, Any]],
    gender: AdministrativeGender,
    birth_date: str,
    birth_sex: Optional[BirthSex] = None,
    race: Optional[FHIRCodeableConcept] = None,
    ethnicity: Optional[FHIRCodeableConcept] = None
) -> McodePatient:
    """Create a properly structured mCODE Patient resource."""
    extensions = []

    if birth_sex:
        extensions.append(BirthSexExtension(valueCode=birth_sex))
    if race:
        extensions.append(USCoreRaceExtension(valueCodeableConcept=race))
    if ethnicity:
        extensions.append(USCoreEthnicityExtension(valueCodeableConcept=ethnicity))

    return McodePatient(
        resourceType="Patient",
        id=id,
        name=[FHIRHumanName(**n) for n in name],
        gender=gender,
        birthDate=birth_date,
        extension=extensions if extensions else None
    )


def create_cancer_condition(
    patient_id: str,
    cancer_code: CancerConditionCode,
    clinical_status: str = "active",
    histology_behavior: Optional[HistologyMorphologyBehavior] = None,
    laterality: Optional[FHIRCodeableConcept] = None
) -> CancerCondition:
    """Create a properly structured mCODE Cancer Condition."""
    extensions = []

    if histology_behavior:
        extensions.append(HistologyMorphologyBehaviorExtension(
            valueCodeableConcept=FHIRCodeableConcept(
                coding=[{
                    "system": "http://snomed.info/sct",
                    "code": histology_behavior.value,
                    "display": histology_behavior.name
                }]
            )
        ))

    if laterality:
        extensions.append(LateralityExtension(valueCodeableConcept=laterality))

    return CancerCondition(
        resourceType="Condition",
        subject=FHIRReference(reference=f"Patient/{patient_id}"),
        clinicalStatus=FHIRCodeableConcept(
            coding=[{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": clinical_status}]
        ),
        category=[FHIRCodeableConcept(
            coding=[{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]
        )],
        code=FHIRCodeableConcept(
            coding=[{
                "system": "http://snomed.info/sct",
                "code": cancer_code.value,
                "display": cancer_code.name.replace('_', ' ').title()
            }]
        ),
        extension=extensions if extensions else None
    )


class McodeElement(BaseModel):
    """Individual mCODE element mapping."""

    element_type: str = Field(
        ...,
        description="Type of mCODE element (e.g., 'CancerCondition', 'CancerTreatment')",
    )
    code: Optional[str] = Field(None, description="Element code")
    display: Optional[str] = Field(None, description="Human-readable display name")
    system: Optional[str] = Field(None, description="Coding system")
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score for the mapping"
    )
    evidence_text: Optional[str] = Field(None, description="Supporting evidence text")


# Export all models
__all__ = [
    # Value sets
    "AdministrativeGender",
    "BirthSex",
    "CancerConditionCode",
    "TNMStageGroupEnum",
    "ECOGPerformanceStatus",
    "ReceptorStatus",
    "HistologyMorphologyBehavior",

    # Base FHIR resources
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

    # Extension models
    "McodeExtension",
    "BirthSexExtension",
    "USCoreRaceExtension",
    "USCoreEthnicityExtension",
    "HistologyMorphologyBehaviorExtension",
    "LateralityExtension",
    "RelatedConditionExtension",
    "ConditionRelatedExtension",

    # mCODE profile models
    "McodePatient",
    "CancerCondition",
    "CancerStaging",
    "TNMStageGroup",
    "TumorMarkerTest",
    "ECOGPerformanceStatusObservation",
    "CancerRelatedMedicationStatement",
    "CancerRelatedSurgicalProcedure",
    "CancerRelatedRadiationProcedure",

    # Core mCODE elements
    "McodeElement",

    # Versioning integration
    "VersionedMcodeResource",

    # Validation and business logic
    "McodeValidator",

    # Utility functions
    "create_mcode_patient",
    "create_cancer_condition",
]
