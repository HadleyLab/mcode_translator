"""
Data Quality Validation Gates for mCODE Processing Pipelines

This module provides comprehensive validation checks for data quality in mCODE processing workflows.
It validates mCODE element completeness, required elements, coding systems, FHIR resource structure,
and mCODE profile compliance. Generates quality reports with coverage metrics and prevents
processing completion if critical elements are missing.

Key Features:
- Patient validation gates (demographics, cancer conditions, coding systems)
- Trial validation gates (metadata, coding systems, mCODE compliance)
- Quality report generation with coverage metrics
- Fail-fast validation that prevents processing with missing critical elements
- Support for SNOMED CT, LOINC, RxNorm, and other required coding systems
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .mcode_models import McodeValidator
from .mcode_versioning import McodeVersion


class ValidationSeverity(str, Enum):
    """Validation severity levels."""
    CRITICAL = "critical"  # Prevents processing
    WARNING = "warning"    # Allows processing but logs issue
    INFO = "info"         # Informational only


class CodingSystem(str, Enum):
    """Supported coding systems for mCODE."""
    SNOMED_CT = "http://snomed.info/sct"
    LOINC = "http://loinc.org"
    RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
    ICD10 = "http://hl7.org/fhir/sid/icd-10"
    ICD9 = "http://hl7.org/fhir/sid/icd-9"
    CVX = "http://hl7.org/fhir/sid/cvx"
    CLINICAL_TRIALS_GOV = "https://clinicaltrials.gov"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    element_path: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive quality report for mCODE data."""
    total_elements: int
    valid_elements: int
    critical_issues: int
    warning_issues: int
    info_issues: int
    coverage_percentage: float
    issues: List[ValidationIssue]
    coding_system_coverage: Dict[str, float]
    completeness_score: float
    can_proceed: bool

    def __str__(self) -> str:
        """String representation of quality report."""
        return (
            f"Quality Report: {self.valid_elements}/{self.total_elements} elements valid "
            f"({self.coverage_percentage:.1f}% coverage). "
            f"Issues: {self.critical_issues} critical, {self.warning_issues} warnings, "
            f"{self.info_issues} info. "
            f"Completeness: {self.completeness_score:.2f}. "
            f"Can proceed: {self.can_proceed}"
        )


class DataQualityValidator:
    """
    Comprehensive data quality validator for mCODE processing pipelines.

    Validates mCODE element completeness, required elements, coding systems,
    FHIR resource structure, and mCODE profile compliance.
    """

    def __init__(self, version: Optional[McodeVersion] = None):
        self.version = version or McodeVersion.latest()
        self.mcode_validator = McodeValidator(version)

        # Required elements for patients
        self.required_patient_elements = {
            "Patient": ["gender", "birthDate"],
            "CancerCondition": ["code", "clinicalStatus"],
            "Demographics": ["name", "gender", "birthDate"]
        }

        # Required elements for trials
        self.required_trial_elements = {
            "TrialMetadata": ["nct_id", "brief_title", "overall_status"],
            "EligibilityCriteria": ["minimum_age", "maximum_age", "sex"],
            "Conditions": ["conditions"],
            "Interventions": ["interventions"]
        }

        # Required coding systems
        self.required_coding_systems = {
            "conditions": [CodingSystem.SNOMED_CT, CodingSystem.ICD10],
            "observations": [CodingSystem.LOINC, CodingSystem.SNOMED_CT],
            "medications": [CodingSystem.RXNORM],
            "procedures": [CodingSystem.SNOMED_CT, CodingSystem.CVX],
            "immunizations": [CodingSystem.CVX],
            "allergies": [CodingSystem.SNOMED_CT, CodingSystem.RXNORM]
        }

    def validate_patient_data(
        self,
        patient_data: Dict[str, Any],
        mcode_elements: Dict[str, Any]
    ) -> QualityReport:
        """
        Validate patient data quality and generate comprehensive report.

        Args:
            patient_data: Raw patient FHIR bundle
            mcode_elements: Extracted mCODE elements

        Returns:
            QualityReport: Comprehensive validation results
        """
        issues = []
        total_elements = 0
        valid_elements = 0

        # Validate patient demographics
        demo_issues, demo_valid, demo_total = self._validate_patient_demographics(patient_data)
        issues.extend(demo_issues)
        total_elements += demo_total
        valid_elements += demo_valid

        # Validate cancer conditions
        condition_issues, condition_valid, condition_total = self._validate_cancer_conditions(mcode_elements)
        issues.extend(condition_issues)
        total_elements += condition_total
        valid_elements += condition_valid

        # Validate coding systems
        coding_issues, coding_valid, coding_total = self._validate_coding_systems(mcode_elements)
        issues.extend(coding_issues)
        total_elements += coding_total
        valid_elements += coding_valid

        # Validate FHIR structure
        fhir_issues, fhir_valid, fhir_total = self._validate_fhir_structure(patient_data)
        issues.extend(fhir_issues)
        total_elements += fhir_total
        valid_elements += fhir_valid

        # Validate mCODE profile compliance
        profile_issues, profile_valid, profile_total = self._validate_mcode_profile_compliance(mcode_elements)
        issues.extend(profile_issues)
        total_elements += profile_total
        valid_elements += profile_valid

        # Calculate metrics
        coverage_percentage = (valid_elements / total_elements * 100) if total_elements > 0 else 0
        critical_issues = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        warning_issues = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        info_issues = len([i for i in issues if i.severity == ValidationSeverity.INFO])

        # Calculate completeness score (weighted by severity)
        completeness_score = max(0, 1.0 - (critical_issues * 0.5 + warning_issues * 0.2 + info_issues * 0.1))

        # Determine if processing can proceed
        can_proceed = critical_issues == 0

        # Calculate coding system coverage
        coding_system_coverage = self._calculate_coding_system_coverage(mcode_elements)

        return QualityReport(
            total_elements=total_elements,
            valid_elements=valid_elements,
            critical_issues=critical_issues,
            warning_issues=warning_issues,
            info_issues=info_issues,
            coverage_percentage=coverage_percentage,
            issues=issues,
            coding_system_coverage=coding_system_coverage,
            completeness_score=completeness_score,
            can_proceed=can_proceed
        )

    def validate_trial_data(
        self,
        trial_data: Dict[str, Any],
        mcode_elements: Dict[str, Any]
    ) -> QualityReport:
        """
        Validate trial data quality and generate comprehensive report.

        Args:
            trial_data: Raw trial data
            mcode_elements: Extracted mCODE elements

        Returns:
            QualityReport: Comprehensive validation results
        """
        issues = []
        total_elements = 0
        valid_elements = 0

        # Validate trial metadata
        meta_issues, meta_valid, meta_total = self._validate_trial_metadata(trial_data)
        issues.extend(meta_issues)
        total_elements += meta_total
        valid_elements += meta_valid

        # Validate eligibility criteria
        eligibility_issues, eligibility_valid, eligibility_total = self._validate_eligibility_criteria(trial_data)
        issues.extend(eligibility_issues)
        total_elements += eligibility_total
        valid_elements += eligibility_valid

        # Validate coding systems
        coding_issues, coding_valid, coding_total = self._validate_trial_coding_systems(mcode_elements)
        issues.extend(coding_issues)
        total_elements += coding_total
        valid_elements += coding_valid

        # Validate mCODE compliance
        mcode_issues, mcode_valid, mcode_total = self._validate_trial_mcode_compliance(mcode_elements)
        issues.extend(mcode_issues)
        total_elements += mcode_total
        valid_elements += mcode_valid

        # Calculate metrics
        coverage_percentage = (valid_elements / total_elements * 100) if total_elements > 0 else 0
        critical_issues = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        warning_issues = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        info_issues = len([i for i in issues if i.severity == ValidationSeverity.INFO])

        completeness_score = max(0, 1.0 - (critical_issues * 0.5 + warning_issues * 0.2 + info_issues * 0.1))
        can_proceed = critical_issues == 0
        coding_system_coverage = self._calculate_coding_system_coverage(mcode_elements)

        return QualityReport(
            total_elements=total_elements,
            valid_elements=valid_elements,
            critical_issues=critical_issues,
            warning_issues=warning_issues,
            info_issues=info_issues,
            coverage_percentage=coverage_percentage,
            issues=issues,
            coding_system_coverage=coding_system_coverage,
            completeness_score=completeness_score,
            can_proceed=can_proceed
        )

    def _validate_patient_demographics(
        self, patient_data: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate patient demographics completeness."""
        issues = []
        valid_count = 0
        total_count = 0

        # Find Patient resource
        patient_resource = None
        for entry in patient_data.get("entry", []):
            if entry.get("resource", {}).get("resourceType") == "Patient":
                patient_resource = entry["resource"]
                break

        if not patient_resource:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="Patient Demographics",
                message="No Patient resource found in FHIR bundle",
                element_path="entry[*].resource[resourceType=Patient]",
                suggested_fix="Ensure FHIR bundle contains a Patient resource"
            ))
            return issues, 0, 1

        total_count += 1

        # Validate required demographics
        required_fields = self.required_patient_elements["Patient"]
        for field in required_fields:
            total_count += 1
            if field not in patient_resource or patient_resource[field] is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Patient Demographics",
                    message=f"Required field '{field}' is missing",
                    element_path=f"Patient.{field}",
                    suggested_fix=f"Add {field} to Patient resource"
                ))
            else:
                valid_count += 1

        # Validate gender format
        if "gender" in patient_resource:
            gender = patient_resource["gender"]
            if gender not in ["male", "female", "other", "unknown"]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Patient Demographics",
                    message=f"Gender '{gender}' not in standard AdministrativeGender values",
                    element_path="Patient.gender",
                    suggested_fix="Use standard values: male, female, other, unknown"
                ))
            else:
                valid_count += 1

        # Validate birth date format
        if "birthDate" in patient_resource:
            birth_date = patient_resource["birthDate"]
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", birth_date):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Patient Demographics",
                    message="Birth date not in ISO format (YYYY-MM-DD)",
                    element_path="Patient.birthDate",
                    suggested_fix="Format birth date as YYYY-MM-DD"
                ))
            else:
                valid_count += 1

        return issues, valid_count, total_count

    def _validate_cancer_conditions(
        self, mcode_elements: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate cancer condition elements."""
        issues = []
        valid_count = 0
        total_count = 0

        cancer_conditions = mcode_elements.get("CancerCondition", [])
        if isinstance(cancer_conditions, dict):
            cancer_conditions = [cancer_conditions]

        total_count += 1
        if not cancer_conditions:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="Cancer Conditions",
                message="No cancer conditions found",
                element_path="CancerCondition",
                suggested_fix="Extract cancer conditions from patient data"
            ))
            return issues, 0, total_count

        valid_count += 1

        # Validate each cancer condition
        for i, condition in enumerate(cancer_conditions):
            total_count += 1
            condition_path = f"CancerCondition[{i}]"

            # Check for required fields
            if not isinstance(condition, dict):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Cancer Conditions",
                    message=f"Cancer condition {i} is not a valid object",
                    element_path=condition_path,
                    suggested_fix="Ensure cancer conditions are properly structured"
                ))
                continue

            # Validate code
            if "code" not in condition:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Cancer Conditions",
                    message=f"Cancer condition {i} missing code",
                    element_path=f"{condition_path}.code",
                    suggested_fix="Add cancer diagnosis code"
                ))
            else:
                valid_count += 1

            # Validate clinical status
            if "clinicalStatus" not in condition:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Cancer Conditions",
                    message=f"Cancer condition {i} missing clinical status",
                    element_path=f"{condition_path}.clinicalStatus",
                    suggested_fix="Add clinical status (active, remission, etc.)"
                ))
            else:
                valid_count += 1

        return issues, valid_count, total_count

    def _validate_coding_systems(
        self, mcode_elements: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate coding systems usage."""
        issues = []
        valid_count = 0
        total_count = 0

        # Check each element type for proper coding systems
        for element_type, required_systems in self.required_coding_systems.items():
            elements = mcode_elements.get(element_type, [])
            if isinstance(elements, dict):
                elements = [elements]

            for i, element in enumerate(elements):
                total_count += 1
                element_path = f"{element_type}[{i}]"

                if not isinstance(element, dict):
                    continue

                # Check coding information
                coding = element.get("coding", [])
                if isinstance(coding, dict):
                    coding = [coding]

                if not coding:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Coding Systems",
                        message=f"{element_type} element {i} has no coding information",
                        element_path=f"{element_path}.coding",
                        suggested_fix=f"Add coding from required systems: {[s.value for s in required_systems]}"
                    ))
                    continue

                # Check if any required system is present
                has_required_system = False
                for code_info in coding:
                    if isinstance(code_info, dict):
                        system = code_info.get("system")
                        if system in [s.value for s in required_systems]:
                            has_required_system = True
                            break

                if has_required_system:
                    valid_count += 1
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Coding Systems",
                        message=f"{element_type} element {i} uses non-standard coding system",
                        element_path=f"{element_path}.coding[*].system",
                        suggested_fix=f"Use one of: {[s.value for s in required_systems]}"
                    ))

        return issues, valid_count, total_count

    def _validate_fhir_structure(
        self, patient_data: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate FHIR resource structure."""
        issues = []
        valid_count = 0
        total_count = 0

        # Check bundle structure
        total_count += 1
        if not isinstance(patient_data, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="FHIR Structure",
                message="Patient data is not a valid object",
                element_path="root",
                suggested_fix="Ensure patient data is a valid FHIR Bundle object"
            ))
            return issues, 0, total_count

        if "resourceType" not in patient_data:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="FHIR Structure",
                message="Missing resourceType - not a valid FHIR resource",
                element_path="resourceType",
                suggested_fix="Add resourceType field to FHIR resource"
            ))
        else:
            valid_count += 1

        # Check entries
        entries = patient_data.get("entry", [])
        total_count += 1
        if not isinstance(entries, list):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="FHIR Structure",
                message="Bundle entries is not an array",
                element_path="entry",
                suggested_fix="Ensure entry field is an array of resources"
            ))
        else:
            valid_count += 1

            # Validate each entry
            for i, entry in enumerate(entries):
                total_count += 1
                entry_path = f"entry[{i}]"

                if not isinstance(entry, dict):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="FHIR Structure",
                        message=f"Entry {i} is not a valid object",
                        element_path=entry_path,
                        suggested_fix="Ensure all entries are valid objects"
                    ))
                    continue

                resource = entry.get("resource")
                if not resource:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="FHIR Structure",
                        message=f"Entry {i} missing resource",
                        element_path=f"{entry_path}.resource",
                        suggested_fix="Add resource field to entry"
                    ))
                    continue

                if "resourceType" not in resource:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="FHIR Structure",
                        message=f"Resource {i} missing resourceType",
                        element_path=f"{entry_path}.resource.resourceType",
                        suggested_fix="Add resourceType to resource"
                    ))
                else:
                    valid_count += 1

        return issues, valid_count, total_count

    def _validate_mcode_profile_compliance(
        self, mcode_elements: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate mCODE profile compliance."""
        issues = []
        valid_count = 0
        total_count = 0

        # Check for required mCODE elements
        required_elements = ["Patient", "CancerCondition"]
        for element in required_elements:
            total_count += 1
            if element not in mcode_elements:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="mCODE Compliance",
                    message=f"Required mCODE element '{element}' is missing",
                    element_path=element,
                    suggested_fix=f"Extract {element} from patient data"
                ))
            else:
                valid_count += 1

        # Validate element structure
        for element_name, element_data in mcode_elements.items():
            total_count += 1

            # Check if element follows expected structure
            if isinstance(element_data, dict):
                # Single element
                if "system" in element_data and "code" in element_data:
                    valid_count += 1
                elif "display" in element_data:
                    valid_count += 1
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="mCODE Compliance",
                        message=f"mCODE element '{element_name}' lacks proper coding structure",
                        element_path=element_name,
                        suggested_fix="Ensure element has system, code, and/or display fields"
                    ))
            elif isinstance(element_data, list):
                # Multiple elements
                for i, item in enumerate(element_data):
                    if isinstance(item, dict):
                        if "system" in item and "code" in item:
                            valid_count += 1
                        elif "display" in item:
                            valid_count += 1
                        else:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category="mCODE Compliance",
                                message=f"mCODE element '{element_name}[{i}]' lacks proper coding structure",
                                element_path=f"{element_name}[{i}]",
                                suggested_fix="Ensure element has system, code, and/or display fields"
                            ))
                    else:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="mCODE Compliance",
                            message=f"mCODE element '{element_name}[{i}]' is not a valid object",
                            element_path=f"{element_name}[{i}]",
                            suggested_fix="Ensure all elements are properly structured objects"
                        ))

        return issues, valid_count, total_count

    def _validate_trial_metadata(
        self, trial_data: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate trial metadata completeness."""
        issues = []
        valid_count = 0
        total_count = 0

        protocol_section = trial_data.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        status = protocol_section.get("statusModule", {})

        # Check required metadata fields
        required_fields = [
            ("nct_id", identification, "identificationModule.nctId"),
            ("brief_title", identification, "identificationModule.briefTitle"),
            ("overall_status", status, "statusModule.overallStatus")
        ]

        for field_name, section, path in required_fields:
            total_count += 1
            if field_name not in section or not section[field_name]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Trial Metadata",
                    message=f"Required field '{field_name}' is missing",
                    element_path=path,
                    suggested_fix=f"Add {field_name} to trial metadata"
                ))
            else:
                valid_count += 1

        # Validate NCT ID format
        nct_id = identification.get("nctId")
        if nct_id:
            total_count += 1
            if not re.match(r"^NCT\d{8}$", nct_id):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Trial Metadata",
                    message=f"NCT ID '{nct_id}' does not match expected format",
                    element_path="identificationModule.nctId",
                    suggested_fix="Ensure NCT ID follows format NCT########"
                ))
            else:
                valid_count += 1

        return issues, valid_count, total_count

    def _validate_eligibility_criteria(
        self, trial_data: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate trial eligibility criteria."""
        issues = []
        valid_count = 0
        total_count = 0

        protocol_section = trial_data.get("protocolSection", {})
        eligibility = protocol_section.get("eligibilityModule", {})

        # Check eligibility criteria text
        total_count += 1
        criteria_text = eligibility.get("eligibilityCriteria")
        if not criteria_text or len(criteria_text.strip()) < 50:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Eligibility Criteria",
                message="Eligibility criteria text is too short or missing",
                element_path="eligibilityModule.eligibilityCriteria",
                suggested_fix="Provide detailed eligibility criteria text"
            ))
        else:
            valid_count += 1

        # Check age criteria
        age_fields = ["minimumAge", "maximumAge"]
        for field in age_fields:
            total_count += 1
            if field in eligibility:
                age_value = eligibility[field]
                if isinstance(age_value, (int, float)) and age_value >= 0:
                    valid_count += 1
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Eligibility Criteria",
                        message=f"Invalid {field}: {age_value}",
                        element_path=f"eligibilityModule.{field}",
                        suggested_fix=f"Provide valid numeric age for {field}"
                    ))
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="Eligibility Criteria",
                    message=f"{field} not specified",
                    element_path=f"eligibilityModule.{field}",
                    suggested_fix=f"Consider specifying {field}"
                ))

        # Check sex criteria
        total_count += 1
        sex = eligibility.get("sex")
        if sex:
            valid_sexes = ["ALL", "FEMALE", "MALE"]
            if sex.upper() in valid_sexes:
                valid_count += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Eligibility Criteria",
                    message=f"Invalid sex criteria: {sex}",
                    element_path="eligibilityModule.sex",
                    suggested_fix=f"Use one of: {', '.join(valid_sexes)}"
                ))
        else:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="Eligibility Criteria",
                message="Sex criteria not specified",
                element_path="eligibilityModule.sex",
                suggested_fix="Specify sex eligibility criteria"
            ))

        return issues, valid_count, total_count

    def _validate_trial_coding_systems(
        self, mcode_elements: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate coding systems in trial mCODE elements."""
        issues = []
        valid_count = 0
        total_count = 0

        # Trial-specific coding validation
        trial_coding_systems = {
            "TrialCancerConditions": [CodingSystem.SNOMED_CT, CodingSystem.ICD10],
            "TrialInterventions": [CodingSystem.RXNORM, CodingSystem.SNOMED_CT]
        }

        for element_type, required_systems in trial_coding_systems.items():
            elements = mcode_elements.get(element_type, [])
            if isinstance(elements, dict):
                elements = [elements]

            for i, element in enumerate(elements):
                total_count += 1
                element_path = f"{element_type}[{i}]"

                if not isinstance(element, dict):
                    continue

                coding = element.get("coding", [])
                if isinstance(coding, dict):
                    coding = [coding]

                if not coding:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Trial Coding Systems",
                        message=f"{element_type} element {i} has no coding information",
                        element_path=f"{element_path}.coding",
                        suggested_fix=f"Add coding from required systems: {[s.value for s in required_systems]}"
                    ))
                    continue

                has_required_system = False
                for code_info in coding:
                    if isinstance(code_info, dict):
                        system = code_info.get("system")
                        if system in [s.value for s in required_systems]:
                            has_required_system = True
                            break

                if has_required_system:
                    valid_count += 1
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Trial Coding Systems",
                        message=f"{element_type} element {i} uses non-standard coding system",
                        element_path=f"{element_path}.coding[*].system",
                        suggested_fix=f"Use one of: {[s.value for s in required_systems]}"
                    ))

        return issues, valid_count, total_count

    def _validate_trial_mcode_compliance(
        self, mcode_elements: Dict[str, Any]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate trial mCODE compliance."""
        issues = []
        valid_count = 0
        total_count = 0

        # Check for trial-specific required elements
        required_trial_elements = ["TrialMetadata", "TrialCancerConditions"]
        for element in required_trial_elements:
            total_count += 1
            if element not in mcode_elements:
                severity = ValidationSeverity.CRITICAL if element == "TrialMetadata" else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    severity=severity,
                    category="Trial mCODE Compliance",
                    message=f"Required trial mCODE element '{element}' is missing",
                    element_path=element,
                    suggested_fix=f"Extract {element} from trial data"
                ))
            else:
                valid_count += 1

        # Validate trial metadata structure
        metadata = mcode_elements.get("TrialMetadata")
        if metadata:
            total_count += 1
            if isinstance(metadata, dict):
                required_meta_fields = ["nct_id", "brief_title", "overall_status"]
                meta_valid = all(field in metadata and metadata[field] for field in required_meta_fields)
                if meta_valid:
                    valid_count += 1
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Trial mCODE Compliance",
                        message="Trial metadata missing required fields",
                        element_path="TrialMetadata",
                        suggested_fix="Ensure TrialMetadata contains nct_id, brief_title, and overall_status"
                    ))
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Trial mCODE Compliance",
                    message="TrialMetadata is not a valid object",
                    element_path="TrialMetadata",
                    suggested_fix="Ensure TrialMetadata is a properly structured object"
                ))

        return issues, valid_count, total_count

    def _calculate_coding_system_coverage(
        self, mcode_elements: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate coverage percentage for each coding system."""
        coverage = {}

        # Collect all coding systems used
        used_systems = set()
        total_codings = 0

        def collect_systems(obj):
            nonlocal total_codings
            if isinstance(obj, dict):
                if "system" in obj and "code" in obj:
                    total_codings += 1
                    used_systems.add(obj["system"])
                for value in obj.values():
                    collect_systems(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect_systems(item)

        collect_systems(mcode_elements)

        # Calculate coverage for each required system
        for system in CodingSystem:
            system_url = system.value
            coverage[system.name] = (1.0 if system_url in used_systems else 0.0)

        return coverage

    def generate_quality_report_markdown(self, report: QualityReport) -> str:
        """Generate a detailed markdown quality report."""
        lines = [
            "# mCODE Data Quality Report",
            "",
            f"## Summary",
            f"- **Coverage**: {report.valid_elements}/{report.total_elements} elements valid ({report.coverage_percentage:.1f}%)",
            f"- **Completeness Score**: {report.completeness_score:.2f}",
            f"- **Can Proceed**: {'✅ Yes' if report.can_proceed else '❌ No'}",
            "",
            f"## Issues by Severity",
            f"- **Critical**: {report.critical_issues} (prevents processing)",
            f"- **Warnings**: {report.warning_issues}",
            f"- **Info**: {report.info_issues}",
            "",
            f"## Coding System Coverage",
        ]

        for system, coverage in report.coding_system_coverage.items():
            status = "✅" if coverage > 0 else "❌"
            lines.append(f"- {system}: {status} ({coverage:.0%})")

        if report.issues:
            lines.extend([
                "",
                "## Detailed Issues",
                "",
                "| Severity | Category | Message | Path | Suggestion |",
                "|----------|----------|---------|------|------------|",
            ])

            for issue in report.issues:
                severity_icon = {
                    ValidationSeverity.CRITICAL: "🚫",
                    ValidationSeverity.WARNING: "⚠️",
                    ValidationSeverity.INFO: "ℹ️"
                }.get(issue.severity, "")

                lines.append(
                    f"| {severity_icon} {issue.severity.value.title()} | {issue.category} | "
                    f"{issue.message} | {issue.element_path or 'N/A'} | {issue.suggested_fix or 'N/A'} |"
                )

        return "\n".join(lines)