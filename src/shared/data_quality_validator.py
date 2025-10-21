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

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .mcode_models import McodeValidator
from .mcode_versioning import McodeVersion
from .models import ClinicalTrialData, McodeElement, PatientData


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
        patient_data: "PatientData",
        mcode_elements: List["McodeElement"]
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
        trial_data: ClinicalTrialData,
        mcode_elements: List[McodeElement]
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
        self, mcode_elements: List[McodeElement]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate cancer condition elements using McodeElement instances."""
        issues = []
        valid_count = 0
        total_count = 0

        # Extract cancer conditions from McodeElement list
        cancer_conditions = [elem for elem in mcode_elements if elem.element_type == "CancerCondition"]

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

            # Validate McodeElement has required fields
            if condition.code:
                valid_count += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Cancer Conditions",
                    message=f"Cancer condition {i} missing code",
                    element_path=f"{condition_path}.code",
                    suggested_fix="Add cancer diagnosis code"
                ))

        return issues, valid_count, total_count

    def _validate_coding_systems(
        self, mcode_elements: List[McodeElement]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate coding systems usage in McodeElement instances."""
        issues = []
        valid_count = 0
        total_count = 0

        # Group elements by type for validation
        elements_by_type: Dict[str, List[McodeElement]] = {}
        for element in mcode_elements:
            if element.element_type not in elements_by_type:
                elements_by_type[element.element_type] = []
            elements_by_type[element.element_type].append(element)

        # Check each element type for proper coding systems
        for element_type, required_systems in self.required_coding_systems.items():
            elements = elements_by_type.get(element_type, [])

            for i, element in enumerate(elements):
                total_count += 1
                element_path = f"{element_type}[{i}]"

                # Validate McodeElement has proper coding
                if element.system and element.code:
                    system_url = element.system
                    if system_url in [s.value for s in required_systems]:
                        valid_count += 1
                    else:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="Coding Systems",
                            message=f"{element_type} element {i} uses non-standard coding system: {system_url}",
                            element_path=f"{element_path}.system",
                            suggested_fix=f"Use one of: {[s.value for s in required_systems]}"
                        ))
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Coding Systems",
                        message=f"{element_type} element {i} has no coding information",
                        element_path=f"{element_path}.system",
                        suggested_fix=f"Add system and code fields with values from: {[s.value for s in required_systems]}"
                    ))

        return issues, valid_count, total_count

    def _validate_fhir_structure(
        self, patient_data: "PatientData"
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate FHIR resource structure using PatientData model."""
        issues = []
        valid_count = 0
        total_count = 0

        # Check bundle structure
        total_count += 1
        if not patient_data.bundle:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="FHIR Structure",
                message="Patient data bundle is missing",
                element_path="bundle",
                suggested_fix="Ensure patient data contains a valid FHIR Bundle"
            ))
            return issues, 0, total_count

        valid_count += 1

        # Check bundle has resourceType
        total_count += 1
        if patient_data.bundle.resourceType != "Bundle":
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="FHIR Structure",
                message="Bundle resourceType is not 'Bundle'",
                element_path="bundle.resourceType",
                suggested_fix="Ensure bundle has resourceType 'Bundle'"
            ))
        else:
            valid_count += 1

        # Check entries
        total_count += 1
        if not patient_data.bundle.entry:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="FHIR Structure",
                message="Bundle has no entries",
                element_path="bundle.entry",
                suggested_fix="Ensure bundle contains FHIR resources"
            ))
        else:
            valid_count += 1

            # Validate each entry
            for i, entry in enumerate(patient_data.bundle.entry):
                total_count += 1
                entry_path = f"bundle.entry[{i}]"

                if not entry.resource:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="FHIR Structure",
                        message=f"Entry {i} missing resource",
                        element_path=f"{entry_path}.resource",
                        suggested_fix="Add resource field to entry"
                    ))
                    continue

                if "resourceType" not in entry.resource:
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
        self, mcode_elements: List[McodeElement]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate mCODE profile compliance using McodeElement instances."""
        issues = []
        valid_count = 0
        total_count = 0

        # Check for required mCODE elements
        required_elements = ["Patient", "CancerCondition"]
        element_types = {elem.element_type for elem in mcode_elements}

        for element in required_elements:
            total_count += 1
            if element in element_types:
                valid_count += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="mCODE Compliance",
                    message=f"Required mCODE element '{element}' is missing",
                    element_path=element,
                    suggested_fix=f"Extract {element} from patient data"
                ))

        # Validate element structure
        for element in mcode_elements:
            total_count += 1
            if element.system and element.code:
                valid_count += 1
            elif element.display:
                valid_count += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="mCODE Compliance",
                    message="mCODE element lacks proper coding structure",
                    element_path=f"{element.element_type}",
                    suggested_fix="Ensure element has system, code, and/or display fields"
                ))

        return issues, valid_count, total_count

    def _validate_trial_metadata(
        self, trial_data: ClinicalTrialData
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate trial metadata completeness using ClinicalTrialData model."""
        issues = []
        valid_count = 0
        total_count = 0

        # Extract modules from ClinicalTrialData model
        identification = trial_data.protocol_section.identification_module
        status_module = trial_data.protocol_section.status_module

        # Validate identification module
        if not identification:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="Trial Metadata",
                message="Identification module is missing",
                element_path="protocolSection.identificationModule",
                suggested_fix="Ensure trial has identification module with NCT ID and title"
            ))
            return issues, 0, 1

        total_count += 1
        valid_count += 1  # We have an identification module

        # Check required identification fields
        required_id_fields = [
            ("nct_id", "nctId", "ClinicalTrials.gov identifier"),
            ("brief_title", "briefTitle", "Brief trial title")
        ]

        for field_name, alias, description in required_id_fields:
            total_count += 1
            field_value = getattr(identification, field_name, None)
            if not field_value:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="Trial Metadata",
                    message=f"Required field '{description}' is missing",
                    element_path=f"identificationModule.{alias}",
                    suggested_fix=f"Add {field_name} to trial identification module"
                ))
            else:
                valid_count += 1

        # Validate NCT ID format
        if identification.nct_id:
            total_count += 1
            if not re.match(r"^NCT\d{8}$", identification.nct_id):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Trial Metadata",
                    message=f"NCT ID '{identification.nct_id}' does not match expected format",
                    element_path="identificationModule.nctId",
                    suggested_fix="Ensure NCT ID follows format NCT########"
                ))
            else:
                valid_count += 1

        # Validate status module if available
        if status_module and isinstance(status_module, dict) and "overallStatus" in status_module:
            total_count += 1
            if status_module["overallStatus"]:
                valid_count += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Trial Metadata",
                    message="Overall status is missing from status module",
                    element_path="statusModule.overallStatus",
                    suggested_fix="Add overall status to trial status module"
                ))

        return issues, valid_count, total_count

    def _validate_eligibility_criteria(
        self, trial_data: ClinicalTrialData
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate trial eligibility criteria using ClinicalTrialData model."""
        issues = []
        valid_count = 0
        total_count = 0

        # Extract eligibility module from ClinicalTrialData model
        eligibility_module = trial_data.protocol_section.eligibility_module

        # Check if eligibility module exists
        if not eligibility_module:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Eligibility Criteria",
                message="Eligibility module is missing",
                element_path="protocolSection.eligibilityModule",
                suggested_fix="Ensure trial has eligibility module with criteria"
            ))
            return issues, 0, 1

        total_count += 1
        valid_count += 1  # We have an eligibility module

        # Check eligibility criteria text
        total_count += 1
        if eligibility_module.eligibility_criteria:
            if len(eligibility_module.eligibility_criteria.strip()) >= 50:
                valid_count += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Eligibility Criteria",
                    message="Eligibility criteria text is too short",
                    element_path="eligibilityModule.eligibilityCriteria",
                    suggested_fix="Provide detailed eligibility criteria text (at least 50 characters)"
                ))
        else:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Eligibility Criteria",
                message="Eligibility criteria text is missing",
                element_path="eligibilityModule.eligibilityCriteria",
                suggested_fix="Provide detailed eligibility criteria text"
            ))

        # Check age criteria
        age_fields = [
            ("minimum_age", "minimumAge", "Minimum age"),
            ("maximum_age", "maximumAge", "Maximum age")
        ]

        for field_name, alias, description in age_fields:
            total_count += 1
            age_value = getattr(eligibility_module, field_name, None)
            if age_value:
                # Try to parse as number for validation
                try:
                    numeric_age = float(str(age_value).strip())
                    if numeric_age >= 0:
                        valid_count += 1
                    else:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="Eligibility Criteria",
                            message=f"Invalid {description}: {age_value}",
                            element_path=f"eligibilityModule.{alias}",
                            suggested_fix=f"Provide valid non-negative age for {field_name}"
                        ))
                except (ValueError, TypeError):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Eligibility Criteria",
                        message=f"Invalid {description} format: {age_value}",
                        element_path=f"eligibilityModule.{alias}",
                        suggested_fix=f"Provide valid numeric age for {field_name}"
                    ))
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="Eligibility Criteria",
                    message=f"{description} not specified",
                    element_path=f"eligibilityModule.{alias}",
                    suggested_fix=f"Consider specifying {field_name}"
                ))

        # Check sex criteria
        total_count += 1
        if eligibility_module.sex:
            valid_sexes = ["ALL", "FEMALE", "MALE"]
            if eligibility_module.sex.upper() in valid_sexes:
                valid_count += 1
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Eligibility Criteria",
                    message=f"Invalid sex criteria: {eligibility_module.sex}",
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
        self, mcode_elements: List[McodeElement]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate coding systems in trial mCODE elements using McodeElement instances."""
        issues = []
        valid_count = 0
        total_count = 0

        # Trial-specific coding validation
        trial_coding_systems = {
            "TrialCancerConditions": [CodingSystem.SNOMED_CT, CodingSystem.ICD10],
            "TrialInterventions": [CodingSystem.RXNORM, CodingSystem.SNOMED_CT]
        }

        # Group elements by type for validation
        elements_by_type: Dict[str, List[McodeElement]] = {}
        for element in mcode_elements:
            element_type = element.element_type
            if element_type not in elements_by_type:
                elements_by_type[element_type] = []
            elements_by_type[element_type].append(element)

        for element_type, required_systems in trial_coding_systems.items():
            elements = elements_by_type.get(element_type, [])

            for i, element in enumerate(elements):
                total_count += 1
                element_path = f"{element_type}[{i}]"

                # Validate McodeElement has proper coding
                if element.system and element.code:
                    system_url = element.system
                    if system_url in [s.value for s in required_systems]:
                        valid_count += 1
                    else:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="Trial Coding Systems",
                            message=f"{element_type} element {i} uses non-standard coding system: {system_url}",
                            element_path=f"{element_path}.system",
                            suggested_fix=f"Use one of: {[s.value for s in required_systems]}"
                        ))
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="Trial Coding Systems",
                        message=f"{element_type} element {i} missing coding information",
                        element_path=f"{element_path}.system",
                        suggested_fix=f"Add system and code fields with values from: {[s.value for s in required_systems]}"
                    ))

        return issues, valid_count, total_count

    def _validate_trial_mcode_compliance(
        self, mcode_elements: List[McodeElement]
    ) -> Tuple[List[ValidationIssue], int, int]:
        """Validate trial mCODE compliance using McodeElement instances."""
        issues = []
        valid_count = 0
        total_count = 0

        # Check for trial-specific required elements
        element_types = {elem.element_type for elem in mcode_elements}

        required_trial_elements = ["TrialMetadata", "TrialCancerConditions"]
        for element in required_trial_elements:
            total_count += 1
            if element in element_types:
                valid_count += 1
            else:
                severity = ValidationSeverity.CRITICAL if element == "TrialMetadata" else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    severity=severity,
                    category="Trial mCODE Compliance",
                    message=f"Required trial mCODE element '{element}' is missing",
                    element_path=element,
                    suggested_fix=f"Extract {element} from trial data"
                ))

        # Validate trial metadata structure
        metadata_elements = [elem for elem in mcode_elements if elem.element_type == "TrialMetadata"]
        if metadata_elements:
            total_count += 1
            # For now, assume metadata is valid if present (could add more detailed validation)
            valid_count += 1

        return issues, valid_count, total_count

    def _calculate_coding_system_coverage(
        self, mcode_elements: List[McodeElement]
    ) -> Dict[str, float]:
        """Calculate coverage percentage for each coding system using McodeElement instances."""
        coverage = {}

        # Collect all coding systems used
        used_systems = set()
        for element in mcode_elements:
            if element.system and element.code:
                used_systems.add(element.system)

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
            "## Summary",
            f"- **Coverage**: {report.valid_elements}/{report.total_elements} elements valid ({report.coverage_percentage:.1f}%)",
            f"- **Completeness Score**: {report.completeness_score:.2f}",
            f"- **Can Proceed**: {'✅ Yes' if report.can_proceed else '❌ No'}",
            "",
            "## Issues by Severity",
            f"- **Critical**: {report.critical_issues} (prevents processing)",
            f"- **Warnings**: {report.warning_issues}",
            f"- **Info**: {report.info_issues}",
            "",
            "## Coding System Coverage",
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
