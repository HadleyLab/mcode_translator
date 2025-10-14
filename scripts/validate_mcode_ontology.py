#!/usr/bin/env python3
"""
mCODE Ontology Completeness Validation Script

This script validates the completeness of mCODE ontology implementation against
sample breast cancer patient and trial data. It assesses which mCODE elements
are present vs missing, validates data against mCODE models, and generates
comprehensive completeness reports.

Usage:
    python scripts/validate_mcode_ontology.py

Requirements:
    - pydantic
    - json
    - pathlib
    - typing
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from shared.mcode_models import (
        McodePatient, CancerCondition, CancerStaging, TNMStageGroup,
        TumorMarkerTest, ECOGPerformanceStatusObservation,
        CancerRelatedMedicationStatement, CancerRelatedSurgicalProcedure,
        CancerRelatedRadiationProcedure, McodeValidator,
        AdministrativeGender, BirthSex, CancerConditionCode,
        TNMStageGroup as TNMStageGroupEnum, ECOGPerformanceStatus,
        ReceptorStatus, HistologyMorphologyBehavior
    )
except ImportError:
    # Fallback: define minimal classes for validation
    print("Warning: Could not import mCODE models, using fallback validation")

    class McodePatient:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CancerCondition:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CancerStaging:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class TNMStageGroup:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class TumorMarkerTest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class ECOGPerformanceStatusObservation:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class McodeValidator:
        pass

    # Enums
    class AdministrativeGender:
        MALE = "male"
        FEMALE = "female"
        OTHER = "other"
        UNKNOWN = "unknown"

    class BirthSex:
        M = "M"
        F = "F"
        UNK = "UNK"

    class CancerConditionCode:
        BREAST_CANCER = "254837009"

    class TNMStageGroupEnum:
        STAGE_0 = "0"
        STAGE_I = "I"
        STAGE_II = "II"
        STAGE_III = "III"
        STAGE_IV = "IV"

    class ECOGPerformanceStatus:
        FULLY_ACTIVE = "0"
        RESTRICTED_ACTIVITY = "1"
        UNABLE_HEAVY_WORK = "2"
        CAPABLE_ONLY_SELFCARE = "3"
        COMPLETELY_DISABLED = "4"
        DEAD = "5"

    class ReceptorStatus:
        POSITIVE = "positive"
        NEGATIVE = "negative"
        UNKNOWN = "unknown"
        NOT_PERFORMED = "not-performed"

    class HistologyMorphologyBehavior:
        MALIGNANT = "3"
        BENIGN = "2"
        UNCERTAIN = "1"


@dataclass
class ValidationResult:
    """Result of validating a single resource."""
    resource_type: str
    resource_id: str
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mcode_elements_present: Set[str] = field(default_factory=set)
    mcode_elements_missing: Set[str] = field(default_factory=set)


@dataclass
class CompletenessReport:
    """Comprehensive completeness report."""
    total_resources: int
    valid_resources: int
    invalid_resources: int
    validation_errors: List[str]
    validation_warnings: List[str]

    # mCODE element coverage
    mcode_elements_present: Dict[str, int]
    mcode_elements_missing: Dict[str, int]
    coverage_percentages: Dict[str, float]

    # Category coverage
    patient_demographics_coverage: float
    cancer_condition_coverage: float
    staging_coverage: float
    biomarkers_coverage: float
    procedures_coverage: float
    medications_coverage: float
    social_determinants_coverage: float

    # Trial-specific coverage
    trial_identification_coverage: float
    trial_characteristics_coverage: float
    trial_sponsors_coverage: float
    trial_design_coverage: float
    trial_eligibility_coverage: float
    trial_interventions_coverage: float
    trial_outcomes_coverage: float

    recommendations: List[str]


class McodeOntologyValidator:
    """Validator for mCODE ontology completeness."""

    # mCODE Patient elements
    PATIENT_ELEMENTS = {
        "Patient": "Core patient demographics",
        "BirthSex": "Birth sex extension",
        "USCoreRace": "US Core race extension",
        "USCoreEthnicity": "US Core ethnicity extension",
        "AdministrativeGender": "Administrative gender",
        "BirthDate": "Birth date",
        "Age": "Calculated age",
        "MaritalStatus": "Marital status",
        "Language": "Language preferences",
        "Address": "Address information"
    }

    # mCODE Cancer Condition elements
    CANCER_CONDITION_ELEMENTS = {
        "CancerCondition": "Primary cancer condition",
        "HistologyMorphologyBehavior": "Histology behavior extension",
        "Laterality": "Laterality extension",
        "RelatedCondition": "Related condition extension",
        "ClinicalStatus": "Clinical status",
        "VerificationStatus": "Verification status",
        "OnsetDateTime": "Diagnosis date",
        "BodySite": "Primary site",
        "Stage": "Cancer stage",
        "Evidence": "Supporting evidence"
    }

    # mCODE Staging elements
    STAGING_ELEMENTS = {
        "CancerStaging": "Cancer staging observation",
        "TNMStageGroup": "TNM stage group",
        "TNMPrimaryTumor": "TNM T category",
        "TNMRegionalNodes": "TNM N category",
        "TNMDistantMetastases": "TNM M category"
    }

    # mCODE Biomarker elements
    BIOMARKER_ELEMENTS = {
        "TumorMarkerTest": "Tumor marker test",
        "ERReceptorStatus": "Estrogen receptor status",
        "PRReceptorStatus": "Progesterone receptor status",
        "HER2ReceptorStatus": "HER2 receptor status",
        "ECOGPerformanceStatus": "ECOG performance status"
    }

    # mCODE Procedure elements
    PROCEDURE_ELEMENTS = {
        "CancerRelatedSurgicalProcedure": "Cancer-related surgery",
        "CancerRelatedRadiationProcedure": "Cancer-related radiation",
        "CancerRelatedBiopsyProcedure": "Cancer-related biopsy"
    }

    # mCODE Medication elements
    MEDICATION_ELEMENTS = {
        "CancerRelatedMedicationStatement": "Cancer-related medication",
        "ChemotherapyRegimen": "Chemotherapy regimen",
        "HormoneTherapy": "Hormone therapy",
        "TargetedTherapy": "Targeted therapy"
    }

    # mCODE Trial elements
    TRIAL_ELEMENTS = {
        "Trial": "Clinical trial",
        "TrialTitle": "Trial title",
        "TrialStatus": "Trial status",
        "TrialPhase": "Trial phase",
        "TrialType": "Trial type",
        "TrialPurpose": "Trial purpose",
        "TrialAllocation": "Trial allocation",
        "TrialMasking": "Trial masking",
        "TrialSponsor": "Trial sponsor",
        "TrialResponsibleParty": "Trial responsible party",
        "TrialCondition": "Trial condition",
        "TrialIntervention": "Trial intervention",
        "TrialEligibility": "Trial eligibility criteria",
        "TrialOutcome": "Trial outcome measures",
        "TrialArm": "Trial arm/group",
        "TrialEnrollment": "Trial enrollment info"
    }

    def __init__(self):
        self.validator = McodeValidator()

    def load_ndjson_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load NDJSON file and return list of records."""
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
        return records

    def validate_patient_resource(self, resource: Dict[str, Any]) -> ValidationResult:
        """Validate a patient resource against mCODE models."""
        result = ValidationResult(
            resource_type="Patient",
            resource_id=resource.get("id", "unknown"),
            valid=False
        )

        try:
            # Try to create mCODE Patient
            patient = McodePatient(**resource)

            # Check for extensions
            extensions = resource.get("extension", [])
            if any(isinstance(ext, dict) and
                   ext.get("url") == "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-birth-sex"
                   for ext in extensions):
                result.mcode_elements_present.add("BirthSex")

            if any(isinstance(ext, dict) and
                   ext.get("url") == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
                   for ext in extensions):
                result.mcode_elements_present.add("USCoreRace")

            if any(isinstance(ext, dict) and
                   ext.get("url") == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
                   for ext in extensions):
                result.mcode_elements_present.add("USCoreEthnicity")

            # Check basic demographics
            if patient.gender:
                result.mcode_elements_present.add("AdministrativeGender")
            if patient.birthDate:
                result.mcode_elements_present.add("BirthDate")
            if patient.name:
                result.mcode_elements_present.add("Patient")
            if patient.address:
                result.mcode_elements_present.add("Address")
            if patient.maritalStatus:
                result.mcode_elements_present.add("MaritalStatus")
            if patient.communication:
                result.mcode_elements_present.add("Language")

            result.valid = True

        except Exception as e:
            result.valid = False
            result.errors.append(f"Patient validation failed: {str(e)}")

        # Determine missing elements
        all_patient_elements = set(self.PATIENT_ELEMENTS.keys())
        result.mcode_elements_missing = all_patient_elements - result.mcode_elements_present

        return result

    def validate_condition_resource(self, resource: Dict[str, Any]) -> ValidationResult:
        """Validate a condition resource against mCODE CancerCondition model."""
        result = ValidationResult(
            resource_type="Condition",
            resource_id=resource.get("id", "unknown"),
            valid=False
        )

        try:
            # Try to create CancerCondition
            condition = CancerCondition(**resource)
            result.mcode_elements_present.add("CancerCondition")

            # Check for extensions
            extensions = resource.get("extension", [])
            if any(isinstance(ext, dict) and
                   ext.get("url") == "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-histology-morphology-behavior"
                   for ext in extensions):
                result.mcode_elements_present.add("HistologyMorphologyBehavior")

            if any(isinstance(ext, dict) and
                   ext.get("url") == "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-laterality"
                   for ext in extensions):
                result.mcode_elements_present.add("Laterality")

            if any(isinstance(ext, dict) and
                   ext.get("url") == "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-related-condition"
                   for ext in extensions):
                result.mcode_elements_present.add("RelatedCondition")

            # Check other fields
            if condition.clinicalStatus:
                result.mcode_elements_present.add("ClinicalStatus")
            if condition.verificationStatus:
                result.mcode_elements_present.add("VerificationStatus")
            if condition.onsetDateTime:
                result.mcode_elements_present.add("OnsetDateTime")
            if condition.bodySite:
                result.mcode_elements_present.add("BodySite")
            if condition.stage:
                result.mcode_elements_present.add("Stage")
            if condition.evidence:
                result.mcode_elements_present.add("Evidence")

            result.valid = True

        except Exception as e:
            result.valid = False
            result.errors.append(f"CancerCondition validation failed: {str(e)}")

        # Determine missing elements
        all_condition_elements = set(self.CANCER_CONDITION_ELEMENTS.keys())
        result.mcode_elements_missing = all_condition_elements - result.mcode_elements_present

        return result

    def validate_observation_resource(self, resource: Dict[str, Any]) -> ValidationResult:
        """Validate an observation resource against various mCODE observation models."""
        result = ValidationResult(
            resource_type="Observation",
            resource_id=resource.get("id", "unknown"),
            valid=False
        )

        # Try different observation types
        observation_types = [
            ("CancerStaging", CancerStaging),
            ("TNMStageGroup", TNMStageGroup),
            ("TumorMarkerTest", TumorMarkerTest),
            ("ECOGPerformanceStatusObservation", ECOGPerformanceStatusObservation)
        ]

        for obs_type, model_class in observation_types:
            try:
                observation = model_class(**resource)
                result.mcode_elements_present.add(obs_type)
                result.valid = True
                break
            except Exception:
                continue

        if not result.valid:
            result.errors.append("Observation does not match any mCODE observation profile")

        # Determine missing elements based on what we found
        if "CancerStaging" in result.mcode_elements_present:
            result.mcode_elements_missing = set(self.STAGING_ELEMENTS.keys()) - result.mcode_elements_present
        elif "TumorMarkerTest" in result.mcode_elements_present:
            result.mcode_elements_missing = set(self.BIOMARKER_ELEMENTS.keys()) - result.mcode_elements_present

        return result

    def validate_trial_data(self, trial_data: Dict[str, Any]) -> ValidationResult:
        """Validate trial data against mCODE trial elements."""
        result = ValidationResult(
            resource_type="ResearchStudy",
            resource_id=trial_data.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "unknown"),
            valid=False
        )

        # Extract mCODE mappings from McodeResults if present
        mcode_results = trial_data.get("McodeResults", {})
        mappings = mcode_results.get("mcode_mappings", [])

        for mapping in mappings:
            element_type = mapping.get("element_type")
            if element_type:
                result.mcode_elements_present.add(element_type)

        # Check protocol section for additional elements
        protocol = trial_data.get("protocolSection", {})

        # Identification
        if protocol.get("identificationModule"):
            ident = protocol["identificationModule"]
            if ident.get("nctId"):
                result.mcode_elements_present.add("Trial")
            if ident.get("briefTitle"):
                result.mcode_elements_present.add("TrialTitle")

        # Status
        if protocol.get("statusModule"):
            status = protocol["statusModule"]
            if status.get("overallStatus"):
                result.mcode_elements_present.add("TrialStatus")

        # Design
        if protocol.get("designModule"):
            design = protocol["designModule"]
            if design.get("phases"):
                result.mcode_elements_present.add("TrialPhase")
            if design.get("studyType"):
                result.mcode_elements_present.add("TrialType")
            if design.get("designInfo"):
                design_info = design["designInfo"]
                if design_info.get("primaryPurpose"):
                    result.mcode_elements_present.add("TrialPurpose")
                if design_info.get("allocation"):
                    result.mcode_elements_present.add("TrialAllocation")
                if design_info.get("maskingInfo"):
                    result.mcode_elements_present.add("TrialMasking")

        # Sponsor
        if protocol.get("sponsorCollaboratorsModule"):
            sponsor = protocol["sponsorCollaboratorsModule"]
            if sponsor.get("leadSponsor"):
                result.mcode_elements_present.add("TrialSponsor")
            if sponsor.get("responsibleParty"):
                result.mcode_elements_present.add("TrialResponsibleParty")

        # Conditions
        if protocol.get("conditionsModule"):
            conditions = protocol["conditionsModule"]
            if conditions.get("conditions"):
                result.mcode_elements_present.add("TrialCondition")

        # Arms/Interventions
        if protocol.get("armsInterventionsModule"):
            arms = protocol["armsInterventionsModule"]
            if arms.get("armGroups"):
                result.mcode_elements_present.add("TrialArm")
            if arms.get("interventions"):
                result.mcode_elements_present.add("TrialIntervention")

        # Eligibility
        if protocol.get("eligibilityModule"):
            eligibility = protocol["eligibilityModule"]
            if eligibility.get("eligibilityCriteria"):
                result.mcode_elements_present.add("TrialEligibility")

        # Outcomes
        if protocol.get("outcomesModule"):
            outcomes = protocol["outcomesModule"]
            if outcomes.get("primaryOutcomes") or outcomes.get("secondaryOutcomes"):
                result.mcode_elements_present.add("TrialOutcome")

        # Enrollment
        if protocol.get("designModule", {}).get("enrollmentInfo"):
            enrollment = protocol["designModule"]["enrollmentInfo"]
            if enrollment.get("count"):
                result.mcode_elements_present.add("TrialEnrollment")

        result.valid = True

        # Determine missing elements
        all_trial_elements = set(self.TRIAL_ELEMENTS.keys())
        result.mcode_elements_missing = all_trial_elements - result.mcode_elements_present

        return result

    def validate_bundle(self, bundle: Dict[str, Any]) -> List[ValidationResult]:
        """Validate all resources in a FHIR Bundle."""
        results = []

        entries = bundle.get("entry", [])
        for entry in entries:
            resource = entry.get("resource", {})
            resource_type = resource.get("resourceType")

            if resource_type == "Patient":
                result = self.validate_patient_resource(resource)
            elif resource_type == "Condition":
                result = self.validate_condition_resource(resource)
            elif resource_type == "Observation":
                result = self.validate_observation_resource(resource)
            else:
                # Unknown resource type
                result = ValidationResult(
                    resource_type=resource_type or "Unknown",
                    resource_id=resource.get("id", "unknown"),
                    valid=False,
                    errors=[f"Unknown resource type: {resource_type}"]
                )

            results.append(result)

        return results

    def calculate_coverage_percentages(self, elements_present: Dict[str, int],
                                    elements_missing: Dict[str, int]) -> Dict[str, float]:
        """Calculate coverage percentages for different categories."""
        percentages = {}

        # Patient demographics
        patient_elements = set(self.PATIENT_ELEMENTS.keys())
        patient_present = sum(elements_present.get(elem, 0) for elem in patient_elements)
        patient_total = len(patient_elements)
        percentages["patient_demographics"] = (patient_present / patient_total) * 100 if patient_total > 0 else 0

        # Cancer condition
        condition_elements = set(self.CANCER_CONDITION_ELEMENTS.keys())
        condition_present = sum(elements_present.get(elem, 0) for elem in condition_elements)
        condition_total = len(condition_elements)
        percentages["cancer_condition"] = (condition_present / condition_total) * 100 if condition_total > 0 else 0

        # Staging
        staging_elements = set(self.STAGING_ELEMENTS.keys())
        staging_present = sum(elements_present.get(elem, 0) for elem in staging_elements)
        staging_total = len(staging_elements)
        percentages["staging"] = (staging_present / staging_total) * 100 if staging_total > 0 else 0

        # Biomarkers
        biomarker_elements = set(self.BIOMARKER_ELEMENTS.keys())
        biomarker_present = sum(elements_present.get(elem, 0) for elem in biomarker_elements)
        biomarker_total = len(biomarker_elements)
        percentages["biomarkers"] = (biomarker_present / biomarker_total) * 100 if biomarker_total > 0 else 0

        # Procedures
        procedure_elements = set(self.PROCEDURE_ELEMENTS.keys())
        procedure_present = sum(elements_present.get(elem, 0) for elem in procedure_elements)
        procedure_total = len(procedure_elements)
        percentages["procedures"] = (procedure_present / procedure_total) * 100 if procedure_total > 0 else 0

        # Medications
        medication_elements = set(self.MEDICATION_ELEMENTS.keys())
        medication_present = sum(elements_present.get(elem, 0) for elem in medication_elements)
        medication_total = len(medication_elements)
        percentages["medications"] = (medication_present / medication_total) * 100 if medication_total > 0 else 0

        # Trial elements
        trial_elements = set(self.TRIAL_ELEMENTS.keys())
        trial_present = sum(elements_present.get(elem, 0) for elem in trial_elements)
        trial_total = len(trial_elements)
        percentages["trial_identification"] = (trial_present / trial_total) * 100 if trial_total > 0 else 0

        return percentages

    def generate_recommendations(self, report: CompletenessReport) -> List[str]:
        """Generate recommendations based on the completeness report."""
        recommendations = []

        # Patient recommendations
        if report.patient_demographics_coverage < 50:
            recommendations.append("Patient demographics coverage is low. Consider adding birth sex, race, ethnicity, and address extensions.")

        if report.cancer_condition_coverage < 50:
            recommendations.append("Cancer condition coverage is low. Add histology morphology behavior, laterality, and staging information.")

        if report.staging_coverage < 30:
            recommendations.append("Staging information is minimal. Implement TNM staging observations for better clinical decision support.")

        if report.biomarkers_coverage < 40:
            recommendations.append("Biomarker testing coverage is low. Add receptor status (ER/PR/HER2) and performance status observations.")

        if report.procedures_coverage < 20:
            recommendations.append("Procedure coverage is very low. Implement cancer-related surgical, radiation, and biopsy procedures.")

        if report.medications_coverage < 20:
            recommendations.append("Medication coverage is very low. Add cancer-related medication statements and treatment regimens.")

        if report.trial_identification_coverage < 60:
            recommendations.append("Trial identification coverage is moderate. Ensure all trial metadata (phases, design, sponsors) is captured.")

        # General recommendations
        if len(report.validation_errors) > 0:
            recommendations.append(f"Fix {len(report.validation_errors)} validation errors to improve data quality.")

        if report.valid_resources / report.total_resources < 0.8:
            recommendations.append("Resource validation rate is below 80%. Review data mapping and model compliance.")

        return recommendations

    def validate_ontology_completeness(self, patient_file: Path, trial_file: Path) -> CompletenessReport:
        """Main method to validate ontology completeness against sample data."""
        print("Loading patient data...")
        patient_records = self.load_ndjson_file(patient_file)
        print(f"Loaded {len(patient_records)} patient records")

        print("Loading trial data...")
        trial_records = self.load_ndjson_file(trial_file)
        print(f"Loaded {len(trial_records)} trial records")

        all_results = []
        mcode_elements_present = defaultdict(int)
        mcode_elements_missing = defaultdict(int)

        # Validate patient data
        for record in patient_records:
            if "entry" in record:  # FHIR Bundle
                bundle_results = self.validate_bundle(record)
                all_results.extend(bundle_results)
            elif "McodeResults" in record:  # Direct mCODE results
                # Handle summarized format
                filtered_elements = record.get("filtered_mcode_elements", {})
                for key in filtered_elements.keys():
                    if key != "CancerCondition":  # Skip nested
                        mcode_elements_present[key] += 1

                cancer_condition = filtered_elements.get("CancerCondition", {})
                for key in cancer_condition.keys():
                    mcode_elements_present[key] += 1

        # Validate trial data
        for record in trial_records:
            trial_result = self.validate_trial_data(record)
            all_results.append(trial_result)

            # Count elements from trial validation
            for element in trial_result.mcode_elements_present:
                mcode_elements_present[element] += 1
            for element in trial_result.mcode_elements_missing:
                mcode_elements_missing[element] += 1

        # Count elements from resource validations
        for result in all_results:
            for element in result.mcode_elements_present:
                mcode_elements_present[element] += 1
            for element in result.mcode_elements_missing:
                mcode_elements_missing[element] += 1

        # Calculate totals
        total_resources = len(all_results)
        valid_resources = sum(1 for r in all_results if r.valid)
        invalid_resources = total_resources - valid_resources

        validation_errors = []
        validation_warnings = []
        for result in all_results:
            validation_errors.extend(result.errors)
            validation_warnings.extend(result.warnings)

        # Calculate coverage percentages
        coverage_percentages = self.calculate_coverage_percentages(
            dict(mcode_elements_present), dict(mcode_elements_missing)
        )

        # Create report
        report = CompletenessReport(
            total_resources=total_resources,
            valid_resources=valid_resources,
            invalid_resources=invalid_resources,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            mcode_elements_present=dict(mcode_elements_present),
            mcode_elements_missing=dict(mcode_elements_missing),
            coverage_percentages=coverage_percentages,
            patient_demographics_coverage=coverage_percentages.get("patient_demographics", 0),
            cancer_condition_coverage=coverage_percentages.get("cancer_condition", 0),
            staging_coverage=coverage_percentages.get("staging", 0),
            biomarkers_coverage=coverage_percentages.get("biomarkers", 0),
            procedures_coverage=coverage_percentages.get("procedures", 0),
            medications_coverage=coverage_percentages.get("medications", 0),
            social_determinants_coverage=0.0,  # Not implemented yet
            trial_identification_coverage=coverage_percentages.get("trial_identification", 0),
            trial_characteristics_coverage=0.0,  # Not implemented yet
            trial_sponsors_coverage=0.0,  # Not implemented yet
            trial_design_coverage=0.0,  # Not implemented yet
            trial_eligibility_coverage=0.0,  # Not implemented yet
            trial_interventions_coverage=0.0,  # Not implemented yet
            trial_outcomes_coverage=0.0,  # Not implemented yet
            recommendations=[]
        )

        # Generate recommendations
        report.recommendations = self.generate_recommendations(report)

        return report

    def print_report(self, report: CompletenessReport):
        """Print the completeness report in a readable format."""
        print("\n" + "="*80)
        print("mCODE ONTOLOGY COMPLETENESS REPORT")
        print("="*80)

        print("\nRESOURCE VALIDATION SUMMARY:")
        print(f"  Total Resources: {report.total_resources}")
        print(f"  Valid Resources: {report.valid_resources} ({report.valid_resources/report.total_resources*100:.1f}%)")
        print(f"  Invalid Resources: {report.invalid_resources} ({report.invalid_resources/report.total_resources*100:.1f}%)")

        print("\nVALIDATION ISSUES:")
        print(f"  Errors: {len(report.validation_errors)}")
        for error in report.validation_errors[:5]:  # Show first 5
            print(f"    - {error}")
        if len(report.validation_errors) > 5:
            print(f"    ... and {len(report.validation_errors) - 5} more")

        print(f"  Warnings: {len(report.validation_warnings)}")
        for warning in report.validation_warnings[:5]:  # Show first 5
            print(f"    - {warning}")
        if len(report.validation_warnings) > 5:
            print(f"    ... and {len(report.validation_warnings) - 5} more")

        print("\nmCODE ELEMENT COVERAGE:")

        # Patient elements
        print("  PATIENT DEMOGRAPHICS:")
        patient_elements = {k: v for k, v in report.mcode_elements_present.items()
                          if k in self.PATIENT_ELEMENTS}
        for element, count in sorted(patient_elements.items()):
            description = self.PATIENT_ELEMENTS.get(element, "Unknown")
            print(f"    ✓ {element}: {description} ({count} instances)")

        missing_patient = {k for k in self.PATIENT_ELEMENTS.keys()
                          if k not in patient_elements}
        for element in sorted(missing_patient):
            description = self.PATIENT_ELEMENTS[element]
            print(f"    ✗ {element}: {description}")

        # Cancer condition elements
        print("\n  CANCER CONDITION:")
        condition_elements = {k: v for k, v in report.mcode_elements_present.items()
                            if k in self.CANCER_CONDITION_ELEMENTS}
        for element, count in sorted(condition_elements.items()):
            description = self.CANCER_CONDITION_ELEMENTS.get(element, "Unknown")
            print(f"    ✓ {element}: {description} ({count} instances)")

        missing_condition = {k for k in self.CANCER_CONDITION_ELEMENTS.keys()
                           if k not in condition_elements}
        for element in sorted(missing_condition):
            description = self.CANCER_CONDITION_ELEMENTS[element]
            print(f"    ✗ {element}: {description}")

        # Trial elements
        print("\n  CLINICAL TRIAL:")
        trial_elements = {k: v for k, v in report.mcode_elements_present.items()
                        if k in self.TRIAL_ELEMENTS}
        for element, count in sorted(trial_elements.items()):
            description = self.TRIAL_ELEMENTS.get(element, "Unknown")
            print(f"    ✓ {element}: {description} ({count} instances)")

        missing_trial = {k for k in self.TRIAL_ELEMENTS.keys()
                       if k not in trial_elements}
        for element in sorted(missing_trial):
            description = self.TRIAL_ELEMENTS[element]
            print(f"    ✗ {element}: {description}")

        print("\nCOVERAGE PERCENTAGES:")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")
        print(".1f")

        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

        print("\n" + "="*80)


def main():
    """Main entry point."""
    # File paths
    patient_file = Path("data/summarized_breast_cancer_patient.ndjson")
    trial_file = Path("data/summarized_breast_cancer_trial.ndjson")

    # Check if files exist
    if not patient_file.exists():
        print(f"Error: Patient data file not found: {patient_file}")
        return 1

    if not trial_file.exists():
        print(f"Error: Trial data file not found: {trial_file}")
        return 1

    # Create validator and run validation
    validator = McodeOntologyValidator()
    report = validator.validate_ontology_completeness(patient_file, trial_file)

    # Print report
    validator.print_report(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())