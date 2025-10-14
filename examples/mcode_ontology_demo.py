#!/usr/bin/env python3
"""
mCODE Ontology Usage Examples

This script demonstrates comprehensive usage of the mCODE (Minimal Common Oncology Data Elements)
ontology implementation in the mCODE Translator. It shows how to:

1. Create mCODE-compliant patient and condition resources
2. Use versioning and validation features
3. Generate FHIR bundles from mCODE data
4. Validate ontology completeness
5. Integrate with the processing pipeline

Requirements:
    - pydantic
    - The mCODE Translator codebase
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from shared.mcode_models import (
        McodePatient, CancerCondition, TNMStageGroup, TumorMarkerTest,
        ECOGPerformanceStatusObservation, CancerRelatedMedicationStatement,
        CancerRelatedSurgicalProcedure, AdministrativeGender, BirthSex,
        CancerConditionCode, TNMStageGroup as TNMStageGroupEnum,
        ECOGPerformanceStatus, ReceptorStatus, HistologyMorphologyBehavior,
        BirthSexExtension, USCoreRaceExtension, USCoreEthnicityExtension,
        HistologyMorphologyBehaviorExtension, LateralityExtension,
        FHIRCodeableConcept, FHIRReference, create_mcode_patient, create_cancer_condition
    )
    from shared.mcode_versioning import McodeVersionManager, McodeVersion, McodeProfile, get_profile_url
    from shared.mcode_models import McodeValidator
    from scripts.validate_mcode_ontology import McodeOntologyValidator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the mCODE Translator root directory")
    sys.exit(1)


def example_1_create_mcode_patient():
    """
    Example 1: Create a comprehensive mCODE Patient resource

    Demonstrates creating a patient with all required and optional mCODE extensions.
    """
    print("=== Example 1: Creating mCODE Patient ===")

    # Create patient using utility function
    patient = create_mcode_patient(
        id="patient-001",
        name=[{
            "use": "official",
            "family": "Smith",
            "given": ["Jane", "Marie"]
        }],
        gender=AdministrativeGender.FEMALE,
        birth_date="1975-06-15",
        birth_sex=BirthSex.F,
        race=FHIRCodeableConcept(
            coding=[{
                "system": "urn:oid:2.16.840.1.113883.6.238",
                "code": "2106-3",
                "display": "White"
            }]
        ),
        ethnicity=FHIRCodeableConcept(
            coding=[{
                "system": "urn:oid:2.16.840.1.113883.6.238",
                "code": "2186-5",
                "display": "Not Hispanic or Latino"
            }]
        )
    )

    print(f"Created patient: {patient.full_name}")
    print(f"Gender: {patient.gender.value}")
    print(f"Birth Date: {patient.birth_date}")
    print(f"Extensions: {len(patient.extension or [])} present")

    # Show extensions
    if patient.extension:
        for ext in patient.extension:
            if isinstance(ext, BirthSexExtension):
                print(f"Birth Sex: {ext.valueCode.value}")
            elif isinstance(ext, USCoreRaceExtension):
                print(f"Race: {ext.valueCodeableConcept.coding[0]['display']}")
            elif isinstance(ext, USCoreEthnicityExtension):
                print(f"Ethnicity: {ext.valueCodeableConcept.coding[0]['display']}")

    return patient


def example_2_create_cancer_condition():
    """
    Example 2: Create a comprehensive Cancer Condition

    Demonstrates creating a cancer diagnosis with histology and laterality.
    """
    print("\n=== Example 2: Creating Cancer Condition ===")

    # Create cancer condition using utility function
    condition = create_cancer_condition(
        patient_id="patient-001",
        cancer_code=CancerConditionCode.BREAST_CANCER,
        clinical_status="active",
        histology_behavior=HistologyMorphologyBehavior.MALIGNANT,
        laterality=FHIRCodeableConcept(
            coding=[{
                "system": "http://snomed.info/sct",
                "code": "7771000",
                "display": "Left"
            }]
        )
    )

    print(f"Cancer Type: {condition.code.coding[0]['display']}")
    print(f"Clinical Status: {condition.clinicalStatus.coding[0]['code']}")
    print(f"Extensions: {len(condition.extension or [])} present")

    # Show extensions
    if condition.extension:
        for ext in condition.extension:
            if isinstance(ext, HistologyMorphologyBehaviorExtension):
                print(f"Histology: {ext.valueCodeableConcept.coding[0]['display']}")
            elif isinstance(ext, LateralityExtension):
                print(f"Laterality: {ext.valueCodeableConcept.coding[0]['display']}")

    return condition


def example_3_staging_observation():
    """
    Example 3: Create TNM Staging Observation

    Demonstrates creating a TNM stage group observation.
    """
    print("\n=== Example 3: Creating TNM Staging ===")

    # Create TNM staging observation
    staging = TNMStageGroup(
        resourceType="Observation",
        id="staging-001",
        status="final",
        category=[FHIRCodeableConcept(
            coding=[{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "exam"}]
        )],
        code=FHIRCodeableConcept(
            coding=[{"system": "http://loinc.org", "code": "21908-9", "display": "Stage group"}]
        ),
        subject=FHIRReference(reference="Patient/patient-001"),
        effectiveDateTime="2024-01-15",
        valueCodeableConcept=FHIRCodeableConcept(
            coding=[{"system": "urn:oid:2.16.840.1.113883.3.26.1.1", "code": "IIA", "display": "Stage IIA"}]
        )
    )

    print(f"Staging Observation ID: {staging.id}")
    print(f"Stage Group: {staging.valueCodeableConcept.coding[0]['display']}")
    print(f"Effective Date: {staging.effectiveDateTime}")

    return staging


def example_4_tumor_marker_test():
    """
    Example 4: Create Tumor Marker Test

    Demonstrates creating receptor status testing observations.
    """
    print("\n=== Example 4: Creating Tumor Marker Tests ===")

    # ER positive test
    er_test = TumorMarkerTest(
        resourceType="Observation",
        id="er-test-001",
        status="final",
        category=[FHIRCodeableConcept(
            coding=[{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]
        )],
        code=FHIRCodeableConcept(
            coding=[{"system": "http://loinc.org", "code": "85310-0", "display": "Estrogen receptor Ag [Presence] in Breast cancer specimen by Immune stain"}]
        ),
        subject=FHIRReference(reference="Patient/patient-001"),
        effectiveDateTime="2024-01-10",
        valueCodeableConcept=FHIRCodeableConcept(
            coding=[{"system": "http://snomed.info/sct", "code": "10828004", "display": "Positive"}]
        )
    )

    print(f"ER Test Result: {er_test.valueCodeableConcept.coding[0]['display']}")

    # HER2 negative test
    her2_test = TumorMarkerTest(
        resourceType="Observation",
        id="her2-test-001",
        status="final",
        category=[FHIRCodeableConcept(
            coding=[{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]
        )],
        code=FHIRCodeableConcept(
            coding=[{"system": "http://loinc.org", "code": "85319-2", "display": "HER2 [Presence] in Breast cancer specimen by Immune stain"}]
        ),
        subject=FHIRReference(reference="Patient/patient-001"),
        effectiveDateTime="2024-01-10",
        valueCodeableConcept=FHIRCodeableConcept(
            coding=[{"system": "http://snomed.info/sct", "code": "260385009", "display": "Negative"}]
        )
    )

    print(f"HER2 Test Result: {her2_test.valueCodeableConcept.coding[0]['display']}")

    return [er_test, her2_test]


def example_5_performance_status():
    """
    Example 5: Create ECOG Performance Status

    Demonstrates creating performance status observations.
    """
    print("\n=== Example 5: Creating ECOG Performance Status ===")

    ecog = ECOGPerformanceStatusObservation(
        resourceType="Observation",
        id="ecog-001",
        status="final",
        category=[FHIRCodeableConcept(
            coding=[{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "exam"}]
        )],
        code=FHIRCodeableConcept(
            coding=[{"system": "http://loinc.org", "code": "89247-1", "display": "ECOG Performance Status score"}]
        ),
        subject=FHIRReference(reference="Patient/patient-001"),
        effectiveDateTime="2024-01-15",
        valueCodeableConcept=FHIRCodeableConcept(
            coding=[{"system": "urn:oid:2.16.840.1.113883.3.26.1.1", "code": "1", "display": "Restricted in physically strenuous activity"}]
        )
    )

    print(f"ECOG Score: {ecog.valueCodeableConcept.coding[0]['code']} - {ecog.valueCodeableConcept.coding[0]['display']}")

    return ecog


def example_6_treatment_resources():
    """
    Example 6: Create Treatment Resources

    Demonstrates creating medication and procedure resources.
    """
    print("\n=== Example 6: Creating Treatment Resources ===")

    # Chemotherapy medication
    chemo = CancerRelatedMedicationStatement(
        resourceType="MedicationStatement",
        id="chemo-001",
        status="active",
        category=FHIRCodeableConcept(
            coding=[{"system": "http://terminology.hl7.org/CodeSystem/medication-statement-category", "code": "patientspecified"}]
        ),
        medicationCodeableConcept=FHIRCodeableConcept(
            coding=[{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "1736854", "display": "paclitaxel"}]
        ),
        subject=FHIRReference(reference="Patient/patient-001"),
        effectivePeriod={"start": "2024-02-01", "end": "2024-05-01"}
    )

    print(f"Chemotherapy: {chemo.medicationCodeableConcept.coding[0]['display']}")

    # Surgical procedure
    surgery = CancerRelatedSurgicalProcedure(
        resourceType="Procedure",
        id="surgery-001",
        status="completed",
        category=FHIRCodeableConcept(
            coding=[{"system": "http://snomed.info/sct", "code": "387713003", "display": "Surgical procedure"}]
        ),
        code=FHIRCodeableConcept(
            coding=[{"system": "http://snomed.info/sct", "code": "392021009", "display": "Mastectomy"}]
        ),
        subject=FHIRReference(reference="Patient/patient-001"),
        performedDateTime="2024-02-15"
    )

    print(f"Surgery: {surgery.code.coding[0]['display']}")

    return [chemo, surgery]


def example_7_version_management():
    """
    Example 7: Version Management and Profile URLs

    Demonstrates using the versioning system.
    """
    print("\n=== Example 7: Version Management ===")

    # Create version manager
    version_manager = McodeVersionManager()

    print(f"Latest mCODE Version: {McodeVersion.latest().value}")

    # Get profile URLs for different versions
    patient_url_stu3 = version_manager.get_profile_url(McodeProfile.PATIENT, McodeVersion.STU3)
    patient_url_stu4 = version_manager.get_profile_url(McodeProfile.CANCER_CONDITION, McodeVersion.STU4)

    print(f"Patient Profile (STU3): {patient_url_stu3}")
    print(f"Cancer Condition Profile (STU4): {patient_url_stu4}")

    # Check compatibility
    compatibility = version_manager.check_compatibility(McodeVersion.STU3, McodeVersion.STU4)
    print(f"STU3 → STU4 Compatible: {compatibility.compatible}")
    if compatibility.breaking_changes:
        print("Breaking Changes:")
        for change in compatibility.breaking_changes[:2]:  # Show first 2
            print(f"  - {change}")

    return version_manager


def example_8_validation():
    """
    Example 8: Validation and Quality Assurance

    Demonstrates validation of mCODE resources.
    """
    print("\n=== Example 8: Validation ===")

    # Create validator
    validator = McodeValidator()

    # Create a patient for validation
    patient = example_1_create_mcode_patient()

    # Validate patient eligibility
    criteria = {"min_age": 18, "max_age": 75, "gender": "female"}
    eligibility_result = validator.validate_patient_eligibility(patient, criteria)

    print(f"Patient Eligible: {eligibility_result['eligible']}")
    print(f"Met Criteria: {len(eligibility_result['criteria_met'])}")
    print(f"Not Met Criteria: {len(eligibility_result['criteria_not_met'])}")

    # Create and validate cancer condition
    condition = example_2_create_cancer_condition()
    condition_validation = validator.validate_cancer_condition(condition)

    print(f"Condition Valid: {condition_validation['valid']}")
    print(f"Quality Score: {condition_validation['quality_score']:.1%}")
    if condition_validation['issues']:
        print("Issues Found:")
        for issue in condition_validation['issues'][:2]:  # Show first 2
            print(f"  - {issue}")

    return validator


def example_9_fhir_bundle_generation():
    """
    Example 9: Generate FHIR Bundle

    Demonstrates creating a FHIR bundle from mCODE resources.
    """
    print("\n=== Example 9: FHIR Bundle Generation ===")

    # Create all resources
    patient = example_1_create_mcode_patient()
    condition = example_2_create_cancer_condition()
    staging = example_3_staging_observation()
    tumor_tests = example_4_tumor_marker_test()
    ecog = example_5_performance_status()
    treatments = example_6_treatment_resources()

    # Create FHIR bundle
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": patient.model_dump(exclude_none=True)},
            {"resource": condition.model_dump(exclude_none=True)},
            {"resource": staging.model_dump(exclude_none=True)},
            {"resource": ecog.model_dump(exclude_none=True)},
        ]
    }

    # Add tumor tests
    for test in tumor_tests:
        bundle["entry"].append({"resource": test.model_dump(exclude_none=True)})

    # Add treatments
    for treatment in treatments:
        bundle["entry"].append({"resource": treatment.model_dump(exclude_none=True)})

    print(f"Created FHIR Bundle with {len(bundle['entry'])} resources:")
    resource_types = {}
    for entry in bundle["entry"]:
        resource_type = entry["resource"]["resourceType"]
        resource_types[resource_type] = resource_types.get(resource_type, 0) + 1

    for resource_type, count in resource_types.items():
        print(f"  - {resource_type}: {count}")

    return bundle


def example_10_ontology_validation():
    """
    Example 10: Ontology Completeness Validation

    Demonstrates validating ontology completeness against sample data.
    Note: This requires actual data files to be present.
    """
    print("\n=== Example 10: Ontology Completeness Validation ===")

    # Check if sample data files exist
    patient_file = Path("data/summarized_breast_cancer_patient.ndjson")
    trial_file = Path("data/summarized_breast_cancer_trial.ndjson")

    if not patient_file.exists() or not trial_file.exists():
        print("Sample data files not found. Skipping completeness validation.")
        print("To run this example, ensure the following files exist:")
        print(f"  - {patient_file}")
        print(f"  - {trial_file}")
        return None

    # Create validator and run completeness check
    validator = McodeOntologyValidator()
    report = validator.validate_ontology_completeness(patient_file, trial_file)

    print("Ontology Completeness Report:")
    print(f"  Total Resources: {report.total_resources}")
    print(f"  Valid Resources: {report.valid_resources} ({report.valid_resources/report.total_resources*100:.1f}%)")
    print(f"  Patient Coverage: {report.patient_demographics_coverage:.1%}")
    print(f"  Cancer Condition Coverage: {report.cancer_condition_coverage:.1%}")
    print(f"  Staging Coverage: {report.staging_coverage:.1%}")
    print(f"  Biomarkers Coverage: {report.biomarkers_coverage:.1%}")

    if report.recommendations:
        print("Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):  # Show first 3
            print(f"  {i}. {rec}")

    return report


def main():
    """Run all mCODE ontology examples."""
    print("mCODE Ontology Usage Examples")
    print("=" * 50)

    try:
        # Run examples
        patient = example_1_create_mcode_patient()
        condition = example_2_create_cancer_condition()
        staging = example_3_staging_observation()
        tumor_tests = example_4_tumor_marker_test()
        ecog = example_5_performance_status()
        treatments = example_6_treatment_resources()
        version_manager = example_7_version_management()
        validator = example_8_validation()
        bundle = example_9_fhir_bundle_generation()
        report = example_10_ontology_validation()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("mCODE resources created and validated.")
        print("=" * 50)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()