# mCODE Validation & Quality Gates

## Overview

The mCODE Translator implements comprehensive validation and quality assurance mechanisms to ensure mCODE ontology compliance, data integrity, and interoperability. This document outlines the validation framework, quality gates, and automated quality assurance processes.

## Validation Framework Architecture

### Core Validation Components

#### 1. Profile Validation (`McodeValidator`)

Comprehensive validation for individual mCODE profiles:

```python
from shared.mcode_models import McodeValidator, CancerCondition

validator = McodeValidator()

# Validate cancer condition
result = validator.validate_cancer_condition(cancer_condition)
print(f"Valid: {result['valid']}, Quality Score: {result['quality_score']:.1%}")
```

#### 2. Ontology Completeness Validation (`McodeOntologyValidator`)

End-to-end validation against sample datasets:

```python
from scripts.validate_mcode_ontology import McodeOntologyValidator

validator = McodeOntologyValidator()
report = validator.validate_ontology_completeness("data/patients.ndjson", "data/trials.ndjson")

print(f"Patient Coverage: {report.patient_demographics_coverage:.1%}")
print(f"Cancer Condition Coverage: {report.cancer_condition_coverage:.1%}")
```

#### 3. Data Quality Validation (`DataQualityValidator`)

General data quality checks integrated with mCODE validation:

```python
from shared.data_quality_validator import DataQualityValidator

quality_validator = DataQualityValidator()
quality_report = quality_validator.validate_dataset(dataset)

print(f"Overall Quality: {quality_report.overall_score:.1%}")
```

## Quality Gates

### Automated Quality Gates

The system implements multiple quality gates that must pass before code integration:

#### 1. Syntax Validation Gate

**Purpose**: Ensure all Python files compile without syntax errors

**Implementation**:
```bash
# Run syntax validation
python -m py_compile src/shared/mcode_models.py
python -m py_compile src/shared/mcode_versioning.py
```

**Success Criteria**:
- ✅ All Python files compile successfully
- ✅ No syntax errors reported
- ✅ Import statements valid

#### 2. Import Validation Gate

**Purpose**: Verify all modules can be imported successfully

**Implementation**:
```bash
# Test imports
python -c "from shared.mcode_models import McodePatient, CancerCondition"
python -c "from shared.mcode_versioning import McodeVersionManager"
python -c "from scripts.validate_mcode_ontology import McodeOntologyValidator"
```

**Success Criteria**:
- ✅ All mCODE modules import without errors
- ✅ Dependencies resolved correctly
- ✅ No circular import issues

#### 3. Type Checking Gate (MyPy)

**Purpose**: Validate type annotations and catch type errors

**Implementation**:
```bash
# Run MyPy type checking
mypy src/shared/mcode_models.py --strict
mypy src/shared/mcode_versioning.py --strict
mypy scripts/validate_mcode_ontology.py --strict
```

**Success Criteria**:
- ✅ No type errors reported
- ✅ All type annotations correct
- ✅ Strict mode compliance

#### 4. Code Quality Gate (Ruff)

**Purpose**: Enforce coding standards and style consistency

**Implementation**:
```bash
# Run Ruff linting and formatting
ruff check src/shared/mcode_*.py
ruff format src/shared/mcode_*.py --check
```

**Success Criteria**:
- ✅ No linting errors (E, W, F levels)
- ✅ Code formatting compliant
- ✅ Import organization correct

#### 5. Import Organization Gate (isort)

**Purpose**: Ensure consistent import ordering and organization

**Implementation**:
```bash
# Check import organization
isort --check-only --diff src/shared/mcode_*.py
```

**Success Criteria**:
- ✅ Imports properly organized
- ✅ Alphabetical ordering maintained
- ✅ Section separation correct

#### 6. mCODE Profile Validation Gate

**Purpose**: Validate mCODE profile implementations against specification

**Implementation**:
```python
# Automated profile validation
validator = McodeValidator()

# Test all profile models
profiles_to_test = [
    McodePatient, CancerCondition, TNMStageGroup,
    TumorMarkerTest, ECOGPerformanceStatusObservation,
    CancerRelatedMedicationStatement, CancerRelatedSurgicalProcedure
]

for profile_class in profiles_to_test:
    # Create test instance
    test_instance = create_test_instance(profile_class)
    result = validator.validate_resource_compatibility(test_instance, profile_class.__name__)
    assert result, f"Profile validation failed for {profile_class.__name__}"
```

**Success Criteria**:
- ✅ All mCODE profiles validate successfully
- ✅ Required fields present
- ✅ Extensions properly structured
- ✅ Value sets correctly used

#### 7. Ontology Completeness Gate

**Purpose**: Ensure comprehensive mCODE element coverage

**Implementation**:
```python
# Run completeness validation
validator = McodeOntologyValidator()
report = validator.validate_ontology_completeness(test_patient_file, test_trial_file)

# Check minimum coverage thresholds
assert report.patient_demographics_coverage >= 0.80, "Patient coverage below 80%"
assert report.cancer_condition_coverage >= 0.70, "Cancer condition coverage below 70%"
assert report.staging_coverage >= 0.50, "Staging coverage below 50%"
```

**Success Criteria**:
- ✅ Patient demographics ≥ 80% coverage
- ✅ Cancer conditions ≥ 70% coverage
- ✅ Staging information ≥ 50% coverage
- ✅ Biomarkers ≥ 40% coverage
- ✅ Treatments ≥ 30% coverage

## Validation Rules

### Profile-Specific Validation Rules

#### Patient Profile Validation

**Required Fields**:
- `resourceType`: Must be "Patient"
- `gender`: Administrative gender (required for mCODE)
- `birthDate`: ISO format date

**Extension Validation**:
- Birth sex extension URL: `http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-birth-sex`
- Race extension URL: `http://hl7.org/fhir/us/core/StructureDefinition/us-core-race`
- Ethnicity extension URL: `http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity`

**Business Rules**:
- Administrative gender must be from value set: male, female, other, unknown
- Birth date must be valid ISO format
- Extensions must have exactly one value field

#### Cancer Condition Validation

**Required Fields**:
- `resourceType`: Must be "Condition"
- `subject`: Reference to patient
- `category`: Must include "problem-list-item"
- `code`: Cancer diagnosis code

**Extension Validation**:
- Histology extension URL: `http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-histology-morphology-behavior`
- Laterality extension URL: `http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-laterality`

**Business Rules**:
- Code must represent a cancer diagnosis (SNOMED CT codes)
- Category must include problem-list-item coding
- Clinical status must be from valid value set

#### Observation Profile Validation

**Common Requirements**:
- `resourceType`: Must be "Observation"
- `status`: Must be from valid status values
- `category`: Appropriate category coding
- `code`: Observation type code
- `subject`: Reference to patient

**Profile-Specific Rules**:

**Tumor Marker Test**:
- Category must include "laboratory"
- Code must be from LOINC or equivalent

**ECOG Performance Status**:
- Value must be from ECOG value set (0-5)
- Code must be LOINC 89247-1

**TNM Staging**:
- Code must be LOINC 21908-9 (Stage group)
- Value must be from TNM stage group value set

#### Procedure Profile Validation

**Common Requirements**:
- `resourceType`: Must be "Procedure"
- `status`: Must be from valid status values
- `subject`: Reference to patient

**Profile-Specific Rules**:

**Cancer-Related Surgical Procedure**:
- Category must indicate surgical procedure
- Code must represent cancer-related surgery

**Cancer-Related Radiation Procedure**:
- Category must indicate radiation procedure
- Code must represent radiation therapy

#### Medication Profile Validation

**Cancer-Related Medication Statement**:
- `resourceType`: Must be "MedicationStatement"
- `status`: Must be from valid status values
- `category`: Must indicate cancer-related
- `subject`: Reference to patient

### Value Set Validation

#### Controlled Terminologies

**Administrative Gender**:
```python
valid_genders = {"male", "female", "other", "unknown"}
```

**Birth Sex**:
```python
valid_birth_sex = {"M", "F", "UNK"}
```

**Cancer Condition Codes** (SNOMED CT):
```python
cancer_codes = {
    "254837009",  # Breast cancer
    "363358000",  # Lung cancer
    "363406005",  # Colorectal cancer
    # ... additional codes
}
```

**TNM Stage Groups**:
```python
tnm_stages = {"0", "I", "II", "III", "IV", "IA", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IVA", "IVB"}
```

**ECOG Performance Status**:
```python
ecog_scores = {"0", "1", "2", "3", "4", "5"}
```

### Version Compatibility Validation

#### Version Validation Rules

**Supported Versions**:
- STU1 (1.0.0): Basic oncology profiles
- STU2 (2.0.0): Added genomics support
- STU3 (3.0.0): Enhanced medication tracking
- STU4 (4.0.0): Pediatric oncology, risk assessment

**Compatibility Checking**:
```python
def validate_version_compatibility(source_version: str, target_version: str) -> bool:
    """Validate version compatibility for migration."""
    compatibility = check_version_compatibility(source_version, target_version)
    return compatibility.compatible
```

**Profile Availability**:
```python
def validate_profile_availability(profile: McodeProfile, version: McodeVersion) -> bool:
    """Check if profile is available in specified version."""
    try:
        url = get_profile_url(profile, version)
        return True
    except ValueError:
        return False
```

## Quality Metrics

### Coverage Metrics

#### Element Coverage Calculation

```python
def calculate_element_coverage(elements_present: Dict[str, int], elements_total: int) -> float:
    """Calculate percentage coverage for element category."""
    present_count = sum(elements_present.values())
    return (present_count / elements_total) * 100 if elements_total > 0 else 0.0
```

#### Category Coverage Thresholds

- **Patient Demographics**: ≥ 80%
- **Cancer Conditions**: ≥ 70%
- **Staging**: ≥ 50%
- **Biomarkers**: ≥ 40%
- **Procedures**: ≥ 30%
- **Medications**: ≥ 30%
- **Trial Information**: ≥ 60%

### Quality Scoring

#### Profile Quality Score

```python
def calculate_profile_quality_score(validation_result: Dict[str, Any]) -> float:
    """Calculate quality score for a profile instance."""
    base_score = 1.0

    # Deduct for missing required fields
    if 'missing_required' in validation_result:
        base_score -= len(validation_result['missing_required']) * 0.2

    # Deduct for validation errors
    if 'errors' in validation_result:
        base_score -= len(validation_result['errors']) * 0.1

    # Deduct for warnings
    if 'warnings' in validation_result:
        base_score -= len(validation_result['warnings']) * 0.05

    return max(0.0, min(1.0, base_score))
```

#### Dataset Quality Score

```python
def calculate_dataset_quality_score(report: CompletenessReport) -> float:
    """Calculate overall quality score for dataset."""
    weights = {
        'patient_demographics': 0.25,
        'cancer_condition': 0.25,
        'staging': 0.20,
        'biomarkers': 0.15,
        'procedures': 0.10,
        'medications': 0.05
    }

    score = 0.0
    for category, weight in weights.items():
        coverage = getattr(report, f"{category}_coverage", 0.0)
        score += coverage * weight

    return score
```

## Automated Testing Integration

### Unit Tests for Validation

```python
def test_mcode_profile_validation():
    """Test mCODE profile validation."""
    validator = McodeValidator()

    # Test valid patient
    valid_patient = create_mcode_patient(...)
    result = validator.validate_patient_eligibility(valid_patient, {})
    assert result['eligible'] == True

    # Test invalid patient (missing gender)
    invalid_patient = create_mcode_patient(gender=None, ...)
    result = validator.validate_patient_eligibility(invalid_patient, {})
    assert result['eligible'] == False
    assert 'Gender criteria' in result['criteria_not_met'][0]
```

### Integration Tests for Quality Gates

```python
def test_quality_gates():
    """Test all quality gates pass."""
    # Syntax validation
    assert run_syntax_validation() == True

    # Import validation
    assert run_import_validation() == True

    # Type checking
    assert run_mypy_validation() == True

    # Code quality
    assert run_ruff_validation() == True

    # Ontology completeness
    report = run_completeness_validation()
    assert report.patient_demographics_coverage >= 0.80
    assert report.cancer_condition_coverage >= 0.70
```

## Continuous Integration

### CI/CD Pipeline Integration

#### GitHub Actions Quality Checks

```yaml
name: Quality Gates
on: [push, pull_request]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install mypy ruff isort

      - name: Syntax validation
        run: python -m py_compile src/shared/mcode_*.py

      - name: Import validation
        run: python -c "from shared.mcode_models import *; from shared.mcode_versioning import *"

      - name: Type checking
        run: mypy src/shared/mcode_*.py --strict

      - name: Code quality
        run: ruff check src/shared/mcode_*.py

      - name: Import organization
        run: isort --check-only src/shared/mcode_*.py

      - name: Ontology validation
        run: python scripts/validate_mcode_ontology.py
```

#### Quality Gate Status Reporting

```python
def generate_quality_report() -> Dict[str, Any]:
    """Generate comprehensive quality report."""
    return {
        "syntax_validation": run_syntax_check(),
        "import_validation": run_import_check(),
        "type_checking": run_mypy_check(),
        "code_quality": run_ruff_check(),
        "ontology_completeness": run_completeness_check(),
        "overall_status": "PASS" if all_checks_pass() else "FAIL"
    }
```

## Troubleshooting Validation Issues

### Common Validation Problems

#### Profile Validation Errors

**Problem**: "Extension must have exactly one value field"
**Solution**: Ensure extensions have only one value field (valueCode, valueString, etc.)

**Problem**: "Required field missing"
**Solution**: Check profile requirements and add missing required fields

**Problem**: "Invalid code system"
**Solution**: Use correct coding system URLs (SNOMED CT, LOINC, etc.)

#### Ontology Completeness Issues

**Problem**: Low coverage percentages
**Solution**:
1. Review data mapping logic
2. Add missing mCODE elements
3. Update validation rules
4. Enhance data extraction

**Problem**: Validation script fails
**Solution**:
1. Check file paths
2. Verify data format (NDJSON)
3. Ensure dependencies installed
4. Check Python path configuration

#### Quality Gate Failures

**Problem**: MyPy type errors
**Solution**:
1. Add missing type annotations
2. Fix type inconsistencies
3. Update type stubs

**Problem**: Ruff formatting errors
**Solution**:
1. Run `ruff format` to auto-fix
2. Review and fix remaining issues
3. Update configuration if needed

## Best Practices

### Validation Implementation

1. **Early Validation**: Validate data as early as possible in the pipeline
2. **Fail Fast**: Stop processing on critical validation errors
3. **Comprehensive Reporting**: Provide detailed error messages and suggestions
4. **Version Awareness**: Use appropriate validation rules for each mCODE version

### Quality Assurance

1. **Automated Gates**: Never bypass quality gates
2. **Regular Testing**: Run validation regularly, not just on commits
3. **Performance Monitoring**: Track validation performance impact
4. **Continuous Improvement**: Update validation rules as mCODE evolves

### Maintenance

1. **Version Updates**: Keep validation rules current with mCODE specification
2. **Test Data**: Maintain comprehensive test datasets
3. **Documentation**: Keep validation documentation synchronized
4. **Monitoring**: Monitor validation success rates and failure patterns

This comprehensive validation and quality assurance framework ensures that mCODE implementations maintain high standards of data quality, interoperability, and compliance with oncology data standards.