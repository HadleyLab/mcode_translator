# Comprehensive Test Documentation and Guidelines

This document provides comprehensive documentation and guidelines for testing in the mCODE Translator project.

## Overview

The mCODE Translator testing framework follows a modular, component-based approach that promotes code reuse, eliminates redundancy, and simplifies maintenance. This documentation outlines the testing structure, guidelines, and best practices.

## Testing Structure

### Directory Organization

```
tests/
├── shared/                    # Shared test components and utilities
│   ├── __init__.py
│   ├── test_components.py     # Reusable test components
│   ├── test_fixtures.py       # Shared pytest fixtures
│   ├── test_data_generators.py # Test data utilities
│   └── test_utils.py          # Common testing utilities
├── unit/                      # Unit tests (all pytest-based)
│   ├── __init__.py
│   ├── test_code_extraction.py
│   ├── test_fetcher.py
│   ├── test_mcode_mapping_engine.py
│   └── test_breast_cancer_profile.py
├── integration/               # Integration tests (all pytest-based)
│   ├── __init__.py
│   ├── test_api_connectivity.py
│   └── test_live_api.py
├── component/                 # Component-based tests
│   ├── __init__.py
│   ├── test_mcode_elements.py
│   ├── test_nlp_engines.py
│   └── test_data_processing.py
├── conftest.py                # Global pytest configuration and fixtures
└── test_runner.py             # Main test runner script
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test interactions between components
3. **Component Tests**: Test related functionality as modular components
4. **End-to-End Tests**: Test complete workflows and pipelines

## Testing Guidelines

### 1. Test Naming Conventions

Follow clear, descriptive naming conventions:

```python
# Good examples
def test_code_extractor_identifies_icd10cm_codes_success():
    pass

def test_mcode_mapper_handles_unknown_concepts_gracefully():
    pass

def test_breast_cancer_profile_matches_er_positive_patients():
    pass

# Avoid
def test_1():
    pass

def test_mapper():
    pass
```

### 2. Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_code_extraction_with_valid_input():
    """Test code extraction with valid input"""
    # Arrange
    code_extractor = CodeExtractionModule()
    text = "Patient diagnosed with breast cancer (ICD-10-CM: C50.911)"
    
    # Act
    codes = code_extractor.process_criteria_for_codes(text)
    
    # Assert
    assert 'extracted_codes' in codes
    assert 'ICD10CM' in codes['extracted_codes']
    assert codes['extracted_codes']['ICD10CM'][0]['code'] == 'C50.911'
```

### 3. Fixture Usage

Use fixtures to reduce duplication and improve test maintainability:

```python
# In conftest.py
@pytest.fixture
def code_extractor():
    """Provide a CodeExtractionModule instance for testing"""
    return CodeExtractionModule()

# In test file
def test_code_extraction(code_extractor, sample_eligibility_criteria):
    """Test code extraction with shared fixture"""
    codes = code_extractor.process_criteria_for_codes(sample_eligibility_criteria)
    assert len(codes['extracted_codes']) > 0
```

### 4. Parameterized Testing

Use parameterized tests for similar scenarios:

```python
@pytest.mark.parametrize("cancer_type,expected_code", [
    ("breast", "C50.911"),
    ("lung", "C34.90"),
    ("colorectal", "C18.9"),
])
def test_cancer_code_extraction(code_extractor, cancer_type, expected_code):
    """Test extraction of cancer codes for different cancer types"""
    text = f"Patient diagnosed with {cancer_type} cancer (ICD-10-CM: {expected_code})"
    codes = code_extractor.process_criteria_for_codes(text)
    
    assert 'ICD10CM' in codes['extracted_codes']
    extracted_codes = codes['extracted_codes']['ICD10CM']
    assert len(extracted_codes) >= 1
    assert extracted_codes[0]['code'] == expected_code
```

### 5. Mocking and Stubbing

Use mocks for external dependencies:

```python
def test_api_search_with_mock(mock_clinical_trials_api, mock_cache_manager):
    """Test API search with mocked fixtures"""
    from src.data_fetcher.fetcher import search_trials
    
    result = search_trials("cancer", max_results=2)
    
    assert "studies" in result
    assert len(result["studies"]) == 2
    mock_clinical_trials_api.return_value.get_study_fields.assert_called()
```

## Test Data Management

### 1. Test Data Generators

Use test data generators for consistent, realistic test data:

```python
class ClinicalTrialDataGenerator:
    """Generates clinical trial data for testing"""
    
    @staticmethod
    def generate_eligibility_criteria(complexity="simple", cancer_type="breast"):
        """Generate sample eligibility criteria text"""
        # Implementation here
        pass
```

### 2. Sample Data Fixtures

Use fixtures for common test data:

```python
@pytest.fixture
def sample_eligibility_criteria():
    """Provide sample eligibility criteria text for testing"""
    return """
    INCLUSION CRITERIA:
    - Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
    - Eastern Cooperative Oncology Group (ECOG) performance status of 0 or 1
    """
```

## Component-Based Testing

### 1. mCODE Elements Testing

Test mCODE elements in isolation:

```python
class TestPrimaryCancerCondition:
    """Test the Primary Cancer Condition mCODE element"""
    
    def test_condition_creation(self, mcode_mapper):
        """Test creating a primary cancer condition"""
        condition = mcode_mapper._create_mcode_resource({
            'mcode_element': 'Condition',
            'primary_code': {'system': 'ICD10CM', 'code': 'C50.911'},
            'mapped_codes': {'SNOMEDCT': '254837009'}
        })
        
        assert condition['resourceType'] == 'Condition'
        assert 'meta' in condition
        assert 'profile' in condition['meta']
        assert 'mcode-primary-cancer-condition' in condition['meta']['profile'][0]
```

### 2. NLP Engine Testing

Test each NLP engine independently:

```python
class TestRegexNLPEngine:
    """Test the Regex NLP Engine"""
    
    def test_basic_entity_extraction(self, regex_nlp_engine):
        """Test basic entity extraction with regex patterns"""
        text = "Patient diagnosed with breast cancer (ICD-10-CM: C50.911)"
        result = regex_nlp_engine.process_criteria(text)
        
        assert 'entities' in result
        assert len(result['entities']) > 0
        assert any(entity['text'] == 'breast cancer' for entity in result['entities'])
```

### 3. Data Processing Testing

Test data processing components:

```python
class TestCodeExtraction:
    """Test code extraction components"""
    
    def test_icd10cm_extraction(self, code_extractor):
        """Test extraction of ICD-10-CM codes"""
        text = "Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)"
        codes = code_extractor.process_criteria_for_codes(text)
        
        assert 'extracted_codes' in codes
        extracted = codes['extracted_codes']
        assert 'ICD10CM' in extracted
        assert len(extracted['ICD10CM']) >= 1
        assert extracted['ICD10CM'][0]['code'] == 'C50.911'
```

## Performance Testing

### 1. Benchmarking

Include performance benchmarks in tests:

```python
def test_performance_benchmark(self, regex_nlp_engine):
    """Test performance of regex engine"""
    import time
    text = "Patient with BRCA1 mutation, ER+ HER2- breast cancer, stage IIA"
    
    start_time = time.time()
    for _ in range(100):
        result = regex_nlp_engine.process_criteria(text)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    assert avg_time < 0.01  # Should process in less than 10ms on average
```

### 2. Load Testing

Test with multiple iterations:

```python
@pytest.mark.slow
def test_load_testing(self, code_extractor):
    """Test code extractor under load"""
    import time
    text = ClinicalTrialDataGenerator.generate_eligibility_criteria("complex")
    
    start_time = time.time()
    for i in range(1000):
        codes = code_extractor.process_criteria_for_codes(text)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / 1000
    assert avg_time < 0.1  # Should process in less than 100ms on average
```

## Test Coverage and Quality

### 1. Coverage Requirements

Maintain high test coverage:

```python
# Minimum coverage requirements
# Unit tests: 90% coverage
# Integration tests: 85% coverage
# Component tests: 95% coverage
```

### 2. Quality Metrics

Track quality metrics:

```python
def test_code_quality_metrics(self):
    """Test code quality metrics"""
    # Test for:
    # - Code duplication
    # - Cyclomatic complexity
    # - Maintainability index
    # - Test coverage
    pass
```

## Continuous Integration

### 1. GitHub Actions

Configure CI/CD pipelines:

```yaml
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python tests/test_runner.py
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: test_results.json
```

### 2. Test Reporting

Generate comprehensive reports:

```python
def generate_summary_report(self, unit_success: bool, integration_success: bool, component_success: bool):
    """Generate summary report of test execution"""
    # Generate HTML report
    # Generate JSON report
    # Generate coverage report
    pass
```

## Best Practices

### 1. Test Isolation

Ensure tests are independent:

```python
# Good - Each test sets up its own data
def test_code_extraction(code_extractor):
    text = "Patient diagnosed with breast cancer (ICD-10-CM: C50.911)"
    codes = code_extractor.process_criteria_for_codes(text)
    assert len(codes['extracted_codes']) > 0

# Avoid - Tests depend on each other
test_data = None

def test_setup():
    global test_data
    test_data = "some data"

def test_with_setup():
    assert test_data is not None  # Depends on test_setup running first
```

### 2. Clear Assertions

Use clear, specific assertions:

```python
# Good
def test_code_extraction():
    codes = extractor.process_criteria_for_codes(text)
    assert 'ICD10CM' in codes['extracted_codes']
    assert codes['extracted_codes']['ICD10CM'][0]['code'] == 'C50.911'

# Avoid
def test_code_extraction():
    codes = extractor.process_criteria_for_codes(text)
    assert codes  # Not specific enough
```

### 3. Descriptive Test Names

Use descriptive names that explain what is being tested:

```python
# Good
def test_code_extractor_handles_multiple_code_types():
    pass

def test_mcode_mapper_validates_required_fields():
    pass

# Avoid
def test_extractor():
    pass

def test_mapper_1():
    pass
```

## Benefits of Comprehensive Testing Guidelines

1. **Consistency**: All tests follow the same patterns and conventions
2. **Maintainability**: Clear guidelines make it easier to maintain tests
3. **Quality**: Comprehensive guidelines ensure high-quality tests
4. **Onboarding**: New team members can quickly understand testing practices
5. **Scalability**: Guidelines support growth of the test suite
6. **Reliability**: Consistent practices lead to more reliable tests
7. **Performance**: Guidelines include performance considerations
8. **Documentation**: Guidelines serve as documentation for testing practices