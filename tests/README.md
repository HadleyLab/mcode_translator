# mCODE Translator Testing Strategy

This document outlines the comprehensive testing strategy for the mCODE Translator project, including unit testing, integration testing, performance testing, and test data management.

## Overview

The testing strategy follows these key principles:
- **Separation of Concerns**: Mock tests vs. live tests with clear delineation
- **Comprehensive Coverage**: Target 90%+ code coverage across all modules
- **Realistic Test Data**: Use fixtures and factories for generating test data
- **Performance Monitoring**: Benchmark critical operations with real data
- **CI/CD Integration**: Automated testing with conditional live test execution

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── pytest.ini              # Pytest configuration with markers and coverage
├── data/                   # Test data files
│   ├── sample_trial.json
│   └── sample_patient.json
├── unit/                   # Unit tests with mocks
│   ├── test_dependency_container.py
│   ├── test_mcode_llm.py
│   ├── test_summarizer.py
│   ├── test_patient_generator.py
│   └── test_*.py
├── integration/            # Integration tests with real data
│   └── test_pipeline_integration.py
├── performance/            # Performance benchmarks
│   └── test_performance.py
└── README.md              # This file
```

## Test Categories

### 1. Unit Tests (`pytest -m unit`)

Unit tests isolate individual functions and methods using mocks for external dependencies.

**Key Features:**
- Use `unittest.mock` for API calls, file I/O, and external services
- Target 90%+ code coverage
- Test error handling and edge cases
- Fast execution suitable for development workflow

**Example:**
```python
@pytest.mark.mock
def test_mcode_mapping_success(mock_llm_response):
    mapper = McodeMapper()
    result = mapper.map_to_mcode("clinical text")

    assert "mcode_elements" in result
    assert len(result["mcode_elements"]) > 0
```

### 2. Integration Tests (`pytest -m integration`)

Integration tests validate end-to-end functionality with real data sources.

**Key Features:**
- Use actual data files and configurations
- Test data persistence and retrieval
- Validate API integrations (when ENABLE_LIVE_TESTS=true)
- Idempotent operations that don't alter production data

**Example:**
```python
@pytest.mark.live
def test_pipeline_with_real_data(sample_trial_data):
    pipeline = ClinicalTrialPipeline()
    result = pipeline.process(sample_trial_data)

    assert result.success is True
    assert result.data is not None
```

### 3. Performance Tests (`pytest -m performance`)

Performance benchmarks measure execution times and resource usage.

**Key Features:**
- Use `pytest-benchmark` for timing measurements
- Compare mock vs. real data performance
- Identify bottlenecks in critical paths
- Generate performance reports

**Example:**
```python
@pytest.mark.performance
def test_summarizer_performance(benchmark, large_trial_data):
    summarizer = McodeSummarizer()

    def run_summary():
        return summarizer.create_trial_summary(large_trial_data)

    result = benchmark(run_summary)
    assert result is not None
```

## Test Data Management

### Fixtures and Factories

**Shared Fixtures** (`conftest.py`):
- `sample_trial_data`: Realistic clinical trial data
- `sample_patient_data`: FHIR patient bundle
- `mock_config`: Mocked configuration object
- `temp_cache_dir`: Temporary directory for cache testing

**Test Data Factories:**
```python
def create_test_trial(nct_id="NCT12345678", **kwargs):
    """Factory for creating test trial data."""
    return {
        "nct_id": nct_id,
        "title": kwargs.get("title", "Test Trial"),
        "status": kwargs.get("status", "Recruiting"),
        # ... more fields
    }
```

### Test Data Files

Located in `tests/data/`:
- `sample_trial.json`: Complete clinical trial data
- `sample_patient.json`: FHIR patient bundle
- Additional data files for specific test scenarios

## Running Tests

### Basic Commands

```bash
# Run all tests
python run_tests.py all

# Run unit tests only
python run_tests.py unit

# Run integration tests only
python run_tests.py integration

# Run performance tests
python run_tests.py performance

# Run with coverage report
python run_tests.py coverage
```

### Advanced Options

```bash
# Run specific test file
pytest tests/unit/test_summarizer.py -v

# Run tests with specific marker
pytest -m mock -v

# Run live tests (requires ENABLE_LIVE_TESTS=true)
ENABLE_LIVE_TESTS=true pytest -m live -v

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only
```

### CI/CD Integration

**Environment Variables:**
- `ENABLE_LIVE_TESTS`: Set to `true` to run integration tests with real APIs
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD`: Disable auto-loading of plugins

**GitHub Actions Example:**
```yaml
- name: Run Unit Tests
  run: |
    source activate mcode_translator
    python run_tests.py unit

- name: Run Integration Tests
  run: |
    source activate mcode_translator
    ENABLE_LIVE_TESTS=true python run_tests.py integration
  env:
    API_KEY: ${{ secrets.API_KEY }}
```

## Test Markers

### Pytest Markers

- `@pytest.mark.mock`: Tests using mocked dependencies (default)
- `@pytest.mark.live`: Tests using real data/services
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.performance`: Performance benchmark tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests

### Automatic Marker Handling

Live tests are automatically skipped unless `ENABLE_LIVE_TESTS=true`:

```python
# conftest.py
LIVE_TESTS_ENABLED = os.getenv("ENABLE_LIVE_TESTS", "false").lower() == "true"

def pytest_collection_modifyitems(config, items):
    if not LIVE_TESTS_ENABLED:
        skip_live = pytest.mark.skip(reason="Live tests disabled")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
```

## Coverage Requirements

**Targets:**
- **Unit Tests**: ≥80% coverage
- **Critical Modules**: ≥90% coverage
- **New Code**: 100% coverage required

**Coverage Configuration** (`pytest.ini`):
```ini
[tool:pytest]
addopts =
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-report=xml
    --cov-fail-under=90
```

## Performance Benchmarking

### Benchmark Categories

1. **API Operations**: LLM calls, data fetching
2. **Data Processing**: JSON parsing, data transformation
3. **Memory Usage**: Large data structure handling
4. **I/O Operations**: File reading/writing, caching

### Benchmark Results

Results are saved to `reports/` directory:
- `benchmark_results.json`: Raw benchmark data
- `benchmark_report.html`: Visual performance report

### Example Benchmark Output

```
test_summarizer_performance
  Min: 45.2ms
  Max: 67.8ms
  Mean: 52.1ms
  StdDev: 8.3ms
```

## Error Handling and Edge Cases

### Test Error Scenarios

- **Invalid Input**: Malformed JSON, missing fields
- **Network Failures**: API timeouts, connection errors
- **File System Issues**: Permission errors, disk space
- **Resource Limits**: Memory constraints, rate limits

### Exception Testing

```python
def test_invalid_trial_data():
    summarizer = McodeSummarizer()

    with pytest.raises(ValueError, match="Trial data is missing"):
        summarizer.create_trial_summary({})
```

## Test Maintenance

### Adding New Tests

1. **Identify Test Category**: Unit, integration, or performance
2. **Create Test File**: Follow naming convention `test_*.py`
3. **Use Appropriate Markers**: `@pytest.mark.mock`, `@pytest.mark.live`, etc.
4. **Add Fixtures**: Use existing fixtures or create new ones
5. **Update Documentation**: Add test descriptions and examples

### Test Data Updates

1. **Version Control**: Keep test data in version control
2. **Realistic Data**: Use anonymized real data when possible
3. **Data Factories**: Prefer factories over static files for dynamic data
4. **Cleanup**: Ensure test data cleanup after test execution

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure conda environment is activated
source activate mcode_translator
pip install -e .
```

**Live Test Failures:**
```bash
# Check environment variables
echo $ENABLE_LIVE_TESTS
export ENABLE_LIVE_TESTS=true
```

**Coverage Issues:**
```bash
# Run coverage for specific module
pytest --cov=src.services.summarizer tests/unit/test_summarizer.py
```

### Debug Mode

```bash
# Run tests with debug output
pytest -v -s --tb=long

# Run specific test with debugging
pytest tests/unit/test_summarizer.py::TestMcodeSummarizer::test_create_trial_summary -v -s
```

## Contributing

When adding new features:

1. **Write Tests First**: Follow TDD principles
2. **Mock External Dependencies**: Keep tests fast and isolated
3. **Add Performance Tests**: For performance-critical code
4. **Update Documentation**: Keep this README current
5. **Run Full Test Suite**: Ensure no regressions

## Metrics and Reporting

### Test Reports

- **HTML Coverage**: `htmlcov/index.html`
- **XML Coverage**: `coverage.xml` (for CI tools)
- **Benchmark Reports**: `reports/benchmark_report.html`
- **Test Results**: `reports/test_report.html`

### Quality Gates

- ✅ All unit tests pass
- ✅ Coverage ≥90%
- ✅ No critical security issues
- ✅ Performance benchmarks within thresholds
- ✅ Integration tests pass in staging environment

---

This testing strategy ensures robust, maintainable, and performant code for the mCODE Translator project.