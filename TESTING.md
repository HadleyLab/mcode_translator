# Testing Strategy for mCODE Translator

This document outlines the comprehensive testing strategy implemented for the mCODE Translator project, ensuring robust, maintainable, and high-quality code through multiple layers of testing.

## Overview

The testing strategy follows a multi-layered approach with clear separation between different types of tests:

- **Unit Tests**: Isolated function/method testing with mocked dependencies
- **Integration Tests**: End-to-end testing with real data sources
- **Performance Tests**: Benchmarking and load testing
- **Test Data Management**: Realistic fixtures and factories

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── data/                    # Test data files
│   ├── sample_trial.json
│   └── sample_patient.json
├── unit/                    # Unit tests with mocks
│   ├── test_dependency_container.py
│   └── test_mcode_llm.py
├── integration/             # Integration tests with real data
│   └── test_pipeline_integration.py
└── performance/             # Performance benchmarks
    └── test_performance.py
```

## 1. Unit Testing with Mocks

### Purpose
Unit tests verify individual functions and methods in isolation, ensuring each component works correctly without external dependencies.

### Key Features
- **Mocked Dependencies**: All external services (APIs, databases, file systems) are mocked
- **90% Coverage Target**: Comprehensive coverage of all public APIs and core logic
- **Error Handling**: Tests for edge cases, invalid inputs, and error conditions
- **Input Validation**: Verification of parameter validation and type checking

### Running Unit Tests
```bash
# Run all unit tests
python scripts/run_tests.py unit

# Run with coverage
python scripts/run_tests.py unit --coverage

# Run specific test file
source activate mcode_translator && python -m pytest tests/unit/test_dependency_container.py -v
```

**Note**: The test runner script is located at `scripts/run_tests.py` in the project root.

### Example Unit Test
```python
@pytest.mark.mock
def test_dependency_container_creation(self, mock_config_class):
    """Test creating dependency container with mocked config."""
    # Arrange
    mock_config = Mock()
    mock_config_class.return_value = mock_config

    # Act
    container = DependencyContainer()

    # Assert
    assert container.config == mock_config
```

## 2. Integration Testing with Real Data

### Purpose
Integration tests validate that components work together correctly with real data sources, ensuring end-to-end functionality.

### Key Features
- **Real Data Sources**: Tests interact with actual databases, APIs, and file systems
- **Idempotent Operations**: Tests don't alter production data
- **Sandbox Environments**: Use staging/test environments for live testing
- **Conditional Execution**: Live tests are disabled by default in CI/CD

### Running Integration Tests
```bash
# Run with mock data (default)
python scripts/run_tests.py integration

# Run with live data sources (requires ENABLE_LIVE_TESTS=true)
python scripts/run_tests.py integration --live
```

### Environment Variables
```bash
# Enable live tests
export ENABLE_LIVE_TESTS=true

# Run integration tests
python scripts/run_tests.py integration --live
```

## 3. Performance and Load Testing

### Purpose
Performance tests measure execution times, memory usage, and scalability under load.

### Key Features
- **Benchmarking**: Measure execution times for key operations
- **Memory Profiling**: Track memory usage patterns
- **Load Simulation**: Test under simulated load conditions
- **Comparison Analysis**: Compare mock vs. real data performance

### Running Performance Tests
```bash
# Run performance tests
python scripts/run_tests.py performance

# Run only benchmarks
python scripts/run_tests.py performance --benchmark
```

### Example Performance Test
```python
@pytest.mark.performance
def test_summarizer_performance(self, benchmark, large_trial_data):
    """Benchmark trial summary generation."""
    summarizer = McodeSummarizer()

    result = benchmark(
        lambda: summarizer.create_trial_summary(large_trial_data)
    )

    assert result is not None
    assert isinstance(result, str)
```

## 4. Test Data Management

### Purpose
Test data management provides realistic, consistent test data across all test suites.

### Key Features
- **Fixtures**: Reusable test data objects
- **Factories**: Dynamic test data generation
- **Cleanup**: Automatic test data cleanup
- **Realistic Data**: Production-like test scenarios

### Shared Fixtures (conftest.py)
```python
@pytest.fixture
def sample_trial_data() -> Dict[str, Any]:
    """Sample clinical trial data for testing."""
    return {
        "nct_id": "NCT12345678",
        "title": "Sample Clinical Trial",
        "eligibility": {"criteria": "Age >= 18"},
        "conditions": ["Breast Cancer"]
    }

@pytest.fixture
def mock_config() -> Mock:
    """Mock configuration object."""
    config = Mock()
    config.get_api_key.return_value = "test-key"
    return config
```

### Test Data Files
- `tests/data/sample_trial.json`: Realistic clinical trial data
- `tests/data/sample_patient.json`: FHIR patient bundle data

## 5. Test Execution and Reporting

### Running All Tests
```bash
# Run complete test suite
python scripts/run_tests.py all

# Run with coverage and fail-fast
python scripts/run_tests.py all --coverage --fail-fast
```

### Coverage Reporting
```bash
# Generate coverage report
python scripts/run_tests.py coverage

# View HTML report
open htmlcov/index.html
```

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python scripts/run_tests.py unit --coverage
    python scripts/run_tests.py integration  # Mock only in CI

- name: Run Performance Tests
  run: |
    python scripts/run_tests.py performance --benchmark
```

## 6. Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
addopts =
    --strict-markers
    --disable-warnings
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=90
markers =
    mock: Tests using mocked dependencies
    live: Tests using real data/services
    slow: Slow-running tests
    performance: Performance benchmark tests
```

### Environment Setup
```bash
# Activate conda environment
source activate mcode_translator

# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-benchmark

# For performance testing
pip install pytest-benchmark
```

## 7. Best Practices

### Test Organization
- **Clear Naming**: `test_function_name_should_do_something`
- **Descriptive Docstrings**: Explain what and why the test validates
- **Arrange-Act-Assert**: Clear test structure
- **Single Responsibility**: Each test validates one behavior

### Mock Strategy
- **Minimal Mocking**: Mock only external dependencies
- **Realistic Mocks**: Return realistic data structures
- **Verification**: Assert mock interactions when relevant
- **Isolation**: Tests should not depend on each other

### Integration Testing
- **Idempotent**: Tests should not have side effects
- **Isolated**: Use test databases/APIs
- **Fast**: Keep integration tests reasonably fast
- **Conditional**: Skip live tests in CI by default

### Performance Testing
- **Realistic Load**: Use production-like data sizes
- **Multiple Runs**: Use benchmark libraries for statistical analysis
- **Baseline Comparison**: Track performance regressions
- **Resource Monitoring**: Monitor memory and CPU usage

## 8. Quality Gates

### Coverage Requirements
- **Unit Tests**: ≥90% coverage
- **Critical Modules**: ≥95% coverage
- **New Code**: 100% coverage required

### Performance Benchmarks
- **Response Time**: <100ms for typical operations
- **Memory Usage**: <50MB for standard workloads
- **Scalability**: Linear performance scaling

### CI/CD Quality Gates
- All unit tests pass
- Coverage requirements met
- No performance regressions
- Linting and type checking pass

## 9. Troubleshooting

### Common Issues

**Mock Not Working**
```python
# Wrong: patching after import
from src.module import Class
@patch('src.module.Class')

# Correct: patch at import location
@patch('module.Class')
```

**Live Tests Failing**
```bash
# Check environment variable
export ENABLE_LIVE_TESTS=true

# Verify service availability
curl https://api.example.com/health
```

**Coverage Not Generated**
```bash
# Clear cache and rerun
rm -rf .pytest_cache htmlcov/
python scripts/run_tests.py coverage
```

### Debug Commands
```bash
# Run single test with debug output
source activate mcode_translator && python -m pytest tests/unit/test_example.py::TestClass::test_method -v -s

# Run with coverage details
source activate mcode_translator && python -m pytest --cov=src --cov-report=term-missing tests/

# Profile test performance
source activate mcode_translator && python -m pytest tests/performance/ --benchmark-only --benchmark-histogram
```

## 10. Maintenance

### Adding New Tests
1. Identify the test type (unit/integration/performance)
2. Create test file in appropriate directory
3. Use existing fixtures from `conftest.py`
4. Follow naming conventions
5. Add appropriate markers (`@pytest.mark.mock`, etc.)

### Updating Test Data
1. Add new fixtures to `conftest.py`
2. Create data files in `tests/data/`
3. Update factories for dynamic data generation
4. Ensure cleanup in teardown methods

### Performance Monitoring
1. Run performance tests regularly
2. Track benchmark results over time
3. Set up alerts for performance regressions
4. Update benchmarks when optimizing code

This testing strategy ensures the mCODE Translator maintains high quality, reliability, and performance through comprehensive automated testing at multiple levels.