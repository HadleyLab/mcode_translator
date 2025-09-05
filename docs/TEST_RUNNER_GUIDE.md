# Test Runner Guide

## Overview

This document describes the new logical testing structure for the Mcode Translator project. The testing has been reorganized to be more explicit, with breast cancer as the default focus and clear extensibility for other cancer types.

## Directory Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── pipeline/           # Pipeline component tests
│   ├── utils/             # Utility function tests
│   └── optimization/      # Optimization framework tests
├── integration/           # Integration tests
│   ├── pipeline_integration.py
│   └── optimization_integration.py
├── e2e/                   # End-to-end tests
│   ├── test_pipeline_simple.py          # Simple pipeline functionality tests
│   ├── test_pipeline_with_prompts.py    # Pipeline tests with prompt integration
│   └── multi_cancer/      # Multi-cancer extensible tests
│       ├── test_multi_cancer_pipeline.py
├── data/                  # Test data
│   ├── gold_standard/
│   │   ├── breast_cancer.json
│   │   └── multi_cancer.json
│   └── test_cases/
│       ├── breast_cancer.json
│       └── multi_cancer.json
└── runners/               # Test runners
    ├── run_all_tests.py
    └── run_breast_cancer_tests.py
```

## Running Tests

### Quick Start (Breast Cancer - DEFAULT)
```bash
# Run all breast cancer tests (recommended for most development)
python -m pytest tests/e2e/breast_cancer/ -v

# Run specific pipeline tests
python -m pytest tests/e2e/test_pipeline_simple.py -v
python -m pytest tests/e2e/test_pipeline_with_prompts.py -v
```

### Comprehensive Testing
```bash
# Run all tests (unit, integration, e2e)
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/unit/ -v

# Run integration tests only
python -m pytest tests/integration/ -v

# Run E2E tests only
python -m pytest tests/e2e/ -v
```

### Multi-Cancer Testing
```bash
# Run multi-cancer pipeline tests
python -m pytest tests/e2e/multi_cancer/test_multi_cancer_pipeline.py -v
```

## Test Types

### 1. Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Fast execution
- No external dependencies
- Focus on specific functionality

### 2. Integration Tests (`tests/integration/`)
- Test interactions between components
- Moderate execution speed
- May have some external dependencies
- Focus on component integration

### 3. End-to-End Tests (`tests/e2e/`)
- Test complete pipeline functionality
- Slower execution
- May use external APIs/services
- Focus on real-world scenarios

## Default Configuration

**Breast cancer is the default focus** for testing. This provides:
- Consistent baseline for development
- Well-defined gold standard data
- Comprehensive test coverage
- Faster feedback cycles

## Extending to Other Cancer Types

To add support for additional cancer types:

1. **Create cancer-specific directory**:
   ```bash
   mkdir tests/e2e/lung_cancer/
   ```

2. **Add gold standard data**:
   ```bash
   cp tests/data/gold_standard/breast_cancer.json tests/data/gold_standard/lung_cancer.json
   # Edit with lung cancer specific entities
   ```

3. **Create test cases**:
   ```bash
   cp tests/data/test_cases/breast_cancer.json tests/data/test_cases/lung_cancer.json
   # Edit with lung cancer test cases
   ```

4. **Create test files** (follow breast cancer pattern):
   ```python
   # tests/e2e/lung_cancer/test_lung_cancer_pipeline.py
   # Copy and adapt from breast cancer tests
   ```

5. **Update test runners** to include new cancer type

## Pruned Functionality

The following legacy features have been removed:
- Duplicate breast cancer test files
- Mixed test types in single files
- Unclear naming conventions
- Redundant test data structures

## Verification

After reorganization, verify all tests work:
```bash
# Run comprehensive test suite
python -m pytest tests/ -v

# Check specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/e2e/ -v
```

## Best Practices

1. **Default to breast cancer** for most development work
2. **Run unit tests frequently** during development
3. **Run integration tests** before committing
4. **Run E2E tests** before major releases
5. **Add new cancer types** following the established pattern
6. **Keep gold standard data** up to date with clinical standards

## Running Tests with Live API Calls

### DeepSeek LLM Integration

The E2E tests support live API calls to DeepSeek LLM for real-time entity extraction and mapping. To use live API calls:

#### Prerequisites
1. **API Key Configuration**: Set your DeepSeek API key as an environment variable:
   ```bash
   export DEEPSEEK_API_KEY="your_api_key_here"
   ```

2. **Environment Setup**: Ensure you're using the correct conda environment:
   ```bash
   conda activate mcode_translator
   ```

#### Running Tests with Live API

```bash
# Run pipeline tests with live API calls
python -m pytest tests/e2e/test_pipeline_with_prompts.py -v

# Run simple pipeline tests with live API
python -m pytest tests/e2e/test_pipeline_simple.py -v

# Run all pipeline tests with live API
python -m pytest tests/e2e/ -v
```

#### Expected Behavior with Live API
- **API Calls**: Tests will make real HTTP requests to DeepSeek API
- **Response Time**: Each API call takes approximately 45-60 seconds
- **JSON Parsing**: Automatic JSON repair for malformed API responses
- **Results**: Test results saved to `results/breast_cancer_test/breast_cancer_test_result.json`

#### Performance Metrics
Typical performance with live DeepSeek API:
- **Extraction**: 7-10 entities extracted from clinical trial data
- **Mapping**: 7 Mcode elements mapped with SNOMEDCT/LOINC codes
- **Metrics**: Precision/Recall/F1 scores calculated against gold standard
- **Success Rate**: High success rate with automatic JSON repair for parsing issues

#### Troubleshooting
- **API Key Issues**: Verify `DEEPSEEK_API_KEY` is set correctly
- **JSON Parsing Errors**: System automatically attempts JSON repair
- **Timeout Issues**: API calls have built-in timeout handling
- **Results Location**: Check `results/` directory for output files

#### Best Practices for Live Testing
1. **Run selectively**: Use live API tests for validation, not frequent development
2. **Monitor costs**: Be aware of API usage costs with DeepSeek
3. **Check results**: Review generated JSON files for extraction quality
4. **Update gold standard**: Keep gold standard data current with API behavior changes