# Testing Reorganization Plan

## Current Issues Identified

1. **Duplication**: Multiple breast cancer pipeline test files
2. **Inconsistent Structure**: Tests scattered across root and src/testing
3. **Unclear Naming**: File names don't clearly indicate purpose
4. **Mixed Concerns**: Integration, unit, and E2E tests mixed together

## Proposed New Structure

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

## File Renaming Strategy

### Current → New
- `test_comprehensive_cancer_cases.py` → `tests/e2e/multi_cancer/test_multi_cancer_pipeline.py`
- `test_optimization_framework.py` → `tests/unit/test_optimization_framework.py`

## Data File Organization

### Gold Standard Files
- `examples/gold_standard/breast_cancer_gold_standard.json` → `tests/data/gold_standard/breast_cancer.json`
- `examples/gold_standard/various_cancers_gold_standard.json` → `tests/data/gold_standard/multi_cancer.json`

### Test Case Files
- `examples/test_cases/breast_cancer_test_case.json` → `tests/data/test_cases/breast_cancer.json`
- `examples/test_cases/various_cancers_test_cases.json` → `tests/data/test_cases/multi_cancer.json`

## Key Principles

1. **Breast Cancer as Default**: All default configurations and runners focus on breast cancer
2. **Extensibility**: Clear comments and patterns for extending to other cancers
3. **Separation of Concerns**: Unit, integration, and E2E tests separated
4. **Explicit Naming**: Files clearly indicate their purpose and scope

## Implementation Steps

1. Create new directory structure
2. Move and rename files according to plan
3. Update imports and references
4. Remove duplicate files
5. Create test runners
6. Update documentation
7. Verify all tests work

## Dead Code to Prune

- Duplicate breast cancer pipeline test files
- Unused legacy test data files
- Redundant test case structures

## Extensibility Comments

All breast cancer tests should include comments like:
```python
# DEFAULT: Breast cancer testing
# EXTEND: To add support for other cancer types, follow this pattern:
# 1. Create new test file in tests/e2e/{cancer_type}/
# 2. Add corresponding gold standard data
# 3. Update test runners to include new cancer type