# Testing Cleanup Plan

## Overview
This document outlines the plan for cleaning up the testing infrastructure in the mCODE Translator project by:
1. Moving bare tests to proper pytest infrastructure
2. Removing legacy or obsolete code and data files
3. Consolidating redundant or duplicate files
4. Ensuring all tests follow strict implementation principles
5. Verifying a clean, minimal codebase structure

## Files to Move to Pytest Infrastructure

### Root Directory Test Files
These files should be moved to the proper pytest infrastructure in `tests/` directory:

1. `test_all_benchmarks.py` - Move to `tests/unit/`
2. `test_benchmark_fix.py` - Move to `tests/unit/`
3. `test_mcode_validation.py` - Move to `tests/unit/`
4. `test_model_library_demo.py` - Move to `tests/unit/`
5. `test_multi_model_token_usage.py` - Move to `tests/unit/`
6. `test_pipeline_methods.py` - Move to `tests/unit/`
7. `test_prompt_validation.py` - Move to `tests/unit/`
8. `test_simple_metrics.py` - Move to `tests/unit/`
9. `test_token_usage.py` - Move to `tests/unit/`
10. `test_token_usage_inline.py` - Move to `tests/unit/`
11. `test_validation_fix.py` - Move to `tests/unit/`
12. `test_validation_optimization.py` - Move to `tests/unit/`
13. `debug_validation.py` - Move to `tests/debug/` or remove if obsolete
14. `evaluate_prompts_pipeline.py` - Move to `tests/integration/` or remove if obsolete
15. `strict_validation_test.py` - Move to `tests/unit/`
16. `test_validation_optimization.py` - Move to `tests/unit/`

### Files to Remove (Obsolete/Legacy)
These files appear to be obsolete or legacy and should be removed:

1. `mcode-cli.py` - Appears to be an older version, replaced by `mcode-cli.py` in root
2. Any duplicate or redundant test files

## Directory Restructuring

### Current Structure
```
.
├── test_all_benchmarks.py
├── test_benchmark_fix.py
├── test_mcode_validation.py
├── test_model_library_demo.py
├── test_multi_model_token_usage.py
├── test_pipeline_methods.py
├── test_prompt_validation.py
├── test_simple_metrics.py
├── test_token_usage.py
├── test_token_usage_inline.py
├── test_validation_fix.py
├── test_validation_optimization.py
├── debug_validation.py
├── evaluate_prompts_pipeline.py
├── strict_validation_test.py
├── mcode-cli.py
├── tests/
│   ├── __init__.py
│   ├── test_combined_functionality.py
│   ├── test_prompt_library_integration.py
│   ├── data/
│   ├── e2e/
│   ├── results/
│   ├── runners/
│   ├── unit/
│   └── ...
```

### Proposed Structure
```
.
├── tests/
│   ├── __init__.py
│   ├── test_combined_functionality.py
│   ├── test_prompt_library_integration.py
│   ├── data/
│   ├── debug/
│   │   └── debug_validation.py
│   ├── e2e/
│   ├── integration/
│   │   └── evaluate_prompts_pipeline.py
│   ├── results/
│   ├── runners/
│   ├── unit/
│   │   ├── test_all_benchmarks.py
│   │   ├── test_benchmark_fix.py
│   │   ├── test_mcode_validation.py
│   │   ├── test_model_library_demo.py
│   │   ├── test_multi_model_token_usage.py
│   │   ├── test_pipeline_methods.py
│   │   ├── test_prompt_validation.py
│   │   ├── test_simple_metrics.py
│   │   ├── test_token_usage.py
│   │   ├── test_token_usage_inline.py
│   │   ├── test_validation_fix.py
│   │   ├── test_validation_optimization.py
│   │   └── strict_validation_test.py
│   └── ...
```

## Implementation Steps

### Step 1: Create Missing Directories
1. Create `tests/debug/` directory
2. Create `tests/integration/` directory

### Step 2: Move Files
1. Move all test files from root to appropriate directories in `tests/`
2. Update import statements in moved files
3. Update any relative path references

### Step 3: Update Test Imports
1. Update all import statements to reflect new file locations
2. Ensure all tests can still import required modules

### Step 4: Remove Obsolete Files
1. Remove `mcode-cli.py` from root (duplicate)
2. Remove any other truly obsolete files

### Step 5: Update Documentation
1. Update any documentation referencing old file locations
2. Update README files to reflect new structure

### Step 6: Verify Functionality
1. Run all tests to ensure nothing is broken
2. Verify that all functionality still works as expected

## Files That Need Special Attention

### `test_token_usage.py` and related files
These files are specifically for testing token usage tracking and should be moved to `tests/unit/` with appropriate renaming to follow pytest conventions.

### `evaluate_prompts_pipeline.py`
This appears to be an integration test and should be moved to `tests/integration/`.

### `debug_validation.py`
This is a debug script and should be moved to `tests/debug/`.

## Timeline
1. **Day 1**: Create directory structure and move files
2. **Day 2**: Update imports and fix any broken references
3. **Day 3**: Remove obsolete files and update documentation
4. **Day 4**: Verify all tests pass and functionality is intact

## Success Criteria
1. All tests pass with the new structure
2. No broken imports or references
3. Cleaner, more organized codebase
4. Proper separation of concerns in testing infrastructure
5. Removal of all obsolete/legacy code