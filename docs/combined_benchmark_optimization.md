# Combined Benchmark + Optimization Framework

## Overview

The MCODE Translator now features a **combined benchmark and optimization framework** that integrates the best of both worlds:

- **Benchmarking**: Performance validation against gold standards with proper metrics
- **Optimization**: Experimental testing of different prompt configurations and API setups

## Key Features

### 1. Integrated Metrics Calculation
- Uses actual gold standard data for metric calculation
- No more placeholder metrics (0.8 precision, 0.7 recall)
- Real semantic similarity-based evaluation
- Proper extraction and mapping accuracy metrics

### 2. Unified Configuration
- Single configuration file for prompt variants
- Support for multiple API configurations
- Test cases with embedded gold standard data
- Comprehensive results reporting

### 3. Combined CLI Command
```bash
python mcode-cli.py combined optimize-and-benchmark \
    --test-cases examples/test_cases.json \
    --prompt-variants examples/combined_optimization_config.json \
    --api-configs examples/config/api_configs.json \
    --output ./combined_results \
    --cases case1 case2
```

## How It Works

### Before (Separate Systems)
- **Benchmark**: Used GoldStandardTester with proper metrics but only for validation
- **Optimization**: Used placeholder metrics (0.8 precision, 0.7 recall) for experimental testing

### After (Combined System)
- **Both**: Use actual gold standard validation for ALL testing
- **Both**: Get real metrics for both validation and optimization
- **Both**: Single unified framework for all performance testing

## Configuration Files

### 1. Prompt Variants Configuration
```json
{
  "variants": [
    {
      "name": "Basic Extraction Prompt",
      "prompt_type": "nlp_extraction",
      "prompt_key": "basic_extraction_prompt",
      "description": "Simple extraction prompt",
      "version": "1.0.0",
      "tags": ["extraction", "basic"],
      "parameters": {}
    }
  ]
}
```

### 2. Test Cases with Gold Standard
```json
{
  "test_cases": {
    "case1": {
      "clinical_data": { ... },
      "expected_extraction": {
        "entities": [
          {"text": "breast cancer", "type": "condition"},
          {"text": "chemotherapy", "type": "treatment"}
        ]
      },
      "expected_mcode_mappings": {
        "mapped_elements": [
          {"element_name": "CancerCondition", "value": "breast cancer"},
          {"element_name": "CancerRelatedMedicationAdministered", "value": "chemotherapy"}
        ]
      }
    }
  }
}
```

## Benefits

1. **Accurate Metrics**: No more guessing - real gold standard validation
2. **Efficiency**: Single framework for both validation and optimization
3. **Consistency**: Same evaluation methodology across all testing
4. **Better Insights**: Real metrics help identify truly better configurations
5. **Time Savings**: No need to run separate benchmark and optimization tests

## Usage Examples

### Basic Combined Testing
```bash
python mcode-cli.py combined optimize-and-benchmark \
    --test-cases test_data/gold_standard_cases.json \
    --prompt-config prompts/prompts_config.json \
    --api-configs config/api_configs.json \
    --output results/combined_test
```

### Specific Test Cases
```bash
python mcode-cli.py combined optimize-and-benchmark \
    --test-cases test_data/gold_standard_cases.json \
    --prompt-config prompts/prompts_config.json \
    --api-configs config/api_configs.json \
    --output results/specific_cases \
    --cases breast_cancer_case lung_cancer_case
```

## Results Output

The combined framework generates comprehensive results including:
- Precision, Recall, F1 scores for extraction
- Mapping accuracy metrics
- Token usage and performance timing
- Detailed per-case and per-configuration breakdowns
- CSV and JSON output formats

## Migration Guide

If you were previously using separate benchmark and optimization commands:

1. **Update your test cases**: Ensure they include gold standard data
2. **Create prompt variants config**: Use the new JSON format
3. **Use the combined command**: Replace separate benchmark/optimize calls
4. **Review results**: Now with actual metrics instead of placeholders

## Performance Impact

The combined approach may take slightly longer to run since it performs actual gold standard validation for each test, but the results are significantly more accurate and meaningful for optimization decisions.