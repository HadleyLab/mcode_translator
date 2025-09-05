# Mcode Optimize CLI - Comprehensive Documentation

## Overview

The `Mcode-optimize.py` CLI provides a unified interface for prompt and model optimization with the ability to set the best performing combinations as defaults in their respective libraries. This replaces the older `Mcode-cli.py` for optimization tasks.

## Installation and Setup

```bash
# Make the script executable
chmod +x Mcode-optimize.py

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Available Commands

### 1. Run Full Optimization

```bash
# Run optimization across all prompts and models
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --prompt-config prompts/prompts_config.json \
  --model-config models/models_config.json \
  --output optimization_results \
  --metric f1_score \
  --top-n 5
```

**Options:**
- `--test-cases`: Path to clinical trial test cases JSON file (required)
- `--gold-standard`: Path to gold standard validation JSON file (required)
- `--prompt-config`: Path to prompt configuration file (default: prompts/prompts_config.json)
- `--model-config`: Path to model configuration file (default: models/models_config.json)
- `--output`: Output directory for results (required)
- `--metric`: Metric to optimize for (f1_score, precision, recall, compliance_score) (default: f1_score)
- `--top-n`: Number of top combinations to consider (default: 5)
- `--prompt-filter`: Filter prompts by name (can be used multiple times)
- `--prompt-type-filter`: Filter prompts by type (can be used multiple times)
- `--model-filter`: Filter models by name (can be used multiple times)
- `--model-type-filter`: Filter models by type (can be used multiple times)

### 2. Set Default Prompts/Models

```bash
# Set specific prompt as default
python Mcode-optimize.py set-default \
  --prompt-type NLP_EXTRACTION \
  --prompt-name comprehensive_extraction

# Set specific model as default
python Mcode-optimize.py set-default \
  --model-name gpt-4-turbo

# Set defaults based on best results from optimization
python Mcode-optimize.py set-default \
  --from-results \
  --results-dir optimization_results \
  --metric f1_score
```

**Options:**
- `--prompt-type`: Prompt type to set as default (NLP_EXTRACTION/MCODE_MAPPING)
- `--prompt-name`: Specific prompt name to set as default
- `--model-name`: Specific model name to set as default
- `--from-results`: Use best results from latest optimization
- `--metric`: Metric to use for selection (f1_score, precision, recall, compliance_score) (default: f1_score)
- `--results-dir`: Directory containing benchmark results

### 3. View Optimization Results

```bash
# View results in table format
python Mcode-optimize.py view-results \
  --results-dir optimization_results \
  --metric f1_score \
  --top-n 10 \
  --format table

# Export results to JSON
python Mcode-optimize.py view-results \
  --results-dir optimization_results \
  --format json \
  --export results.json

# Export results to CSV
python Mcode-optimize.py view-results \
  --results-dir optimization_results \
  --format csv \
  --export results.csv
```

**Options:**
- `--results-dir`: Directory containing benchmark results JSON files (required)
- `--metric`: Metric to sort by (f1_score, precision, recall, compliance_score) (default: f1_score)
- `--top-n`: Number of top combinations to show (default: 10)
- `--format`: Output format (table, json, csv) (default: table)
- `--export`: Export results to file

### 4. List Available Prompts

```bash
# List all prompts
python Mcode-optimize.py list-prompts

# List only extraction prompts
python Mcode-optimize.py list-prompts --type NLP_EXTRACTION

# List only production prompts
python Mcode-optimize.py list-prompts --status production

# List only default prompts
python Mcode-optimize.py list-prompts --default-only
```

**Options:**
- `--type`: Filter by prompt type (extraction/mapping)
- `--status`: Filter by status (production/experimental)
- `--config`: Path to prompt configuration file (default: prompts/prompts_config.json)
- `--default-only`: Show only default prompts

### 5. Show Prompt Content

```bash
# Show raw prompt content
python Mcode-optimize.py show-prompt comprehensive_extraction

# Show formatted prompt with examples
python Mcode-optimize.py show-prompt comprehensive_extraction --format

# Show prompt requirements
python Mcode-optimize.py show-prompt comprehensive_extraction --requirements
```

**Options:**
- `--config`: Path to prompt configuration file (default: prompts/prompts_config.json)
- `--format`: Format the prompt with example placeholders
- `--requirements`: Show prompt template requirements

### 6. List Available Models

```bash
# List all models
python Mcode-optimize.py list-models

# List models by type
python Mcode-optimize.py list-models --type openai

# List only production models
python Mcode-optimize.py list-models --status production

# List only default models
python Mcode-optimize.py list-models --default-only
```

**Options:**
- `--type`: Filter by model type
- `--status`: Filter by status (production/experimental)
- `--config`: Path to model configuration file (default: models/models_config.json)
- `--default-only`: Show only default models

### 7. Show Model Configuration

```bash
# Show model configuration
python Mcode-optimize.py show-model gpt-4-turbo
```

**Options:**
- `--config`: Path to model configuration file (default: models/models_config.json)

## Usage Examples

### Example 1: Complete Optimization Workflow

```bash
# Step 1: Run full optimization
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output optimization_results_2024 \
  --metric f1_score \
  --top-n 10

# Step 2: View results
python Mcode-optimize.py view-results \
  --results-dir optimization_results_2024 \
  --metric f1_score \
  --top-n 5

# Step 3: Set best performing combination as defaults
python Mcode-optimize.py set-default \
  --from-results \
  --results-dir optimization_results_2024 \
  --metric f1_score
```

### Example 2: Testing Specific Prompts

```bash
# Test a specific prompt against all test cases
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output prompt_test_results \
  --metric precision

# View detailed results for the prompt test
python Mcode-optimize.py view-results \
  --results-dir prompt_test_results \
  --format json \
  --export prompt_performance.json
```

### Example 3: Model Comparison

```bash
# Compare different models with the same prompt
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output model_comparison \
  --metric recall

# Analyze model performance
python Mcode-optimize.py view-results \
  --results-dir model_comparison \
  --metric recall \
  --top-n 15
```

### Example 4: Using Prompt and Model Filters

```bash
# Run optimization with specific prompts only
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output filtered_prompt_results \
  --prompt-filter comprehensive_extraction \
  --prompt-filter standard_mapping

# Run optimization with specific prompt types only
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output extraction_only_results \
  --prompt-type-filter NLP_EXTRACTION

# Run optimization with specific models only
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output filtered_model_results \
  --model-filter gpt-4 \
  --model-filter claude

# Run optimization with specific model types only
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output openai_only_results \
  --model-type-filter openai

# Combine prompt and model filters
python Mcode-optimize.py run \
  --test-cases tests/data/test_cases/clinical.json \
  --gold-standard tests/data/gold_standard/multi_cancer.json \
  --output combined_filter_results \
  --prompt-type-filter NLP_EXTRACTION \
  --model-type-filter openai
```

## Output Files

The optimization process generates several output files:

1. **`optimization_results.json`**: Complete results of all test runs
2. **Performance reports**: Detailed analysis of metrics and performance
3. **Configuration backups**: Snapshots of prompt and model configurations

## Best Practices

1. **Regular Optimization**: Run optimization regularly as new prompts and models are added
2. **Version Control**: Keep optimization results under version control for historical comparison
3. **Gold Standard Updates**: Update gold standard files as new test cases are developed
4. **Performance Monitoring**: Track performance metrics over time to identify trends
5. **Backup Configurations**: Always backup configurations before making changes
6. **Use Filters for Targeted Optimization**: Use prompt and model filters to focus optimization on specific components rather than running all combinations

## Troubleshooting

### Common Issues

1. **Missing test cases**: Ensure test case files exist and are properly formatted
2. **API configuration errors**: Verify API keys and configuration in `.env` file
3. **Prompt validation errors**: Check that prompts contain required placeholders
4. **Memory issues**: For large test sets, consider running in batches

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python Mcode-optimize.py run --test-cases ... --gold-standard ...
```

## Migration from Mcode-cli.py

The new `Mcode-optimize.py` replaces the optimization functionality in `Mcode-cli.py`. Key differences:

1. **Unified interface**: All optimization commands in one place
2. **Model integration**: Support for model library integration
3. **Default management**: Built-in support for setting defaults
4. **Enhanced reporting**: More detailed performance analysis

For validation and testing tasks, continue using `Mcode-cli.py validate` commands.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the generated log files
3. Consult the prompt and model library documentation
4. Check the test case and gold standard file formats