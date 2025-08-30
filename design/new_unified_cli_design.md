# New Unified CLI Design

## Overview
This document outlines the design for a new unified command-line interface that will obsolete the current `mcode-cli.py` and provide comprehensive optimization capabilities across all prompts and models.

## Current CLI Limitations
The current CLI has several limitations:
1. Scattered functionality across multiple command groups
2. Limited optimization capabilities
3. No direct way to set default prompts
4. Complex command structure

## New CLI Design Principles
1. **Unified Interface**: Single entry point for all operations
2. **Intuitive Commands**: Clear, concise command structure
3. **Comprehensive Optimization**: Full optimization across prompts and models
4. **Default Management**: Ability to set and manage default prompts
5. **Detailed Reporting**: Comprehensive results and analysis

## Command Structure
```
mcode-optimize [OPTIONS] COMMAND [ARGS]...

Options:
 --help  Show this message and exit.

Commands:
  run        Run full optimization across all prompts and models
  set-default Set the best performing prompt as default
  view-results View optimization results and best combinations
  list-prompts List available prompts in the library
  show-prompt Show the content of a specific prompt
  benchmark  Run performance benchmarks
  validate   Validate against gold standard
```

## Detailed Command Specifications

### 1. run
```
mcode-optimize run [OPTIONS]

Run full optimization across all prompts and models.

Options:
  --test-cases PATH        Path to test cases JSON file [required]
  --gold-standard PATH     Path to gold standard JSON file [required]
  --prompt-config PATH     Path to prompt configuration JSON file
  --api-configs PATH       Path to unified configuration JSON file
  --output DIR             Output directory for results [required]
  --metric TEXT            Metric to optimize for (f1_score, precision, recall, compliance_score)
  --top-n INTEGER         Number of top combinations to consider
  --help                   Show this message and exit.
```

### 2. set-default
```
mcode-optimize set-default [OPTIONS]

Set the best performing prompt as default for its type.

Options:
  --prompt-type TEXT      Prompt type (NLP_EXTRACTION, MCODE_MAPPING)
  --prompt-name TEXT      Specific prompt name to set as default
  --from-results          Use best results from latest optimization
  --metric TEXT           Metric to use for selection (f1_score, precision, recall, compliance_score)
  --results-dir PATH      Directory containing benchmark results
  --help                  Show this message and exit.
```

### 3. view-results
```
mcode-optimize view-results [OPTIONS]

View optimization results and best combinations.

Options:
  --results-dir PATH      Directory containing benchmark results
  --metric TEXT           Metric to sort by (f1_score, precision, recall, compliance_score)
  --top-n INTEGER         Number of top combinations to show
 --format TEXT           Output format (table, json, csv)
  --export PATH           Export results to file
  --help                  Show this message and exit.
```

### 4. list-prompts
```
mcode-optimize list-prompts [OPTIONS]

List available prompts in the library.

Options:
  --type TEXT             Filter by prompt type (NLP_EXTRACTION, MCODE_MAPPING)
  --status TEXT           Filter by status (production, experimental)
  --default-only          Show only default prompts
  --config PATH           Path to prompt configuration file
  --help                  Show this message and exit.
```

### 5. show-prompt
```
mcode-optimize show-prompt [OPTIONS] PROMPT_NAME

Show the content of a specific prompt.

Options:
  --config PATH           Path to prompt configuration file
  --format                Format the prompt with example placeholders
  --requirements          Show prompt template requirements
  --help                  Show this message and exit.
```

### 6. benchmark
```
mcode-optimize benchmark [OPTIONS]

Run performance benchmarks.

Options:
  --test-cases PATH       Path to test cases JSON file [required]
  --prompt-config PATH    Path to prompt configuration JSON file
  --api-configs PATH      Path to unified configuration JSON file
  --output DIR            Output directory for results [required]
  --models TEXT           Comma-separated list of models to test
  --prompts TEXT          Comma-separated list of prompts to test
  --help                  Show this message and exit.
```

### 7. validate
```
mcode-optimize validate [OPTIONS]

Validate against gold standard.

Options:
  --test-cases PATH       Path to clinical trial test cases JSON file [required]
  --gold-standard PATH    Path to gold standard validation JSON file [required]
  --output PATH           Save detailed validation results to file
  --help                  Show this message and exit.
```

## Implementation Plan

### Phase 1: Core CLI Structure
1. Create new CLI entry point
2. Implement basic command structure
3. Add help and version information

### Phase 2: Optimization Commands
1. Implement `run` command with full optimization capabilities
2. Add support for different metrics and top-N selection
3. Implement result saving and loading

### Phase 3: Default Management
1. Implement `set-default` command
2. Add logic for selecting best prompts based on metrics
3. Implement persistence for default prompt settings

### Phase 4: Reporting and Analysis
1. Implement `view-results` command with sorting and filtering
2. Add export functionality in multiple formats
3. Implement `list-prompts` and `show-prompt` commands

### Phase 5: Additional Features
1. Implement `benchmark` command for performance testing
2. Add `validate` command for gold standard validation
3. Add comprehensive error handling and logging

## Technical Details

### Configuration Management
The CLI will use the existing configuration files:
- `config.json` for API and model configurations
- `prompts/prompts_config.json` for prompt library
- `models/models_config.json` for model library

### Result Storage
Benchmark results will be stored in JSON format in the specified output directory, with filenames following the pattern `benchmark_{timestamp}.json`.

### Default Prompt Persistence
Default prompt settings will be stored directly in the `prompts/prompts_config.json` file by adding a `default: true` field to the appropriate prompt entries.

## Error Handling
The CLI will implement comprehensive error handling:
1. **File Validation**: Check existence and accessibility of all input files
2. **Configuration Validation**: Validate JSON structure and required fields
3. **Runtime Errors**: Graceful handling of API errors, timeouts, and other runtime issues
4. **User Feedback**: Clear error messages with suggestions for resolution

## Testing Strategy
1. **Unit Tests**: Test individual command functions
2. **Integration Tests**: Test command combinations and workflows
3. **End-to-End Tests**: Test complete optimization workflows
4. **Manual Testing**: Verify CLI behavior with real data

## Documentation
1. **Command Reference**: Detailed documentation for each command
2. **Examples**: Practical examples for common use cases
3. **Troubleshooting**: Common issues and solutions
4. **Best Practices**: Recommendations for optimization workflows

## Rollout Plan
1. Develop core CLI structure
2. Implement optimization commands
3. Add default management functionality
4. Implement reporting and analysis features
5. Add additional features and utilities
6. Comprehensive testing
7. Documentation and user training