# Default Prompt Functionality Design

## Overview
This document outlines the design for implementing default prompt functionality in the mCODE Translator system. The goal is to allow users to set and use default prompts for NLP extraction and mCODE mapping after benchmark validation.

## Current System Architecture
The current system consists of:
1. **Prompt Library**: File-based prompt storage in `prompts/prompts_config.json`
2. **Prompt Loader**: `src/utils/prompt_loader.py` handles loading prompts from the library
3. **Optimization Framework**: `src/optimization/prompt_optimization_framework.py` runs benchmarks and optimization
4. **Optimization UI**: Web interface for managing prompts and running benchmarks
5. **CLI**: Command-line interface for various operations

## Proposed Changes

### 1. Prompt Configuration Structure
Add a `default` field to the prompt configuration to mark default prompts:

```json
{
  "prompt_library": {
    "version": "1.0.0",
    "created_date": "2024-08-26",
    "description": "Strict implementation - only working prompts retained",
    "prompts": {
      "production": {
        "nlp_extraction": [
          {
            "name": "generic_extraction",
            "prompt_type": "NLP_EXTRACTION",
            "prompt_file": "txt/nlp_extraction/generic_extraction.txt",
            "description": "Generic clinical entity extraction prompt with comprehensive structure",
            "version": "1.0.0",
            "author": "Clinical NLP Team",
            "created_date": "2024-01-15",
            "status": "production",
            "default": true,
            "tags": ["generic", "comprehensive", "structured"]
          }
        ],
        "mcode_mapping": [
          {
            "name": "generic_mapping",
            "prompt_type": "MCODE_MAPPING",
            "prompt_file": "txt/mcode_mapping/generic_mapping.txt",
            "description": "Generic mCODE mapping prompt with comprehensive instructions",
            "version": "1.0.0",
            "author": "mCODE Team",
            "created_date": "2024-08-26",
            "status": "production",
            "default": true,
            "tags": ["generic", "comprehensive", "mCODE"]
          }
        ]
      }
    }
  }
}
```

### 2. PromptOptimizationFramework Enhancements
Add methods to:
- Get best performing prompt combinations based on benchmark results
- Set default prompts in the configuration
- Save updated configuration to disk

### 3. OptimizationUI Enhancements
Add UI elements to:
- Display which prompts are currently set as defaults
- Allow users to set prompts as defaults from the results tab
- Show default prompt settings in the configuration tab

### 4. New Unified CLI Design
Create a new CLI with commands for:
- Running full optimization across all prompts and models
- Setting best performing prompts as defaults
- Viewing optimization results and best combinations
- Managing prompt library

## Implementation Plan

### Phase 1: Backend Implementation
1. Modify `PromptOptimizationFramework` to add default prompt functionality
2. Update `PromptLoader` to recognize and load default prompts
3. Add persistence for default prompt settings

### Phase 2: UI Implementation
1. Modify `OptimizationUI` to display and manage default prompts
2. Add options in results tab to set defaults after benchmark validation

### Phase 3: CLI Implementation
1. Design and implement new unified CLI with full optimization capabilities
2. Add commands for setting and viewing default prompts
3. Create comprehensive documentation

## Technical Details

### Default Prompt Selection Criteria
Default prompts will be selected based on:
1. **F1 Score**: Primary metric for extraction and mapping performance
2. **Compliance Score**: Secondary metric for mCODE compliance
3. **Success Rate**: Reliability of the prompt-model combination
4. **Token Usage**: Efficiency consideration

### API Design
```python
# In PromptOptimizationFramework
def get_best_combinations(self, metric: str = 'f1_score', top_n: int = 5) -> pd.DataFrame:
    """Get the best prompt-model combinations based on the specified metric"""

def set_default_prompt(self, prompt_type: str, prompt_name: str) -> None:
    """Set a prompt as default for its type"""

def save_prompt_config(self) -> None:
    """Save updated prompt configuration to disk"""

def get_default_prompt(self, prompt_type: str) -> str:
    """Get the default prompt for a given type"""
```

## CLI Command Structure
```
mCODE-optimize run [--test-cases PATH] [--gold-standard PATH] [--output DIR]
mCODE-optimize set-default [--prompt-type TYPE] [--prompt-name NAME]
mCODE-optimize view-results [--metric METRIC] [--top-n N]
mCODE-optimize list-prompts [--type TYPE] [--default-only]
```

## Testing Strategy
1. Unit tests for new framework methods
2. Integration tests for UI components
3. End-to-end tests for CLI commands
4. Manual testing of default prompt functionality

## Rollout Plan
1. Implement backend functionality
2. Add UI elements
3. Develop new CLI
4. Comprehensive testing
5. Documentation and user training