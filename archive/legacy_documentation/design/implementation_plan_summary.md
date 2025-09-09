# mCODE Translator - Default Prompt Functionality and Unified CLI Implementation Plan

## Project Overview
This document summarizes the implementation plan for adding default prompt functionality to the mCODE Translator system and creating a new unified command-line interface that will obsolete the current `mCODE-cli.py`.

## Key Objectives
1. Enable setting and using default prompts for NLP extraction and mCODE mapping
2. Create a new unified CLI with comprehensive optimization capabilities
3. Allow full optimization across all prompts and models
4. Provide functionality to set the best performing prompts as defaults in the prompts library

## Completed Design Work

### 1. Default Prompt Functionality Design
- **Document**: `design/default_prompt_functionality.md`
- **Status**: Complete
- **Key Features**:
  - Add `default` field to prompt configuration
  - Enhance `PromptOptimizationFramework` with methods to set and save default prompts
  - Update `PromptLoader` to recognize and load default prompts
  - Add UI elements to manage default prompts

### 2. New Unified CLI Design
- **Document**: `design/new_unified_cli_design.md`
- **Status**: Complete
- **Key Features**:
  - Simplified command structure with intuitive commands
  - Comprehensive optimization across all prompts and models
  - Default prompt management capabilities
  - Detailed reporting and analysis features

## Implementation Progress

### Phase 1: Backend Implementation (Completed)
- ✅ Analyzed existing codebase to understand implementation requirements
- ✅ Designed default prompt functionality
- ✅ Designed new unified CLI

### Phase 2: Framework Enhancements (In Progress)
- ✅ Add methods to `PromptOptimizationFramework` to set and save default prompts

### Phase 3: UI Implementation (Pending)
- ⬜ Modify `OptimizationUI` to display and manage default prompts
- ⬜ Update results tab to show options for setting default prompts after benchmark validation

### Phase 4: CLI Implementation (Pending)
- ⬜ Implement CLI command for running optimization across all prompts and models
- ⬜ Add CLI functionality to set the best performing prompts as defaults in the prompts library
- ⬜ Implement CLI command for viewing optimization results and best combinations

### Phase 5: Documentation and Testing (Pending)
- ⬜ Create comprehensive documentation for the new CLI
- ⬜ Implement persistence for default prompt settings

## Technical Approach

### Default Prompt Implementation
1. **Configuration Changes**: Add `default: true` field to prompt entries in `prompts/prompts_config.json`
2. **Framework Methods**: 
   - `get_best_combinations()` - Get best prompt-model combinations based on metrics
   - `set_default_prompt()` - Set a prompt as default for its type
   - `save_prompt_config()` - Save updated configuration to disk
   - `get_default_prompt()` - Get the default prompt for a given type
3. **UI Integration**: Add visual indicators and controls for default prompts

### New CLI Implementation
1. **Core Structure**: Create new CLI entry point with intuitive command structure
2. **Optimization Commands**: Implement comprehensive optimization across all prompts and models
3. **Default Management**: Add commands to set and manage default prompts
4. **Reporting**: Implement detailed reporting and analysis features

## Command Structure Examples

### New CLI Commands
```
# Run full optimization
mCODE-optimize run --test-cases data/test_cases.json --gold-standard data/gold_standard.json --output results/

# Set best prompt as default
mCODE-optimize set-default --from-results --metric f1_score

# View optimization results
mCODE-optimize view-results --metric f1_score --top-n 10

# List prompts
mCODE-optimize list-prompts --default-only
```

## Implementation Timeline

### Week 1: UI Implementation
- Modify `OptimizationUI` to display default prompts
- Add controls for setting defaults in results tab

### Week 2: CLI Development
- Implement core CLI structure
- Add optimization and default management commands

### Week 3: Persistence and Documentation
- Implement persistence for default prompt settings
- Create comprehensive documentation

### Week 4: Testing and Deployment
- Comprehensive testing of all features
- User acceptance testing
- Deployment preparation

## Success Criteria
1. Users can run full optimization across all prompts and models
2. Users can set best performing prompts as defaults
3. Default prompts are properly persisted and loaded
4. New CLI provides intuitive interface for all operations
5. Existing functionality is maintained or improved

## Risk Mitigation
1. **Backward Compatibility**: Ensure existing configurations and workflows continue to work
2. **Data Integrity**: Implement robust error handling for configuration updates
3. **Performance**: Optimize optimization algorithms to handle large prompt/model combinations
4. **User Experience**: Provide clear feedback and progress indicators during long-running operations

## Next Steps
1. Implement UI modifications for default prompt management
2. Begin CLI implementation with core command structure
3. Continue framework enhancements for default prompt functionality