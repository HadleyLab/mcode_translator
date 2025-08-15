# Refactoring Plan: Aligning NLP Implementation with Documentation

## Goals
1. Rename `llm_interface.py` to `llm_nlp_engine.py` to align with naming conventions
2. Standardize class and file naming across NLP components
3. Improve code organization and structure
4. Ensure consistency between documentation and implementation
5. Enhance maintainability and extensibility of the NLP module

## Current State Analysis

### Key Files and Components
1. **src/llm_interface.py** - Contains `LLMInterface` class for LLM-based extraction
2. **src/simple_nlp_engine.py** - Contains `SimpleNLPEngine` class with regex-based extraction
3. **Documentation** - References a generic "NLP Engine" without specific implementation details

### Identified Discrepancies
1. Naming inconsistency between documentation ("NLP Engine") and implementation (`llm_interface.py`)
2. Lack of clear abstraction between different NLP implementations
3. Missing factory pattern or interface definition for NLP components
4. Limited documentation on the relationship between different NLP implementations

## Refactoring Steps

### 1. File and Class Renaming
- Rename `llm_interface.py` to `llm_nlp_engine.py`
- Keep `simple_nlp_engine.py` as is (represents a simpler implementation)
- Create `__init__.py` in src/ to define module exports

### 2. Interface Definition
- Define a common `NLP_Engine_Interface` abstract base class
- Define core methods that all NLP implementations must implement:
  - `extract_mcode_features(criteria_text: str) -> Dict`
  - `clean_text(text: str) -> str`
  - `identify_sections(text: str) -> Dict[str, str]`

### 3. Code Organization Improvements
- Move NLP-related files to a dedicated `nlp` directory
- Create submodules for different implementation approaches:
  - `nlp/llm/` - LLM-based implementation
  - `nlp/simple/` - Regex-based implementation
- Add utility modules for common NLP functions

### 4. Documentation Updates
- Update architecture diagrams to show both NLP implementations
- Clarify use cases for each NLP implementation in the documentation
- Update API references to reflect new naming and structure
- Add implementation notes about versioning and evolution

### 5. Dependency Management
- Update imports in dependent modules (extraction_pipeline.py, etc.)
- Add version tracking for NLP implementations
- Implement backward compatibility layer if needed

### 6. Testing Strategy
- Add unit tests for renamed/refactored components
- Add integration tests to verify functionality remains unchanged
- Add regression tests for previously identified edge cases

## Migration Path

### Short-Term (Current Task)
1. Rename llm_interface.py to llm_nlp_engine.py
2. Update class name from LLMInterface to LLMNLPEnigne
3. Update imports in referencing files
4. Verify existing tests pass

### Mid-Term (Next Iteration)
1. Implement common interface for NLP engines
2. Restructure codebase to support multiple implementations
3. Add proper module documentation

### Long-Term (Future Enhancement)
1. Add factory pattern for selecting NLP implementation
2. Implement configuration-driven engine selection
3. Add metrics collection for comparing different implementations

## Impact Assessment

### Affected Components
- src/extraction_pipeline.py (references LLMInterface)
- src/nicegui_interface.py (may reference NLP components)
- test_llm_mcode_extraction.py
- Documentation files (architecture, system documentation)

### Risk Mitigation Strategies
1. Comprehensive testing before and after refactoring
2. Maintain backward compatibility where possible
3. Clear versioning to track changes
4. Detailed migration guide for developers

## Versioning Considerations
- This refactoring will be considered a minor version update (1.x -> 1.x+1)
- Breaking changes will be documented in release notes
- Backward compatibility layer can be added if needed

## Success Criteria
1. All existing functionality works as before
2. Code structure better reflects documentation
3. Both NLP implementations can coexist and be selected at runtime
4. All tests pass
5. Updated documentation accurately reflects implementation