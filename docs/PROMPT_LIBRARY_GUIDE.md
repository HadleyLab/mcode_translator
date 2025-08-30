# Prompt Library Implementation Guide

## Introduction

This guide provides comprehensive documentation for the new file-based prompt library system implemented in the mCODE Translator project. The library centralizes all LLM prompts, eliminates hardcoded strings, and provides a robust system for prompt management and experimentation.

## Architecture Overview

### Before (Hardcoded System)
```
┌─────────────────┐
│   Source Code   │
│                 │
│  Hardcoded      │
│  Prompt Strings │
│                 │
└─────────────────┘
```

### After (File-Based Library)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source Code   │    │   PromptLoader  │    │  File Library   │
│                 │    │                 │    │                 │
│  prompt_loader. │───▶│   load_prompt() │───▶│  prompts/       │
│  load_prompt()  │    │   (cached)      │    │   ├── config.json
│                 │    │                 │    │   └── txt/      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Benefits

1. **Centralized Management**: All prompts in one location
2. **Version Control**: Track prompt changes through git
3. **Experimentation**: Easy A/B testing of prompt variations
4. **Reusability**: Prompts can be shared across components
5. **Maintainability**: No hardcoded strings to search for
6. **Documentation**: Built-in descriptions and metadata

## Implementation Details

### File Structure

```
prompts/
├── prompts_config.json          # Master configuration
├── txt/
│   ├── nlp_extraction/          # 6 NLP extraction prompts
│   └── mcode_mapping/           # 6 mCODE mapping prompts
└── README.md                    # Library documentation
```

### PromptLoader Class

The [`PromptLoader`](../src/utils/prompt_loader.py) class provides:

- **Caching**: Loads prompts once and caches them
- **Relative Path Handling**: Works across different environments
- **Formatting Support**: Handles template placeholders
- **Error Handling**: Validates prompt existence
- **Nested Config Support**: Handles complex configuration structures

### Configuration Schema

```json
{
  "category": {
    "prompt_key": {
      "file": "relative/path/to/prompt.txt",
      "description": "Prompt description",
      "metadata": {}  // Optional additional metadata
    }
  }
}
```

## Migration Process

### Step 1: Identify Hardcoded Prompts
Located all hardcoded prompts across the codebase:
- [`nlp_engine.py`](../src/pipeline/nlp_engine.py)
- [`mcode_mapper.py`](../src/pipeline/mcode_mapper.py) 
- [`strict_prompt_optimization_framework.py`](../src/optimization/strict_prompt_optimization_framework.py)
- [`prompts_config.json`](../prompts/prompts_config.json)

### Step 2: Create File Structure
Organized prompts into logical categories:
- `nlp_extraction/` - Entity extraction prompts
- `mcode_mapping/` - mCODE mapping prompts

### Step 3: Migrate Content
Moved all prompt content from code to text files while preserving:
- Original prompt content and structure
- Placeholder formatting ({variable_name})
- Intent and purpose of each prompt

### Step 4: Update Code References
Replaced all hardcoded prompt references with:
```python
# Before
prompt = "Extract entities from: {text}..."

# After  
prompt = self.prompt_loader.load_prompt("generic_extraction", text=input_text)
```

### Step 5: Test and Validate
Created comprehensive tests to ensure:
- All prompts load successfully
- Formatting works correctly
- No functionality regression
- Performance is maintained

## Usage Examples

### Basic Prompt Loading
```python
from src.utils.prompt_loader import PromptLoader

loader = PromptLoader()
prompt = loader.load_prompt("generic_extraction")
```

### Prompt with Formatting
```python
formatted_prompt = loader.load_prompt(
    "generic_extraction",
    text=clinical_text,
    json_schema=entity_schema,
    examples=example_entities
)
```

### In Pipeline Components
```python
# NLP Engine
def extract_entities(self, text: str, prompt_name: str = "generic_extraction"):
    template = self.prompt_loader.load_prompt(prompt_name)
    formatted = template.format(text=text)
    return self.llm_client.generate(formatted)

# mCODE Mapper
def map_to_mcode(self, entities: List, prompt_name: str = "generic_mapping"):
    template = self.prompt_loader.load_prompt(prompt_name)
    formatted = template.format(entities=json.dumps(entities))
    return self.llm_client.generate(formatted)
```

### Optimization Framework (Strict Mode)
```python
# Create test variants for strict optimization framework
variants = [
    PromptVariant(
        prompt_type=PromptType.NLP_EXTRACTION,
        template=prompt_loader.load_prompt("basic_extraction"),
        description="Basic extraction prompt"
    ),
    PromptVariant(
        prompt_type=PromptType.MCODE_MAPPING,
        template=prompt_loader.load_prompt("comprehensive_mapping"),
        description="Comprehensive mapping prompt"
    )
]
```

## Prompt Categories

### NLP Extraction Prompts (6 total)
- **generic_extraction**: Main clinical entity extraction
- **comprehensive_extraction**: Detailed extraction with full schema
- **minimal_extraction**: Minimalist extraction approach
- **structured_extraction**: Structured output formatting
- **basic_extraction**: Optimization framework baseline
- **minimal_extraction_optimization**: Minimal optimization variant

### mCODE Mapping Prompts (6 total)  
- **generic_mapping**: Primary mCODE mapping
- **standard_mapping**: Standard mapping approach
- **detailed_mapping**: Comprehensive mapping instructions
- **error_robust_mapping**: Error handling and fallbacks
- **comprehensive_mapping**: Optimization framework comprehensive
- **simple_mapping**: Optimization framework simplified

## Performance Considerations

### Caching Strategy
The `PromptLoader` implements a simple caching mechanism:
- Loads each prompt file once on first access
- Stores loaded content in memory for subsequent requests
- No performance penalty for multiple calls to same prompt

### Memory Usage
- Prompt files are typically 1-5KB each
- Total memory footprint: ~50KB for all prompts
- Negligible impact on overall application memory

### File I/O
- Single read operation per prompt on first access
- Subsequent accesses use cached content
- Optimal for production deployment

## Error Handling

The library includes comprehensive error handling:

### Configuration Errors
- Validates prompt configuration existence
- Checks file path accessibility
- Provides clear error messages for missing prompts

### File Access Errors
- Handles file not found errors gracefully
- Provides fallback behavior where appropriate
- Logs detailed error information for debugging

### Formatting Errors
- Validates placeholder consistency
- Provides clear error messages for missing format arguments
- Supports optional placeholders with default values

### Performance Metrics
- Tracks token usage for each prompt through unified token tracker
- Monitors API call performance and resource consumption
- Provides detailed metrics for cost optimization

## Testing and Validation

### Unit Tests
```python
def test_prompt_loading():
    loader = PromptLoader()
    
    # Test all prompts load successfully
    for prompt_name in EXPECTED_PROMPTS:
        prompt = loader.load_prompt(prompt_name)
        assert prompt is not None
        assert len(prompt) > 0
        
    # Test formatting
    formatted = loader.load_prompt("generic_extraction", text="test")
    assert "test" in formatted
```

### Integration Tests
- Verify prompts work with actual LLM calls
- Test end-to-end pipeline functionality
- Validate performance characteristics

## Best Practices

### Prompt Naming Convention
- Use descriptive, consistent names
- Follow `category_purpose_variant` pattern
- Avoid ambiguous or generic names

### Documentation
- Include descriptions in config file
- Document placeholder requirements
- Note any special formatting requirements

### Version Control
- Treat prompts as code
- Use meaningful commit messages for prompt changes
- Consider prompt versioning for major changes

### Testing
- Test new prompts before deployment
- Validate formatting with realistic data
- Monitor performance impact

## Future Enhancements

### Planned Features
1. **Prompt Versioning**: Track multiple versions of prompts
2. **A/B Testing Framework**: Built-in support for experimentation
3. **Performance Metrics**: Track prompt effectiveness
4. **Template Validation**: Validate placeholder consistency
5. **Internationalization**: Support for multiple languages

### Extension Points
1. **Custom Loaders**: Support for different storage backends
2. **Dynamic Prompt Generation**: Programmatic prompt creation
3. **Prompt Analytics**: Usage tracking and performance monitoring
4. **Access Control**: Role-based prompt access

## Migration Checklist

- [x] Identify all hardcoded prompts
- [x] Create directory structure
- [x] Migrate prompt content to files
- [x] Create configuration file
- [x] Implement PromptLoader utility
- [x] Update all code references
- [x] Test functionality
- [x] Validate performance
- [x] Create documentation
- [x] Remove obsolete prompt files

## Conclusion

The file-based prompt library provides a robust, maintainable solution for prompt management that eliminates technical debt from hardcoded strings while enabling better experimentation, version control, and collaboration across the development team.