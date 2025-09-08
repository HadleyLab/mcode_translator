# Prompt Library Documentation

## Overview

This prompt library provides a centralized, file-based system for managing all LLM prompts used in the mCODE Translator project. The library eliminates hardcoded prompts and provides a single source of truth for prompt management, versioning, and experimentation.

## Directory Structure

```
prompts/
├── prompts_config.json          # Master configuration file
├── txt/
│   ├── nlp_extraction/          # NLP entity extraction prompts
│   │   ├── generic_extraction.txt
│   │   ├── comprehensive_extraction.txt
│   │   ├── minimal_extraction.txt
│   │   ├── structured_extraction.txt
│   │   ├── basic_extraction.txt
│   │   └── minimal_extraction_optimization.txt
│   └── mcode_mapping/           # mCODE mapping prompts
│       ├── generic_mapping.txt
│       ├── standard_mapping.txt
│       ├── detailed_mapping.txt
│       ├── error_robust_mapping.txt
│       ├── comprehensive_mapping.txt
│       └── simple_mapping.txt
└── README.md                    # This documentation
```

## Configuration File Format

The `prompts_config.json` file uses external file references to point to prompt text files:

```json
{
  "nlp_extraction": {
    "generic_extraction": {
      "file": "txt/nlp_extraction/generic_extraction.txt",
      "description": "Main extraction prompt for clinical entity recognition"
    },
    "comprehensive_extraction": {
      "file": "txt/nlp_extraction/comprehensive_extraction.txt",
      "description": "Comprehensive extraction with detailed JSON schema"
    }
    // ... more prompts
  },
  "mcode_mapping": {
    "generic_mapping": {
      "file": "txt/mcode_mapping/generic_mapping.txt",
      "description": "Main mCODE mapping prompt"
    }
    // ... more prompts
  }
}
```

## Prompt Categories

### NLP Extraction Prompts
- **generic_extraction**: Main extraction prompt for clinical entity recognition
- **comprehensive_extraction**: Detailed extraction with comprehensive JSON schema
- **minimal_extraction**: Minimalist extraction prompt
- **structured_extraction**: Structured extraction with specific formatting
- **basic_extraction**: Basic extraction for optimization framework
- **minimal_extraction_optimization**: Minimal extraction for optimization

### mCODE Mapping Prompts
- **generic_mapping**: Main mCODE mapping prompt
- **standard_mapping**: Standard mCODE mapping
- **detailed_mapping**: Detailed mapping with comprehensive instructions
- **error_robust_mapping**: Error-robust mapping with fallback handling
- **comprehensive_mapping**: Comprehensive mapping for optimization
- **simple_mapping**: Simple mapping for optimization

## Usage

### Loading Prompts Programmatically

```python
from src.utils.prompt_loader import PromptLoader

# Initialize the prompt loader
prompt_loader = PromptLoader()

# Load a specific prompt
extraction_prompt = prompt_loader.load_prompt("generic_extraction")
mapping_prompt = prompt_loader.load_prompt("generic_mapping")

# Load with formatting (if prompt contains {placeholders})
formatted_prompt = prompt_loader.load_prompt("generic_extraction", patient_data=patient_info)
```

### In Pipeline Components

```python
# In NLP Engine
def extract_entities(self, text: str, prompt_name: str = "generic_extraction") -> Dict:
    template = self.prompt_loader.load_prompt(prompt_name)
    # Use template with LLM

# In mCODE Mapper  
def map_to_mcode(self, entities: List[Dict], prompt_name: str = "generic_mapping") -> Dict:
    template = self.prompt_loader.load_prompt(prompt_name)
    # Use template with LLM
```

### In Optimization Framework

```python
# Create prompt variants for testing
prompt_variants = [
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

## Adding New Prompts

1. **Create the prompt text file** in the appropriate directory:
   ```bash
   # For NLP extraction
   touch prompts/txt/nlp_extraction/new_extraction_prompt.txt
   
   # For mCODE mapping  
   touch prompts/txt/mcode_mapping/new_mapping_prompt.txt
   ```

2. **Add the prompt content** to the text file

3. **Register the prompt** in `prompts_config.json`:
   ```json
   {
     "nlp_extraction": {
       "new_extraction_prompt": {
         "file": "txt/nlp_extraction/new_extraction_prompt.txt",
         "description": "Description of the new prompt"
       }
     }
   }
   ```

4. **Use the prompt** in your code:
   ```python
   prompt = prompt_loader.load_prompt("new_extraction_prompt")
   ```

## Best Practices

1. **Single Source of Truth**: All prompts should be managed through this library
2. **Version Control**: Prompt changes are tracked through git
3. **Descriptive Names**: Use clear, descriptive names for prompt keys
4. **Documentation**: Include descriptions in the config file
5. **Testing**: Test new prompts before adding to production
6. **Backward Compatibility**: Avoid breaking changes to existing prompt names

## Prompt Formatting

Prompts can contain placeholders for dynamic content:

```
Extract medical entities from the following clinical text:

{text}

Return the results in JSON format with the following structure:
{json_schema}
```

Use the `load_prompt()` method with keyword arguments to format:

```python
formatted = prompt_loader.load_prompt(
    "generic_extraction",
    text=clinical_text,
    json_schema=entity_schema
)
```

## Performance Considerations

- The `PromptLoader` caches loaded prompts for performance
- File I/O is minimal since prompts are loaded once and cached
- Relative file paths ensure portability across environments

## Migration from Hardcoded Prompts

All hardcoded prompts have been migrated to this library. The following files were updated:

- `src/pipeline/nlp_extractor.py` - Now uses prompt loader
- `src/pipeline/Mcode_mapper.py` - Now uses prompt loader  
- `src/optimization/prompt_optimization_framework.py` - Now loads from file library
- `src/pipeline/strict_dynamic_extraction_pipeline.py` - Accepts prompt names instead of templates

## Testing

Run the test script to verify all prompts load correctly:

```bash
python -m pytest tests/unit/test_prompt_loading.py -v
```

This will validate that all 12 prompts in the configuration can be loaded successfully.