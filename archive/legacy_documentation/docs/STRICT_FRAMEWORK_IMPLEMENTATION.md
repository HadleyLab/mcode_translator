# STRICT Prompt Optimization Framework Implementation

## Overview

This document describes the implementation of the **Strict Prompt Optimization Framework** - a lean, performant version of the original framework that **fails hard on invalid configurations** with no fallback mechanisms.

## Key Changes from Original Framework

### 1. Removed Fallback Mechanisms
- ✅ **No mock pipeline callbacks** - only uses real pipeline processing
- ✅ **No minimal test case creation** - requires proper test case JSON files
- ✅ **No environment variable fallbacks** - uses API keys directly from JSON configs
- ✅ **No cached/mock responses** - requires actual API calls to succeed

### 2. Strict Validation
- ✅ **API key validation** - rejects placeholder/example API keys
- ✅ **Base URL validation** - requires proper HTTP/HTTPS URLs
- ✅ **Configuration validation** - fails immediately on any invalid config

### 3. Lean & Performant
- ✅ **Minimal dependencies** - only essential imports
- ✅ **Direct API key usage** - no environment variable fallback logic
- ✅ **Exception-based error handling** - fails fast instead of silent fallbacks

## Files Created/Modified

### New Files
- [`src/optimization/prompt_optimization_framework.py`](src/optimization/prompt_optimization_framework.py) - Core strict framework
- [`examples/config/valid_api_configs.json`](examples/config/valid_api_configs.json) - Example valid config
- [`tests/unit/test_strict_framework.py`](tests/unit/test_strict_framework.py) - Validation tests

### Modified Files
- [`src/optimization/prompt_optimization_framework.py`](src/optimization/prompt_optimization_framework.py) - Enhanced validation logic

## Validation Patterns Detected

The strict framework detects and rejects the following placeholder patterns:

### API Key Patterns
- `""` (empty string)
- `"your-*-api-key-here"`
- `"ollama"` (local testing placeholder)
- `"your-api-key-here"`
- `"your-openai-api-key-here"`
- `"fake-api-key-123"`
- `"test-api-key-456"`
- `"example-api-key"`
- `"dummy-api-key"`
- `"mock-api-key"`
- `"sk-original"` (OpenAI style)
- `"sk-test"` (OpenAI style)
- `"sk-example"` (OpenAI style)
- `"sk-dummy"` (OpenAI style)
- `"sk-mock"` (OpenAI style)
- `"sk-placeholder"` (OpenAI style)
- `"sk-your"` (OpenAI style)

### URL Patterns
- Invalid URLs (must start with `http://` or `https://`)

## Usage

### 1. Running the Strict Demo
```bash
python mCODE-cli.py optimization demo
```

**Expected Behavior**: Fails immediately if any API configuration contains placeholder keys.

### 2. Setting Up Valid Configurations

Update [`examples/config/api_configs.json`](examples/config/api_configs.json) with real API keys:

```json
{
  "api_configurations": [
    {
      "name": "deepseek_local",
      "base_url": "http://localhost:11434/v1",
      "api_key": "your-actual-ollama-key",  // Replace with real key
      "model": "deepseek-coder",
      ...
    },
    {
      "name": "openai_gpt4", 
      "base_url": "https://api.openai.com/v1",
      "api_key": "sk-real-openai-key-123",  // Replace with real key
      "model": "gpt-4",
      ...
    }
  ]
}
```

### 3. Running Validation Tests
```bash
python -m pytest tests/unit/test_strict_framework.py -v
```

## Error Messages

The framework provides clear, actionable error messages:

```
❌ STRICT Demo FAILED: INVALID API KEY for config 'deepseek_local'. 
Update examples/config/api_configs.json with valid credentials.

To fix this:
1. Update examples/config/api_configs.json with valid API keys
2. Ensure all required JSON configuration files exist  
3. Set valid DEEPSEEK_API_KEY environment variable if needed
```

## Benefits

1. **Production Ready**: No silent failures - issues are caught immediately
2. **Security**: Prevents accidental use of placeholder API keys
3. **Performance**: No fallback logic overhead
4. **Maintainability**: Clear error messages and validation rules
5. **Reliability**: Consistent behavior across environments

## Migration from Original Framework

To migrate from the original framework to the strict version:

1. Replace all placeholder API keys with real credentials
2. Update imports from `prompt_optimization_framework` to `prompt_optimization_framework`
3. Remove any fallback/retry logic from your code
4. Handle validation exceptions appropriately

## Testing

The framework includes comprehensive validation tests that verify:
- ✅ Valid API configurations pass validation
- ✅ Invalid API configurations fail with appropriate errors  
- ✅ Configuration loading from JSON files works correctly
- ✅ Placeholder patterns are properly detected and rejected

Run tests with: `python -m pytest tests/unit/test_strict_framework.py -v`