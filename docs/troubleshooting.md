# Troubleshooting Guide

## Common Issues and Solutions

### API Key Issues

#### Problem: "No API key available for [model]"
```
ValueError: No API key available for gpt-4
```

**Solutions:**
1. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Pass via CLI:
   ```bash
   python mcode_cli.py --api-key "your-heysol-key" trials process --input data.ndjson
   ```

3. Check configuration:
   ```bash
   python mcode_cli.py config check
   ```

#### Problem: "API key not configured for model"
```
ValueError: API key not configured for model gpt-4
```

**Solutions:**
- Verify the API key environment variable name matches the configuration
- Check that the API key is valid and not expired
- Ensure the model name is spelled correctly

### Memory/Storage Issues

#### Problem: "Unable to connect to HeySol API"
```
ConnectionError: Unable to connect to HeySol API
```

**Solutions:**
1. Verify HeySol API key:
   ```bash
   export HEYSOL_API_KEY="your-key"
   ```

2. Check network connectivity:
   ```bash
   curl -H "Authorization: Bearer $HEYSOL_API_KEY" https://core.heysol.ai/api/v1/spaces
   ```

3. Test system status:
   ```bash
   python mcode_cli.py status
   ```

#### Problem: "Memory space not found"
```
ValueError: Memory space 'OncoCore_Trials' not found
```

**Solutions:**
- Spaces are auto-created by default
- Check memory configuration in `src/config/core_memory_config.json`
- Manually create space if needed

### Processing Errors

#### Problem: "Empty LLM response"
```
ValueError: Empty LLM response
```

**Solutions:**
- Check API key validity
- Verify model availability
- Try different model or prompt
- Check rate limits

#### Problem: "JSON parsing error"
```
json.JSONDecodeError: Expecting ',' delimiter
```

**Solutions:**
- This often indicates model returned malformed JSON
- Try different model (DeepSeek models can be more reliable)
- Use regex engine as fallback: `--engine regex`
- Check prompt configuration

#### Problem: "Trial data missing required format"
```
ValueError: Trial data missing required format
```

**Solutions:**
- Ensure trial data has `protocolSection` with required fields
- Validate NCT ID presence
- Check eligibility criteria exist
- Use `validate_trial_data()` method to check format

### Configuration Issues

#### Problem: "Configuration file not found"
```
FileNotFoundError: src/config/llms_config.json not found
```

**Solutions:**
- Ensure all configuration files exist in `src/config/`
- Check file permissions
- Reinstall or restore missing files

#### Problem: "Invalid JSON in configuration"
```
json.JSONDecodeError: Invalid JSON
```

**Solutions:**
- Validate JSON syntax in configuration files
- Use online JSON validator
- Check for trailing commas or syntax errors

### Performance Issues

#### Problem: "Processing is slow"
**Solutions:**
1. Enable caching:
   - Ensure API cache is configured
   - Check cache TTL settings

2. Use concurrent processing:
   ```bash
   python mcode_cli.py trials process --workers 4 --input data.ndjson
   ```

3. Try different model:
   - DeepSeek models are often faster than GPT-4
   - Use regex engine for simple cases

#### Problem: "Memory usage is high"
**Solutions:**
- Process data in smaller batches
- Clear caches periodically
- Monitor memory usage with `--memory-stats`
- Use streaming processing for large datasets

### CLI Issues

#### Problem: "Command not found"
```
bash: mcode_cli.py: command not found
```

**Solutions:**
1. Use full path:
   ```bash
   python src/cli/__init__.py trials process --input data.ndjson
   ```

2. Add to PATH or create symlink

3. Use python module:
   ```bash
   python -m src.cli trials process --input data.ndjson
   ```

#### Problem: "Permission denied"
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
- Check file permissions on input/output files
- Ensure write permissions for output directories
- Run with appropriate user permissions

### Data Format Issues

#### Problem: "Invalid trial data format"
**Required fields for clinical trial data:**
```json
{
  "protocolSection": {
    "identificationModule": {
      "nctId": "NCT123456"
    },
    "eligibilityModule": {
      "eligibilityCriteria": "text here..."
    }
  }
}
```

**Solutions:**
- Validate data structure before processing
- Use `validate_trial_data()` method
- Check data source API documentation

#### Problem: "Invalid FHIR bundle format"
**Required FHIR bundle structure:**
```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "123"
      }
    }
  ]
}
```

**Solutions:**
- Validate FHIR compliance
- Check bundle structure
- Ensure Patient resources exist

### Rate Limiting Issues

#### Problem: "Rate limit exceeded"
```
ValueError: Rate limit exceeded for model gpt-4
```

**Solutions:**
1. Implement exponential backoff (automatic in the code)
2. Reduce concurrent workers: `--workers 1`
3. Add delays between requests
4. Switch to different model with higher limits

#### Problem: "Too many requests"
```
HTTPError: 429 Too Many Requests
```

**Solutions:**
- Reduce processing speed
- Use caching to avoid repeated API calls
- Implement request queuing
- Check API provider limits

### Validation Issues

#### Problem: "Validation failed"
```
ValidationError: mCODE compliance score below threshold
```

**Solutions:**
- Check validation configuration in `validation_config.json`
- Adjust confidence thresholds
- Review input data quality
- Use different processing model

#### Problem: "Missing required mCODE elements"
```
ValidationError: Required elements missing: CancerCondition
```

**Solutions:**
- Ensure input data contains relevant clinical information
- Try different prompts or models
- Check if data matches expected format
- Review validation requirements

## Diagnostic Tools

### System Status Check
```bash
python mcode_cli.py status
```

### Comprehensive Diagnostics
```bash
python mcode_cli.py doctor
```

### Configuration Validation
```bash
python mcode_cli.py config check
python mcode_cli.py config validate
```

### Memory Diagnostics
```bash
python mcode_cli.py memory status
python mcode_cli.py memory stats
```

## Logging and Debugging

### Enable Debug Logging
```bash
export LOG_LEVEL=DEBUG
python mcode_cli.py trials process --input data.ndjson
```

### Check Log Files
```bash
tail -f logs/mcode_translator.log
```

### Verbose CLI Output
```bash
python mcode_cli.py --verbose trials process --input data.ndjson
```

## Recovery Procedures

### Clear Caches
```bash
python mcode_cli.py memory clear
# Or manually
rm -rf .cache/
```

### Reset Configuration
```bash
# Backup current config
cp src/config/ src/config.backup/

# Restore defaults
git checkout src/config/
```

### Clean Reinstall
```bash
# Remove caches and temp files
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
rm -rf .pytest_cache/

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Getting Help

### Check Version and Help
```bash
python mcode_cli.py --help
python mcode_cli.py version
```

### Report Issues
- Check existing issues on GitHub
- Provide complete error messages
- Include system information
- Share configuration (without API keys)

### Community Support
- GitHub Discussions for questions
- Documentation wiki for tutorials
- Stack Overflow with `mcode-translator` tag

## Performance Tuning

### Optimize for Speed
```bash
# Use multiple workers
python mcode_cli.py trials process --workers 8 --input data.ndjson

# Use faster model
python mcode_cli.py trials process --model deepseek-coder --input data.ndjson

# Use regex for simple cases
python mcode_cli.py trials process --engine regex --input data.ndjson
```

### Optimize for Memory
```bash
# Process in batches
python mcode_cli.py trials process --batch-size 10 --input data.ndjson

# Disable caching if memory constrained
export DISABLE_CACHE=true
```

### Monitor Performance
```bash
# Enable performance logging
export LOG_LEVEL=DEBUG
export ENABLE_PERFORMANCE_LOGGING=true

# Check system resources
python mcode_cli.py doctor
```

## Advanced Troubleshooting

### Network Issues
```bash
# Test connectivity
curl -v https://api.openai.com/v1/models
curl -v https://core.heysol.ai/api/v1/spaces

# Check DNS
nslookup api.openai.com
nslookup core.heysol.ai

# Test with proxy
export HTTPS_PROXY=http://proxy.company.com:8080
```

### SSL/TLS Issues
```bash
# Disable SSL verification (not recommended for production)
export REQUESTS_CA_BUNDLE=""
export SSL_VERIFY=false
```

### Database/Storage Issues
```bash
# Check disk space
df -h

# Check file permissions
ls -la data/
ls -la logs/

# Test file I/O
echo "test" > /tmp/test_write
rm /tmp/test_write
```

### Python Environment Issues
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(pydantic|openai|requests|typer)"

# Check virtual environment
which python
echo $VIRTUAL_ENV

# Reinstall in clean environment
python -m venv venv_clean
source venv_clean/bin/activate
pip install -r requirements.txt