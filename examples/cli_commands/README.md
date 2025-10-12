# CLI Commands Workflow Example

This example demonstrates the complete CLI workflow for the mCODE Translator, showing how to use various commands for data processing, analysis, and management.

## What You'll Learn

- Complete CLI command workflow from data ingestion to analysis
- Command execution patterns and best practices
- Error handling and troubleshooting
- Configuration management
- Performance monitoring

## Quick Start

```bash
cd examples/cli_commands
python cli_workflow_demo.py
```

## Expected Output

```
🚀 mCODE Translator - CLI Commands Workflow Demo
======================================================================

This demo shows the complete CLI workflow:
1. System status check
2. Trial data ingestion
3. mCODE processing
4. Optimization
5. Results analysis

🔧 Step 1: Checking system status and connectivity
   Command: python mcode-cli.py status
   ------------------------------------------------------------
✅ Success
   Output preview:
   🔍 System Status Check
   ✅ Python Version: 3.10+
   ✅ Dependencies: All required packages installed
   ✅ API Connectivity: ClinicalTrials.gov API accessible

🔧 Step 2: Ingesting breast cancer trial data (RegexEngine)
   Command: python mcode-cli.py data ingest-trials --cancer-type 'breast' --limit 2 --engine 'regex'
   ------------------------------------------------------------
✅ Success
   Output preview:
   🚀 Starting trial ingestion pipeline...
   📥 Fetching trials for: breast cancer
   🧪 Processing with RegexEngine
   ✅ Ingested 2 trials successfully

... (additional steps)

🎉 CLI Workflow Demo completed!

💡 Key CLI Commands Demonstrated:
   • status          - System health check
   • data ingest-trials - Fetch and process trial data
   • mcode summarize  - Extract mCODE from specific trials
   • mcode optimize-trials - Optimize processing parameters
   • memory stats     - CORE Memory usage statistics
   • config show      - Current configuration display
```

## CLI Command Categories

### Data Operations
```bash
# Ingest clinical trials
python mcode-cli.py data ingest-trials --cancer-type "breast" --limit 10

# Process patient data
python mcode-cli.py patients pipeline --input-file patients.ndjson

# Fetch specific trials
python mcode-cli.py trials pipeline --trial-ids NCT123456 NCT789012
```

### mCODE Processing
```bash
# Extract mCODE from single trial
python mcode-cli.py mcode summarize NCT02364999 --engine regex

# Compare processing engines
python mcode-cli.py mcode summarize NCT02364999 --compare-engines

# Optimize processing parameters
python mcode-cli.py mcode optimize-trials --trials-file trials.ndjson
```

### Memory Operations
```bash
# View memory statistics
python mcode-cli.py memory stats

# Search stored data
python mcode-cli.py memory search "breast cancer trials"

# Ingest data into memory
python mcode-cli.py memory ingest --data-file processed_trials.ndjson
```

### Configuration Management
```bash
# Show current configuration
python mcode-cli.py config show

# Setup configuration interactively
python mcode-cli.py config setup

# Validate configuration
python mcode-cli.py config validate
```

## Common CLI Patterns

### Batch Processing
```bash
# Process multiple trials efficiently
python mcode-cli.py data ingest-trials \
  --cancer-type "lung" \
  --limit 50 \
  --batch-size 10 \
  --engine regex
```

### Engine Selection
```bash
# Fast processing (free, deterministic)
python mcode-cli.py mcode summarize NCT123456 --engine regex

# Intelligent processing (AI-powered, flexible)
python mcode-cli.py mcode summarize NCT123456 --engine llm --model deepseek-coder
```

### Output Control
```bash
# Save results to file
python mcode-cli.py mcode summarize NCT123456 --output results.json

# Enable verbose logging
python mcode-cli.py data ingest-trials --verbose --log-level DEBUG

# Store in CORE Memory
python mcode-cli.py data ingest-trials --store-in-memory
```

## Files in This Example

- `cli_workflow_demo.py` - Main demonstration script
- `README.md` - This documentation

## Configuration Requirements

### Environment Variables
```bash
# Required for LLM features
export HEYSOL_API_KEY="your-api-key-here"

# Optional: logging level
export LOG_LEVEL="INFO"
```

### Configuration Files
The CLI automatically uses configuration from:
- `src/config/apis_config.json` - API endpoints
- `src/config/core_memory_config.json` - Storage settings
- `src/config/llms_config.json` - LLM configurations

## Error Handling

The CLI provides comprehensive error handling:

- **Network Issues**: Automatic retry with exponential backoff
- **API Limits**: Rate limiting and quota management
- **Invalid Data**: Validation with detailed error messages
- **Configuration Errors**: Clear setup instructions

## Performance Tips

1. **Use RegexEngine for large batches**: Faster and free
2. **Enable batch processing**: Reduces API calls
3. **Configure appropriate timeouts**: Based on network conditions
4. **Monitor memory usage**: For large datasets
5. **Use CORE Memory**: For result persistence and querying

## Troubleshooting

### Common Issues

- **Command not found**: Ensure you're in the project root directory
- **Import errors**: Check Python path and dependencies
- **API key issues**: Verify HEYSOL_API_KEY environment variable
- **Network timeouts**: Increase timeout values or check connectivity
- **Memory errors**: Reduce batch sizes or increase system memory

### Getting Help

```bash
# General help
python mcode-cli.py --help

# Command-specific help
python mcode-cli.py data --help
python mcode-cli.py mcode --help

# Subcommand help
python mcode-cli.py data ingest-trials --help
```

## Next Steps

After running this example:

1. **Explore Individual Commands**: Try each CLI command separately
2. **Customize Workflows**: Modify parameters for your use cases
3. **Script Automation**: Create shell scripts using these commands
4. **Integration**: Use CLI commands in larger data processing pipelines
5. **Monitoring**: Set up logging and monitoring for production use