# Prompt Optimization Framework Documentation

A comprehensive framework for optimizing LLM prompts in clinical NLP and Mcode mapping pipelines.

## Overview

The Prompt Optimization Framework provides a systematic approach to:
- Manage multiple API configurations for different LLM providers
- Version control and compare prompt variants
- Run automated benchmarks across different configurations
- Analyze performance metrics (accuracy, latency, compliance)
- Visualize results through a web-based interface

## Architecture

### Core Components

1. **PromptOptimizationFramework** - Main orchestrator class
2. **APIConfig** - API endpoint configurations
3. **PromptVariant** - Versioned prompt templates
4. **BenchmarkResult** - Performance measurement storage
5. **OptimizationUI** - Web-based management interface

### Supported Prompt Types

- **NLP_EXTRACTION**: Clinical entity extraction from trial protocols
- **MCODE_MAPPING**: Mapping extracted entities to Mcode standard

## Installation & Setup

### Prerequisites

```bash
pip install -r requirements.txt
# Includes: nicegui, pandas, aiohttp, etc.
```

### Quick Start

1. **Run the test harness:**
```bash
python -m pytest tests/unit/test_optimization_framework.py -v
```

2. **Launch the web interface:**
```bash
python -m src.optimization.optimization_ui
```

3. **Access the UI:** http://localhost:8081

## Configuration

### API Configurations

Manage multiple LLM API endpoints:

```python
from src.optimization.prompt_optimization_framework import APIConfig

config = APIConfig(
    name="deepseek_prod",
    base_url="https://api.deepseek.com/v1",
    api_key="your-api-key",
    model="deepseek-coder",
    temperature=0.2,
    max_tokens=4000
)
```

### Prompt Variants

Create and version prompt templates:

```python
from src.optimization.prompt_optimization_framework import PromptVariant, PromptType

variant = PromptVariant(
    name="comprehensive_extraction",
    prompt_type=PromptType.NLP_EXTRACTION,
    template="Your prompt template...",
    description="Detailed extraction prompt",
    version="1.2.0"
)
```

### Test Cases

Load clinical trial data for benchmarking:

```python
framework.load_test_cases_from_file("test_data/clinical_trials.json")
```

## Usage Examples

### Basic Benchmarking

```python
import asyncio
from src.optimization.prompt_optimization_framework import PromptOptimizationFramework, PromptType

async def run_benchmark():
    framework = PromptOptimizationFramework()
    
    # Configure your API and prompts first
    # framework.add_api_config(...)
    # framework.add_prompt_variant(...)
    # framework.load_test_cases_from_file(...)
    
    results = await framework.run_comprehensive_benchmark(
        prompt_type=PromptType.NLP_EXTRACTION,
        api_config_names=["deepseek_local"],
        prompt_variant_ids=["variant_1", "variant_2"],
        test_case_ids=["test_case_1", "test_case_2"]
    )
    
    framework.save_benchmark_results()
    print(f"Completed {len(results)} experiments")

asyncio.run(run_benchmark())
```

### Results Analysis

```python
framework.load_benchmark_results()
df = framework.get_results_dataframe()

# Basic statistics
print(f"Success rate: {df['success'].mean():.1%}")
print(f"Average duration: {df['duration_ms'].mean():.1f} ms")

# Group by prompt variant
prompt_stats = df.groupby('prompt_name').agg({
    'success': 'mean',
    'duration_ms': 'mean',
    'entities_extracted': 'mean'
})
```

### Performance Report

```python
report = framework.generate_performance_report()

print(f"Total experiments: {report['total_experiments']}")
print(f"Best prompt: {report['best_configs']['prompt_variant']['name']}")
print(f"Best API config: {report['best_configs']['api_config']['name']}")
```

## Web Interface

The NiceGUI-based interface provides:

### Configuration Tab
- Manage API configurations
- Create and version prompt variants
- Load test cases
- Real-time configuration validation

### Benchmarking Tab
- Select configurations for experiments
- Monitor progress with live updates
- Run comprehensive benchmarks

### Results Tab
- Load and view benchmark results
- Export to CSV format
- Generate performance reports
- Summary statistics

### Visualizations Tab
- Performance charts (placeholder for now)
- Comparative analysis
- Trend visualization

## Integration with Existing Pipeline

### Using Existing Prompt Templates

```python
from src.pipeline.nlp_extractor import GENERIC_EXTRACTION_PROMPT_TEMPLATE
from src.pipeline.mcode_mapper import MCODE_MAPPING_PROMPT_TEMPLATE

# Create variants from existing templates
framework.add_prompt_variant(PromptVariant(
    name="existing_extraction",
    prompt_type=PromptType.NLP_EXTRACTION,
    template=GENERIC_EXTRACTION_PROMPT_TEMPLATE,
    description="Original extraction template",
    version="1.0.0"
))
```

### Custom Integration

```python
# Use optimized prompts in your pipeline
best_prompt = framework.get_best_performing_prompt(PromptType.NLP_EXTRACTION)

# Integrate with existing NLP engine
from src.pipeline.nlp_extractor import NlpBase

engine = NlpBase()
engine.set_prompt_template(best_prompt.template)
```

## Performance Metrics

The framework tracks:

### Primary Metrics
- **Success Rate**: Percentage of successful API calls
- **Duration**: Processing time in milliseconds
- **Entities Extracted**: Number of clinical entities identified
- **Compliance Score**: Mcode standard compliance (0-1)

### Secondary Metrics
- **Token Usage**: Input/output tokens consumed (tracked via unified token tracker)
- **Cost Estimation**: API call cost approximation
- **Error Rates**: Detailed error categorization

### Model Library Integration
- **File-Based Model Configuration**: Centralized management of LLM model configurations
- **Cross-Provider Compatibility**: Support for multiple LLM providers (DeepSeek, OpenAI, etc.)
- **Dynamic Model Selection**: Runtime selection of optimal models for specific tasks
- **Performance Benchmarking**: Model-specific performance metrics and comparisons

## Best Practices

### Prompt Design
1. **Version Control**: Always version your prompt variants
2. **Descriptive Names**: Use clear, descriptive names for variants
3. **Incremental Changes**: Make small, testable changes between versions
4. **Documentation**: Include detailed descriptions for each variant

### Benchmarking
1. **Representative Data**: Use diverse test cases that represent real usage
2. **Multiple Configurations**: Test across different API providers and models
3. **Statistical Significance**: Run sufficient iterations for reliable results
4. **Baseline Comparison**: Always include a baseline configuration

### Analysis
1. **Holistic View**: Consider multiple metrics (not just accuracy)
2. **Cost-Benefit**: Balance performance with cost considerations
3. **Error Analysis**: Investigate failure patterns
4. **Trend Monitoring**: Track performance over time

## File Structure

```
src/optimization/
├── prompt_optimization_framework.py  # Core framework implementation
├── optimization_ui.py                # Web interface
└── __init__.py

tests/unit/test_optimization_framework.py        # Test harness
docs/prompt_optimization_framework.md # This documentation
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check API keys and endpoints
   - Verify network connectivity
   - Validate model availability

2. **Prompt Template Issues**
   - Ensure proper JSON formatting in prompts
   - Validate template variable substitution
   - Check for syntax errors

3. **Performance Problems**
   - Monitor API rate limits
   - Consider batch processing
   - Implement retry mechanisms

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned features:
- [ ] Advanced visualization with matplotlib/seaborn
- [ ] Automated A/B testing capabilities
- [ ] Cost optimization algorithms
- [ ] Integration with more LLM providers
- [ ] Real-time monitoring dashboard
- [ ] Export to various formats (JSON, Excel, PDF)
- [ ] Collaborative features for team usage

## Contributing

1. Follow existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for changes
4. Use descriptive commit messages
5. Consider backward compatibility

## License

This framework is part of the Mcode Translator project. See the main project [`LICENSE`](../LICENSE) file for details.