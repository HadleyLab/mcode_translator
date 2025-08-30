# Prompt Optimization Framework

A comprehensive system for optimizing LLM prompts in clinical NLP and mCODE mapping pipelines. This framework enables systematic experimentation, benchmarking, and analysis of prompt variations across different API configurations.

## 🚀 Quick Start

### Installation
```bash
# Ensure you have the required dependencies
pip install nicegui pandas aiohttp
```

### Run Demo
```bash
python mcode-cli.py optimization demo
```

### Launch Web Interface
```bash
python -m src.optimization.optimization_ui
```
Access: http://localhost:8081

## 📋 Features

### Core Capabilities
- **Multi-API Management**: Configure and test across different LLM providers
- **Prompt Versioning**: Track and compare prompt variants with semantic versioning
- **Automated Benchmarking**: Run comprehensive experiments across configurations
- **Performance Metrics**: Track success rate, latency, extraction quality, compliance
- **Web-Based UI**: NiceGUI interface for configuration and analysis
- **Result Export**: CSV, JSON, and report generation

### Supported Prompt Types
- **NLP Extraction**: Clinical entity extraction from trial protocols
- **MCODE Mapping**: Transformation to mCODE standard format

## 🏗️ Architecture

### Key Components
- **StrictPromptOptimizationFramework**: Main orchestrator class
- **APIConfig**: API endpoint configurations with rate limiting
- **PromptVariant**: Versioned prompt templates with metadata
- **BenchmarkResult**: Structured performance measurement storage
- **OptimizationUI**: Web-based management interface

### File Structure
```
src/optimization/
├── strict_prompt_optimization_framework.py  # Core framework logic
├── optimization_ui.py                # Web interface
└── __init__.py

tests/unit/test_optimization_framework.py        # Comprehensive test harness
docs/prompt_optimization_framework.md # Detailed documentation
```

## 🎯 Usage Examples

### Basic Configuration
```python
from src.optimization.strict_prompt_optimization_framework import APIConfig, PromptVariant, PromptType

# Create API configuration
api_config = APIConfig(
    name="deepseek_prod",
    base_url="https://api.deepseek.com/v1",
    api_key="your-key",
    model="deepseek-coder",
    temperature=0.2,
    max_tokens=4000
)

# Create prompt variant
prompt_variant = PromptVariant(
    name="comprehensive_extraction",
    prompt_type=PromptType.NLP_EXTRACTION,
    template="Your prompt template...",
    description="Detailed clinical extraction",
    version="1.0.0"
)

# Add to framework
framework = StrictPromptOptimizationFramework()
framework.add_api_config(api_config)
framework.add_prompt_variant(prompt_variant)
```

### Running Benchmarks
```python
import asyncio

async def run_experiment():
    results = await framework.run_comprehensive_benchmark(
        prompt_type=PromptType.NLP_EXTRACTION,
        api_config_names=["deepseek_prod"],
        prompt_variant_ids=["comprehensive_extraction"],
        test_case_ids=["test_case_1", "test_case_2"]
    )
    print(f"Completed {len(results)} experiments")

asyncio.run(run_experiment())
```

### Results Analysis
```python
# Load and analyze results
framework.load_benchmark_results()
df = framework.get_results_dataframe()

# Basic statistics
print(f"Success Rate: {df['success'].mean():.1%}")
print(f"Avg Duration: {df['duration_ms'].mean():.1f} ms")

# Generate comprehensive report
report = framework.generate_performance_report()
```

## 📊 Performance Metrics

The framework tracks comprehensive metrics:

### Primary Metrics
- **Success Rate**: API call success percentage
- **Duration**: Processing time (milliseconds)
- **Entities Extracted**: Number of clinical entities identified
- **Compliance Score**: mCODE standard adherence (0-1 scale)

### Secondary Metrics
- **Token Usage**: Input/output token consumption
- **Cost Estimation**: Approximate API call costs
- **Error Analysis**: Categorized failure reasons

## 🌐 Web Interface

The NiceGUI-based interface provides four main tabs:

### 1. Configuration Tab
- Manage API configurations with form validation
- Create and version prompt variants
- Load test cases from JSON files
- Real-time configuration preview

### 2. Benchmarking Tab
- Select configurations for experiments
- Monitor progress with live updates
- Run comprehensive benchmarks
- Progress tracking with visual indicators

### 3. Results Tab
- Load and view benchmark results
- Interactive data table with filtering
- Export to CSV format
- Generate performance reports
- Summary statistics dashboard

### 4. Visualizations Tab
- Performance charts (placeholder implementation)
- Comparative analysis views
- Trend visualization

## 🔧 Integration

### With Existing Pipeline
```python
# Use optimized prompts in production
from src.pipeline.nlp_engine import NLPEngine
from src.optimization.strict_prompt_optimization_framework import StrictPromptOptimizationFramework

framework = StrictPromptOptimizationFramework()
framework.load_benchmark_results()

# Get best performing prompt
best_prompt = framework.get_best_performing_prompt(PromptType.NLP_EXTRACTION)

# Integrate with NLP engine
engine = NLPEngine()
engine.set_prompt_template(best_prompt.template)
```

### Comprehensive Cancer Test Cases

The framework includes extensive test data covering multiple cancer types:

#### Available Test Cases
- **Pancreatic Cancer**: Advanced metastatic scenarios
- **Ovarian Cancer**: Platinum-sensitive/resistant variants
- **Glioblastoma**: MGMT methylation status variations
- **Bladder Cancer**: Muscle-invasive and non-muscle invasive
- **Multiple Myeloma**: Relapsed/refractory cases
- **Head & Neck Cancer**: HPV-positive and negative
- **Gastric Cancer**: HER2-positive and negative
- **Renal Cell Carcinoma**: Clear cell and variants
- **Traditional Cancers**: Breast, Lung, Colon, Prostate, Melanoma

#### File Locations
```bash
# Original test cases
examples/test_cases/clinical_test_cases.json

# Comprehensive cancer test cases
examples/test_cases/various_cancers_test_cases.json
```

#### Example Test Case Structure
```json
{
  "pancreatic_cancer_advanced": {
    "protocolSection": {
      "identificationModule": {
        "briefTitle": "Phase III Study of Gemcitabine + Nab-Paclitaxel in Metastatic Pancreatic Cancer",
        "nctId": "NCT12345678"
      },
      "descriptionModule": {
        "briefSummary": "Randomized phase III trial comparing gemcitabine plus nab-paclitaxel versus gemcitabine alone in patients with metastatic pancreatic adenocarcinoma...",
        "detailedDescription": "Patients must have histologically confirmed metastatic pancreatic adenocarcinoma, measurable disease per RECIST 1.1, ECOG performance status 0-1, adequate organ function..."
      },
      "eligibilityModule": {
        "eligibilityCriteria": "Inclusion Criteria:\n- Age ≥18 years\n- Histologically confirmed pancreatic adenocarcinoma\n- Metastatic disease\n- ECOG PS 0-1\n\nExclusion Criteria:\n- Prior chemotherapy for metastatic disease\n- Brain metastases\n- Significant comorbidities"
      }
    }
  }
}
```

## 🧪 Testing & Cancer-Specific Analysis

### Comprehensive Cancer Testing
```bash
# Run comprehensive cancer test demonstration
python mcode-cli.py benchmark comprehensive

# Expected output:
# 🩺 COMPREHENSIVE CANCER TEST CASE DEMONSTRATION
# 📋 Loading comprehensive cancer test configurations...
# 🎯 Cancer types distribution:
#    - pancreatic: 2 cases
#    - ovarian: 2 cases
#    - glioblastoma: 2 cases
#    - ... etc.
# 🧪 Running comprehensive cancer benchmark...
```

### Cancer Type Detection
The framework automatically detects cancer types from clinical trial titles and provides:
- Cancer-specific entity extraction patterns
- Type-aware performance analysis
- Distribution visualization in web UI
- Specialized mock responses for different cancers

### Test Harness
The optimization framework tests provide:
- Integration testing with existing components
- Demonstration of all framework features
- Performance benchmarking examples
- Result analysis and reporting

### Running Tests
```bash
# Run comprehensive test suite
python -m pytest tests/unit/test_optimization_framework.py -v

# Expected output:
# 🚀 Starting Prompt Optimization Framework Demo
# 📋 Configuration Overview...
# 🧪 Running NLP Extraction Benchmark...
# ✅ Completed X experiments
# 📊 Performance Report...
```

## 📈 Best Practices

### Prompt Design
1. **Semantic Versioning**: Use meaningful version numbers (1.0.0, 1.1.0, etc.)
2. **Descriptive Names**: "comprehensive_extraction" vs "prompt_v2"
3. **Incremental Changes**: Small, testable modifications between versions
4. **Comprehensive Documentation**: Include purpose, changes, and expected behavior

### Benchmarking
1. **Diverse Test Data**: Include various clinical scenarios and complexity levels
2. **Multiple Configurations**: Test across API providers, models, and parameters
3. **Statistical Significance**: Ensure sufficient sample size for reliable results
4. **Cost Awareness**: Balance performance improvements with cost considerations

### Analysis
1. **Holistic Evaluation**: Consider multiple metrics, not just accuracy
2. **Error Investigation**: Analyze failure patterns and root causes
3. **Trend Monitoring**: Track performance over time and across versions
4. **Actionable Insights**: Focus on improvements that matter for production use

## 🚨 Troubleshooting

### Common Issues
- **API Connection Errors**: Verify API keys, endpoints, and network connectivity
- **Prompt Template Issues**: Check JSON formatting and variable substitution
- **Performance Problems**: Monitor rate limits and implement retry logic
- **Memory Issues**: Large result sets may require optimized storage

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

### Planned Features
- [x] Comprehensive cancer test case library ✅
- [x] Cancer type detection and analysis ✅
- [x] Specialized visualization for oncology data ✅
- [ ] Advanced visualization with matplotlib/seaborn integration
- [ ] Automated A/B testing capabilities
- [ ] Cost optimization algorithms
- [ ] Additional LLM provider integrations
- [ ] Real-time monitoring dashboard
- [ ] Collaborative features for team usage
- [ ] Advanced export formats (Excel, PDF, interactive reports)
- [ ] Cancer-specific performance benchmarks
- [ ] Oncology-focused prompt templates

### Integration Opportunities
- **CI/CD Pipeline**: Automated prompt testing in deployment workflows
- **Monitoring**: Real-time performance tracking in production
- **Alerting**: Notifications for performance degradation
- **Data Lake**: Long-term result storage and analysis

## 📚 Documentation

### Comprehensive Guides
- **Framework Overview**: Architecture and design principles
- **API Reference**: Detailed class and method documentation
- **Usage Examples**: Practical code samples and patterns
- **Best Practices**: Optimization strategies and recommendations

### Quick References
- **Configuration Examples**: Sample API configs and prompt variants
- **Troubleshooting Guide**: Common issues and solutions
- **Performance Metrics**: Explanation of tracked metrics
- **Integration Patterns**: How to use with existing systems

## 🤝 Contributing

### Development Guidelines
1. Follow existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for all changes
4. Use descriptive commit messages
5. Consider backward compatibility

### Testing Requirements
- Unit tests for new functionality
- Integration tests with existing components
- Performance benchmarking
- Documentation updates

## 📄 License

Part of the mCODE Translator project. See the main project [`LICENSE`](LICENSE) file for details.

---

**Note**: This framework is designed for systematic prompt optimization and should be used as part of a comprehensive LLM deployment strategy. Always test thoroughly before deploying optimized prompts to production environments.