# mCODE Translator - Unified Processor System

A comprehensive clinical trial data processing system that offers users the choice between fast, cost-effective rule-based processing and intelligent, flexible LLM-enhanced processing.

## 🎯 Overview

The mCODE Translator CLI now provides a **unified processor interface** that allows users to choose between two distinct processing approaches based on their specific needs:

- **🧪 Rule-Based Processing**: Fast, deterministic, cost-effective structured data extraction
- **🤖 LLM-Enhanced Processing**: Intelligent, flexible, AI-powered pattern recognition

## 🚀 Quick Start

### Basic Usage (Rule-Based - Default)
```bash
# Ingest clinical trials with fast processing
python mcode-cli.py data ingest-trials --cancer-type "breast" --limit 10

# Summarize a specific trial
python mcode-cli.py mcode summarize NCT02314481
```

### Advanced Usage (LLM-Enhanced)
```bash
# Ingest with intelligent processing
python mcode-cli.py data ingest-trials \
  --cancer-type "breast" \
  --limit 10 \
  --method "llm_based" \
  --llm-model "deepseek-coder"

# Compare both methods
python mcode-cli.py mcode summarize NCT02314481 --compare-methods
```

## 🔧 Processing Methods

### Rule-Based Processing
**Best for:** Structured data, speed-critical applications, cost optimization

- ⚡ **Ultra-fast processing** (milliseconds)
- 💰 **Cost-effective** (no API costs)
- 🎯 **Deterministic results** (consistent output)
- 🔧 **Reliable** (no external dependencies)

**Use when:**
- Processing well-formatted clinical trial data
- Speed is critical for real-time applications
- Cost optimization is important
- Consistent, predictable results are required

### LLM-Enhanced Processing
**Best for:** Complex data, unstructured content, advanced pattern recognition

- 🧠 **Intelligent analysis** (AI-powered insights)
- 🔍 **Flexible processing** (handles varied formats)
- 💡 **Advanced patterns** (recognizes complex relationships)
- 🌐 **Adaptive** (learns from diverse data)

**Use when:**
- Processing unstructured or complex clinical text
- Advanced pattern recognition is needed
- Handling edge cases or unusual data formats
- Maximum flexibility is more important than speed

## 📊 Performance Comparison

| Feature | Rule-Based | LLM-Enhanced | Best For |
|---------|------------|--------------|----------|
| **Speed** | ⚡ Ultra-fast (~0.000s) | 🧠 Intelligent (~0.005s) | **Rule-based** |
| **Cost** | 💰 Free | 💳 API costs | **Rule-based** |
| **Consistency** | ✅ Deterministic | ✅ Intelligent | **Both methods** |
| **Flexibility** | 📋 Structured data | 🌐 Any data type | **LLM-enhanced** |

## 🎛️ Configuration Options

### Data Ingestion
```bash
python mcode-cli.py data ingest-trials \
  --cancer-type "breast" \
  --limit 50 \
  --method "rule_based" \
  --batch-size 10 \
  --verbose
```

**Parameters:**
- `--method`: Choose `"rule_based"` or `"llm_based"` (default: `"rule_based"`)
- `--llm-model`: LLM model for enhanced processing (default: `"deepseek-coder"`)
- `--llm-prompt`: Prompt template for enhanced processing
- `--batch-size`: Number of trials to process per batch
- `--verbose`: Enable detailed output

### Trial Summarization
```bash
python mcode-cli.py mcode summarize NCT02314481 \
  --method "llm_based" \
  --compare-methods \
  --store-memory \
  --verbose
```

**Parameters:**
- `--compare-methods`: Show comparison and recommendation
- `--store-memory`: Store results in CORE Memory
- `--format`: Output format (`"text"`, `"json"`, `"ndjson"`)

## 💡 Method Selection Guide

### Choose Rule-Based When You Need:
- ✅ **Maximum speed** for real-time processing
- ✅ **Cost-effective** processing (no API costs)
- ✅ **Deterministic results** (consistent output)
- ✅ **Reliable processing** (no external dependencies)
- ✅ **Batch processing** of large datasets

### Choose LLM-Enhanced When You Need:
- ✅ **Intelligent analysis** of complex patterns
- ✅ **Flexible processing** of varied data formats
- ✅ **Advanced insights** from unstructured text
- ✅ **Adaptive learning** from diverse content
- ✅ **Sophisticated pattern** recognition

## 🔍 Method Comparison & Benchmarking

### Automatic Method Recommendation
The system analyzes trial characteristics and recommends the optimal method:

```bash
python mcode-cli.py mcode summarize NCT02314481 --compare-methods
```

### Performance Benchmarking
Compare both methods with detailed metrics:

```python
from services.unified_processor import UnifiedTrialProcessor

processor = UnifiedTrialProcessor()
benchmark = await processor.benchmark_methods(trial_data, iterations=5)
print(f"Rule-based: {benchmark['rule_based']['avg_time']:.3f}s avg")
print(f"LLM-enhanced: {benchmark['llm_based']['avg_time']:.3f}s avg")
```

## 🏗️ Architecture

### Unified Processor Design
```
┌─────────────────────────────────────────────────────────┐
│                 UnifiedTrialProcessor                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐    │
│  │           RuleBasedProcessor                    │    │
│  │  • Uses McodeSummarizer                         │    │
│  │  • Fast, deterministic processing               │    │
│  │  • No LLM calls, cost-effective                │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │           LLMProcessor                          │    │
│  │  • Uses McodePipeline                           │    │
│  │  • Intelligent, flexible processing            │    │
│  │  • AI-powered pattern recognition              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Key Components
- **`UnifiedTrialProcessor`**: Main interface for method selection
- **`RuleBasedProcessor`**: Fast, structured data extraction
- **`LLMProcessor`**: Intelligent, AI-enhanced processing
- **`ProcessingResult`**: Standardized result format for both methods

## 🔧 Technical Implementation

### Method Selection Logic
```python
def recommend_method(self, trial_data: Dict[str, Any]) -> str:
    """Recommend optimal method based on trial characteristics."""
    # Analyze trial structure and complexity
    has_design_module = "designModule" in trial_data.get("protocolSection", {})
    has_status_module = "statusModule" in trial_data.get("protocolSection", {})

    # Well-structured data works best with rule-based
    if has_design_module and has_status_module:
        return "rule_based"

    # Complex data benefits from LLM enhancement
    return "llm_based"
```

### Async Processing Support
Both methods support async processing for scalability:

```python
# Process single trial
result = await processor.process_trial(trial_data, method="rule_based")

# Process multiple trials
results = await processor.process_batch(trials_data, method="llm_based")
```

## 📈 Performance Optimization

### Rule-Based Optimizations
- **Batch Processing**: Process multiple trials efficiently
- **Caching**: Cache frequent operations
- **Minimal Overhead**: No external API calls

### LLM-Enhanced Optimizations
- **Connection Reuse**: Maintain API connections
- **Response Caching**: Cache identical requests
- **Rate Limiting**: Intelligent retry logic with exponential backoff

## 🔒 Security & Cost Management

### API Key Configuration
```bash
# Set API key for LLM-enhanced processing
export HEYSOL_API_KEY="your_api_key_here"

# Or use CLI parameter
python mcode-cli.py data ingest-trials --method "llm_based" --api-key "your_key"
```

### Cost Monitoring
- **Token Tracking**: Monitor API usage
- **Budget Controls**: Set usage limits
- **Usage Reports**: Track costs by method and operation

## 🚨 Troubleshooting

### Common Issues

**Rule-Based Processing Issues:**
- Verify trial data format matches expected schema
- Check for required fields (`protocolSection`, `identificationModule`)
- Enable verbose mode for detailed error information

**LLM-Enhanced Processing Issues:**
- Verify API key configuration
- Check model availability and quota limits
- Review prompt template compatibility

**Method Comparison Issues:**
- Ensure trial data is valid for both methods
- Check async environment compatibility
- Verify sufficient timeout for LLM operations

## 📚 Examples & Notebooks

### Demo Scripts
- **`examples/unified_processor_demo.py`**: Complete demonstration of all features
- **`examples/CLI_demo.py`**: CLI usage examples
- **`examples/clinical_trials_demo.py`**: Clinical trial processing examples

### Jupyter Notebooks
- **`examples/clinical_trials_demo.ipynb`**: Interactive trial processing demo
- **`examples/patient_matching_demo.ipynb`**: Patient-trial matching examples
- **`examples/core_memory_integration_demo.ipynb`**: Memory integration examples

## 🔗 Related Documentation

- **[CLI Reference](README.md)**: Complete CLI documentation
- **[Processing Methods Guide](PROCESSING_METHODS_README.md)**: Detailed method comparison
- **[API Reference](../src/pipeline/README.md)**: Pipeline API details
- **[Configuration Guide](../src/config/README.md)**: Configuration options

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd mcode_translator

# Install dependencies
pip install -e .

# Run demos
python examples/unified_processor_demo.py
```

### Testing
```bash
# Run the unified processor demo
python examples/unified_processor_demo.py

# Test specific functionality
python -m pytest tests/unit/test_unified_processor.py -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HeySol API**: For providing the CORE Memory platform
- **ClinicalTrials.gov**: For clinical trial data access
- **mCODE Community**: For standardization efforts

---

**For questions or support, please refer to the main project documentation or submit an issue in the project repository.**