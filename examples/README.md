# 🚀 mCODE Translator Examples

This directory contains examples demonstrating the new engine-based architecture and key features of the mCODE Translator.

## 📁 Example Files

### Core Demos

- **`engine_demo.py`** - Comprehensive demonstration of the RegexEngine and LLMEngine architecture
- **`simple_unified_demo.py`** - Simple informational demo showing key concepts and usage patterns

### Specialized Examples

- **`clinical_trials_demo.py`** - End-to-end clinical trial processing workflow
- **`patient_matching_demo.py`** - Patient-trial matching using mCODE standards
- **`core_memory_integration_demo.py`** - Integration with CORE Memory for persistent storage

### Jupyter Notebooks

- **`comprehensive_demo.ipynb`** - Interactive comprehensive demonstration
- **`clinical_trials_ingestion.ipynb`** - Interactive trial ingestion workflow
- **`patient_matching_demo.ipynb`** - Interactive patient matching examples

## 🎯 Key Features Demonstrated

### Engine Architecture
- **RegexEngine**: Fast, deterministic structured data extraction
- **LLMEngine**: Intelligent, flexible processing with AI enhancement
- **Drop-in Replacement**: Both engines work in the same pipeline
- **Unified Interface**: Single API for choosing processing methods

### Performance & Flexibility
- **Speed vs. Intelligence**: Choose the right tool for your use case
- **Batch Processing**: Efficient handling of large datasets
- **Error Handling**: Robust processing with comprehensive error management
- **Memory Integration**: Seamless storage in CORE Memory

## 🚀 Quick Start

### Basic Usage
```bash
# Fast processing with RegexEngine
python mcode-cli.py data ingest-trials --cancer-type "breast" --engine "regex"

# Intelligent processing with LLMEngine
python mcode-cli.py data ingest-trials --cancer-type "breast" --engine "llm" --model "deepseek-coder"

# Compare both engines
python mcode-cli.py mcode summarize NCT02314481 --compare-engines
```

### Running Examples
```bash
# Run the engine architecture demo
python examples/engine_demo.py

# Run the simple informational demo
python examples/simple_unified_demo.py
```

## 📊 Performance Characteristics

| Feature | RegexEngine | LLMEngine | Best For |
|---------|-------------|-----------|----------|
| Speed | ⚡ Ultra-fast | 🐌 Slower | Large datasets |
| Cost | 💰 Free | 💳 API costs | Budget-conscious |
| Accuracy | 🎯 Deterministic | 🧠 Intelligent | Structured data |
| Flexibility | 🔧 Structured | 🌊 Any format | Complex text |

## 🛠️ Architecture Benefits

### For Developers
- **Clean Separation**: LLM and Regex code are completely separate
- **Easy Extension**: New engines can be added without changing pipeline
- **Consistent Interface**: Same API regardless of processing method
- **Shared Infrastructure**: Common pipeline, summarizer, and storage

### For Users
- **Choice**: Select the best tool for each use case
- **Performance**: Optimize for speed, cost, or intelligence
- **Reliability**: Both methods thoroughly tested and validated
- **Future-Proof**: Architecture supports new processing methods

## 📚 Learning Path

1. **Start Here**: Read `simple_unified_demo.py` for concepts
2. **See It In Action**: Run `engine_demo.py` for demonstrations
3. **Try It Yourself**: Use the Jupyter notebooks for interactive exploration
4. **Go Deep**: Study the CLI commands and integration examples

## 🔧 Troubleshooting

### Common Issues
- **Import Errors**: Ensure you're running from the project root directory
- **API Key Issues**: Set `HEYSOL_API_KEY` environment variable for LLM features
- **Path Issues**: Use absolute paths or run from the correct working directory

### Getting Help
- Check the main `README.md` for detailed documentation
- Review `PROCESSING_METHODS_README.md` for in-depth method comparison
- Examine `README_UNIFIED_PROCESSOR.md` for architecture details

## 🎉 What's New

This example collection showcases the latest engine-based architecture that provides:

- **🔄 Drop-in Engine Replacement**: Switch between processing methods seamlessly
- **⚡ Performance Optimization**: Choose the fastest method for your needs
- **🧠 AI Enhancement**: Leverage LLM intelligence when needed
- **💰 Cost Management**: Use free regex processing for simple cases
- **🔧 Easy Extension**: Add new processing engines without changing existing code

The examples demonstrate how this architecture gives you the best of both worlds: the speed and reliability of structured processing combined with the intelligence and flexibility of AI-powered analysis.