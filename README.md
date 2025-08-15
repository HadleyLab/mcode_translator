# mCODE Translator

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI Status](https://github.com/yourusername/mcode-translator/actions/workflows/python-app.yml/badge.svg)](https://github.com/yourusername/mcode-translator/actions)

A clinical trial matching system that translates eligibility criteria into structured mCODE (Minimal Common Oncology Data Elements) format using multiple NLP approaches.

## Features

- **Multi-engine NLP processing**:
  - LLM-based (DeepSeek API)
  - spaCy medical NLP
  - Regex pattern matching
- **Breast cancer specialization** with genomic feature extraction
- **Structured output** in mCODE FHIR format
- **Modular architecture** for easy extension

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interface
python src/nicegui_interface.py
```

## Documentation

- [System Overview](system_documentation.md)
- [Architecture Design](mcode_translator_architecture.md) 
- [NLP Engine Details](nlp_criteria_parsing_design.md)
- [Contribution Guide](CONTRIBUTING.md)

## Examples

```python
from src.llm_nlp_engine import LLMNLPEngine

engine = LLMNLPEngine()
result = engine.extract_mcode_features("ER+ breast cancer, HER2-negative")
```

## License

MIT License - See [LICENSE](LICENSE) for details.