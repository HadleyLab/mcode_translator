# 🚀 MCODE Translator Examples

Comprehensive collection of examples demonstrating MCODE Translator's clinical data processing, patient-trial matching, and CORE Memory integration capabilities.

## 📋 Examples Overview

This directory contains practical examples showing how to use MCODE Translator for various clinical and research applications:

| Example | Description | Use Case | Complexity |
|---------|-------------|----------|------------|
| [`quick_start.py`](quick_start.py) | Quick start guide | Getting started | 🟢 Beginner |
| [`patients_demo.py`](patients_demo.py) | Patient data processing | Clinical care | 🟡 Intermediate |
| [`clinical_trials_demo.py`](clinical_trials_demo.py) | Trial data analysis | Research | 🟡 Intermediate |
| [`core_memory_integration_demo.py`](core_memory_integration_demo.py) | Memory operations | Advanced features | 🔴 Advanced |
| [`patient_matching_demo.py`](patient_matching_demo.py) | Patient-trial matching | Precision medicine | 🔴 Advanced |
| [`comprehensive_demo.py`](comprehensive_demo.py) | Full capabilities | Production demo | 🔴 Advanced |

## 🎯 Quick Start Examples

### Basic Setup
```bash
# 1. Set your API key
export HEYSOL_API_KEY="your-api-key-here"

# 2. Run the quick start
python quick_start.py
```

### Interactive Learning
```bash
# Open the interactive notebook
jupyter notebook quick_start.ipynb
```

## 👥 Patient Data Examples

### Patient Profile Processing
```python
python examples/patients_demo.py
```

**What it demonstrates:**
- Patient data ingestion with rich metadata
- Automated patient summarization
- Patient classification and analytics
- Clinical research cohort identification
- Patient outcome analysis

**Key features:**
- Multi-format patient data processing
- Automated clinical summarization
- Statistical analysis and insights
- Research cohort identification
- Quality metrics and validation

## 🧪 Clinical Trials Examples

### Trial Landscape Analysis
```python
python examples/clinical_trials_demo.py
```

**What it demonstrates:**
- Clinical trial data ingestion and processing
- Trial eligibility criteria analysis
- Trial enrollment optimization
- Competitive intelligence gathering
- Trial outcome analysis and reporting

**Key features:**
- Multi-source trial data integration
- Automated eligibility screening
- Enrollment trend analysis
- Strategic trial recommendations
- Outcome prediction modeling

## 🧠 CORE Memory Integration Examples

### 💾 Save-to-Memory Functionality
```python
python examples/core_memory_integration_demo.py
```

**What it demonstrates:**
- **Persistent Data Storage**: Save clinical data to CORE Memory for long-term retention
- Multi-space memory architecture with domain-specific organization
- **Persistent Storage Across Sessions**: Data remains available across different sessions and restarts
- Advanced semantic search capabilities across memory spaces
- Knowledge graph relationship discovery and pattern analysis
- Cross-domain data integration and correlation
- Memory performance optimization and monitoring

**Key Save-to-Memory Features:**
- **Long-term Clinical Knowledge Storage**: Patient data, trial results, and research persist indefinitely
- **Advanced Search and Retrieval**: Semantic search across all stored clinical data
- **Relationship and Pattern Discovery**: Find connections between patients, treatments, and outcomes
- **Cross-Domain Clinical Intelligence**: Integrate patient data with trial results and research evidence
- **Performance Monitoring and Optimization**: Track memory usage and query performance
- **Session Continuity**: Maintain clinical context across different interactions

**Save-to-Memory Workflow:**
1. **Data Ingestion**: Clinical data is processed and saved to appropriate memory spaces
2. **Persistent Storage**: Information is stored with rich metadata for future retrieval
3. **Knowledge Accumulation**: Build long-term memory for AI applications and clinical decision support
4. **Cross-Session Availability**: Access historical data across different sessions and time periods
5. **Evidence Integration**: Combine current and historical clinical data for comprehensive analysis

## 🎯 Patient-Trial Matching Examples

### Intelligent Matching Algorithms
```python
python examples/patient_matching_demo.py
```

**What it demonstrates:**
- Advanced patient-trial matching algorithms
- Multi-criteria eligibility assessment
- Quantitative matching score calculation
- Knowledge graph-enhanced matching
- Clinical decision support recommendations
- Matching analytics and insights

**Key features:**
- Comprehensive patient profile analysis
- Clinical trial eligibility parsing
- Intelligent matching algorithms
- Evidence-based recommendations
- Strategic enrollment insights

## 🚀 Comprehensive Examples

### Full Capabilities Showcase
```python
python examples/comprehensive_demo.py
```

**What it demonstrates:**
- Complete MCODE Translator workflow
- Multi-domain clinical data processing
- Advanced search and analytics
- Production-ready patterns
- Clinical health monitoring
- Strategic recommendations

**Key features:**
- End-to-end clinical data workflows
- Multi-domain architecture patterns
- Production deployment strategies
- Comprehensive error handling
- Performance optimization techniques

## 🏥 Clinical Use Cases

### For Clinicians
- **Treatment Planning**: Find similar cases and evidence-based treatments
- **Trial Identification**: Match patients to appropriate clinical trials
- **Outcome Prediction**: Analyze expected outcomes based on patient characteristics
- **Evidence Access**: Quick access to relevant research and guidelines

### For Researchers
- **Cohort Building**: Identify patient populations for research studies
- **Trial Design**: Analyze existing trial landscapes and gaps
- **Evidence Synthesis**: Combine multiple evidence sources for analysis
- **Knowledge Discovery**: Find patterns and relationships in clinical data

### For Administrators
- **Quality Improvement**: Track clinical outcomes and process metrics
- **Resource Optimization**: Optimize clinical trial enrollment processes
- **Compliance Monitoring**: Ensure adherence to clinical standards
- **Performance Analytics**: Monitor system usage and effectiveness

## 🔧 Technical Features Demonstrated

### Data Processing
- ✅ Multi-format clinical data ingestion
- ✅ Rich metadata extraction and enrichment
- ✅ Data quality validation and cleaning
- ✅ Cross-domain data integration
- ✅ Real-time data processing capabilities

### Search and Discovery
- ✅ Semantic search across clinical domains
- ✅ Multi-domain query optimization
- ✅ Knowledge graph relationship discovery
- ✅ Advanced filtering and ranking
- ✅ Real-time search performance

### Analytics and Insights
- ✅ Automated summarization and reporting
- ✅ Statistical analysis and pattern recognition
- ✅ Predictive modeling capabilities
- ✅ Clinical decision support generation
- ✅ Performance monitoring and optimization

### Production Features
- ✅ Comprehensive error handling
- ✅ Resource management and cleanup
- ✅ Health monitoring and alerting
- ✅ Audit trail maintenance
- ✅ Scalable architecture patterns

## 💾 Save-to-Memory Functionality

All MCODE Translator examples demonstrate **persistent data storage** in CORE Memory:

### **What is Save-to-Memory?**
- **Persistent Storage**: Clinical data is automatically saved to CORE Memory for long-term retention
- **Session Continuity**: Data remains available across different sessions and system restarts
- **Knowledge Accumulation**: Build comprehensive clinical knowledge bases over time
- **Cross-Domain Integration**: Connect patient data, trial results, and research evidence

### **Save-to-Memory in Action**
Every example demonstrates saving clinical data to CORE Memory:

```python
# Patient data is automatically saved to persistent memory
result = client.ingest(
    message=patient_content,
    space_id=patients_space_id,
    metadata=patient_metadata
)
# 💾 Data is now stored in CORE Memory for future retrieval
```

### **Memory Benefits Demonstrated**
- **Long-term Storage**: Patient profiles, trial results, and research persist indefinitely
- **Advanced Search**: Semantic search across all stored clinical data
- **Knowledge Graphs**: Discover relationships between patients, treatments, and outcomes
- **Evidence Integration**: Combine current and historical data for comprehensive analysis
- **Performance Optimization**: Efficient storage and retrieval of clinical information

## 📚 Learning Path

### 🟢 Beginner Level
1. Start with [`quick_start.py`](quick_start.py) or [`quick_start.ipynb`](quick_start.ipynb)
2. Understand basic concepts and setup
3. Learn core functionality and navigation
4. **See save-to-memory in action with your first data ingestion**

### 🟡 Intermediate Level
1. Explore [`patients_demo.py`](patients_demo.py) for patient data workflows
2. Study [`clinical_trials_demo.py`](clinical_trials_demo.py) for trial analysis
3. Practice with real clinical data scenarios
4. **Experience persistent storage across multiple clinical domains**

### 🔴 Advanced Level
1. Master [`core_memory_integration_demo.py`](core_memory_integration_demo.py) for persistent storage
2. Implement [`patient_matching_demo.py`](patient_matching_demo.py) for precision medicine
3. Deploy [`comprehensive_demo.py`](comprehensive_demo.py) for production systems
4. **Build comprehensive clinical knowledge bases with persistent memory**

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- MCODE Translator installed: `pip install -e .`
- Valid HeySol API key configured
- Jupyter notebook (optional, for interactive examples)

### Running Examples
```bash
# Quick start (recommended first)
python quick_start.py

# Patient data processing
python examples/patients_demo.py

# Clinical trials analysis
python examples/clinical_trials_demo.py

# Memory integration (advanced)
python examples/core_memory_integration_demo.py

# Patient-trial matching (advanced)
python examples/patient_matching_demo.py

# Comprehensive demo (full capabilities)
python examples/comprehensive_demo.py
```

### Interactive Learning
```bash
# Start Jupyter notebook server
jupyter notebook

# Open quick_start.ipynb for interactive learning
# Navigate through examples in your browser
```

## 💡 Tips for Success

### Best Practices
- **Start Simple**: Begin with quick_start.py before moving to advanced examples
- **Build Gradually**: Master basic concepts before tackling complex scenarios
- **Experiment Safely**: Use demo spaces for testing and learning
- **Document Learnings**: Keep notes on what works for your use cases

### Troubleshooting
- **API Key Issues**: Verify your HEYSOL_API_KEY environment variable
- **Import Errors**: Ensure MCODE Translator is properly installed
- **Performance Issues**: Check your internet connection for API calls
- **Memory Issues**: Close unused notebooks and restart if needed

### Customization
- **Modify Examples**: Adapt examples for your specific clinical domain
- **Add Data**: Include your own clinical data for realistic testing
- **Extend Functionality**: Build upon examples for custom applications
- **Integrate Systems**: Connect examples with your existing clinical systems

## 📖 Documentation References

- **[Main README](../README.md)**: Complete project documentation
- **[API Documentation](../docs/)**: Technical API reference
- **[Clinical Guidelines](./clinical_trials_demo.py)**: Clinical use case examples
- **[Research Protocols](./patients_demo.py)**: Research workflow examples

## 🤝 Contributing

Found an issue or want to improve the examples? Here's how:

1. **Report Issues**: Use the GitHub issue tracker
2. **Suggest Improvements**: Create a feature request
3. **Submit Examples**: Add new examples for common use cases
4. **Fix Documentation**: Improve clarity and completeness

## 📞 Support

- **Documentation**: Check the main README and inline comments
- **Examples**: Use these examples as starting points for your applications
- **Community**: Join discussions for help and best practices
- **Issues**: Report problems or request features

---

**Happy Learning!** 🎉

These examples will help you master MCODE Translator and build powerful clinical applications. Start with the basics and progress to advanced features as you become more comfortable with the system.