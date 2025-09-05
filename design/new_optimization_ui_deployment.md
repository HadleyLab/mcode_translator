# New Optimization UI Deployment Plan

## Overview
This document outlines the deployment plan for the new Modern Optimization UI, including installation requirements, deployment steps, and migration from the legacy UI.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended)
- 100MB free disk space for application files
- Internet connectivity for API access

### Python Dependencies
The new UI requires the following packages:
- nicegui>=1.4.0 (for web interface)
- pandas>=1.5.0 (for data analysis)
- psutil>=5.9.0 (for system monitoring)
- Additional dependencies from requirements.txt

### External Services
- Access to configured LLM providers (DeepSeek, OpenAI, etc.)
- Proper API keys in environment variables
- Cache directory with write permissions

## Installation Steps

### 1. Environment Setup
```bash
# Create a new conda environment
conda create -n Mcode_optimization python=3.9
conda activate Mcode_optimization

# Install core dependencies
pip install -r requirements.txt

# Install additional UI dependencies
pip install nicegui pandas psutil
```

### 2. Configuration
Ensure the following configuration files are properly set up:
- `config.json` - Main application configuration
- `prompts/prompts_config.json` - Prompt library configuration
- `models/models_config.json` - Model library configuration
- Environment variables for API keys

### 3. Directory Structure
```
mcode_translator/
├── src/
│   └── optimization/
│       ├── new_optimization_ui.py          # Main UI implementation
│       ├── ui_components/                  # UI component modules
│       │   ├── config_manager.py
│       │   ├── benchmark_runner.py
│       │   ├── results_analyzer.py
│       │   └── system_monitor.py
│       └── ui_utils/                       # Utility modules
│           ├── data_loader.py
│           └── validation.py
├── prompts/                                # Prompt library
│   ├── prompts_config.json
│   └── txt/
├── models/                                 # Model library
│   └── models_config.json
├── cache/                                  # Cache directory
├── results/                                # Results directory
└── config.json                            # Main configuration
```

## Deployment Process

### 1. Code Deployment
```bash
# Clone or copy the latest codebase
git clone <repository-url> mcode_translator
cd mcode_translator

# Verify all required files are present
ls -la src/optimization/new_optimization_ui.py
ls -la src/optimization/ui_components/
ls -la src/optimization/ui_utils/
```

### 2. Configuration Verification
```bash
# Check configuration files
cat config.json
cat prompts/prompts_config.json
cat models/models_config.json

# Verify environment variables
echo $DEEPSEEK_API_KEY
echo $OPENAI_API_KEY
```

### 3. Test Run
```bash
# Run the UI in development mode
python src/optimization/new_optimization_ui.py

# Access the UI at http://localhost:8082
```

### 4. Production Deployment
For production deployment, consider using:
- A WSGI server like Gunicorn
- A reverse proxy like Nginx
- Process management with systemd or supervisor

Example Gunicorn deployment:
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8082 src.optimization.new_optimization_ui:app
```

## Migration from Legacy UI

### 1. Backup Current Implementation
```bash
# Backup existing UI files
cp src/optimization/optimization_ui.py src/optimization/optimization_ui.py.backup
```

### 2. Data Migration
- Benchmark results are compatible between UI versions
- Configuration files remain unchanged
- No database migration required

### 3. User Transition
- Update documentation and user guides
- Provide training for new interface features
- Maintain legacy UI temporarily during transition

## Rollback Plan

### If Issues Occur
1. Stop the new UI service
2. Restore the backup of the legacy UI
3. Restart the legacy UI service
4. Investigate and fix issues in the new UI

### Rollback Commands
```bash
# Stop new UI
pkill -f new_optimization_ui.py

# Restore legacy UI
mv src/optimization/optimization_ui.py.backup src/optimization/optimization_ui.py

# Start legacy UI
python src/optimization/optimization_ui.py
```

## Monitoring and Maintenance

### Health Checks
- Monitor application logs for errors
- Check resource usage (CPU, memory, disk)
- Verify API connectivity to LLM providers
- Ensure cache directory has sufficient space

### Log Locations
- Application logs: `logs/` directory
- System logs: Check system journal or log files
- Error logs: stderr output from the application

### Regular Maintenance Tasks
1. **Cache Cleanup**
   ```bash
   # Periodically clean cache to free space
   python -c "from src.optimization.new_optimization_ui import ModernOptimizationUI; ui = ModernOptimizationUI(); ui._clear_cache()"
   ```

2. **Configuration Reload**
   ```bash
   # Reload configurations without restart
   curl -X POST http://localhost:8082/reload-config
   ```

3. **Result Archiving**
   ```bash
   # Archive old results to save space
   tar -czf results_archive_$(date +%Y%m%d).tar.gz results/
   rm -rf results/*.json
   ```

## Security Considerations

### API Key Management
- Store API keys in environment variables, not in code
- Use different keys for development and production
- Rotate keys periodically

### Network Security
- Restrict access to the UI port (8082) with firewall rules
- Use HTTPS in production with SSL certificates
- Implement authentication if needed for multi-user environments

### Data Protection
- Ensure benchmark results are stored securely
- Protect configuration files with appropriate permissions
- Regular backups of important data

## Performance Optimization

### Caching Strategy
- Enable file-based caching for LLM responses
- Configure appropriate TTL values
- Monitor cache size and performance

### Resource Management
- Set appropriate concurrency levels for benchmarks
- Configure timeouts to prevent hanging operations
- Monitor system resources during heavy usage

### UI Performance
- Use pagination for large datasets
- Implement lazy loading for visualizations
- Optimize data loading and processing

## Troubleshooting Guide

### Common Issues

1. **UI Not Starting**
   - Check Python dependencies
   - Verify configuration files
   - Review error logs

2. **LLM API Connection Failures**
   - Verify API keys
   - Check network connectivity
   - Confirm model configurations

3. **Performance Issues**
   - Monitor system resources
   - Adjust concurrency settings
   - Check cache configuration

### Diagnostic Commands
```bash
# Check Python environment
python --version
pip list | grep nicegui

# Check configuration files
ls -la config.json prompts/prompts_config.json models/models_config.json

# Check logs
tail -f logs/app.log

# Test API connectivity
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/models
```

## Support and Maintenance

### Documentation
- User guides for new UI features
- API documentation for integration
- Troubleshooting guides

### Updates
- Regular updates for security patches
- Feature enhancements based on user feedback
- Compatibility testing with new LLM models

### Contact
For support, contact the development team through:
- GitHub issues
- Internal support channels
- Documentation resources