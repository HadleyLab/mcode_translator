# Modern Optimization UI User Guide

## Overview
The Modern Optimization UI is a web-based interface for managing and executing prompt optimization experiments. It provides a comprehensive set of tools for configuring experiments, running benchmarks, analyzing results, and monitoring system status.

## Accessing the UI
To access the UI, start the application and navigate to `http://localhost:8082` in your web browser.

```bash
# Start the UI
python src/optimization/new_optimization_ui.py
```

## Main Interface

The UI is organized into four main tabs:

1. **Library Management** - Manage prompt and model libraries
2. **Benchmark Execution** - Configure and run benchmark experiments
3. **Results Analysis** - Analyze and visualize benchmark results
4. **System Status** - Monitor system resources and configuration

## Library Management Tab

### Prompt Library Browser
This section displays all available prompts from the prompt library:

- **Filtering**: Use the dropdown filters to narrow down prompts by type (NLP_EXTRACTION, MCODE_MAPPING) or status (production, experimental)
- **Default Management**: Set prompts as default for their type using the action buttons
- **Details**: View prompt descriptions and metadata

### Model Library Browser
This section displays all available models from the model library:

- **Filtering**: Filter models by type (CODE_GENERATION, GENERAL_CONVERSATION, etc.) or status
- **Default Management**: Set models as default using the action buttons
- **Capabilities**: View model capabilities and specifications

## Benchmark Execution Tab

### Experiment Configuration
Configure your benchmark experiment by selecting:

- **Prompts**: Choose one or more prompts to test
- **Models**: Select the LLM models to use
- **Test Cases**: Pick the test cases for evaluation

### Advanced Options
Fine-tune your experiment with advanced settings:

- **Optimization Metric**: Choose the primary metric for evaluation (F1 Score, Precision, Recall, Compliance Score)
- **Concurrency Level**: Set how many experiments run simultaneously
- **Timeout**: Configure maximum execution time per experiment

### Execution Controls
- **Run Benchmark**: Start the benchmark execution
- **Stop Benchmark**: Cancel the current execution
- **Progress Tracking**: Monitor real-time progress and metrics

### Real-time Metrics
During execution, view live metrics for the current experiment:
- Prompt and model being tested
- Test case information
- Duration and performance metrics
- Entity extraction and mapping results

## Results Analysis Tab

### Loading Results
- **Load Latest Results**: Import the most recent benchmark results
- **Export to CSV**: Save results for external analysis
- **Generate Report**: Create a comprehensive performance report
- **Clear Results**: Remove current results from the UI

### Summary Statistics
View key metrics across all experiments:
- Total runs and success rate
- Average duration and performance scores
- Models and prompts tested

### Performance Visualizations
Generate charts to visualize performance data:
- Success rates by prompt variant
- Compliance scores by prompt variant
- Comparative performance analysis

### Detailed Results Table
Browse individual experiment results with columns for:
- Run ID and configuration details
- Performance metrics (duration, entities, compliance, F1 score)
- Success status

## System Status Tab

### Resource Usage
Monitor system performance:
- CPU and memory utilization
- Disk usage statistics

### Cache Status
Manage the application cache:
- View cache size and file count
- Clear cache to free space
- Reload cache configuration

### Configuration Overview
View and manage application configuration:
- Configuration file status and last modified dates
- Reload configurations without restarting the application

## Best Practices

### Experiment Design
1. **Start Small**: Begin with a limited set of prompts, models, and test cases
2. **Use Default Settings**: Leverage default configurations for initial experiments
3. **Monitor Progress**: Keep an eye on real-time metrics during execution
4. **Analyze Results**: Use visualizations to identify performance patterns

### Performance Optimization
1. **Concurrency**: Adjust concurrency levels based on system resources
2. **Caching**: Enable caching to reduce API calls and execution time
3. **Timeouts**: Set appropriate timeouts to prevent hanging operations
4. **Resource Monitoring**: Watch system resources during heavy usage

### Data Management
1. **Regular Exports**: Export results periodically to prevent data loss
2. **Cache Maintenance**: Clean cache regularly to free disk space
3. **Configuration Backups**: Keep backups of important configuration files

## Troubleshooting

### Common Issues

1. **UI Not Loading**
   - Verify the application is running
   - Check network connectivity
   - Review application logs for errors

2. **Benchmark Failures**
   - Check API key configuration
   - Verify model and prompt availability
   - Review error messages in the status display

3. **Performance Issues**
   - Reduce concurrency levels
   - Monitor system resources
   - Check cache configuration

### Getting Help
For additional support:
- Review the documentation
- Check application logs for detailed error information
- Contact the development team for assistance

## Advanced Features

### Custom Prompt Integration
Add new prompts to the library by:
1. Creating prompt files in the `prompts/txt/` directory
2. Updating `prompts/prompts_config.json` with prompt metadata
3. Refreshing the prompt library in the UI

### Model Configuration
Add new models by:
1. Updating `models/models_config.json` with model details
2. Refreshing the model library in the UI
3. Testing with benchmark experiments

### API Integration
The UI integrates with:
- LLM providers (DeepSeek, OpenAI, etc.)
- File-based prompt and model libraries
- System monitoring tools

## Security Considerations

### API Keys
- Store API keys in environment variables
- Use different keys for development and production
- Rotate keys regularly for security

### Network Access
- Restrict access to the UI port in production
- Use HTTPS with SSL certificates
- Implement authentication for multi-user environments

## Feedback and Support

### Reporting Issues
Report bugs or issues through:
- GitHub issues
- Internal support channels
- Email support

### Feature Requests
Submit feature requests for future enhancements:
- GitHub feature requests
- User feedback forms
- Direct communication with the development team

### Contributing
Contribute to the project by:
- Forking the repository
- Creating pull requests
- Following coding standards and guidelines