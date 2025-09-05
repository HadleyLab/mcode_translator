# Benchmark Task Tracker Design

## Overview
This document outlines the design for a new benchmark task tracker that extends the existing functionality with Mcode-optimize integration. The new system will provide a GUI interface for running benchmark validations using the Mcode-optimize CLI functionality.

## Requirements
1. Extend the existing task management functionality to handle benchmark validation tasks
2. Implement a control panel for selecting prompts, models, and trials
3. Add live color logging for all LLM calls, caches, etc.
4. Integrate the Mcode-optimize CLI functionality with the GUI
5. Use the provided test files for trials and gold standard validation

## Architecture

### Core Components

1. **BenchmarkTaskTrackerUI** - Main UI class that extends the existing AdvancedTaskTrackerUI
2. **ControlPanel** - Interface for selecting prompts, models, and trials
3. **LiveLogger** - Component for displaying live color logging
4. **BenchmarkRunner** - Integration with Mcode-optimize framework
5. **ResultsAnalyzer** - Component for analyzing and displaying benchmark results

### File Structure
```
src/optimization/
├── benchmark_task_tracker.py
├── benchmark_task_tracker_design.md
└── ... (existing files)
```

## Implementation Plan

### Phase 1: Core UI Extension
- Create BenchmarkTaskTrackerUI class that inherits from AdvancedTaskTrackerUI
- Add benchmark-specific task types
- Implement control panel for prompt/model/trial selection

### Phase 2: Integration with Mcode-optimize
- Integrate PromptOptimizationFramework
- Implement benchmark execution functionality
- Add live logging capabilities

### Phase 3: Advanced Features
- Implement results analysis and visualization
- Add export functionality for benchmark results
- Implement configuration management

## Detailed Design

### BenchmarkTaskTrackerUI Class
```python
class BenchmarkTaskTrackerUI(AdvancedTaskTrackerUI):
    def __init__(self):
        super().__init__()
        self.framework = PromptOptimizationFramework()
        self.available_prompts = {}
        self.available_models = {}
        self.trial_data = {}
        self.gold_standard_data = {}
        self.live_log_display = None
        self.control_panel = None
        
    def setup_ui(self):
        # Extend the existing UI with benchmark-specific components
        super().setup_ui()
        self._setup_control_panel()
        self._setup_live_logger()
        
    def _setup_control_panel(self):
        # Implementation for control panel with prompt/model/trial selection
        
    def _setup_live_logger(self):
        # Implementation for live color logging display
```

### Control Panel
The control panel will include:
- Prompt selection (NLP extraction and Mcode mapping prompts)
- Model selection (various LLM models)
- Trial selection (clinical trial data)
- Run configuration options (metrics, top N combinations, etc.)

### Live Logger
The live logger will:
- Display real-time logging from the optimization framework
- Use color coding for different log levels (INFO, WARNING, ERROR)
- Show LLM calls, cache hits/misses, token usage, etc.

### Benchmark Runner Integration
Integration points with Mcode-optimize:
- Loading prompt and model libraries
- Running benchmark combinations
- Processing gold standard validation
- Capturing and displaying real-time metrics

## Data Flow

1. User configures benchmark parameters via control panel
2. System loads selected prompts, models, and trial data
3. Benchmark tasks are created and added to the task queue
4. Tasks are executed using the Mcode-optimize framework
5. Live logging is captured and displayed in real-time
6. Results are collected and analyzed
7. User can view detailed results and export data

## UI Components

### Task List
- Extended to show benchmark-specific information (prompt, model, trial)
- Progress indicators for long-running optimizations
- Status badges for different benchmark states

### Control Panel
- Tabbed interface for different configuration sections
- Multi-select components for prompts, models, and trials
- Advanced options for benchmark execution

### Live Logger
- Color-coded log display with real-time updates
- Filter options for different log levels
- Export functionality for log data

### Results Display
- Summary statistics for benchmark runs
- Detailed results table with sorting and filtering
- Visualization components for performance comparisons

## Integration Points

### Prompt Library
- Load available prompts from prompt configuration
- Display prompt information (type, description, version)
- Allow selection of multiple prompts for benchmarking

### Model Library
- Load available models from model configuration
- Display model information (type, capabilities, version)
- Allow selection of multiple models for benchmarking

### Trial Data
- Load clinical trial data from JSON files
- Parse and display trial information
- Associate with gold standard validation data

### Gold Standard Validation
- Load gold standard data for validation
- Compare benchmark results against gold standard
- Calculate accuracy metrics (F1, precision, recall)

## Error Handling

### Configuration Errors
- Invalid prompt or model selections
- Missing trial or gold standard data
- Configuration file parsing errors

### Runtime Errors
- LLM API call failures
- Timeout during benchmark execution
- Validation errors with gold standard data

### Recovery Strategies
- Graceful degradation for non-critical errors
- Clear error messages with recovery suggestions
- Logging of all errors for debugging

## Performance Considerations

### Memory Management
- Efficient loading and caching of prompt/model data
- Cleanup of benchmark results after display
- Memory usage monitoring for long-running benchmarks

### Concurrency
- Limit concurrent benchmark executions to prevent resource exhaustion
- Queue management for large numbers of benchmark tasks
- Progress reporting for queued tasks

### Optimization
- Caching of frequently accessed data
- Efficient data structures for result storage
- Lazy loading of UI components when possible

## Testing Strategy

### Unit Tests
- Test individual components (control panel, logger, etc.)
- Mock external dependencies (LLM APIs, file system)
- Verify data flow between components

### Integration Tests
- Test end-to-end benchmark execution
- Verify integration with Mcode-optimize framework
- Validate results against gold standard data

### UI Tests
- Test UI interactions and state changes
- Verify responsive design across different screen sizes
- Test accessibility features

## Deployment Considerations

### Dependencies
- NiceGUI for web interface
- Mcode-optimize framework
- Required Python packages (pandas, etc.)

### Configuration
- Default paths for prompt/model/trial data
- Logging configuration
- Port and other server settings

### Scalability
- Support for running on different hardware configurations
- Options for distributed benchmark execution
- Resource monitoring and management

## Future Enhancements

### Advanced Features
- Automated prompt optimization
- Model comparison reports
- Historical benchmark tracking

### UI Improvements
- Drag-and-drop configuration
- Advanced visualization options
- Customizable dashboard layouts

### Integration Extensions
- Support for additional LLM providers
- Integration with experiment tracking tools
- API for programmatic benchmark execution