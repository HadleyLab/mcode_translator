# Benchmarking Metrics Collection Design for Pipeline Task Tracker

## Overview
This document details the design for collecting benchmarking metrics in the pipeline task tracker. The goal is to gather performance data that can be used to analyze and optimize pipeline execution.

## Metrics to Collect

### 1. Time-based Metrics
- **Total Pipeline Execution Time**: Time from pipeline start to completion
- **NLP Extraction Time**: Time for entity extraction step (NLP to Mcode pipeline)
- **Mcode Mapping Time**: Time for Mcode mapping step
- **Per-Task Processing Time**: Individual task processing duration

### 2. Resource Usage Metrics
- **Token Usage**: Total tokens consumed by LLM calls
  - Prompt tokens
  - Completion tokens
  - Total tokens per step
- **Memory Usage**: RAM consumption during processing (if measurable)
- **CPU Usage**: Processing power consumption (if measurable)

### 3. Quality Metrics
- **Entity Count**: Number of entities extracted
- **Mapping Count**: Number of Mcode elements mapped
- **Validation Scores**: Precision, recall, F1-score for both extraction and mapping
- **Compliance Score**: Percentage of mappings that comply with Mcode standards

### 4. Reliability Metrics
- **Success Rate**: Percentage of successful pipeline executions
- **Error Rate**: Percentage of failed pipeline executions
- **Error Types**: Categorization of failure reasons
- **Retry Count**: Number of retries needed for successful completion

## Data Collection Approach

### 1. Timing Measurements
- Use `time.time()` or `asyncio.get_event_loop().time()` for precise timing
- Record start and end times for each pipeline step
- Calculate durations for all relevant operations

### 2. Token Usage Tracking
- Extract token usage information from LLM API responses
- Aggregate token counts across all LLM calls in a pipeline
- Store token usage data with pipeline results

### 3. Quality Metrics Calculation
- Implement validation logic that compares results with gold standard
- Calculate precision, recall, and F1-score using standard formulas
- Determine compliance based on Mcode validation rules

### 4. Resource Usage Monitoring
- Use system monitoring libraries (e.g., `psutil`) to track resource usage
- Record peak resource consumption during pipeline execution
- Store resource usage data with pipeline results

## Data Storage

### 1. In-Memory Storage
- Extend existing data classes to include benchmarking fields
- Store metrics with task results for real-time display

### 2. Persistent Storage
- Save benchmarking results to JSON files for historical analysis
- Organize results by timestamp, pipeline type, and configuration
- Include environment information (model, prompts, etc.)

## Integration Points

### 1. Pipeline Processing
- Add timing measurements around pipeline execution
- Extract token usage from LLM responses
- Integrate validation logic with pipeline results

### 2. Task Management
- Associate benchmarking data with individual tasks
- Update task status with performance metrics
- Provide access to metrics through task APIs

### 3. UI Display
- Show metrics in task detail views
- Create summary dashboards for performance analysis
- Enable filtering and sorting by metric values

## Implementation Considerations

### 1. Performance Impact
- Minimize overhead from metrics collection
- Use efficient data structures for storing metrics
- Avoid blocking operations during metric collection

### 2. Accuracy
- Use high-resolution timers for precise measurements
- Handle time zone differences consistently
- Account for system clock adjustments

### 3. Scalability
- Design metrics collection to handle high concurrency
- Implement efficient aggregation for summary statistics
- Support incremental updates for long-running processes

## Privacy and Security
- Ensure benchmarking data does not contain sensitive information
- Anonymize any identifying information in stored metrics
- Follow data protection guidelines for performance data

## Future Enhancements
- Add support for custom metrics defined by users
- Implement real-time streaming of metrics to external systems
- Create alerting mechanisms for performance degradation
- Add comparative analysis between different pipeline configurations