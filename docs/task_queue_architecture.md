# Task Queue Architecture

## Overview

The task queue architecture provides a centralized system for managing and executing pipeline processing tasks with concurrent worker support. It ensures efficient resource utilization while maintaining proper task tracking and metric calculation.

## Components

### PipelineTaskQueue

The main task queue class that manages:

1. **Task Management**
   - Queues benchmark tasks for processing
   - Tracks task status and results
   - Maintains task statistics

2. **Worker Management**
   - Manages a pool of asynchronous workers
   - Handles worker lifecycle (start/stop)
   - Distributes tasks among available workers

3. **Metric Calculation**
   - Integrates with shared `BenchmarkResult` class for consistent metric calculation
   - Calculates precision, recall, F1-score, and compliance metrics
   - Uses gold standard data for validation

### BenchmarkTask

Data structure representing a single benchmark task with:

- Task identification and metadata
- Input data (trial data, expected entities/mappings)
- Execution status and results
- Performance metrics
- Live logging capability

## Workflow

1. **Task Creation**
   - Tasks are created with trial data and gold standard expectations
   - Tasks are added to the queue for processing

2. **Task Processing**
   - Workers pick up tasks from the queue
   - Appropriate pipeline is selected based on task type
   - Trial data is processed through the pipeline
   - Results are collected and stored in the task

3. **Metric Calculation**
   - Extracted entities and mCODE mappings are compared against gold standard
   - Performance metrics are calculated using the shared `BenchmarkResult` class
   - Metrics are stored in the task for reporting

4. **Result Tracking**
   - Task status is updated throughout processing
   - Live logs are maintained for monitoring
   - Statistics are updated for progress tracking

## Integration Points

### With Shared Components

- **BenchmarkResult**: Used for consistent metric calculation across the system
- **Pipeline Classes**: NlpMcodePipeline and McodePipeline for actual processing
- **Logging**: Centralized logging through the application's logging system

### With Optimization Framework

- Tasks can be created from optimization experiments
- Results feed back into the optimization process for model/prompt selection

## Benefits

1. **Concurrency**: Multiple tasks can be processed simultaneously
2. **Resource Efficiency**: Worker pool management prevents resource exhaustion
3. **Consistency**: Shared components ensure uniform metric calculation
4. **Scalability**: Architecture supports increasing workload by adjusting worker count
5. **Monitoring**: Live logging and status tracking enable real-time monitoring

## Usage

```python
# Create task queue
task_queue = PipelineTaskQueue(max_workers=5)

# Start workers
await task_queue.start_workers()

# Create and add tasks
task = BenchmarkTask(...)
await task_queue.add_task(task)

# Wait for completion
await task_queue.shutdown()

# Retrieve results
results = task_queue.get_all_tasks()