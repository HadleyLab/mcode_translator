# Benchmarking Metrics Calculation Implementation Plan

## Overview
This document outlines the implementation plan for adding benchmarking metrics calculation to the pipeline task tracker. This will enable the system to collect and display performance metrics for pipeline execution.

## Implementation Steps

### 1. Extend Data Classes
Add benchmarking metrics fields to the data classes:

```python
@dataclass
class LLMCallTask:
    """Represents a single LLM call task (extraction or mapping)"""
    name: str  # "NLP Extraction" or "mCODE Mapping"
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    details: Optional[str] = None
    
    # Validation metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    compliance_score: Optional[float] = None
    
    # Benchmarking metrics
    processing_time_ms: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate duration of the task"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

@dataclass
class PipelineTask:
    """Represents a complete pipeline task with two LLM call sub-tasks"""
    id: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    nlp_extraction: LLMCallTask = None
    mcode_mapping: LLMCallTask = None
    error_message: Optional[str] = None
    trial_data: Optional[Dict[str, Any]] = None
    pipeline_type: str = "NLP to mCODE"  # Store the pipeline type used for this task
    test_case_name: str = "unknown"  # Store the test case name for display
    prompt_info: Optional[Dict[str, str]] = None  # Store prompt information for display
    
    # Benchmarking metrics
    total_processing_time_ms: Optional[float] = None
    total_token_usage: Optional[int] = None
    
    def __post_init__(self):
        """Initialize sub-tasks if not provided"""
        if not self.nlp_extraction:
            self.nlp_extraction = LLMCallTask(name="NLP Extraction")
        if not self.mcode_mapping:
            self.mcode_mapping = LLMCallTask(name="mCODE Mapping")
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate duration of the pipeline task"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
```

### 2. Add Benchmarking Metrics Collection
Implement methods to collect benchmarking metrics during pipeline execution:

```python
def _collect_benchmarking_metrics(self, task: PipelineTask, result: Any) -> None:
    """Collect benchmarking metrics for a pipeline task"""
    try:
        # Collect processing time metrics
        if task.nlp_extraction and task.nlp_extraction.start_time and task.nlp_extraction.end_time:
            task.nlp_extraction.processing_time_ms = (task.nlp_extraction.end_time - task.nlp_extraction.start_time) * 1000
        
        if task.mcode_mapping and task.mcode_mapping.start_time and task.mcode_mapping.end_time:
            task.mcode_mapping.processing_time_ms = (task.mcode_mapping.end_time - task.mcode_mapping.start_time) * 1000
        
        # Collect total processing time
        if task.start_time and task.end_time:
            task.total_processing_time_ms = (task.end_time - task.start_time) * 1000
        
        # Collect token usage metrics
        total_tokens = 0
        
        # From nlp_extraction
        if task.nlp_extraction and task.nlp_extraction.token_usage:
            total_tokens += task.nlp_extraction.token_usage.get('total_tokens', 0)
        
        # From mcode_mapping
        if task.mcode_mapping and task.mcode_mapping.token_usage:
            total_tokens += task.mcode_mapping.token_usage.get('total_tokens', 0)
        
        # From result metadata
        if hasattr(result, 'metadata') and 'token_usage' in result.metadata:
            total_tokens += result.metadata['token_usage'].get('total_tokens', 0)
        
        task.total_token_usage = total_tokens
        
        logger.info(f"Benchmarking metrics collected for task {task.id}")
        logger.info(f"  Total processing time: {task.total_processing_time_ms:.2f}ms")
        logger.info(f"  Total token usage: {task.total_token_usage}")
        
    except Exception as e:
        logger.error(f"Failed to collect benchmarking metrics for task {task.id}: {e}")

def _collect_resource_usage_metrics(self) -> Dict[str, Any]:
    """Collect system resource usage metrics"""
    try:
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Collect memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        # Collect CPU usage
        cpu_percent = process.cpu_percent()
        
        return {
            'memory_usage_mb': memory_mb,
            'cpu_percent': cpu_percent
        }
    except Exception as e:
        logger.error(f"Failed to collect resource usage metrics: {e}")
        return {}
```

### 3. Update Pipeline Processing
Modify the `_process_pipeline_task` method to include benchmarking metrics collection:

```python
async def _process_pipeline_task(self, task: PipelineTask):
    """Process a single pipeline task"""
    logger.info(f"Processing pipeline task {task.id}")
    
    # Record start time for benchmarking
    task_start_time = asyncio.get_event_loop().time()
    
    # Update task status
    task.status = TaskStatus.RUNNING
    task.start_time = task_start_time
    if self.status_label:
        self.status_label.set_text(f"Running task {task.id}")
    
    try:
        # Get gold standard data for validation
        expected_entities = []
        expected_mappings = []
        
        if self.gold_standard_data and task.test_case_name in self.gold_standard_data:
            gold_data = self.gold_standard_data[task.test_case_name]
            expected_entities = gold_data.get('expected_extraction', {}).get('entities', [])
            expected_mappings = gold_data.get('expected_mcode_mappings', {}).get('mapped_elements', [])
        
        # Create pipeline instance with selected prompts
        if self.pipeline_selector.value == 'Direct to mCODE':
            prompt_name = self.direct_mcode_prompt_selector.value
            pipeline = McodePipeline(prompt_name=prompt_name)
        else:
            extraction_prompt = self.nlp_prompt_selector.value
            mapping_prompt = self.mcode_prompt_selector.value
            pipeline = NlpMcodePipeline(
                extraction_prompt_name=extraction_prompt,
                mapping_prompt_name=mapping_prompt
            )
        
        # Process the clinical trial with validation
        result = await self._run_pipeline_with_tracking(pipeline, task, expected_entities, expected_mappings)
        
        # Perform validation if gold standard data is available
        if result and (expected_entities or expected_mappings):
            self._perform_validation(task, result, expected_entities, expected_mappings)
        
        # Collect benchmarking metrics
        task_end_time = asyncio.get_event_loop().time()
        task.end_time = task_end_time
        self._collect_benchmarking_metrics(task, result)
        
        # Update task with result
        task.status = TaskStatus.SUCCESS
        self._add_notification(f"Task {task.id} completed", 'positive')
        logger.info(f"Pipeline task {task.id} completed successfully")
        
    except Exception as e:
        # Handle task failure
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.end_time = asyncio.get_event_loop().time()
        self._add_notification(f"Task {task.id} failed: {str(e)}", 'negative')
        logger.error(f"Pipeline task {task.id} failed: {e}")
```

### 4. Add Summary Statistics
Implement methods to calculate and display summary statistics:

```python
def _calculate_summary_statistics(self) -> Dict[str, Any]:
    """Calculate summary statistics for all completed tasks"""
    try:
        completed_tasks = [task for task in self.tasks if task.status == TaskStatus.SUCCESS]
        
        if not completed_tasks:
            return {}
        
        # Calculate processing time statistics
        processing_times = [task.total_processing_time_ms for task in completed_tasks if task.total_processing_time_ms is not None]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        min_processing_time = min(processing_times) if processing_times else 0
        max_processing_time = max(processing_times) if processing_times else 0
        
        # Calculate token usage statistics
        token_usages = [task.total_token_usage for task in completed_tasks if task.total_token_usage is not None]
        avg_token_usage = sum(token_usages) / len(token_usages) if token_usages else 0
        total_token_usage = sum(token_usages) if token_usages else 0
        
        # Calculate validation metrics statistics
        f1_scores = [task.mcode_mapping.f1_score for task in completed_tasks if task.mcode_mapping and task.mcode_mapping.f1_score is not None]
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        
        precision_scores = [task.mcode_mapping.precision for task in completed_tasks if task.mcode_mapping and task.mcode_mapping.precision is not None]
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        
        recall_scores = [task.mcode_mapping.recall for task in completed_tasks if task.mcode_mapping and task.mcode_mapping.recall is not None]
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        
        compliance_scores = [task.mcode_mapping.compliance_score for task in completed_tasks if task.mcode_mapping and task.mcode_mapping.compliance_score is not None]
        avg_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
        
        return {
            'total_tasks': len(self.tasks),
            'completed_tasks': len(completed_tasks),
            'success_rate': len(completed_tasks) / len(self.tasks) if self.tasks else 0,
            'avg_processing_time_ms': avg_processing_time,
            'min_processing_time_ms': min_processing_time,
            'max_processing_time_ms': max_processing_time,
            'avg_token_usage': avg_token_usage,
            'total_token_usage': total_token_usage,
            'avg_f1_score': avg_f1_score,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_compliance_score': avg_compliance
        }
    except Exception as e:
        logger.error(f"Failed to calculate summary statistics: {e}")
        return {}

def _display_summary_statistics(self) -> None:
    """Display summary statistics in the UI"""
    try:
        stats = self._calculate_summary_statistics()
        
        if not stats:
            return
        
        # Create or update a summary card in the UI
        # This would be implemented in the UI setup method
        logger.info("Summary Statistics:")
        logger.info(f"  Total Tasks: {stats['total_tasks']}")
        logger.info(f"  Completed Tasks: {stats['completed_tasks']}")
        logger.info(f"  Success Rate: {stats['success_rate']:.2%}")
        logger.info(f"  Avg Processing Time: {stats['avg_processing_time_ms']:.2f}ms")
        logger.info(f"  Avg Token Usage: {stats['avg_token_usage']:.0f}")
        logger.info(f"  Avg F1 Score: {stats['avg_f1_score']:.3f}")
        logger.info(f"  Avg Precision: {stats['avg_precision']:.3f}")
        logger.info(f"  Avg Recall: {stats['avg_recall']:.3f}")
        logger.info(f"  Avg Compliance: {stats['avg_compliance_score']:.2%}")
        
    except Exception as e:
        logger.error(f"Failed to display summary statistics: {e}")
```

### 5. Add Benchmarking Display to UI
Update the UI to display benchmarking metrics:

```python
def _create_benchmarking_summary_card(self) -> None:
    """Create a card to display benchmarking summary statistics"""
    with ui.card().classes('w-full mt-4'):
        ui.label('Benchmarking Summary').classes('text-lg font-semibold mb-2')
        
        # This would be updated dynamically as tasks complete
        self.benchmarking_summary = ui.markdown().classes('w-full')

def _update_benchmarking_summary(self) -> None:
    """Update the benchmarking summary display"""
    try:
        stats = self._calculate_summary_statistics()
        
        if not stats:
            self.benchmarking_summary.set_content("No benchmarking data available yet.")
            return
        
        content = f"""
### Benchmarking Summary

- **Total Tasks**: {stats['total_tasks']}
- **Completed Tasks**: {stats['completed_tasks']}
- **Success Rate**: {stats['success_rate']:.2%}

#### Performance Metrics
- **Avg Processing Time**: {stats['avg_processing_time_ms']:.2f}ms
- **Min Processing Time**: {stats['min_processing_time_ms']:.2f}ms
- **Max Processing Time**: {stats['max_processing_time_ms']:.2f}ms
- **Total Token Usage**: {stats['total_token_usage']:,}
- **Avg Token Usage**: {stats['avg_token_usage']:.0f}

#### Validation Metrics
- **Avg F1 Score**: {stats['avg_f1_score']:.3f}
- **Avg Precision**: {stats['avg_precision']:.3f}
- **Avg Recall**: {stats['avg_recall']:.3f}
- **Avg Compliance**: {stats['avg_compliance_score']:.2%}
        """
        
        self.benchmarking_summary.set_content(content)
        
    except Exception as e:
        logger.error(f"Failed to update benchmarking summary: {e}")
        self.benchmarking_summary.set_content("Error updating benchmarking summary.")
```

## Testing Plan
1. Verify benchmarking metrics are correctly collected during pipeline execution
2. Test summary statistics calculation with various task completion scenarios
3. Confirm UI displays benchmarking metrics properly
4. Validate error handling for missing or malformed benchmarking data
5. Test performance impact of metrics collection

## Dependencies
- Gold standard loading functionality
- Validation logic implementation
- Existing pipeline processing methods
- UI components for displaying metrics

## Rollout Strategy
1. Extend data classes with benchmarking fields
2. Implement benchmarking metrics collection methods
3. Update pipeline processing to include metrics collection
4. Add summary statistics calculation
5. Update UI to display benchmarking results
6. Test with sample data
7. Document new features