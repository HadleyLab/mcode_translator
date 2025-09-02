# UI Benchmarking Metrics Display Implementation Plan

## Overview
This document outlines the implementation plan for updating the UI to display benchmarking metrics in the pipeline task tracker. This will enable users to see performance metrics directly in the task cards and details.

## Implementation Steps

### 1. Update Task Card Display
Modify the `_create_task_card` method to show benchmarking metrics:

```python
def _create_task_card(self, task: PipelineTask):
    """Create a card for a pipeline task"""
    # Existing code for task card setup...
    
    with ui.card().classes('w-full'):
        # Main task header
        with ui.row().classes('w-full justify-between items-center flex-wrap'):
            ui.label(f'{task.pipeline_type} - Task {task.id}').classes('text-lg font-semibold')
            ui.label(f'Test Case: {test_case_name}').classes('text-sm text-gray-500')
            
            # Status indicator with benchmarking info
            status_color = {
                TaskStatus.PENDING: 'blue',
                TaskStatus.RUNNING: 'orange',
                TaskStatus.SUCCESS: 'green',
                TaskStatus.FAILED: 'red'
            }.get(task.status, 'gray')
            
            status_text = task.status.value.capitalize()
            if task.duration:
                status_text += f" ({task.duration:.2f}s)"
                
            ui.label(status_text).classes(f'text-{status_color}-600 font-medium')
        
        # Benchmarking metrics indicator
        if task.status == TaskStatus.SUCCESS and task.total_processing_time_ms is not None:
            with ui.row().classes('w-full text-sm mt-1'):
                ui.label(f"Time: {task.total_processing_time_ms:.2f}ms").classes('text-gray-600 dark:text-gray-400 mr-4')
                if task.total_token_usage is not None:
                    ui.label(f"Tokens: {task.total_token_usage:,}").classes('text-gray-600 dark:text-gray-400')
        
        # Validation status indicator
        if task.status == TaskStatus.SUCCESS:
            # Show overall validation status
            if task.mcode_mapping and task.mcode_mapping.f1_score is not None:
                validation_color = 'green' if task.mcode_mapping.f1_score > 0.8 else 'yellow' if task.mcode_mapping.f1_score > 0.6 else 'red'
                validation_text = f"Validation: {task.mcode_mapping.f1_score:.2f} F1"
                ui.label(validation_text).classes(f'text-{validation_color}-600 font-medium')
        
        # Additional trial information
        if nct_id != "N/A" or brief_title != "N/A":
            with ui.row().classes('w-full text-sm text-gray-600 dark:text-gray-400'):
                if nct_id != "N/A":
                    ui.label(f'NCT ID: {nct_id}').classes('mr-4')
                if brief_title != "N/A":
                    ui.label(f'Title: {brief_title}').classes('truncate')
        
        # Expandable details
        with ui.expansion('Details', icon='info').classes('w-full'):
            # Pipeline information
            with ui.row().classes('w-full text-sm text-gray-600 dark:text-gray-400 mb-2'):
                if task.pipeline_type == 'Direct to mCODE' and hasattr(task, 'prompt_info'):
                    ui.label(f'Prompt: {task.prompt_info.get("direct_prompt", "N/A")}')
                elif hasattr(task, 'prompt_info'):
                    ui.label(f'Extraction Prompt: {task.prompt_info.get("extraction_prompt", "N/A")}')
                    ui.label(f'Mapping Prompt: {task.prompt_info.get("mapping_prompt", "N/A")}')
            
            # Sub-tasks with validation and benchmarking metrics
            if task.pipeline_type == 'Direct to mCODE':
                self._create_subtask_row(task.mcode_mapping)
            else:
                self._create_subtask_row(task.nlp_extraction)
                self._create_subtask_row(task.mcode_mapping)
            
            # Error message if any
            if task.error_message:
                ui.label(f"Error: {task.error_message}").classes('text-red-600 mt-2')
```

### 2. Update Subtask Row Display
Modify the `_create_subtask_row` method to display benchmarking metrics:

```python
def _create_subtask_row(self, subtask: LLMCallTask):
    """Create a row for a sub-task"""
    with ui.card().classes('w-full bg-gray-50 dark:bg-gray-80 mt-2'):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label(subtask.name).classes('font-medium')
            
            # Status indicator
            status_color = {
                TaskStatus.PENDING: 'blue',
                TaskStatus.RUNNING: 'orange',
                TaskStatus.SUCCESS: 'green',
                TaskStatus.FAILED: 'red'
            }.get(subtask.status, 'gray')
            
            status_text = subtask.status.value.capitalize()
            if subtask.duration:
                status_text += f" ({subtask.duration:.2f}s)"
                
            ui.label(status_text).classes(f'text-{status_color}-600')
        
        # Validation metrics
        if subtask.precision is not None and subtask.recall is not None and subtask.f1_score is not None:
            with ui.row().classes('w-full text-sm mt-1'):
                ui.label(f"Precision: {subtask.precision:.3f}").classes('text-gray-600 dark:text-gray-400 mr-4')
                ui.label(f"Recall: {subtask.recall:.3f}").classes('text-gray-600 dark:text-gray-400 mr-4')
                ui.label(f"F1-Score: {subtask.f1_score:.3f}").classes('text-gray-600 dark:text-gray-400')
        
        # Compliance score
        if subtask.compliance_score is not None:
            compliance_color = 'green' if subtask.compliance_score > 0.8 else 'yellow' if subtask.compliance_score > 0.6 else 'red'
            ui.label(f"Compliance: {subtask.compliance_score:.2%}").classes(f'text-{compliance_color}-600 text-sm mt-1')
        
        # Benchmarking metrics
        if subtask.processing_time_ms is not None:
            ui.label(f"Time: {subtask.processing_time_ms:.2f}ms").classes('text-gray-600 dark:text-gray-400 text-sm mt-1')
        
        # Token usage
        if subtask.token_usage:
            token_text = f"Tokens: {subtask.token_usage.get('total_tokens', 'N/A')} "
            token_text += f"(Prompt: {subtask.token_usage.get('prompt_tokens', 'N/A')}, "
            token_text += f"Completion: {subtask.token_usage.get('completion_tokens', 'N/A')})"
            ui.label(token_text).classes('text-gray-600 dark:text-gray-400 text-sm mt-1')
        
        # Additional details
        if subtask.details or subtask.error_message:
            with ui.column().classes('w-full text-sm mt-1'):
                if subtask.details:
                    ui.label(subtask.details).classes('text-gray-600 dark:text-gray-400')
                
                if subtask.error_message:
                    ui.label(f"Error: {subtask.error_message}").classes('text-red-600')
```

### 3. Add Benchmarking Summary Display
Add a method to display benchmarking summary statistics:

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
        stats = self._calculate_benchmarking_statistics()
        
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

def _calculate_benchmarking_statistics(self) -> Dict[str, Any]:
    """Calculate benchmarking statistics for all completed tasks"""
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
        logger.error(f"Failed to calculate benchmarking statistics: {e}")
        return {}
```

### 4. Update Task List Display
Modify the `_update_task_list` method to update benchmarking summaries:

```python
def _update_task_list(self):
    """Update the task list display"""
    if not self.task_list_container:
        return
        
    self.task_list_container.clear()
    
    with self.task_list_container:
        if not self.tasks:
            ui.label('No pipeline tasks yet. Click "Run Pipeline Task" to start.').classes('text-gray-500 italic')
        else:
            # Display tasks in reverse order (newest first)
            for task in reversed(self.tasks):
                self._create_task_card(task)
            
            # Update validation and benchmarking summaries
            self._update_validation_summary()
            self._update_benchmarking_summary()
```

### 5. Add Benchmarking Summary to UI Setup
Update the `_setup_task_list` method to include benchmarking summary:

```python
def _setup_task_list(self):
    """Setup the task list display area"""
    with ui.card().classes('w-full mt-4'):
        ui.label('Pipeline Tasks').classes('text-lg font-semibold mb-2')
        self.task_list_container = ui.column().classes('w-full gap-2')
        # Initial update to show empty state
        self._update_task_list()
    
    # Add validation summary card
    self._create_validation_summary_card()
    
    # Add benchmarking summary card
    self._create_benchmarking_summary_card()
```

### 6. Add Resource Usage Display (Optional)
If resource usage metrics are collected, add a method to display them:

```python
def _create_resource_usage_card(self) -> None:
    """Create a card to display resource usage statistics"""
    with ui.card().classes('w-full mt-4'):
        ui.label('Resource Usage').classes('text-lg font-semibold mb-2')
        
        # This would be updated dynamically
        self.resource_usage = ui.markdown().classes('w-full')

def _update_resource_usage(self) -> None:
    """Update the resource usage display"""
    try:
        resource_stats = self._collect_resource_usage_metrics()
        
        if not resource_stats:
            self.resource_usage.set_content("Resource usage data not available.")
            return
        
        content = f"""
### Resource Usage

- **Memory Usage**: {resource_stats['memory_usage_mb']:.2f} MB
- **CPU Usage**: {resource_stats['cpu_percent']:.2f}%
        """
        
        self.resource_usage.set_content(content)
        
    except Exception as e:
        logger.error(f"Failed to update resource usage: {e}")
        self.resource_usage.set_content("Error updating resource usage.")
```

## Testing Plan
1. Verify benchmarking metrics are displayed correctly in task cards
2. Test UI updates when tasks complete with benchmarking results
3. Confirm benchmarking summary statistics are calculated and displayed properly
4. Validate error handling for missing or malformed benchmarking data
5. Test responsive design for various screen sizes

## Dependencies
- Gold standard loading functionality
- Validation logic implementation
- Benchmarking metrics calculation
- Existing UI components

## Rollout Strategy
1. Update task card display to show benchmarking metrics
2. Update subtask row display to show benchmarking metrics
3. Add benchmarking summary display
4. Update task list display to include benchmarking summaries
5. Test with sample data
6. Document new features