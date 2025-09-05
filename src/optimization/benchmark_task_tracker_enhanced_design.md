# Enhanced TaskQueue Visualization Design

## Overview
This document outlines the design for enhancing the TaskQueue visualization in the benchmark_task_tracker.py file. The enhancements will include:
1. Real-time visualization of the TaskQueue with automatic updates based on filters
2. Full information display about pipelines, prompts, and models
3. Inline logging of each worker's consumption
4. Slick UI with tooltips, badges, and icons using NiceGUI components

## Current Implementation Analysis

The current `benchmark_task_tracker.py` already has:
- A UI with filters for prompts, models, trials, and pipelines
- A validation display showing task status
- Live logging capabilities
- Results display with benchmark metrics
- Progress tracking

## Enhanced Features

### 1. Real-time TaskQueue Visualization
We'll enhance the existing validation display to show a more detailed view of the TaskQueue with:
- Worker assignment information
- Task status with color-coded badges
- Progress indicators for running tasks
- Automatic updates when filters change

### 2. Worker Consumption Tracking
We'll add tracking for:
- Token usage per worker
- Execution time per task
- Queue depth and worker utilization
- Memory consumption (if available)

### 3. Pipeline and Prompt Information
Enhanced display of:
- Pipeline type (NLP to Mcode vs Direct to Mcode)
- Prompt names and versions
- Model information with provider details
- Trial data context

### 4. Slick UI with NiceGUI Components
Implementation of:
- Interactive tooltips with detailed information
- Status badges with color coding
- Progress bars with percentage indicators
- Expandable task cards with detailed information
- Filter chips for quick selection
- Sortable columns in results tables

## UI Component Design

### Task Queue Visualization Panel
```
+-------------------------------------------------------------+
| Task Queue Visualization                           [Refresh]|
+-------------------------------------------------------------+
| Workers: 5/5 Running    Queue Depth: 3    Utilization: 85%  |
|                                                             |
| [Worker 1] [Worker 2] [Worker 3] [Worker 4] [Worker 5]      |
|    |          |          |          |            |
| Task A    Task B    Task C    Task D    Task E             |
|  (75%)     (40%)     (90%)     (20%)     (60%)             |
+-------------------------------------------------------------+
```

### Enhanced Task Card
```
+-------------------------------------------------------------+
| Task: abc123 - NLP to Mcode Pipeline              [Status: 🔄]|
| Prompt: nlp_extraction_comprehensive              [F1: 0.87] |
| Model: gpt-4-turbo                                [Tokens: 1247]|
+-------------------------------------------------------------+
| Trial: breast_cancer_her2_positive                          |
|                                                             |
| [Worker 1]  🔄 Processing  (45.2s / 60s) [|||||||     ] 75% |
|                                                             |
| Live Consumption Log:                                       |
| 🔄 [12:45:23] Starting NLP extraction                        |
| ✅ [12:45:35] Extracted 15 entities                          |
| 🔄 [12:45:36] Starting Mcode mapping                         |
| ❌ [12:45:45] Mapping failed: API timeout                   |
+-------------------------------------------------------------+
```

### Worker Consumption Dashboard
```
+-------------------------------------------------------------+
| Worker Consumption Dashboard                                |
+-------------------------------------------------------------+
| Worker | Status  | Tasks | Tokens Used | Avg Time | Util%   |
|--------|---------|-------|-------------|----------|---------|
|   1    | Running |   3   |    4,230    |  25.4s   |   85%   |
|   2    | Running |   2   |    3,150    |  18.7s   |   62%   |
|   3    | Idle    |   0   |       0     |   0s     |    0%   |
+-------------------------------------------------------------+
```

## Implementation Plan

### Phase 1: Data Enhancement
1. Extend BenchmarkTask to include worker assignment information
2. Add token usage tracking per worker
3. Implement detailed logging with timestamps
4. Enhance task metadata with pipeline and prompt information

### Phase 2: UI Components
1. Create enhanced task queue visualization panel
2. Implement worker consumption dashboard
3. Design interactive task cards with expandable details
4. Add filter chips and quick selection controls

### Phase 3: Integration
1. Connect enhanced UI components to existing task queue
2. Implement real-time updates based on filters
3. Add tooltips, badges, and icons for better UX
4. Optimize performance for large task queues

## Technical Details

### Enhanced BenchmarkTask Class
```python
@dataclass
class BenchmarkTask:
    # Existing fields...
    
    # Worker consumption tracking
    worker_id: Optional[int] = None
    assigned_at: Optional[datetime] = None
    token_usage_per_worker: Dict[int, int] = field(default_factory=dict)
    execution_times_per_worker: Dict[int, float] = field(default_factory=dict)
    
    # Enhanced metadata
    pipeline_details: Dict[str, Any] = field(default_factory=dict)
    prompt_details: Dict[str, Any] = field(default_factory=dict)
    model_details: Dict[str, Any] = field(default_factory=dict)
```

### Worker Consumption Tracking
```python
class PipelineTaskQueue:
    # Existing fields...
    
    # Worker consumption tracking
    worker_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    def get_worker_consumption(self) -> Dict[int, Dict[str, Any]]:
        """Get consumption statistics for all workers"""
        return self.worker_stats
    
    def update_worker_stats(self, worker_id: int, task: BenchmarkTask) -> None:
        """Update statistics for a specific worker"""
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = {
                'tasks_processed': 0,
                'total_tokens': 0,
                'total_time': 0.0,
                'current_task': None
            }
        
        stats = self.worker_stats[worker_id]
        stats['tasks_processed'] += 1
        stats['total_tokens'] += task.token_usage.get('total_tokens', 0) if task.token_usage else 0
        stats['total_time'] += task.duration_ms / 1000 if task.duration_ms else 0
        stats['current_task'] = task.task_id if task.status == TaskStatus.PROCESSING else None
```

### UI Components with NiceGUI
```python
def _setup_enhanced_task_queue_display(self) -> None:
    """Setup the enhanced task queue visualization"""
    with ui.card().classes('w-full mt-4'):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label('Task Queue Visualization').classes('text-lg font-semibold')
            ui.button('Refresh', on_click=self._refresh_task_queue_display).props('icon=refresh')
        
        # Worker consumption dashboard
        self._create_worker_consumption_dashboard()
        
        # Task queue visualization
        self._create_task_queue_visualization()
        
        # Enhanced task cards
        self._update_enhanced_task_cards()

@ui.refreshable
def _update_enhanced_task_cards(self) -> None:
    """Update enhanced task cards with detailed information"""
    # Filter tasks based on current selections
    filtered_tasks = self._filter_tasks()
    
    # Create enhanced task cards with worker information
    for task in filtered_tasks:
        self._create_enhanced_task_card(task)
```

## Filter Integration

The enhanced visualization will automatically update when any of the existing filters change:
- Pipeline type selection
- Prompt selection
- Model selection
- Trial selection

Each filter change will trigger a refresh of the task queue visualization to show only relevant tasks.

## Performance Considerations

1. **Efficient Updates**: Use NiceGUI's refreshable pattern to minimize DOM updates
2. **Virtual Scrolling**: Implement virtual scrolling for large task lists
3. **Data Pagination**: Paginate task displays for better performance
4. **Caching**: Cache worker statistics and task metadata to reduce computation

## Error Handling

1. **Worker Failures**: Display worker failure status with error details
2. **Task Failures**: Show detailed error information in task cards
3. **Connection Issues**: Handle API connection failures gracefully
4. **Resource Limits**: Warn when approaching system resource limits

## Testing Strategy

1. **Unit Tests**: Test individual components (task cards, worker dashboard)
2. **Integration Tests**: Test filter integration and real-time updates
3. **Performance Tests**: Verify performance with large task queues
4. **UI Tests**: Test responsive design and accessibility

## Future Enhancements

1. **Historical Data**: Store and display historical worker performance
2. **Predictive Analytics**: Predict task completion times based on historical data
3. **Resource Optimization**: Suggest optimal worker count based on task patterns
4. **Export Functionality**: Export worker consumption reports