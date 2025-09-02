# Pipeline Task Tracker - Design Plan

## Objective
Create a new NiceGUI UI application (`pipeline_task_tracker.py`) that focuses on tracking individual pipeline tasks. Each pipeline task consists of two sequential LLM calls:
1. NLP Entity Extraction (via `StrictNlpExtractor`)
2. mCODE Mapping (via `StrictMcodeMapper`)

The UI should display a hierarchical view of these tasks and their sub-tasks with real-time status updates.

## Key Features
1. **Hierarchical Task Display**:
   - Main view lists pipeline tasks.
   - Each pipeline task can be expanded to show its two component LLM calls.
   - Visual indicators for task status (pending, running, success, failed).

2. **Real-time Updates**:
   - UI updates automatically as tasks progress through their stages.
   - Uses NiceGUI's reactive state management.

3. **Task Execution**:
   - Button to trigger a new pipeline task using sample clinical trial data.
   - Queue-based execution for managing tasks (though likely simpler than the benchmark tracker).

4. **Detailed Information**:
   - Show start/end times, duration, and any errors for each task/sub-task.
   - Display token usage if available.

## Architecture Overview

### Data Structures
```python
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import time

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

@dataclass
class LLMCallTask:
    name: str  # "NLP Extraction" or "mCODE Mapping"
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    token_usage: Optional[dict] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

@dataclass
class PipelineTask:
    id: str
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    nlp_extraction: LLMCallTask
    mcode_mapping: LLMCallTask
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.nlp_extraction:
            self.nlp_extraction = LLMCallTask(name="NLP Extraction")
        if not self.mcode_mapping:
            self.mcode_mapping = LLMCallTask(name="mCODE Mapping")

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
```

### UI Components
1. **Main Layout**:
   - Header with title and dark mode toggle.
   - Control panel with "Run Pipeline Task" button.
   - Task list display area.

2. **Task List Item**:
   - Collapsible card for each pipeline task.
   - Summary view showing task ID, overall status, and duration.
   - Expanded view showing details of `nlp_extraction` and `mcode_mapping` sub-tasks.

3. **Task Execution Logic**:
   - A simple queue (e.g., Python list or `asyncio.Queue`) to hold tasks.
   - A single background worker (similar to benchmark tracker) to process tasks sequentially.
   - The worker will:
     - Pick a task from the queue.
     - Update task status to RUNNING.
     - Create a `StrictDynamicExtractionPipeline` instance.
     - Call `process_clinical_trial`, but with hooks or wrappers to update sub-task statuses.
     - Update task status to SUCCESS or FAILED based on result.
     - Handle any exceptions and update error messages.

### Implementation Approach
1. **File Structure**:
   - `src/optimization/pipeline_task_tracker.py`: Main application file.

2. **NiceGUI Features**:
   - Use `ui.card`, `ui.expansion`, `ui.label`, `ui.button`, `ui.icon` for UI elements.
   - Use `ui.timer` for periodic UI refreshes if needed, or rely on reactive updates.
   - Use `background_tasks.create` for the worker task.
   - Use `ui.notify` for user feedback.

3. **Integration with Pipeline**:
   - Import `StrictDynamicExtractionPipeline` from `src.pipeline.strict_dynamic_extraction_pipeline`.
   - Use a fixed sample clinical trial data file (e.g., from `examples/breast_cancer_data`).
   - When executing a task, instantiate the pipeline and call `process_clinical_trial`.

4. **Status Tracking**:
   - Modify the execution flow to track when each LLM call starts and ends.
   - This might require slight modifications to how the pipeline is called or adding logging points.
   - Alternatively, we can infer the stages by wrapping the specific method calls within the worker.

## Next Steps
1. Create the `pipeline_task_tracker.py` file (this will require switching to Code mode).
2. Implement the data structures.
3. Implement the NiceGUI UI layout.
4. Implement the task execution logic with the background worker.
5. Test with sample data.