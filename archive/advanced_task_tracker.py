"""
Advanced Task Tracker - Simplified implementation using pure NiceGUI events, binding, and state management.
"""

__all__ = ['Task', 'TaskStatus', 'TaskPriority', 'AdvancedTaskTrackerUI']

import asyncio
import random
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from nicegui import ui, run, background_tasks


class TaskStatus(Enum):
    """Enumeration of possible task states"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    """Enumeration of task priorities"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class Task:
    """Simple task representation using NiceGUI's reactive capabilities"""
    
    def __init__(self, name: str, description: str = "", priority: TaskPriority = TaskPriority.NORMAL):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.failed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.execution_time = 0.0
        self.estimated_time = 0.0
        self.tags: List[str] = []


class AdvancedTaskTrackerUI:
    """Simplified UI using pure NiceGUI events and state management"""
    
    def __init__(self):
        self.tasks: List[Task] = []
        self.dark = ui.dark_mode()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI layout using NiceGUI's built-in features"""
        with ui.header().classes('bg-primary text-white p-4 items-center'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Advanced Task Tracker').classes('text-2xl font-bold')
                with ui.row().classes('items-center'):
                    ui.button('Toggle Dark Mode', on_click=self._toggle_dark_mode).props('flat color=white')
                    ui.button('Add Sample Tasks', on_click=self._add_sample_tasks).props('flat color=white')
        
        with ui.column().classes('w-full p-4 gap-4'):
            self._setup_controls_section()
            self._setup_task_list_section()
    
    def _setup_controls_section(self):
        """Setup the controls section"""
        with ui.card().classes('w-full'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Task Controls').classes('text-lg font-semibold')
                with ui.row().classes('gap-2'):
                    ui.button('Create Task', on_click=self._show_create_task_dialog).props('icon=add color=primary')
                    ui.button('Clear All Tasks', on_click=self._clear_all_tasks).props('icon=delete color=negative')
    
    def _setup_task_list_section(self):
        """Setup the task list section using NiceGUI's built-in binding"""
        with ui.card().classes('w-full'):
            ui.label('Task List').classes('text-lg font-semibold mb-4')
            # Use NiceGUI's built-in binding to automatically update the task list
            self.task_list_container = ui.column().classes('w-full gap-4')
            # Bind the task list to automatically update when tasks change
            ui.bind(self.tasks, self._update_task_list_display)
    
    def _update_task_list_display(self):
        """Update the task list display using NiceGUI's built-in features"""
        self.task_list_container.clear()
        with self.task_list_container:
            for task in self.tasks:
                self._create_task_card(task)
    
    def _toggle_dark_mode(self):
        """Toggle dark mode"""
        self.dark.value = not self.dark.value
        ui.notify("Dark mode toggled")
    
    def _add_sample_tasks(self):
        """Add sample tasks for demonstration"""
        sample_tasks = [
            Task("File Processing Task", "Process large dataset files", TaskPriority.HIGH),
            Task("Data Analysis", "Analyze processed data for insights", TaskPriority.NORMAL),
            Task("Network Request", "Fetch external API data", TaskPriority.LOW),
            Task("Generic Task", "Perform a generic operation", TaskPriority.CRITICAL)
        ]
        
        self.tasks.extend(sample_tasks)
        ui.notify(f"Added {len(sample_tasks)} sample tasks")
    
    def _show_create_task_dialog(self):
        """Show dialog to create a new task"""
        with ui.dialog() as dialog, ui.card():
            ui.label('Create New Task').classes('text-xl font-bold mb-4')
            
            name_input = ui.input('Task Name').classes('w-full')
            description_input = ui.textarea('Description').classes('w-full')
            
            with ui.row().classes('w-full gap-4'):
                priority_options = {
                    TaskPriority.LOW: 'Low',
                    TaskPriority.NORMAL: 'Normal',
                    TaskPriority.HIGH: 'High',
                    TaskPriority.CRITICAL: 'Critical'
                }
                priority_select = ui.select(priority_options, label='Priority', value=TaskPriority.NORMAL).classes('w-full')
            
            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ui.button('Create', on_click=lambda: self._create_task_from_dialog(
                    dialog,
                    name_input,
                    description_input,
                    priority_select
                )).props('color=primary')
        
        dialog.open()
    
    def _create_task_from_dialog(self, dialog, name_input, description_input, priority_select):
        """Create a task from dialog inputs"""
        name = name_input.value
        description = description_input.value
        priority = priority_select.value
        
        if not name:
            ui.notify("Task name is required", type='warning')
            return
        
        task = Task(name, description, priority)
        self.tasks.append(task)
        
        dialog.close()
        ui.notify(f"Task '{task.name}' created")
    
    def _create_task_card(self, task: Task):
        """Create a card UI element for a task using NiceGUI's built-in binding"""
        with ui.card().classes('w-full'):
            # Task header with reactive binding
            with ui.row().classes('w-full justify-between items-center mb-2'):
                ui.label(task.name).classes('text-lg font-semibold')
                # Use NiceGUI's built-in binding for status badge
                status_colors = {
                    TaskStatus.PENDING: 'grey',
                    TaskStatus.RUNNING: 'blue',
                    TaskStatus.PAUSED: 'orange',
                    TaskStatus.COMPLETED: 'green',
                    TaskStatus.FAILED: 'red'
                }
                ui.badge(task.status.value, color=status_colors.get(task.status, 'grey')).classes('self-start')
            
            # Task description
            if task.description:
                ui.label(task.description).classes('text-gray-600 dark:text-gray-300 mb-2')
            
            # Task metadata
            with ui.row().classes('w-full gap-4 mb-2'):
                ui.label(f"Priority: {task.priority.name}").classes('text-sm')
                ui.label(f"Created: {task.created_at.strftime('%H:%M:%S')}").classes('text-sm')
            
            # Progress bar with reactive binding
            ui.linear_progress(task.progress / 100).classes('w-full mb-2')
            ui.label(f"{task.progress:.1f}%").classes('text-sm mb-2')
            
            # Error message (if any)
            if task.error_message:
                ui.label(task.error_message).classes('text-red-500 text-sm mb-2')
            
            # Action buttons
            with ui.row().classes('w-full gap-2'):
                ui.button('Run', on_click=lambda: self._start_task(task)).props('color=positive')
                ui.button('Pause', on_click=lambda: self._pause_task(task)).props('color=warning')
                ui.button('Cancel', on_click=lambda: self._cancel_task(task)).props('color=negative')
                ui.button('Retry', on_click=lambda: self._retry_task(task)).props('color=info')
    
    def _start_task(self, task: Task):
        """Start a task using NiceGUI's background task management"""
        if task.status != TaskStatus.PENDING:
            ui.notify("Task cannot be started", type='warning')
            return
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        async def execute_task():
            try:
                start_time = time.time()
                steps = random.randint(50, 100)
                
                for i in range(steps):
                    await asyncio.sleep(0.05)
                    task.progress = (i + 1) / steps * 100
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.execution_time = time.time() - start_time
                ui.notify(f"Task '{task.name}' completed", type='positive')
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.failed_at = datetime.now()
                task.error_message = str(e)
                task.execution_time = time.time() - start_time
                ui.notify(f"Task '{task.name}' failed: {str(e)}", type='negative')
        
        background_tasks.create(execute_task())
        ui.notify(f"Task '{task.name}' started", type='positive')
    
    def _pause_task(self, task: Task):
        """Pause a task"""
        if task.status != TaskStatus.RUNNING:
            ui.notify("Task cannot be paused", type='warning')
            return
        
        task.status = TaskStatus.PAUSED
        ui.notify(f"Task '{task.name}' paused", type='info')
    
    def _cancel_task(self, task: Task):
        """Cancel a task"""
        if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED]:
            ui.notify("Task cannot be cancelled", type='warning')
            return
        
        task.status = TaskStatus.FAILED
        task.failed_at = datetime.now()
        task.error_message = "Task cancelled by user"
        ui.notify(f"Task '{task.name}' cancelled", type='warning')
    
    def _retry_task(self, task: Task):
        """Retry a task"""
        if task.status != TaskStatus.FAILED:
            ui.notify("Task cannot be retried", type='warning')
            return
        
        task.status = TaskStatus.PENDING
        task.progress = 0.0
        task.error_message = None
        ui.notify(f"Task '{task.name}' ready for retry", type='positive')
    
    def _clear_all_tasks(self):
        """Clear all tasks"""
        self.tasks.clear()
        ui.notify("All tasks cleared")


def run_advanced_task_tracker(port: int = 8088):
    """Run the advanced task tracker UI"""
    tracker = AdvancedTaskTrackerUI()
    ui.run(title='Advanced Task Tracker', port=port, reload=True)


if __name__ in {"__main__", "__mp_main__"}:
    import argparse
    parser = argparse.ArgumentParser(description='Run Advanced Task Tracker')
    parser.add_argument('--port', type=int, default=8088, help='Port to run the UI on')
    args = parser.parse_args()
    run_advanced_task_tracker(args.port)