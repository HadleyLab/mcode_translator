"""
Pipeline Task Tracker - A NiceGUI UI for tracking individual pipeline tasks.
Each pipeline task consists of two LLM calls: NLP extraction and mCODE mapping.
"""

import sys
import os
import asyncio
import uuid
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

# Add the parent directory to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nicegui import ui, background_tasks, run
from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from src.utils import get_logger

# Configure logging
logger = get_logger(__name__)

class TaskStatus(Enum):
    """Enumeration for task statuses"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"

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

class PipelineTaskTrackerUI:
    """Main UI class for the pipeline task tracker"""
    
    def __init__(self):
        """Initialize the pipeline task tracker UI"""
        # Task management
        self.tasks: List[PipelineTask] = []
        self.pending_tasks: List[PipelineTask] = []
        self.is_worker_running = False
        self.notifications: List[Dict[str, Any]] = []  # Store notifications to be displayed
        
        # Sample data
        self.sample_trial_data = self._load_sample_data()
        
        # UI components
        self.task_list_container = None
        self.run_task_button = None
        self.status_label = None
        self.notification_container = None  # Container for notifications
        
        # Setup UI
        self._setup_ui()
        
        # Start background worker after UI is initialized
        ui.timer(0.1, self._start_worker, once=True)
        # Timer to process notifications
        ui.timer(0.5, self._process_notifications)
    
    def _load_sample_data(self) -> Optional[Dict[str, Any]]:
        """Load sample clinical trial data for testing"""
        try:
            trial_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.trial.json")
            if trial_file.exists():
                with open(trial_file, 'r') as f:
                    data = json.load(f)
                    logger.info("Successfully loaded sample trial data")
                    return data
            else:
                logger.warning("Sample trial data file not found")
                return None
        except Exception as e:
            logger.error(f"Failed to load sample trial data: {e}")
            return None
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        with ui.header().classes('bg-primary text-white p-4 items-center'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Pipeline Task Tracker').classes('text-2xl font-bold')
                ui.button('Toggle Dark Mode', on_click=lambda: ui.dark_mode().toggle()).props('flat color=white')
        
        with ui.column().classes('w-full p-4 gap-4'):
            self._setup_control_panel()
            self._setup_task_list()
    
    def _setup_control_panel(self):
        """Setup the control panel for running tasks"""
        with ui.card().classes('w-full'):
            ui.label('Control Panel').classes('text-lg font-semibold mb-4')
            
            with ui.row().classes('w-full gap-2'):
                self.run_task_button = ui.button(
                    'Run Pipeline Task', 
                    on_click=self._run_pipeline_task
                ).props('icon=play_arrow color=positive')
                
                self.status_label = ui.label('Ready').classes('self-center ml-4')
    
    def _setup_task_list(self):
        """Setup the task list display area"""
        with ui.card().classes('w-full mt-4'):
            ui.label('Pipeline Tasks').classes('text-lg font-semibold mb-2')
            self.task_list_container = ui.column().classes('w-full gap-2')
            # Initial update to show empty state
            self._update_task_list()
    
    def _run_pipeline_task(self):
        """Create and queue a new pipeline task"""
        if not self.sample_trial_data:
            ui.notify("No sample data available", type='warning')
            return
            
        # Extract the correct trial data structure
        trial_data = self.sample_trial_data
        if trial_data and "test_cases" in trial_data:
            test_cases = trial_data["test_cases"]
            if test_cases:
                # Get the first test case data
                first_test_case_key = list(test_cases.keys())[0]
                trial_data = test_cases[first_test_case_key]
        
        # Create a new pipeline task
        task_id = str(uuid.uuid4())[:8]
        task = PipelineTask(
            id=task_id,
            trial_data=trial_data
        )
        
        # Add to pending tasks
        self.pending_tasks.append(task)
        self.tasks.append(task)
        
        # Update UI
        self._update_task_list()
        self._add_notification(f"Pipeline task {task_id} queued", 'positive')
        logger.info(f"Pipeline task {task_id} queued")
    
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
    
    def _create_task_card(self, task: PipelineTask):
        """Create a card for a pipeline task"""
        # Get test case information
        test_case_name = "unknown"
        if task.trial_data and "test_cases" in task.trial_data:
            test_cases = task.trial_data["test_cases"]
            if test_cases:
                test_case_name = list(test_cases.keys())[0]  # Get first test case name
        
        with ui.card().classes('w-full'):
            # Main task header
            with ui.row().classes('w-full justify-between items-center'):
                ui.label(f'Task {task.id}').classes('text-lg font-semibold')
                ui.label(f'Test Case: {test_case_name}').classes('text-sm text-gray-500')
                
                # Status indicator
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
            
            # Expandable details
            with ui.expansion('Details', icon='info').classes('w-full'):
                # NLP Extraction task
                self._create_subtask_row(task.nlp_extraction)
                
                # mCODE Mapping task
                self._create_subtask_row(task.mcode_mapping)
                
                # Error message if any
                if task.error_message:
                    ui.label(f"Error: {task.error_message}").classes('text-red-600 mt-2')
    
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
            
            # Additional details
            if subtask.details or subtask.error_message or subtask.token_usage:
                with ui.column().classes('w-full text-sm mt-1'):
                    if subtask.details:
                        ui.label(subtask.details).classes('text-gray-600 dark:text-gray-400')
                    
                    if subtask.error_message:
                        ui.label(f"Error: {subtask.error_message}").classes('text-red-600')
                    
                    if subtask.token_usage:
                        token_text = f"Tokens: {subtask.token_usage.get('total_tokens', 'N/A')} "
                        token_text += f"(Prompt: {subtask.token_usage.get('prompt_tokens', 'N/A')}, "
                        token_text += f"Completion: {subtask.token_usage.get('completion_tokens', 'N/A')})"
                        ui.label(token_text).classes('text-gray-600 dark:text-gray-400')
    
    def _process_notifications(self):
        """Process and display notifications"""
        while self.notifications:
            notification = self.notifications.pop(0)
            ui.notify(notification['message'], type=notification['type'])
    
    def _add_notification(self, message: str, type: str = 'info'):
        """Add a notification to be displayed"""
        self.notifications.append({'message': message, 'type': type})
    
    def _start_worker(self):
        """Start the background worker task"""
        if not self.is_worker_running:
            self.is_worker_running = True
            background_tasks.create(self._worker())
            logger.info("Pipeline task worker started")
    
    async def _worker(self):
        """Background worker that processes pipeline tasks"""
        logger.info("Pipeline task worker running")
        
        while self.is_worker_running:
            # Check for pending tasks
            if self.pending_tasks:
                task = self.pending_tasks.pop(0)
                await self._process_pipeline_task(task)
                self._update_task_list()
            else:
                # Wait a bit before checking again
                await asyncio.sleep(0.5)
    
    async def _process_pipeline_task(self, task: PipelineTask):
        """Process a single pipeline task"""
        logger.info(f"Processing pipeline task {task.id}")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.start_time = asyncio.get_event_loop().time()
        self.status_label.set_text(f"Running task {task.id}")
        
        try:
            # Create pipeline instance
            pipeline = StrictDynamicExtractionPipeline()
            
            # Process the clinical trial
            # We'll wrap the calls to track individual LLM call progress
            result = await self._run_pipeline_with_tracking(pipeline, task)
            
            # Update task with result
            task.status = TaskStatus.SUCCESS
            task.end_time = asyncio.get_event_loop().time()
            self.status_label.set_text(f"Task {task.id} completed successfully")
            self._add_notification(f"Task {task.id} completed", 'positive')
            logger.info(f"Pipeline task {task.id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = asyncio.get_event_loop().time()
            self.status_label.set_text(f"Task {task.id} failed")
            self._add_notification(f"Task {task.id} failed: {str(e)}", 'negative')
            logger.error(f"Pipeline task {task.id} failed: {e}")
    
    async def _run_pipeline_with_tracking(self, pipeline: StrictDynamicExtractionPipeline, task: PipelineTask):
        """Run pipeline with tracking of individual LLM calls"""
        # Get model and prompt information
        model_name = pipeline.nlp_engine.model_name
        temperature = pipeline.nlp_engine.temperature
        max_tokens = pipeline.nlp_engine.max_tokens
        
        # Get more detailed prompt information
        extraction_prompt = pipeline.nlp_engine.get_prompt_name()
        mapping_prompt = pipeline.llm_mapper.get_prompt_name()
        
        # Update NLP extraction task with detailed information
        task.nlp_extraction.status = TaskStatus.RUNNING
        task.nlp_extraction.start_time = asyncio.get_event_loop().time()
        task.nlp_extraction.details = f"Extracting entities using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={extraction_prompt})..."
        self._update_task_list()
        
        try:
            # Run the pipeline process
            result = await run.io_bound(
                pipeline.process_clinical_trial,
                task.trial_data
            )
            
            # Update NLP extraction completion
            task.nlp_extraction.status = TaskStatus.SUCCESS
            task.nlp_extraction.end_time = asyncio.get_event_loop().time()
            task.nlp_extraction.details = f"Extracted {len(result.extracted_entities)} entities using {model_name} with prompt '{extraction_prompt}'"
            
            # Get token usage if available
            if result.metadata and 'token_usage' in result.metadata:
                task.nlp_extraction.token_usage = result.metadata['token_usage']
            
            # Update mCODE mapping task with detailed information
            task.mcode_mapping.status = TaskStatus.RUNNING
            task.mcode_mapping.start_time = asyncio.get_event_loop().time()
            task.mcode_mapping.details = f"Mapping entities to mCODE using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={mapping_prompt})..."
            self._update_task_list()
            
            # Update mCODE mapping completion (this is part of the same pipeline call)
            task.mcode_mapping.status = TaskStatus.SUCCESS
            task.mcode_mapping.end_time = asyncio.get_event_loop().time()
            task.mcode_mapping.details = f"Mapped {len(result.mcode_mappings)} mCODE elements using {model_name} with prompt '{mapping_prompt}'"
            
            # Get token usage for mapping if available
            if result.metadata and 'token_usage' in result.metadata:
                task.mcode_mapping.token_usage = result.metadata['token_usage']
            
            return result
            
        except Exception as e:
            # Determine which step failed based on what was running
            if task.nlp_extraction.status == TaskStatus.RUNNING:
                task.nlp_extraction.status = TaskStatus.FAILED
                task.nlp_extraction.error_message = str(e)
                task.nlp_extraction.end_time = asyncio.get_event_loop().time()
            elif task.mcode_mapping.status == TaskStatus.RUNNING:
                task.mcode_mapping.status = TaskStatus.FAILED
                task.mcode_mapping.error_message = str(e)
                task.mcode_mapping.end_time = asyncio.get_event_loop().time()
            
            raise e

def run_pipeline_task_tracker(port: int = 8090):
    """Run the pipeline task tracker UI"""
    PipelineTaskTrackerUI()
    ui.run(title='Pipeline Task Tracker', port=port, reload=True)

if __name__ in {"__main__", "__mp_main__"}:
    import argparse
    parser = argparse.ArgumentParser(description='Run Pipeline Task Tracker')
    parser.add_argument('--port', type=int, default=8090, help='Port to run the UI on')
    args = parser.parse_args()
    run_pipeline_task_tracker(args.port)