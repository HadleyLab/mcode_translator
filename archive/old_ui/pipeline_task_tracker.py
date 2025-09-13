"""
Pipeline Task Tracker - A NiceGUI UI for tracking individual pipeline tasks.
Each pipeline task consists of two LLM calls: NLP extraction and mCODE mapping.

This implementation uses a queue-based concurrency approach with multiple worker tasks
to process pipeline tasks concurrently. Tasks are added directly to a queue and
processed by worker tasks based on the selected concurrency level.

Features:
- Concurrency control with adjustable number of worker tasks (1-10)
- Support for different pipeline types (NLP extraction to mCODE mapping and Direct to mCODE)
- Dynamic prompt selection for each pipeline type
- Real-time task status tracking and display
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from nicegui import background_tasks, run, ui

from pipeline.pipeline_base import ProcessingPipeline
from src.pipeline.mcode_pipeline import McodePipeline
from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from src.shared.types import TaskStatus
from src.utils import get_logger
from src.utils.prompt_loader import PromptLoader

# Configure logging
logger = get_logger(__name__)


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

from src.pipeline.task_queue import BenchmarkTask


@dataclass
class PipelineTask(BenchmarkTask):
    """Represents a complete pipeline task with two LLM call sub-tasks and benchmarking capabilities"""
    nlp_extraction: LLMCallTask = None
    mcode_mapping: LLMCallTask = None
    pipeline_type: str = "NLP to mCODE"
    test_case_name: str = "unknown"
    prompt_info: Optional[Dict[str, str]] = None
    pipeline_result: Optional[Dict[str, Any]] = None
    gold_standard_data: Optional[Dict[str, Any]] = None
    benchmark_metrics: Optional[Dict[str, float]] = None



class PipelineTaskTrackerUI:
    """Main UI class for the pipeline task tracker.
    
    This class implements a queue-based concurrency approach with multiple worker tasks
    to process pipeline tasks concurrently. Tasks are added directly to a queue and
    processed by worker tasks based on the selected concurrency level.
    
    The UI provides:
    - Concurrency control with adjustable number of worker tasks (1-10)
    - Support for different pipeline types (NLP extraction to mCODE mapping and Direct to mCODE)
    - Dynamic prompt selection for each pipeline type
    - Real-time task status tracking and display
    """
    
    def __init__(self):
        """Initialize the pipeline task tracker UI"""
        # Task management
        self.tasks: List[PipelineTask] = []
        self.is_worker_running = False
        self.notifications: List[Dict[str, Any]] = []  # Store notifications to be displayed
        
        # Queue-based concurrency
        self.task_queue = asyncio.Queue()
        self.worker_tasks = []
        
        # Sample data
        self.sample_trial_data = self._load_sample_data()
        self.gold_standard_data = self._load_gold_standard_data()
        
        # Prompt loader
        self.prompt_loader = PromptLoader()
        self.available_prompts = self._load_available_prompts()
        
        # UI components
        self.task_list_container = None
        self.run_task_button = None
        self.status_label = None
        self.notification_container = None
        self.pipeline_selector = None
        self.nlp_prompt_selector = None
        self.mcode_prompt_selector = None
        self.direct_mcode_prompt_selector = None
        self.concurrency_selector = None
        self.max_workers = 5  # Default to 5 workers
        self.enable_validation = True  # Enable gold standard validation by default
        
        # Setup UI
        self._setup_ui()
        
        # Track dark mode state
        self.dark_mode = ui.dark_mode()
        self.is_dark_mode = False
        
        # Start background worker after UI is fully initialized
        # Use a timer to start workers after the event loop is available
        ui.timer(0.1, self._start_worker, once=True)
    
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
    
    def _load_gold_standard_data(self) -> Optional[Dict[str, Any]]:
        """Load gold standard data for validation"""
        try:
            gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
            if gold_file.exists():
                with open(gold_file, 'r') as f:
                    data = json.load(f)
                    logger.info("Successfully loaded gold standard data")
                    return data
            else:
                logger.warning("Gold standard data file not found")
                return None
        except Exception as e:
            logger.error(f"Failed to load gold standard data: {e}")
            return None

    def _load_available_prompts(self) -> Dict[str, Dict[str, Any]]:
        """Load all available prompts from the prompt loader"""
        try:
            prompts = self.prompt_loader.list_available_prompts()
            logger.info(f"Loaded {len(prompts)} available prompts")
            return prompts
        except Exception as e:
            logger.error(f"Failed to load available prompts: {e}")
            return {}

    def _validate_pipeline_result(self, task: PipelineTask) -> Dict[str, float]:
        """Validate pipeline result against gold standard data"""
        if not task.pipeline_result or not task.gold_standard_data:
            return {}
        
        try:
            # Extract mCODE mappings from pipeline result
            pipeline_mappings = task.pipeline_result.mcode_mappings if task.pipeline_result else []
            
            # Extract gold standard mappings - correct path based on gold standard structure
            # Gold standard structure: gold_standard.breast_cancer_her2_positive.expected_mcode_mappings.mapped_elements
            gold_mappings = []
            if task.gold_standard_data:
                # Try multiple possible paths to extract gold mappings
                if 'gold_standard' in task.gold_standard_data:
                    gold_standard = task.gold_standard_data['gold_standard']
                    # Handle nested structure with test case name
                    for test_case_name, test_case_data in gold_standard.items():
                        if 'expected_mcode_mappings' in test_case_data:
                            mappings_data = test_case_data['expected_mcode_mappings']
                            if 'mapped_elements' in mappings_data:
                                gold_mappings = mappings_data['mapped_elements']
                                break
                # Fallback: try direct access if nested structure not found
                if not gold_mappings and 'expected_mcode_mappings' in task.gold_standard_data:
                    mappings_data = task.gold_standard_data['expected_mcode_mappings']
                    if 'mapped_elements' in mappings_data:
                        gold_mappings = mappings_data['mapped_elements']
                # Final fallback: direct mcode_mappings key (old format)
                if not gold_mappings:
                    gold_mappings = task.gold_standard_data.get('mcode_mappings', [])
            
            # Calculate validation metrics
            metrics = self._calculate_validation_metrics(pipeline_mappings, gold_mappings)
            
            # Store metrics in task - initialize benchmark_metrics if None
            if task.benchmark_metrics is None:
                task.benchmark_metrics = {}
            task.benchmark_metrics.update(metrics)
            
            logger.info(f"Validation completed for task {task.id}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed for task {task.id}: {e}")
            return {}

    def _calculate_validation_metrics(self, pipeline_mappings: List[Dict], gold_mappings: List[Dict]) -> Dict[str, float]:
        """Calculate precision, recall, and F1-score for mCODE mappings"""
        if not pipeline_mappings or not gold_mappings:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Convert to sets of simplified representations for comparison
        pipeline_set = self._simplify_mappings(pipeline_mappings)
        gold_set = self._simplify_mappings(gold_mappings)
        
        # Calculate true positives, false positives, false negatives
        true_positives = len(pipeline_set.intersection(gold_set))
        false_positives = len(pipeline_set - gold_set)
        false_negatives = len(gold_set - pipeline_set)
        
        # Calculate precision, recall, F1-score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1_score, 3),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_pipeline': len(pipeline_mappings),
            'total_gold': len(gold_mappings)
        }

    def _simplify_mappings(self, mappings: List[Dict]) -> set:
        """Simplify mCODE mappings for comparison by focusing on key elements"""
        simplified = set()
        for mapping in mappings:
            # Create a simplified representation focusing on key elements
            key_parts = []
            
            # Include resource type if available
            if mapping.get('resourceType'):
                key_parts.append(f"resourceType:{mapping['resourceType']}")
            
            # Include code if available
            if mapping.get('code'):
                code = mapping['code']
                if isinstance(code, dict):
                    if code.get('coding'):
                        for coding in code['coding']:
                            if coding.get('code'):
                                key_parts.append(f"code:{coding['code']}")
                            if coding.get('system'):
                                key_parts.append(f"system:{coding['system']}")
                elif isinstance(code, str):
                    key_parts.append(f"code:{code}")
            
            # Include value if available
            if mapping.get('valueString'):
                key_parts.append(f"value:{mapping['valueString']}")
            elif mapping.get('valueCodeableConcept'):
                value_cc = mapping['valueCodeableConcept']
                if value_cc.get('coding'):
                    for coding in value_cc['coding']:
                        if coding.get('code'):
                            key_parts.append(f"value_code:{coding['code']}")
            
            # Sort and join to create a consistent representation
            key_parts.sort()
            simplified.add(tuple(key_parts))
        
        return simplified

    def _calculate_benchmark_metrics(self, task: PipelineTask):
        """Calculate benchmarking metrics for the pipeline task"""
        if not task.start_time or not task.end_time:
            return
        
        # Initialize benchmark_metrics if None
        if task.benchmark_metrics is None:
            task.benchmark_metrics = {}
        
        # Calculate total processing time
        total_time = task.end_time - task.start_time
        task.benchmark_metrics['total_processing_time'] = round(total_time, 3)
        
        # Calculate individual LLM call times
        if task.nlp_extraction and task.nlp_extraction.duration:
            task.benchmark_metrics['nlp_extraction_time'] = round(task.nlp_extraction.duration, 3)
        
        if task.mcode_mapping and task.mcode_mapping.duration:
            task.benchmark_metrics['Mcode_mapping_time'] = round(task.mcode_mapping.duration, 3)
        
        # Calculate total token usage
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        
        if task.nlp_extraction and task.nlp_extraction.token_usage:
            total_tokens += task.nlp_extraction.token_usage.get('total_tokens', 0)
            prompt_tokens += task.nlp_extraction.token_usage.get('prompt_tokens', 0)
            completion_tokens += task.nlp_extraction.token_usage.get('completion_tokens', 0)
        
        if task.mcode_mapping and task.mcode_mapping.token_usage:
            total_tokens += task.mcode_mapping.token_usage.get('total_tokens', 0)
            prompt_tokens += task.mcode_mapping.token_usage.get('prompt_tokens', 0)
            completion_tokens += task.mcode_mapping.token_usage.get('completion_tokens', 0)
        
        if total_tokens > 0:
            task.benchmark_metrics['total_tokens'] = total_tokens
            task.benchmark_metrics['prompt_tokens'] = prompt_tokens
            task.benchmark_metrics['completion_tokens'] = completion_tokens
        
        logger.info(f"Benchmark metrics calculated for task {task.id}: {task.benchmark_metrics}")
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        with ui.header().classes('bg-primary text-white p-4 items-center'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Pipeline Task Tracker').classes('text-2xl font-bold')
                
                # Theme controls
                with ui.row().classes('gap-2'):
                    ui.button('Toggle Dark Mode', on_click=self._toggle_dark_mode).props('flat color=white')
                    ui.button('Reset UI', on_click=self._reset_ui).props('flat color=white')
        
        with ui.column().classes('w-full p-4 gap-4'):
            self._setup_tab_navigation()
            self._setup_task_list()
    
    def _setup_tab_navigation(self):
        """Setup tab navigation for different views"""
        with ui.card().classes('w-full'):
            with ui.tabs().classes('w-full') as tabs:
                self.single_task_tab = ui.tab('Single Task').classes('text-lg')
                self.run_all_tab = ui.tab('Run All').classes('text-lg')
            
            with ui.tab_panels(tabs, value=self.single_task_tab).classes('w-full'):
                with ui.tab_panel(self.single_task_tab):
                    self._setup_single_task_panel()
                with ui.tab_panel(self.run_all_tab):
                    self._setup_run_all_panel()
    
    def _setup_single_task_panel(self):
        """Setup the single task control panel"""
        ui.label('Single Task Control').classes('text-lg font-semibold mb-4')
        
        # Pipeline selection
        with ui.row().classes('w-full gap-2 items-end'):
            self.pipeline_selector = ui.select(
                options=['NLP to mCODE', 'Direct to mCODE'],
                value='NLP to mCODE',
                label='Select Pipeline',
                on_change=self._on_pipeline_change
            ).classes('w-64')
            
            # Concurrency control
            with ui.column():
                ui.label('Concurrency').classes('text-sm text-gray-600')
                self.concurrency_selector = ui.select(
                    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    value=5,
                    label='Workers',
                    on_change=lambda e: setattr(self, 'max_workers', int(e.value))
                ).classes('w-24')
            
            self.run_task_button = ui.button(
                'Run Pipeline Task',
                on_click=self._run_pipeline_task
            ).props('icon=play_arrow color=positive')

            self.status_label = ui.label('Ready').classes('self-center ml-4')
        
        # Prompt selectors (initially hidden)
        with ui.column().classes('w-full mt-4 gap-2').bind_visibility_from(self.pipeline_selector, 'value'):
            # NLP Extraction prompts (for NLP to mCODE pipeline)
            with ui.row().classes('w-full gap-2 items-end').bind_visibility_from(
                self.pipeline_selector, 'value', backward=lambda x: x == 'NLP to mCODE'):
                ui.label('NLP Extraction Prompt:').classes('text-sm self-center')
                nlp_extraction_prompts = [
                    name for name, info in self.available_prompts.items()
                    if info.get('prompt_type') == 'NLP_EXTRACTION'
                ]
                self.nlp_prompt_selector = ui.select(
                    options=nlp_extraction_prompts,
                    value='generic_extraction' if 'generic_extraction' in nlp_extraction_prompts else nlp_extraction_prompts[0] if nlp_extraction_prompts else '',
                    label='Extraction Prompt'
                ).classes('w-48')
                
                ui.label('mCODE Mapping Prompt:').classes('text-sm self-center')
                Mcode_mapping_prompts = [
                    name for name, info in self.available_prompts.items()
                    if info.get('prompt_type') == 'MCODE_MAPPING'
                ]
                self.mcode_prompt_selector = ui.select(
                    options=Mcode_mapping_prompts,
                    value='generic_mapping' if 'generic_mapping' in Mcode_mapping_prompts else Mcode_mapping_prompts[0] if Mcode_mapping_prompts else '',
                    label='Mapping Prompt'
                ).classes('w-48')
            
            # Direct mCODE prompts (for Direct to mCODE pipeline)
            with ui.row().classes('w-full gap-2 items-end').bind_visibility_from(
                self.pipeline_selector, 'value', backward=lambda x: x == 'Direct to mCODE'):
                ui.label('Direct mCODE Prompt:').classes('text-sm self-center')
                direct_mcode_prompts = [
                    name for name, info in self.available_prompts.items()
                    if info.get('prompt_type') == 'DIRECT_MCODE'
                ]
                self.direct_mcode_prompt_selector = ui.select(
                    options=direct_mcode_prompts,
                    value='direct_text_to_mcode_mapping' if 'direct_text_to_mcode_mapping' in direct_mcode_prompts else direct_mcode_prompts[0] if direct_mcode_prompts else '',
                    label='Direct Mapping Prompt'
                ).classes('w-48')
    
        # Test case loading controls
        with ui.row().classes('w-full gap-2 items-end mt-4'):
            ui.label('Test Case Loading:').classes('text-sm font-semibold')
            
            # Test case path input
            ui.label('Test Case Path:').classes('text-sm self-center')
            self.test_case_path_input = ui.input(
                value='examples/breast_cancer_data/breast_cancer_her2_positive.trial.json',
                label='Test Case File',
                placeholder='Path to test case JSON file'
            ).classes('w-64')
            
            # Load test case button
            ui.button(
                'Load Test Case',
                on_click=self._load_test_case_from_ui
            ).props('icon=file_download')
        
        # Gold standard validation controls
        with ui.row().classes('w-full gap-2 items-end mt-4'):
            ui.label('Gold Standard Validation:').classes('text-sm font-semibold')
            
            # Gold standard path input
            ui.label('Gold Standard Path:').classes('text-sm self-center')
            self.gold_standard_path_input = ui.input(
                value='examples/breast_cancer_data/breast_cancer_her2_positive.gold.json',
                label='Gold Standard File',
                placeholder='Path to gold standard JSON file'
            ).classes('w-64')
            
            # Enable validation checkbox
            self.enable_validation_checkbox = ui.checkbox(
                'Enable Validation',
                value=True,
                on_change=lambda e: setattr(self, 'enable_validation', e.value)
            ).classes('self-center')
            
            # Load gold standard button
            ui.button(
                'Load Gold Standard',
                on_click=self._load_gold_standard_from_ui
            ).props('icon=file_download')
        
        
    def _setup_task_list(self):
        """Setup the task list display area"""
        with ui.card().classes('w-full mt-4'):
            ui.label('Pipeline Tasks').classes('text-lg font-semibold mb-2')
            self.task_list_container = ui.column().classes('w-full gap-2')
            # Initial update to show empty state
            self._update_task_list()
    
    def _run_pipeline_task(self):
        """Create and queue a new pipeline task.
        
        This method creates a new pipeline task with the selected configuration
        and adds it directly to the processing queue for concurrent execution.
        """
        if not self.sample_trial_data:
            ui.notify("No sample data available", type='warning')
            return
            
        # Extract the correct trial data structure and test case name
        trial_data = self.sample_trial_data
        test_case_name = "unknown"
        if trial_data and "test_cases" in trial_data:
            test_cases = trial_data["test_cases"]
            if test_cases:
                # Get the first test case data and name
                first_test_case_key = list(test_cases.keys())[0]
                test_case_name = first_test_case_key
                trial_data = test_cases[first_test_case_key]
        
        # Create a new pipeline task
        task_id = str(uuid.uuid4())[:8]
        prompt_info = {}
        if self.pipeline_selector.value == 'Direct to mCODE':
            prompt_info = {
                'direct_prompt': self.direct_mcode_prompt_selector.value
            }
        else:
            prompt_info = {
                'extraction_prompt': self.nlp_prompt_selector.value,
                'mapping_prompt': self.mcode_prompt_selector.value
            }
        
        task = PipelineTask(
            id=task_id,
            trial_data=trial_data,
            pipeline_type=self.pipeline_selector.value,
            test_case_name=test_case_name,
            prompt_info=prompt_info
        )
        
        # Add to tasks list and directly to queue
        self.tasks.append(task)
        background_tasks.create(self._add_task_to_queue_async(task))
        
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
        # Use the stored test case name
        test_case_name = task.test_case_name
        
        # Extract additional information from trial data
        nct_id = "N/A"
        brief_title = "N/A"
        if task.trial_data and "protocolSection" in task.trial_data:
            protocol_section = task.trial_data["protocolSection"]
            if "identificationModule" in protocol_section:
                identification_module = protocol_section["identificationModule"]
                nct_id = identification_module.get("nctId", "N/A")
                brief_title = identification_module.get("briefTitle", "N/A")
        
        with ui.card().classes('w-full'):
            # Main task header
            with ui.row().classes('w-full justify-between items-center flex-wrap'):
                ui.label(f'{task.pipeline_type} - Task {task.id}').classes('text-lg font-semibold')
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
                    
                # Add validation status if available - make it more prominent
                if task.status == TaskStatus.SUCCESS and task.benchmark_metrics:
                    metrics = task.benchmark_metrics
                    if 'f1_score' in metrics:
                        f1_score = metrics['f1_score']
                        if f1_score is not None:
                            if f1_score >= 0.8:
                                status_text += f" ✓ F1: {f1_score:.3f}"
                            elif f1_score >= 0.5:
                                status_text += f" ⚠ F1: {f1_score:.3f}"
                            else:
                                status_text += f" ✗ F1: {f1_score:.3f}"
                        else:
                            status_text += " ✓ F1: N/A"
                    
                ui.label(status_text).classes(f'text-{status_color}-600 font-medium')
            
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
                    if task.prompt_info:
                        if task.pipeline_type == 'Direct to mCODE':
                            ui.label(f'Prompt: {task.prompt_info.get("direct_prompt", "N/A")}')
                        else:
                            ui.label(f'Extraction Prompt: {task.prompt_info.get("extraction_prompt", "N/A")}')
                            ui.label(f'Mapping Prompt: {task.prompt_info.get("mapping_prompt", "N/A")}')
                
                # Sub-tasks
                if task.pipeline_type == 'Direct to mCODE':
                    self._create_subtask_row(task.mcode_mapping)
                else:
                    self._create_subtask_row(task.nlp_extraction)
                    self._create_subtask_row(task.mcode_mapping)
                
                # Validation results if available - make more prominent
                if task.status == TaskStatus.SUCCESS and task.benchmark_metrics:
                    metrics = task.benchmark_metrics
                    if 'precision' in metrics and 'recall' in metrics and 'f1_score' in metrics:
                        # Add validation summary badge to main card
                        with ui.row().classes('w-full items-center gap-2 mt-2'):
                            ui.label('Validation:').classes('text-sm font-semibold')
                            
                            # Color-coded validation badge
                            f1_score = metrics['f1_score']
                            if f1_score is not None:
                                if f1_score >= 0.8:
                                    badge_color = 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                    badge_text = f"Excellent (F1: {f1_score:.3f})"
                                elif f1_score >= 0.5:
                                    badge_color = 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                                    badge_text = f"Good (F1: {f1_score:.3f})"
                                else:
                                    badge_color = 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                                    badge_text = f"Poor (F1: {f1_score:.3f})"
                            else:
                                badge_color = 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
                                badge_text = "Validation: N/A"
                            
                            ui.label(badge_text).classes(f'text-xs px-2 py-1 rounded-full {badge_color}')
                        
                        # Detailed validation metrics
                        with ui.card().classes('w-full bg-blue-50 dark:bg-blue-900 mt-2 border-l-4 border-blue-500'):
                            ui.label('Validation Results').classes('font-semibold mb-2 text-blue-800 dark:text-blue-200')
                            
                            # Main metrics in a prominent row
                            with ui.row().classes('w-full justify-between text-sm font-medium'):
                                precision = metrics["precision"]
                                recall = metrics["recall"]
                                f1_score = metrics["f1_score"]
                                ui.label(f'Precision: {precision:.3f}' if precision is not None else 'Precision: N/A').classes('text-blue-700 dark:text-blue-300')
                                ui.label(f'Recall: {recall:.3f}' if recall is not None else 'Recall: N/A').classes('text-blue-700 dark:text-blue-300')
                                ui.label(f'F1-Score: {f1_score:.3f}' if f1_score is not None else 'F1-Score: N/A').classes('text-blue-700 dark:text-blue-300')
                            
                            # Detailed counts
                            with ui.row().classes('w-full justify-between text-sm text-gray-600 dark:text-gray-400 mt-2'):
                                ui.label(f'True Positives: {metrics.get("true_positives", 0)}')
                                ui.label(f'False Positives: {metrics.get("false_positives", 0)}')
                                ui.label(f'False Negatives: {metrics.get("false_negatives", 0)}')
                            
                            # Totals
                            with ui.row().classes('w-full justify-between text-sm text-gray-600 dark:text-gray-400 mt-1'):
                                ui.label(f'Pipeline Mappings: {metrics.get("total_pipeline", 0)}')
                                ui.label(f'Gold Mappings: {metrics.get("total_gold", 0)}')
                            
                            # Validation status message
                            if f1_score >= 0.8:
                                ui.label('✓ Validation: Excellent match with gold standard').classes('text-green-600 dark:text-green-400 text-xs mt-2')
                            elif f1_score >= 0.5:
                                ui.label('⚠ Validation: Good match with gold standard').classes('text-yellow-600 dark:text-yellow-400 text-xs mt-2')
                            else:
                                ui.label('✗ Validation: Poor match with gold standard').classes('text-red-600 dark:text-red-400 text-xs mt-2')
                
                # Benchmarking metrics if available
                if task.status == TaskStatus.SUCCESS and task.benchmark_metrics:
                    with ui.card().classes('w-full bg-green-50 dark:bg-green-900 mt-2'):
                        ui.label('Benchmarking Metrics').classes('font-semibold mb-2')
                        
                        metrics = task.benchmark_metrics
                        
                        # Processing time metrics
                        if 'total_processing_time' in metrics:
                            with ui.row().classes('w-full justify-between text-sm'):
                                ui.label(f'Total Processing Time: {metrics["total_processing_time"]:.3f}s')
                                
                        if 'nlp_extraction_time' in metrics:
                            ui.label(f'NLP Extraction Time: {metrics["nlp_extraction_time"]:.3f}s').classes('text-sm text-gray-600 dark:text-gray-400')
                            
                        if 'Mcode_mapping_time' in metrics:
                            ui.label(f'mCODE Mapping Time: {metrics["Mcode_mapping_time"]:.3f}s').classes('text-sm text-gray-600 dark:text-gray-400')
                        
                        # Token usage metrics
                        if 'total_tokens' in metrics:
                            with ui.row().classes('w-full justify-between text-sm mt-2'):
                                ui.label(f'Total Tokens: {metrics["total_tokens"]}')
                                
                        if 'prompt_tokens' in metrics:
                            ui.label(f'Prompt Tokens: {metrics["prompt_tokens"]}').classes('text-sm text-gray-600 dark:text-gray-400')
                            
                        if 'completion_tokens' in metrics:
                            ui.label(f'Completion Tokens: {metrics["completion_tokens"]}').classes('text-sm text-gray-600 dark:text-gray-400')
                
                # Error message if any
                if task.error_message:
                    ui.label(f"Error: {task.error_message}").classes('text-red-600 mt-2')
                
                # Live log display
                with ui.column().classes('w-full mt-2'):
                    ui.label('Live Log').classes('text-sm font-semibold')
                    log_area = ui.log().classes('w-full h-48 bg-gray-100 dark:bg-gray-800 p-2 rounded')
                    for log_entry in task.live_log:
                        log_area.push(log_entry)
    
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
        """Process and display notifications immediately"""
        # Process all pending notifications
        notifications_to_process = self.notifications.copy()
        self.notifications.clear()
        
        for notification in notifications_to_process:
            ui.notify(notification['message'], type=notification['type'])
    
    def _refresh_ui_during_processing(self):
        """Refresh UI when tasks are being processed - now handled by reactive updates"""
        # This method is no longer needed as UI updates are handled reactively
        # by NiceGUI's data binding system
        pass
    
    def _add_notification(self, message: str, type: str = 'info'):
        """Add a notification to be displayed"""
        self.notifications.append({'message': message, 'type': type})
    
    def _load_gold_standard_from_ui(self):
        """Load gold standard data from the UI path input"""
        try:
            path = self.gold_standard_path_input.value
            if not path:
                ui.notify("Please enter a gold standard file path", type='warning')
                return
            
            gold_file = Path(path)
            if gold_file.exists():
                with open(gold_file, 'r') as f:
                    data = json.load(f)
                    self.gold_standard_data = data
                    logger.info(f"Successfully loaded gold standard data from {path}")
                    ui.notify(f"Gold standard data loaded from {path}", type='positive')
            else:
                logger.warning(f"Gold standard file not found: {path}")
                ui.notify(f"Gold standard file not found: {path}", type='warning')
                
        except Exception as e:
            logger.error(f"Failed to load gold standard data: {e}")
            ui.notify(f"Failed to load gold standard data: {str(e)}", type='negative')

    def _load_test_case_from_ui(self):
        """Load test case data from the UI path input"""
        try:
            path = self.test_case_path_input.value
            if not path:
                ui.notify("Please enter a test case file path", type='warning')
                return
            
            test_case_file = Path(path)
            if test_case_file.exists():
                with open(test_case_file, 'r') as f:
                    data = json.load(f)
                    self.sample_trial_data = data
                    logger.info(f"Successfully loaded test case data from {path}")
                    ui.notify(f"Test case data loaded from {path}", type='positive')
                    # Update UI to reflect the new test case data
                    self._update_task_list()
            else:
                logger.warning(f"Test case file not found: {path}")
                ui.notify(f"Test case file not found: {path}", type='warning')
                
        except Exception as e:
            logger.error(f"Failed to load test case data: {e}")
            ui.notify(f"Failed to load test case data: {str(e)}", type='negative')

    def _on_pipeline_change(self, event):
        """Handle pipeline selection changes"""
        # This method is called when the pipeline selector changes
        # The UI visibility is handled by the bind_visibility_from bindings
        pass
    
    def _toggle_dark_mode(self):
        """Toggle dark mode with proper state tracking"""
        self.is_dark_mode = not self.is_dark_mode
        if self.is_dark_mode:
            self.dark_mode.enable()
        else:
            self.dark_mode.disable()
        logger.info(f"Dark mode {'enabled' if self.is_dark_mode else 'disabled'}")
    
    def _reset_ui(self):
        """Reset the UI to default state"""
        try:
            # Reset dark mode to light
            self.is_dark_mode = False
            self.dark_mode.disable()
            
            # Clear all tasks
            self.tasks.clear()
            
            # Stop all workers
            self._stop_workers()
            
            # Reset concurrency to default
            self.max_workers = 5
            if self.concurrency_selector:
                self.concurrency_selector.value = 5
            
            # Reset pipeline selection
            if self.pipeline_selector:
                self.pipeline_selector.value = 'NLP to mCODE'
            
            # Reset status label
            if self.status_label:
                self.status_label.set_text('Ready')
            
            # Update task list
            self._update_task_list()
            
            # Clear notifications
            self.notifications.clear()
            
            # Restart workers after reset
            self._start_worker()
            
            ui.notify("UI reset to default state", type='positive')
            logger.info("UI reset to default state")
            
        except Exception as e:
            logger.error(f"Error resetting UI: {e}")
            ui.notify(f"Error resetting UI: {str(e)}", type='negative')
    
    async def _add_task_to_queue_async(self, task: PipelineTask):
        """Add a task to the queue asynchronously"""
        await self.task_queue.put(task)
        logger.info(f"Added task {task.id} to queue")
    
    def _start_worker(self):
        """Start the background worker tasks based on concurrency level.
        
        This method creates multiple worker tasks that process pipeline tasks
        from the queue concurrently based on the selected concurrency level.
        """
        if not self.is_worker_running:
            self.is_worker_running = True
            # Start worker tasks based on concurrency level
            concurrency_level = self.max_workers
            self.worker_tasks = []
            for i in range(concurrency_level):
                worker_task = background_tasks.create(self._pipeline_worker(i + 1))
                self.worker_tasks.append(worker_task)
            
            logger.info(f"Pipeline task workers started with {concurrency_level} workers")
    
    async def _pipeline_worker(self, worker_id: int):
        """Worker task that processes pipeline tasks from the queue.
        
        This worker continuously pulls tasks from the queue and processes them.
        Multiple workers run concurrently based on the selected concurrency level.
        """
        while self.is_worker_running:
            try:
                # Get task from queue with timeout to check for cancellation
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if we should stop
                    if not self.is_worker_running:
                        break
                    continue
                
                if task is None:  # Sentinel value to stop worker
                    break
                
                # Process the pipeline task
                try:
                    await self._process_pipeline_task_wrapper(task)
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed to process task {task.id}: {str(e)}")
                finally:
                    # Mark task as done
                    self.task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                if not self.task_queue.empty():
                    self.task_queue.task_done()
    
    
    async def _process_pipeline_task_wrapper(self, task: PipelineTask):
        """Wrapper to process a pipeline task and handle UI updates"""
        try:
            await self._process_pipeline_task(task)
        except Exception as e:
            # If task processing failed, ensure task status is properly set to FAILED
            # The _process_pipeline_task method should handle this, but we ensure it here
            if task.status != TaskStatus.FAILED:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.end_time = asyncio.get_event_loop().time()
                logger.error(f"Task {task.id} failed: {e}")
                # Add notification for task failure
                self._add_notification(f"Task {task.id} failed: {str(e)}", 'negative')
        finally:
            # Ensure UI is updated after task completion
            self._update_task_list()
    
    def _stop_workers(self):
        """Stop all worker tasks"""
        self.is_worker_running = False
        
        # Clear the queue to stop processing
        async def clear_queue():
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()
                except asyncio.QueueEmpty:
                    break
        
        # Send None sentinel values to stop workers
        async def stop_workers():
            concurrency_level = self.max_workers
            for _ in range(concurrency_level):
                await self.task_queue.put(None)
        
        background_tasks.create(clear_queue)
        background_tasks.create(stop_workers)
        
        logger.info("Pipeline task workers stopped")
    
    async def _process_pipeline_task(self, task: PipelineTask):
        """Process a single pipeline task"""
        logger.info(f"Processing pipeline task {task.id}")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.start_time = asyncio.get_event_loop().time()
        if self.status_label:
            self.status_label.set_text(f"Running task {task.id}")
        
        try:
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
            
            # Process the clinical trial
            # We'll wrap the calls to track individual LLM call progress
            result = await self._run_pipeline_with_tracking(pipeline, task)
            
            # Store pipeline result for benchmarking and validation
            task.pipeline_result = result.to_dict() if hasattr(result, 'to_dict') else result
            
            # Load gold standard data for this task
            task.gold_standard_data = self.gold_standard_data
            
            # Perform validation if gold standard data is available and validation is enabled
            if self.enable_validation and task.gold_standard_data:
                validation_metrics = self._validate_pipeline_result(task)
                logger.info(f"Validation completed for task {task.id}: {validation_metrics}")
                # Add validation notification
                if validation_metrics:
                    f1_score = validation_metrics.get('f1_score')
                    if f1_score is not None:
                        if f1_score >= 0.8:
                            self._add_notification(f"Task {task.id}: Excellent validation (F1: {f1_score:.3f})", 'positive')
                        elif f1_score >= 0.5:
                            self._add_notification(f"Task {task.id}: Good validation (F1: {f1_score:.3f})", 'warning')
                        else:
                            self._add_notification(f"Task {task.id}: Poor validation (F1: {f1_score:.3f})", 'negative')
                    else:
                        self._add_notification(f"Task {task.id}: Validation completed (F1: N/A)", 'info')
            
            # Calculate benchmarking metrics (processing time, token usage, etc.)
            self._calculate_benchmark_metrics(task)
            
            # Update task with result
            task.status = TaskStatus.SUCCESS
            task.end_time = asyncio.get_event_loop().time()
            self._add_notification(f"Task {task.id} completed", 'positive')
            logger.info(f"Pipeline task {task.id} completed successfully")
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = asyncio.get_event_loop().time()
            self._add_notification(f"Task {task.id} failed: {str(e)}", 'negative')
            logger.error(f"Pipeline task {task.id} failed: {e}")
    
    async def _run_pipeline_with_tracking(self, pipeline: 'ProcessingPipeline', task: PipelineTask):
        """Run pipeline with tracking of individual LLM calls"""
        # Get model and prompt information
        if isinstance(pipeline, NlpMcodePipeline):
            model_name = pipeline.nlp_extractor.model_name
            temperature = pipeline.nlp_extractor.temperature
            max_tokens = pipeline.nlp_extractor.max_tokens
        elif isinstance(pipeline, McodePipeline):
            model_name = pipeline.llm_mapper.model_name
            temperature = pipeline.llm_mapper.temperature
            max_tokens = pipeline.llm_mapper.max_tokens
        else:
            model_name = "unknown"
            temperature = "unknown"
            max_tokens = "unknown"
        
        # Get more detailed prompt information
        if isinstance(pipeline, NlpMcodePipeline):
            extraction_prompt = pipeline.extraction_prompt_name
            mapping_prompt = pipeline.mapping_prompt_name
        elif isinstance(pipeline, McodePipeline):
            extraction_prompt = "N/A"
            mapping_prompt = pipeline.prompt_name
        else:
            extraction_prompt = "N/A"
            mapping_prompt = "N/A"
        
        # Update task details based on pipeline type
        if isinstance(pipeline, McodePipeline):
            task.mcode_mapping.name = "Direct to mCODE"
            task.mcode_mapping.status = TaskStatus.RUNNING
            task.mcode_mapping.start_time = asyncio.get_event_loop().time()
            task.mcode_mapping.details = f"Mapping text to mCODE using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={mapping_prompt})..."
            task.nlp_extraction.details = "Not applicable for this pipeline"
            task.nlp_extraction.status = TaskStatus.SUCCESS
        else:
            task.nlp_extraction.status = TaskStatus.RUNNING
            task.nlp_extraction.start_time = asyncio.get_event_loop().time()
            task.nlp_extraction.details = f"Extracting entities using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={extraction_prompt})..."
        
        self._update_task_list()
        
        try:
            # Run the pipeline process
            result = await run.io_bound(
                pipeline.process_clinical_trial,
                task.trial_data,
                task_id=task.id
            )
            
            if result:
                if isinstance(pipeline, NlpMcodePipeline):
                    task.nlp_extraction.status = TaskStatus.SUCCESS
                    task.nlp_extraction.end_time = asyncio.get_event_loop().time()
                    task.nlp_extraction.details = f"Extracted {len(result.extracted_entities)} entities using {model_name} with prompt '{pipeline.extraction_prompt_name}'"
                    if result.metadata and 'token_usage' in result.metadata:
                        task.nlp_extraction.token_usage = result.metadata['token_usage']

                    task.mcode_mapping.status = TaskStatus.RUNNING
                    task.mcode_mapping.start_time = asyncio.get_event_loop().time()
                    task.mcode_mapping.details = f"Mapping entities to mCODE using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={pipeline.mapping_prompt_name})..."
                    self._update_task_list()

                task.mcode_mapping.status = TaskStatus.SUCCESS
                task.mcode_mapping.end_time = asyncio.get_event_loop().time()
                task.mcode_mapping.details = f"Mapped {len(result.mcode_mappings)} mCODE elements using {model_name} with prompt '{pipeline.prompt_name if isinstance(pipeline, McodePipeline) else pipeline.mapping_prompt_name}'"
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

    def _setup_run_all_panel(self):
        """Setup the Run All panel for batch execution"""
        ui.label('Batch Execution Control').classes('text-lg font-semibold mb-4')
        
        # Concurrency control for batch operations
        with ui.row().classes('w-full gap-2 items-end'):
            with ui.column():
                ui.label('Concurrency').classes('text-sm text-gray-600')
                self.batch_concurrency_selector = ui.select(
                    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    value=5,
                    label='Workers',
                    on_change=lambda e: setattr(self, 'max_workers', int(e.value))
                ).classes('w-24')
            
            self.run_all_button = ui.button(
                'Run Selected Tasks',
                on_click=self._run_selected_tasks
            ).props('icon=playlist_play color=positive')
            
            self.batch_status_label = ui.label('Ready').classes('self-center ml-4')
        
        # Pipeline selection hierarchy
        with ui.row().classes('w-full gap-4 mt-4'):
            # NLP to mCODE Pipeline
            with ui.card().classes('flex-1'):
                ui.label('NLP to mCODE Pipeline').classes('text-md font-semibold mb-2')
                
                # Pipeline-level checkbox for NLP Extraction prompts
                self.nlp_pipeline_checkbox = ui.checkbox(
                    'Select All NLP Prompts',
                    on_change=lambda e: self._toggle_all_prompts('NLP_EXTRACTION', e.value)
                )
                
                # NLP Extraction prompts
                with ui.column().classes('ml-6 gap-2'):
                    ui.label('NLP Extraction Prompts:').classes('text-sm font-medium')
                    self.nlp_extraction_checkboxes = {}
                    nlp_extraction_prompts = [
                        (name, info) for name, info in self.available_prompts.items()
                        if info.get('prompt_type') == 'NLP_EXTRACTION'
                    ]
                    for name, info in nlp_extraction_prompts:
                        self.nlp_extraction_checkboxes[name] = ui.checkbox(
                            f"{name} ({info.get('description', 'No description')})",
                            value=False
                        )
                
                # Pipeline-level checkbox for mCODE Mapping prompts
                self.mcode_pipeline_checkbox = ui.checkbox(
                    'Select All mCODE Mapping Prompts',
                    on_change=lambda e: self._toggle_all_prompts('MCODE_MAPPING', e.value)
                ).classes('mt-4')
                
                # mCODE Mapping prompts
                with ui.column().classes('ml-6 gap-2 mt-2'):
                    ui.label('mCODE Mapping Prompts:').classes('text-sm font-medium')
                    self.mcode_mapping_checkboxes = {}
                    Mcode_mapping_prompts = [
                        (name, info) for name, info in self.available_prompts.items()
                        if info.get('prompt_type') == 'MCODE_MAPPING'
                    ]
                    for name, info in Mcode_mapping_prompts:
                        self.mcode_mapping_checkboxes[name] = ui.checkbox(
                            f"{name} ({info.get('description', 'No description')})",
                            value=False
                        )
            
            # Direct to mCODE Pipeline
            with ui.card().classes('flex-1'):
                ui.label('Direct to mCODE Pipeline').classes('text-md font-semibold mb-2')
                
                # Pipeline-level checkbox
                self.direct_pipeline_checkbox = ui.checkbox(
                    'Select All Direct Prompts',
                    on_change=lambda e: self._toggle_all_prompts('DIRECT_MCODE', e.value)
                )
                
                # Direct mCODE prompts
                with ui.column().classes('ml-6 gap-2'):
                    ui.label('Direct mCODE Prompts:').classes('text-sm font-medium')
                    self.direct_mcode_checkboxes = {}
                    direct_mcode_prompts = [
                        (name, info) for name, info in self.available_prompts.items()
                        if info.get('prompt_type') == 'DIRECT_MCODE'
                    ]
                    for name, info in direct_mcode_prompts:
                        self.direct_mcode_checkboxes[name] = ui.checkbox(
                            f"{name} ({info.get('description', 'No description')})",
                            value=False
                        )
    
    def _toggle_all_prompts(self, prompt_type: str, selected: bool):
        """Toggle all prompts of a specific type"""
        if prompt_type == 'NLP_EXTRACTION':
            for checkbox in self.nlp_extraction_checkboxes.values():
                checkbox.value = selected
        elif prompt_type == 'MCODE_MAPPING':
            for checkbox in self.mcode_mapping_checkboxes.values():
                checkbox.value = selected
        elif prompt_type == 'DIRECT_MCODE':
            for checkbox in self.direct_mcode_checkboxes.values():
                checkbox.value = selected
    
    def _run_selected_tasks(self):
        """Run all selected tasks in batch"""
        if not self.sample_trial_data:
            ui.notify("No sample data available", type='warning')
            return
        
        selected_tasks = []
        
        # Get selected NLP to mCODE tasks
        for nlp_prompt_name, nlp_checkbox in self.nlp_extraction_checkboxes.items():
            if nlp_checkbox.value:
                for Mcode_prompt_name, Mcode_checkbox in self.mcode_mapping_checkboxes.items():
                    if Mcode_checkbox.value:
                        selected_tasks.append({
                            'pipeline_type': 'NLP to mCODE',
                            'extraction_prompt': nlp_prompt_name,
                            'mapping_prompt': Mcode_prompt_name
                        })
        
        # Get selected Direct to mCODE tasks
        for direct_prompt_name, direct_checkbox in self.direct_mcode_checkboxes.items():
            if direct_checkbox.value:
                selected_tasks.append({
                    'pipeline_type': 'Direct to mCODE',
                    'direct_prompt': direct_prompt_name
                })
        
        if not selected_tasks:
            ui.notify("No tasks selected", type='warning')
            return
        
        # Create and queue all selected tasks
        for task_config in selected_tasks:
            self._create_and_queue_task(task_config)
        
        self._add_notification(f"Queued {len(selected_tasks)} tasks for execution", 'positive')
        self.batch_status_label.set_text(f"Queued {len(selected_tasks)} tasks")
    

    def _create_and_queue_task(self, task_config: Dict[str, Any]):
        """Create and queue a pipeline task with the given configuration.
        
        This method is used by Run All mode to create tasks with specific
        prompt configurations and add them to the processing queue.
        """
        if not self.sample_trial_data:
            logger.warning("No sample data available for task creation")
            return
            
        # Extract the correct trial data structure and test case name
        trial_data = self.sample_trial_data
        test_case_name = "unknown"
        if trial_data and "test_cases" in trial_data:
            test_cases = trial_data["test_cases"]
            if test_cases:
                # Get the first test case data and name
                first_test_case_key = list(test_cases.keys())[0]
                test_case_name = first_test_case_key
                trial_data = test_cases[first_test_case_key]
        
        # Create a new pipeline task
        task_id = str(uuid.uuid4())[:8]
        pipeline_type = task_config['pipeline_type']
        prompt_info = {}
        
        if pipeline_type == 'Direct to mCODE':
            prompt_info = {
                'direct_prompt': task_config['direct_prompt']
            }
        else:
            prompt_info = {
                'extraction_prompt': task_config['extraction_prompt'],
                'mapping_prompt': task_config['mapping_prompt']
            }
        
        task = PipelineTask(
            name=f"{pipeline_type} - {test_case_name}",
            description=f"Running {pipeline_type} pipeline with prompts: {prompt_info}",
            prompt_key=str(prompt_info),
            model_key="gpt-4",
            trial_id=test_case_name
        )
        task.id = task_id
        task.trial_data = trial_data
        task.pipeline_type = pipeline_type
        task.test_case_name = test_case_name
        task.prompt_info = prompt_info
        
        # Add to tasks list and directly to queue
        self.tasks.append(task)
        background_tasks.create(self._add_task_to_queue_async(task))
        
        # Update UI
        self._update_task_list()
        logger.info(f"Pipeline task {task_id} queued with config: {task_config}")

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