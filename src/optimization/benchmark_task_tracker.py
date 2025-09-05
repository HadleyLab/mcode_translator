
"""
Benchmark Task Tracker - Simplified visualization-only implementation.
Uses centralized pipeline task queue for execution, focuses only on UI.
"""

import sys
import os
# Add the parent directory to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import time
import uuid
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from nicegui import ui, run, background_tasks
import pandas as pd

# Import centralized pipeline components
from src.pipeline import (
    PipelineTaskQueue, BenchmarkTask, TaskStatus,
    get_global_task_queue, initialize_task_queue, shutdown_task_queue
)

# Import cache management
from src.utils.api_manager import APICache

# Import existing components for data loading only
from src.optimization.prompt_optimization_framework import (
    PromptOptimizationFramework, PromptType,
    APIConfig, PromptVariant
)

# Import utilities for data loading
from src.utils import (
    prompt_loader,
    model_loader,
    get_logger,
    UnifiedAPIManager
)


class BenchmarkTaskTrackerUI:
    """Simplified UI for benchmark task visualization using centralized pipeline"""
    
    def __init__(self):
        # Initialize the framework for data loading only
        self.framework = PromptOptimizationFramework()
        self.available_prompts = {}
        self.available_models = {}
        self.trial_data = {}
        self.gold_standard_data = {}
        
        # Centralized task queue
        self.task_queue = None
        
        # UI components
        self.live_log_display = None
        self.control_panel = None
        self.results_display = None
        self.validation_display = None
        self.prompt_selection = None
        self.model_selection = None
        self.trial_selection = None
        self.metric_selection = None
        self.top_n_selection = None
        self.concurrency_selection = None
        self.run_benchmark_button = None
        self.stop_benchmark_button = None
        self.benchmark_progress = None
        self.benchmark_status = None
        self.worker_filter = None
        
        # Benchmark state
        self.is_benchmark_running = False
        self.benchmark_cancelled = False
        self.benchmark_results = []
        self.validation_results = []
        self.active_validations: Dict[str, Dict] = {}  # task_id -> validation data
        
        # Dark mode
        self.dark_mode = ui.dark_mode()
        
        # Task tracking
        self.active_tasks: Dict[str, BenchmarkTask] = {}
        
        # Load libraries and data
        self._load_libraries()
        self._load_test_data()
        
        # Setup UI components
        self._setup_ui()

        # Setup refreshable UI components
        self._setup_refreshable_ui()

    def _setup_ui(self):
        """Setup the main UI layout using NiceGUI's built-in features"""
        with ui.header().classes('bg-primary text-white p-4 items-center'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Benchmark Task Tracker').classes('text-2xl font-bold')
                with ui.row().classes('items-center'):
                    ui.button('Toggle Dark Mode', on_click=self._toggle_dark_mode).props('flat color=white')
                    ui.button('Reset', on_click=self._reset_interface).props('flat color=white')

        with ui.column().classes('w-full p-4 gap-4'):
            self._setup_benchmark_control_panel()
            self._setup_validation_display()
            self._setup_results_display()
            self._setup_live_logger()
    
    def _toggle_dark_mode(self):
        """Toggle dark mode"""
        self.dark_mode.toggle()
        ui.notify("Dark mode toggled")
    
    def _load_libraries(self) -> None:
        """Load prompt and model libraries"""
        try:
            self.available_prompts = prompt_loader.list_available_prompts()
            self.available_models = model_loader.list_available_models()
            logging.info(f"Loaded {len(self.available_prompts)} prompts and {len(self.available_models)} models")
        except Exception as e:
            logging.error(f"Failed to load libraries: {str(e)}")
            raise
    
    def _load_test_data(self) -> None:
        """Load test cases and gold standard data"""
        try:
            # Load trial data
            trial_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.trial.json")
            if trial_file.exists():
                with open(trial_file, 'r') as f:
                    trial_data = json.load(f)
                    self.trial_data = trial_data.get("test_cases", {})
                    logging.info(f"Loaded {len(self.trial_data)} test cases")
            
            # Load gold standard data
            gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
            if gold_file.exists():
                with open(gold_file, 'r') as f:
                    gold_data = json.load(f)
                    self.gold_standard_data = gold_data.get("gold_standard", {})
                    logging.info(f"Loaded gold standard data for {len(self.gold_standard_data)} test cases")
        except Exception as e:
            logging.error(f"Failed to load test data: {str(e)}")
            raise
    
    def _setup_benchmark_control_panel(self) -> None:
        """Setup the benchmark control panel"""
        with ui.card().classes('w-full'):
            ui.label('Benchmark Control Panel').classes('text-lg font-semibold mb-4')
            
            with ui.row().classes('w-full gap-4'):
                # Pipeline type selection - multi-select with both types selected by default
                pipeline_options = {
                    'McodePipeline': 'Direct Mcode Pipeline',
                    'NlpMcodePipeline': 'NLP Extraction + Mapping Pipeline'
                }
                self.pipeline_selection = ui.select(
                    pipeline_options,
                    label='Pipeline Types',
                    multiple=True,
                    value=list(pipeline_options.keys()),
                    on_change=lambda: self._update_prompt_selection()
                ).classes('w-1/4').tooltip('Filter prompts by pipeline type')
                
                # Prompt selection
                self.prompt_selection = ui.select(
                    {},
                    label='Select Prompts',
                    multiple=True,
                    value=[],
                    on_change=lambda: self._update_validation_display()
                ).classes('w-1/4').tooltip('Select one or more prompts to benchmark')
                
                # Model selection
                model_options = {key: info.get('name', key) for key, info in self.available_models.items()}
                self.model_selection = ui.select(
                    model_options,
                    label='Select Models',
                    multiple=True,
                    value=list(model_options.keys()),
                    on_change=lambda: self._update_validation_display()
                ).classes('w-1/4').tooltip('Select one or more models to benchmark')
                
                # Trial selection
                trial_options = {key: key for key in self.trial_data.keys()}
                self.trial_selection = ui.select(
                    trial_options,
                    label='Select Trials',
                    multiple=True,
                    value=list(trial_options.keys()),
                    on_change=lambda: self._update_validation_display()
                ).classes('w-1/4').tooltip('Select one or more trials to benchmark')
            
            with ui.row().classes('w-full gap-4 mt-4'):
                # Metric selection
                metric_options = {
                    'f1_score': 'F1 Score',
                    'precision': 'Precision',
                    'recall': 'Recall',
                    'compliance_score': 'Compliance Score'
                }
                self.metric_selection = ui.select(metric_options, label='Optimization Metric', value='f1_score').classes('w-1/4')
                
                # Top N selection
                self.top_n_selection = ui.number('Top N Combinations', value=5, min=1, max=20).classes('w-1/4')
                
                # Concurrency selection
                self.concurrency_selection = ui.number('Concurrency Level', value=5, min=1, max=10).classes('w-1/4')
                
                # Worker filter
                self.worker_filter = ui.input(
                    placeholder='Filter by worker ID...',
                    on_change=lambda: self._update_validation_display()
                ).classes('w-1/4').tooltip('Filter tasks by worker ID')
            
            with ui.row().classes('w-full gap-2 mt-4'):
                self.run_benchmark_button = ui.button('Run Benchmark', on_click=self._run_benchmark).props('icon=play_arrow color=positive')
                self.stop_benchmark_button = ui.button('Stop Benchmark', on_click=self._stop_benchmark).props('icon=stop color=negative')
                self.stop_benchmark_button.set_visibility(False)
                
                self.benchmark_status = ui.label('Ready to run benchmark').classes('self-center ml-4')
            
            self.benchmark_progress = ui.linear_progress(0).classes('w-full mt-2')
            
            # Initial population of prompts
            self._update_prompt_selection()
    
    def _update_prompt_selection(self) -> None:
        """Update prompt selection based on pipeline filter"""
        selected_pipelines = self.pipeline_selection.value or []
        filtered_prompts = {}
        
        for key, info in self.available_prompts.items():
            prompt_type_str = info.get('prompt_type', 'NLP_EXTRACTION')
            prompt_pipeline_type = 'McodePipeline' if prompt_type_str == 'DIRECT_MCODE' else 'NlpMcodePipeline'
            
            # If no pipelines selected or this prompt's pipeline is selected
            if not selected_pipelines or prompt_pipeline_type in selected_pipelines:
                filtered_prompts[key] = f"{info.get('name', key)} ({info.get('prompt_type', 'Unknown')})"
        
        self.prompt_selection.set_options(filtered_prompts)
        self.prompt_selection.set_value(list(filtered_prompts.keys()))
        # Refresh validation display using NiceGUI's refreshable pattern
        self._update_validation_display.refresh()
    
    def _setup_validation_display(self) -> None:
        """Setup validation results display"""
        with ui.card().classes('w-full mt-4'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Validation Results').classes('text-lg font-semibold')
                ui.label('Updated in real-time as benchmarks run').classes('text-sm text-gray-500')
            # Use refreshable function directly
            self._update_validation_display()
    
    def _create_validation_entry(self, task: BenchmarkTask) -> Dict[str, Any]:
        """Create a validation entry for a task"""
        prompt_type_str = task.prompt_type
        pipeline_type = 'McodePipeline' if prompt_type_str == 'DIRECT_MCODE' else 'NlpMcodePipeline'
        
        return {
            'prompt': task.prompt_name,
            'model': task.model_name,
            'trial': task.trial_id,
            'status': 'Processing',
            'details': 'Queued for execution',
            'status_icon': '🔄',
            'precision': None,
            'recall': None,
            'f1_score': None,
            'duration_ms': None,
            'token_usage': None,
            'compliance_score': None,
            'log': f'INFO: Queued {task.prompt_name} + {task.model_name} on {task.trial_id}',
            'detailed_log': f'🕒 Queued {task.prompt_name} + {task.model_name} on {task.trial_id}',
            'error_message': '',
            'pipeline_type': pipeline_type,
            'live_logs': [],
            'task_id': task.task_id,
            'worker_id': task.worker_id or 'N/A',
            'pipeline_type': task.pipeline_type or pipeline_type,
            'optimization_parameters': task.optimization_parameters or {},
            'prompt_info': task.prompt_info or {}
        }
    
    def _update_validation_status(self, task: BenchmarkTask, status: str, details: str,
                                 status_icon: str, log: str = '', error_message: str = '',
                                 live_log_entry: str = None) -> None:
        """Update the status of a validation entry"""
        if task.task_id not in self.active_validations:
            # Create new validation entry if it doesn't exist
            self.active_validations[task.task_id] = self._create_validation_entry(task)
        
        validation = self.active_validations[task.task_id]
        validation.update({
            'status': status,
            'details': details,
            'status_icon': status_icon,
            'log': log,
            'error_message': error_message,
            'worker_id': task.worker_id or 'N/A',
            'pipeline_type': task.pipeline_type or validation['pipeline_type'],
            'optimization_parameters': task.optimization_parameters or {},
            'prompt_info': task.prompt_info or {}
        })
        
        # Update metrics if task completed successfully
        if status == 'Success':
            validation.update({
                'precision': f"{task.precision:.3f}" if task.precision is not None else 'N/A',
                'recall': f"{task.recall:.3f}" if task.recall is not None else 'N/A',
                'f1_score': f"{task.f1_score:.3f}" if task.f1_score is not None else 'N/A',
                'compliance_score': f"{task.compliance_score:.2%}" if task.compliance_score is not None else 'N/A',
                'duration_ms': f"{task.duration_ms:.1f}" if task.duration_ms else 'N/A',
                'token_usage': str(task.token_usage) if task.token_usage else 'N/A',
            })
        
        if live_log_entry:
            validation['live_logs'].append(live_log_entry)
            if len(validation['live_logs']) > 20:
                validation['live_logs'] = validation['live_logs'][-20:]
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        if status == 'Processing':
            validation['detailed_log'] = f"[{timestamp}] 🔄 {log}"
        elif status == 'Success':
            validation['detailed_log'] = f"[{timestamp}] ✅ {log}"
        elif status == 'Failed':
            validation['detailed_log'] = f"[{timestamp}] ❌ {details}\nError: {error_message}"
        else:
            validation['detailed_log'] = f"[{timestamp}] {log}"
        
        # Refresh displays after updating validation status using NiceGUI's refreshable pattern
        self._update_validation_display.refresh()
        self._update_results_display.refresh()
    
    @ui.refreshable
    def _update_validation_display(self) -> None:
        """Update validation display based on current filters"""
        selected_prompts = self.prompt_selection.value or []
        selected_models = self.model_selection.value or []
        selected_trials = self.trial_selection.value or []
        selected_pipelines = self.pipeline_selection.value or []
        worker_filter = self.worker_filter.value or ''
        
        validation_data = list(self.active_validations.values())
        filtered_results = validation_data
        
        if selected_prompts:
            # Convert prompt keys to prompt names for filtering
            prompt_names = []
            for prompt_key in selected_prompts:
                prompt_info = self.available_prompts.get(prompt_key)
                if prompt_info:
                    prompt_names.append(prompt_info.get('name', prompt_key))
            filtered_results = [r for r in filtered_results if r.get('prompt') in prompt_names]
        if selected_models:
            filtered_results = [r for r in filtered_results if r.get('model') in selected_models]
        if selected_trials:
            filtered_results = [r for r in filtered_results if r.get('trial') in selected_trials]
        if selected_pipelines:
            filtered_results = [r for r in filtered_results if r.get('pipeline_type') in selected_pipelines]
        if worker_filter:
            filtered_results = [r for r in filtered_results if worker_filter.lower() in str(r.get('worker_id', '')).lower()]
        
        def status_priority(status):
            if status == 'Processing': return 1
            else: return 0
        
        filtered_results.sort(key=lambda x: status_priority(x.get('status', 'Completed')))
        
        if not filtered_results:
            ui.label('No validation results match current filters.').classes('text-gray-500 italic')
        else:
            ui.label(f'Showing {len(filtered_results)} results').classes('text-sm text-gray-500 mb-2')
            
            # Create a compact table with all benchmark stats using NiceGUI's grid component
            with ui.grid(columns=13).classes('w-full gap-x-2 gap-y-1 items-center'):
                # Header row
                ui.label('Status').classes('font-bold text-xs')
                ui.label('Prompt').classes('font-bold text-xs')
                ui.label('Model').classes('font-bold text-xs')
                ui.label('Pipeline').classes('font-bold text-xs')
                ui.label('Trial').classes('font-bold text-xs')
                ui.label('Worker').classes('font-bold text-xs')
                ui.label('F1').classes('font-bold text-xs')
                ui.label('Precision').classes('font-bold text-xs')
                ui.label('Recall').classes('font-bold text-xs')
                ui.label('Time').classes('font-bold text-xs')
                ui.label('Tokens').classes('font-bold text-xs')
                ui.label('Compliance').classes('font-bold text-xs')
                ui.label('Log').classes('font-bold text-xs')

                # Data rows
                for result in filtered_results:
                    if result['status'] == 'Success':
                        ui.icon('check_circle', color='green').classes('text-base').tooltip('✅ Task completed successfully')
                    elif result['status'] == 'Failed':
                        ui.icon('error', color='red').classes('text-base').tooltip('❌ Task failed')
                    elif result['status'] == 'Processing':
                        ui.spinner('dots', color='orange').tooltip('🔄 Task in progress')
                    else:
                        ui.icon('pending', color='blue').classes('text-base').tooltip('🔵 Task completed')
                    
                    ui.label(result['prompt']).classes('text-xs truncate')
                    ui.label(result['model']).classes('text-xs truncate')
                    ui.label(result['pipeline_type']).classes('text-xs truncate')
                    ui.label(result['trial']).classes('text-xs truncate')
                    ui.label(result.get('worker_id', 'N/A')).classes('text-xs truncate')
                    
                    ui.label(result.get('f1_score', 'N/A')).classes('text-xs')
                    ui.label(result.get('precision', 'N/A')).classes('text-xs')
                    ui.label(result.get('recall', 'N/A')).classes('text-xs')
                    
                    duration = result.get('duration_ms')
                    duration_str = f"{duration:.1f}ms" if isinstance(duration, (int, float)) and duration > 0 else "N/A"
                    ui.label(duration_str).classes('text-xs')

                    tokens = result.get('token_usage')
                    if isinstance(tokens, dict) and tokens:
                        total_tokens = tokens.get('total_tokens', tokens.get('prompt_tokens', 0) + tokens.get('completion_tokens', 0))
                        token_str = f"{total_tokens}t"
                    elif isinstance(tokens, str) and tokens.strip() and tokens.strip() != '{}':
                         try:
                             token_dict = json.loads(tokens.replace("'", "\""))
                             total_tokens = token_dict.get('total_tokens', 0)
                             token_str = f"{total_tokens}t"
                         except (json.JSONDecodeError, TypeError):
                             token_str = "N/A"
                    else:
                        token_str = "N/A"
                    ui.label(token_str).classes('text-xs')

                    ui.label(result.get('compliance_score', 'N/A')).classes('text-xs')

                    if result['status'] == 'Failed':
                        log_text = f"❌ {result.get('error_message', 'Unknown error')}"
                        log_color = 'text-red-600'
                    elif result.get('live_logs'):
                        log_text = result['live_logs'][-1]
                        log_color = 'text-gray-500'
                    else:
                        log_text = '✅ Completed'
                        log_color = 'text-green-600'
                    ui.label(log_text).classes(f'text-xs truncate {log_color}')
    
    def _setup_live_logger(self) -> None:
        """Setup live logging display with filtering capabilities"""
        with ui.card().classes('w-full mt-4'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Live Task Logging').classes('text-lg font-semibold')
                self.log_filter_input = ui.input(placeholder='Filter logs...').classes('w-1/3')
            self.live_log_display = ui.log(max_lines=100).classes('w-full h-48 bg-gray-100 p-2 rounded')

    def _setup_results_display(self) -> None:
        """Setup results display area"""
        with ui.card().classes('w-full mt-4'):
            ui.label('Benchmark Results').classes('text-lg font-semibold mb-2')
            # Use refreshable function directly
            self._update_results_display()
    
    def _run_benchmark(self) -> None:
        """Run benchmark tasks using centralized pipeline"""
        if self.is_benchmark_running:
            ui.notify("Benchmark is already running", type='warning')
            return
        
        selected_prompts = self.prompt_selection.value or []
        selected_models = self.model_selection.value or []
        selected_trials = self.trial_selection.value or []
        
        if not (selected_prompts and selected_models and selected_trials):
            ui.notify("Please select at least one prompt, model, and trial", type='warning')
            return
        
        self.run_benchmark_button.disable()
        self.stop_benchmark_button.set_visibility(True)
        self.is_benchmark_running = True
        self.benchmark_cancelled = False
        
        async def execute_benchmark():
            try:
                await self._execute_benchmark_async(selected_prompts, selected_models, selected_trials)
            except Exception as e:
                ui.notify(f"Benchmark failed: {str(e)}", type='negative')
                logging.error(f"Benchmark failed: {str(e)}")
            finally:
                self.run_benchmark_button.enable()
                self.stop_benchmark_button.set_visibility(False)
                self.is_benchmark_running = False
                self.benchmark_status.set_text("Benchmark completed")
                self.benchmark_progress.set_value(1.0)
        
        background_tasks.create(execute_benchmark())
        ui.notify("Starting benchmark execution", type='positive')
        self.benchmark_status.set_text("Benchmark running...")
    
    async def _execute_benchmark_async(self, prompt_keys: List[str], model_keys: List[str], trial_ids: List[str]) -> None:
        """Execute benchmark using centralized pipeline"""
        # Initialize task queue
        concurrency = int(self.concurrency_selection.value)
        self.task_queue = get_global_task_queue(max_workers=concurrency)
        await initialize_task_queue(max_workers=concurrency)
        
        total_tasks = len(prompt_keys) * len(model_keys) * len(trial_ids)
        self.benchmark_progress.set_value(0)
        self.task_queue.total_tasks = total_tasks  # Set the total tasks for progress tracking
        
        # Add all tasks to the centralized queue
        for prompt_key in prompt_keys:
            prompt_info = self.available_prompts.get(prompt_key)
            if not prompt_info:
                continue
            
            prompt_name = prompt_info.get('name', prompt_key)
            prompt_type = prompt_info.get('prompt_type', 'NLP_EXTRACTION')
            
            for model_key in model_keys:
                model_info = self.available_models.get(model_key)
                model_name = model_info.get('name', model_key) if model_info else model_key
                
                for trial_id in trial_ids:
                    if self.benchmark_cancelled:
                        break
                    
                    # Get trial data for this trial_id
                    trial_data = self.trial_data.get(trial_id, {})
                    
                    # Create benchmark task with actual trial data and gold standard
                    gold_standard = self.gold_standard_data.get(trial_id, {})
                    expected_entities = gold_standard.get("expected_extraction", {}).get("entities", [])
                    expected_mappings = gold_standard.get("expected_mcode_mappings", {}).get("mapped_elements", [])
                    
                    task = BenchmarkTask(
                        prompt_name=prompt_name,
                        model_name=model_name,
                        trial_id=trial_id,
                        trial_data=trial_data,
                        prompt_type=prompt_type,
                        expected_entities=expected_entities,
                        expected_mappings=expected_mappings,
                        pipeline_type='McodePipeline' if prompt_type == 'DIRECT_MCODE' else 'NlpMcodePipeline',
                        optimization_parameters={
                            'metric': self.metric_selection.value,
                            'top_n': self.top_n_selection.value,
                            'concurrency': self.concurrency_selection.value
                        },
                        prompt_info={
                            'prompt_key': prompt_key,
                            'prompt_name': prompt_name,
                            'prompt_type': prompt_type
                        }
                    )
                    
                    # Add task to centralized queue with callback
                    await self.task_queue.add_task(task, self._task_completion_callback)
                    
                    # Update UI for pending task - validation entry will be created when task is processed
                
                if self.benchmark_cancelled:
                    break
            
            if self.benchmark_cancelled:
                break
        
        # Wait for tasks to complete
        while not self.benchmark_cancelled and self.task_queue.completed_tasks < total_tasks:
            progress = self.task_queue.completed_tasks / total_tasks
            self.benchmark_progress.set_value(progress)
            self.benchmark_status.set_text(f"Running {self.task_queue.completed_tasks}/{total_tasks}")
            await asyncio.sleep(0.1)
        
        # Clean up
        await shutdown_task_queue()
    
    def _task_completion_callback(self, task: BenchmarkTask) -> None:
        """Callback for task completion from centralized pipeline"""
        self.benchmark_results.append(task)
        status = 'Success' if task.status == TaskStatus.SUCCESS else 'Failed'
        
        # Log completion with worker information
        worker_info = f" (Worker: {task.worker_id})" if task.worker_id else ""
        log_message = f"[{status.upper()}] Task {task.task_id} completed. Prompt: {task.prompt_name}, Model: {task.model_name}{worker_info}"
        if self.live_log_display:
            self.live_log_display.push(log_message)

        # Update validation status
        details = f"F1={task.f1_score:.3f}" if task.status == TaskStatus.SUCCESS else task.error_message
        status_icon = '✅' if status == 'Success' else '❌'
        log_message = f"Task {task.task_id} completed. Prompt: {task.prompt_name}, Model: {task.model_name}"
        
        self._update_validation_status(
            task=task,
            status=status,
            details=details,
            status_icon=status_icon,
            log=log_message,
            error_message=task.error_message or '',
        )

        # Refresh both displays to show updated state using NiceGUI's refreshable pattern
        self._update_validation_display.refresh()
        self._update_results_display.refresh()
    
    def _setup_refreshable_ui(self) -> None:
        """Setup refreshable UI components using NiceGUI's built-in refresh functionality"""
        # Refreshable validation display will be automatically updated by NiceGUI
        pass
    
    def _stop_benchmark(self) -> None:
        """Stop benchmark execution"""
        if not self.is_benchmark_running:
            ui.notify("No benchmark is currently running", type='info')
            return
        
        self.benchmark_cancelled = True
        self.benchmark_status.set_text("Benchmark stopping...")
        
        async def stop_queue():
            if self.task_queue:
                await shutdown_task_queue()
        
        background_tasks.create(stop_queue())
        ui.notify("Stopping benchmark execution", type='info')
    
    def _reset_interface(self) -> None:
        """Reset the interface to its initial state"""
        if self.is_benchmark_running:
            self.benchmark_cancelled = True
            self.is_benchmark_running = False
            self.run_benchmark_button.enable()
            self.stop_benchmark_button.set_visibility(False)
            
            async def cleanup():
                await shutdown_task_queue()
            
            background_tasks.create(cleanup())
        
        self.benchmark_results = []
        self.validation_results = []
        self.benchmark_progress.set_value(0)
        self.benchmark_status.set_text("Ready to run benchmark")
        self.benchmark_cancelled = False
        
        # Refresh displays to show reset state using NiceGUI's refreshable pattern
        self._update_validation_display.refresh()
        self._update_results_display.refresh()
        
        if self.prompt_selection:
            self.prompt_selection.set_value(list(self.available_prompts.keys()))
        if self.model_selection:
            self.model_selection.set_value(list(self.available_models.keys()))
        if self.trial_selection:
            self.trial_selection.set_value(list(self.trial_data.keys()))
        
        self.active_validations.clear()
        if self.live_log_display:
            self.live_log_display.clear()
        ui.notify("Interface reset to initial state", type='positive')
        # Refresh displays to show reset state using NiceGUI's refreshable pattern
        self._update_validation_display.refresh()
        self._update_results_display.refresh()

    @ui.refreshable
    def _update_results_display(self) -> None:
        """Update the results display with aggregated data"""
        if not self.benchmark_results:
            ui.label("No successful results yet.").classes('text-gray-500 italic')
            return

        df = pd.DataFrame([
            {
                'prompt': r.prompt_name,
                'model': r.model_name,
                'f1_score': r.f1_score,
                'precision': r.precision,
                'recall': r.recall,
                'duration_ms': r.duration_ms,
                'token_usage': r.token_usage,
            }
            for r in self.benchmark_results if r.status == TaskStatus.SUCCESS
        ])

        if df.empty:
            ui.label("No successful results yet.").classes('text-gray-500 italic')
            return

        # Aggregate results
        # Calculate total tokens for each prompt+model combination
        def calculate_total_tokens(token_usage_list):
            total = 0
            for usage in token_usage_list:
                if isinstance(usage, dict):
                    total += usage.get('total_tokens', usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0))
                elif isinstance(usage, str) and usage.strip() and usage.strip() != '{}':
                    try:
                        token_dict = json.loads(usage.replace("'", "\""))
                        total += token_dict.get('total_tokens', 0)
                    except (json.JSONDecodeError, TypeError):
                        pass
            return total
        
        summary = df.groupby(['prompt', 'model']).agg({
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'duration_ms': 'mean',
            'token_usage': calculate_total_tokens
        }).reset_index().sort_values(by='f1_score', ascending=False)

        top_n = self.top_n_selection.value

        ui.label(f"Top {top_n} Benchmark Combinations").classes('text-md font-semibold')
        with ui.grid(columns=7).classes('w-full gap-2'):
            for col in ['Prompt', 'Model', 'F1 Score', 'Precision', 'Recall', 'Avg. Time (ms)', 'Total Tokens']:
                ui.label(col).classes('font-bold')
            for _, row in summary.head(top_n).iterrows():
                ui.label(row['prompt'])
                ui.label(row['model'])
                ui.label(f"{row['f1_score']:.3f}")
                ui.label(f"{row['precision']:.3f}")
                ui.label(f"{row['recall']:.3f}")
                ui.label(f"{row['duration_ms']:.1f}")
                ui.label(f"{int(row['token_usage'])}")

    def display_cache_stats(self) -> None:
        """Display cache statistics for both API and LLM caches"""
        from src.utils.config import Config
        config = Config()
        
        # Create cache instances using config methods
        api_cache = APICache(config.get_api_cache_directory())
        llm_cache = APICache(".api_cache/llm")
        
        # Get statistics
        api_stats = api_cache.get_stats()
        llm_stats = llm_cache.get_stats()
        
        # Display statistics
        ui.notify("Displaying cache statistics...")
        print(f"API Cache Stats: {api_stats}")
        print(f"LLM Cache Stats: {llm_stats}")

    def clear_caches(self) -> None:
        """Clear both API and LLM caches"""
        from src.utils.config import Config
        config = Config()
        
        # Create cache instances using config methods
        api_cache = APICache(config.get_api_cache_directory())
        llm_cache = APICache(".api_cache/llm")
        
        # Clear both caches
        api_cache.clear()
        llm_cache.clear()
        
        ui.notify("Clearing caches...")
        print("Clearing caches...")


def run_benchmark_task_tracker(port: int = 8089):
    """Run the benchmark task tracker UI"""
    tracker = BenchmarkTaskTrackerUI()
    ui.run(title='Benchmark Task Tracker', port=port, reload=True)


if __name__ in {"__main__", "__mp_main__"}:
    import argparse
    parser = argparse.ArgumentParser(description='Run Benchmark Task Tracker')
    parser.add_argument('--port', type=int, default=8089, help='Port to run the UI on')
    args = parser.parse_args()
    run_benchmark_task_tracker(args.port)