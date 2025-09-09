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

from nicegui import ui, run, background_tasks, app
from src.pipeline.task_queue import PipelineTaskQueue, TaskStatus
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
        # Simple initialization without complex dependencies
        self.framework = None
        
        # Load data and initialize framework synchronously
        self._load_test_data()
        self._load_simple_libraries()
        
        # UI components
        self.live_log_display = None
        self.control_panel = None
        self.metric_selection = None
        self.top_n_selection = None
        self.concurrency_selection = None
        self.run_benchmark_button = None
        self.stop_benchmark_button = None
        self.benchmark_progress = None
        self.benchmark_status = None
        self.worker_filter: Optional[ui.input] = None
        self.cache_stats_dialog: Optional[ui.dialog] = None
        self.cache_stats_title: Optional[ui.label] = None
        self.cache_stats_content: Optional[ui.label] = None

        
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
        
        
        # Pagination state
        self.current_page = 1
        self.page_size = 10  # Show 10 tasks per page
        self.max_pages = 1
        self.pagination_widget = None
        self.page_size_selector = None
        self.page_info_label = None
        
        # Filtering state
        self.status_filter = "all"
        self.search_filter = ""
        self.filtered_tasks = []  # Cached filtered tasks for pagination
        
        # Reactive state variables for data binding
        self.benchmark_status_text = "Ready to run benchmark"
        self.benchmark_progress_value = 0.0
        self.completed_tasks_count = 0
        self.total_tasks_count = 0
        
        # Load data and initialize framework synchronously
        self._load_test_data()
        self._load_libraries()
        
        # Setup UI components
        self._setup_ui()
        
        # Preload tasks after UI setup
        self._preload_all_benchmark_tasks()
        
        # Setup automatic refresh timer for live updates
        ui.timer(2.0, self._refresh_ui_data)
        

    def _refresh_ui_data(self):
        """Refresh UI data and trigger refreshable updates"""
        # Update header badges
        active_count = sum(1 for v in self.active_validations.values() if v.get('status') in ['Queued', 'Processing'])
        completed_count = sum(1 for v in self.active_validations.values() if v.get('status') in ['Success', 'Failed'])
        
        if hasattr(self, 'active_tasks_badge'):
            self.active_tasks_badge.set_text(f'{active_count} Active')
        if hasattr(self, 'completed_tasks_badge'):
            self.completed_tasks_badge.set_text(f'{completed_count} Done')
        
        # Update worker count if benchmark is running
        if self.is_benchmark_running and hasattr(self, 'concurrency_selection'):
            worker_count = int(self.concurrency_selection.value)
            if hasattr(self, 'worker_count_badge'):
                self.worker_count_badge.set_text(f'{worker_count} Workers')
        
        # Trigger refreshable method updates
        if hasattr(self, '_update_validation_display'):
            self._update_validation_display.refresh()
        if hasattr(self, '_update_pagination_controls'):
            self._update_pagination_controls.refresh()
        """Initialize framework synchronously"""
        if not hasattr(self, 'framework') or self.framework is None:
            self.framework = PromptOptimizationFramework()

    def _preload_all_benchmark_tasks(self):
        """Preload all possible benchmark task combinations"""
        try:
            total_tasks = 0
            
            # Simple pipeline definitions for lean implementation
            pipelines = {
                'NlpMcodePipeline': 'NLP_MCODE',
                'McodePipeline': 'DIRECT_MCODE'
            }
            
            # Sample prompts for each pipeline type
            sample_prompts = {
                'NLP_MCODE': ['basic_extraction → simple_mapping', 'detailed_extraction → complex_mapping'],
                'DIRECT_MCODE': ['direct_mcode_v1', 'direct_mcode_v2']
            }
            
            for pipeline_class, pipeline_type in pipelines.items():
                prompts = sample_prompts.get(pipeline_type, [])
                
                for prompt_name in prompts:
                    for model_key, model_info in self.available_models.items():
                        model_name = model_info.get('name', model_key)
                        
                        for trial_id in self.trial_data.keys():
                            gold_standard = self.gold_standard_data.get(trial_id, {})
                            
                            validation_entry = {
                                'prompt': prompt_name,
                                'model': model_name,
                                'trial': trial_id,
                                'status': 'Queued',
                                'details': 'Ready for execution',
                                'status_icon': '🔄',
                                'precision': None,
                                'recall': None,
                                'f1_score': None,
                                'duration_ms': None,
                                'token_usage': None,
                                'compliance_score': None,
                                'log': f'Queued {prompt_name} + {model_name} on {trial_id}',
                                'detailed_log': f'🕒 {prompt_name} + {model_name} + {trial_id}',
                                'error_message': '',
                                'pipeline_type': pipeline_type,
                                'live_logs': [],
                                'worker_id': 'N/A',
                                'optimization_parameters': {'metric': 'f1_score'},
                                'prompt_info': {'prompt_key': prompt_name},
                                'expected_entities': gold_standard.get("expected_extraction", {}).get("entities", []),
                                'expected_mappings': gold_standard.get("expected_mcode_mappings", {}).get("mapped_elements", [])
                            }
                            
                            task_id = str(uuid.uuid4())
                            self.active_validations[task_id] = validation_entry
                            total_tasks += 1
            
            # Update badge immediately
            if hasattr(self, 'preloaded_tasks_badge'):
                self.preloaded_tasks_badge.set_text(f'{total_tasks} Preloaded')
            
            ui.notify(f"Preloaded {total_tasks} tasks", type='positive')
            
        except Exception as e:
            ui.notify(f"Task preloading failed: {e}", type='negative')

    def _setup_ui(self):
        """Setup the main UI layout using NiceGUI's built-in features"""
        self._setup_header()
        self._setup_cache_stats_dialog() # Setup dialog first
        with ui.column().classes('w-full p-4 gap-4'):
            self._setup_benchmark_control_panel()
            self._setup_validation_display()
            self._setup_results_display()

    def _setup_header(self):
        """Sets up the header of the UI."""
        with ui.header().classes('bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 items-center shadow-lg'):
            with ui.row().classes('w-full justify-between items-center'):
                with ui.row().classes('items-center gap-4'):
                    ui.icon('dashboard', size='2rem').classes('text-white')
                    ui.label('Benchmark Task Tracker').classes('text-2xl font-bold')
                    # Status badges
                    with ui.row().classes('items-center gap-2 ml-4'):
                        self.worker_count_badge = ui.badge('0 Workers', color='green').props('floating').tooltip('Active workers')
                        self.active_tasks_badge = ui.badge('0 Active', color='orange').props('floating').tooltip('Tasks in progress')
                        self.completed_tasks_badge = ui.badge('0 Done', color='blue').props('floating').tooltip('Completed tasks')
                        self.preloaded_tasks_badge = ui.badge('0 Preloaded', color='purple').props('floating').tooltip('Preloaded Tasks')
                with ui.row().classes('items-center gap-2'):
                    ui.button(icon='dark_mode', on_click=self._toggle_dark_mode).props('flat round color=white').tooltip('Toggle dark mode')
                    ui.button(icon='refresh', on_click=self._reset_interface).props('flat round color=white').tooltip('Reset interface')
                    ui.button(icon='storage', on_click=self.display_cache_stats).props('flat round color=white').tooltip('Cache statistics')

    def _setup_cache_stats_dialog(self):
        """Create the dialog for displaying cache statistics."""
        with ui.dialog().props('max-width=600px') as self.cache_stats_dialog, ui.card():
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Cache Statistics').classes('text-xl font-bold')
                ui.icon('storage', size='1.5rem').classes('text-gray-500')
            ui.separator().classes('my-2')
            self.cache_stats_content = ui.html().classes('text-base leading-relaxed')
            ui.button('Close', on_click=self.cache_stats_dialog.close).props('color=primary no-caps').classes('mt-4 self-end')
    
    def _toggle_dark_mode(self):
        """Toggle dark mode"""
        self.dark_mode.toggle()
        ui.notify("Dark mode toggled")
    
    def _load_simple_libraries(self) -> None:
        """Load simple mock libraries for demonstration"""
        self.available_prompts = {
            "basic_extraction": {"name": "Basic Extraction", "prompt_type": "NLP_EXTRACTION"},
            "detailed_extraction": {"name": "Detailed Extraction", "prompt_type": "NLP_EXTRACTION"},
            "simple_mapping": {"name": "Simple Mapping", "prompt_type": "MCODE_MAPPING"},
            "complex_mapping": {"name": "Complex Mapping", "prompt_type": "MCODE_MAPPING"},
            "direct_mcode_v1": {"name": "Direct mCODE v1", "prompt_type": "DIRECT_MCODE"},
            "direct_mcode_v2": {"name": "Direct mCODE v2", "prompt_type": "DIRECT_MCODE"}
        }
        
        self.available_models = {
            "gpt-4o": {"name": "GPT-4o"},
            "claude-3-sonnet": {"name": "Claude 3 Sonnet"},
            "deepseek-coder": {"name": "DeepSeek Coder"},
            "llama-3.1-70b": {"name": "Llama 3.1 70B"}
        }
    
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
            else:
                pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    
    def _setup_benchmark_control_panel(self) -> None:
        """Setup the benchmark control panel with individual search and select boxes for each factor"""
        with ui.card().classes('w-full'):
            ui.label('Benchmark Control Panel').classes('text-xl font-semibold mb-6')
            
            # Global search for quick filtering
            with ui.card().classes('w-full mb-4'):
                ui.label('Quick Search').classes('text-lg font-semibold mb-3')
                with ui.row().classes('gap-4 items-center'):
                    ui.button('Clear All', icon='clear_all', on_click=self._clear_all_selections).props('color=negative outline').classes('px-4')
                    ui.button('Select All', icon='select_all', on_click=self._select_all_items).props('color=positive outline').classes('px-4')
            
            with ui.row().classes('w-full gap-4'):
                # Left side for pipeline and prompt selection
                with ui.column().classes('w-1/2'):
                    # Pipelines Selection with searchable select
                    with ui.card().classes('w-full mb-4'):
                        ui.label('Pipelines').classes('text-lg font-semibold mb-3')
                        pipeline_options = {'McodePipeline': 'Direct mCODE', 'NlpMcodePipeline': 'NLP + mCODE'}
                        self.pipeline_selection = ui.select(
                            pipeline_options,
                            label='Select Pipelines',
                            value=list(pipeline_options.keys()),
                            multiple=True,
                            with_input=True,
                            on_change=self._update_prompt_selection
                        ).classes('w-full')
                        self.pipeline_selection_badges = ui.row().classes('gap-2 flex-wrap mt-2')

                    # Prompts Selection with searchable select (hierarchy: pipeline -> prompt(s))
                    with ui.card().classes('w-full mb-4'):
                        ui.label('Prompts').classes('text-lg font-semibold mb-3')
                        self.prompt_container = ui.column().classes('w-full')
                        # Initialize prompt selection but don't set visibility yet
                        self.prompt_selection = ui.select(
                            {},
                            label='Select Prompts',
                            value=[],
                            multiple=True,
                            with_input=True,
                            on_change=self._update_selection_badges
                        ).classes('w-full')
                        self.prompt_selection.set_visibility(False)  # Initially hidden
                        self.prompt_selection_badges = ui.row().classes('gap-2 flex-wrap mt-2')

                # Right side for model and trial selection
                with ui.column().classes('w-1/2'):
                    # Models Selection with searchable select
                    with ui.card().classes('w-full mb-4'):
                        ui.label('Models').classes('text-lg font-semibold mb-3')
                        model_options = {k: v.get('name', k) for k, v in self.available_models.items()}
                        self.model_selection = ui.select(
                            model_options,
                            label='Select Models',
                            value=list(model_options.keys()),
                            multiple=True,
                            with_input=True,
                            on_change=self._update_selection_badges
                        ).classes('w-full')
                        self.model_selection_badges = ui.row().classes('gap-2 flex-wrap mt-2')
                    
                    # Trials Selection with searchable select
                    with ui.card().classes('w-full mb-4'):
                        ui.label('Clinical Trials').classes('text-lg font-semibold mb-3')
                        trial_options = {k: k for k in self.trial_data.keys()}
                        self.trial_selection = ui.select(
                            trial_options,
                            label='Select Trials',
                            value=list(trial_options.keys()),
                            multiple=True,
                            with_input=True,
                            on_change=self._update_selection_badges
                        ).classes('w-full')
                        self.trial_selection_badges = ui.row().classes('gap-2 flex-wrap mt-2')

            # Settings and Run controls
            with ui.card().classes('w-full mt-4'):
                with ui.row().classes('w-full justify-between items-center'):
                    with ui.column():
                        ui.label('Settings').classes('text-lg font-semibold mb-3')
                        with ui.row().classes('gap-4 items-center'):
                            metric_options = {'f1_score': 'F1 Score', 'precision': 'Precision', 'recall': 'Recall', 'compliance_score': 'Compliance Score'}
                            self.metric_selection = ui.select(metric_options, label='Optimization Metric', value='f1_score').classes('w-48')
                            self.top_n_selection = ui.number('Top N Combinations', value=5, min=1, max=20).classes('w-32')
                            self.concurrency_selection = ui.slider(min=1, max=10, value=5, step=1, on_change=self._update_worker_count).props('label-always color=primary').classes('w-48')
                        with ui.row().classes('gap-4 items-center mt-2'):
                            ui.label().bind_text_from(self.concurrency_selection, 'value', lambda v: f'Workers: {int(v)}').classes('text-sm')
                            self.worker_filter = ui.input(placeholder='Filter by worker ID...', on_change=self._update_validation_display.refresh).props('prepend-icon=filter_list').classes('w-64').tooltip('Filter tasks by worker ID')
                    
                    with ui.column().classes('items-end'):
                        self.total_tasks_label = ui.label('Total Tasks: 0').classes('text-lg font-semibold mb-2')
                        with ui.row().classes('gap-4 items-center'):
                            self.stop_benchmark_button = ui.button('Stop Benchmark', icon='stop').props('size=lg color=negative no-caps').classes('px-8')
                            self.run_benchmark_button = ui.button('Run Benchmark', icon='play_arrow').props('size=lg color=positive no-caps').classes('px-8')
                        self.benchmark_status = ui.label().classes('text-sm text-gray-500 mt-1')
            
            # Set initial visibility and handlers
            self.stop_benchmark_button.set_visibility(False)
            self.stop_benchmark_button.on('click', self._stop_benchmark)
            self.run_benchmark_button.on('click', self._run_benchmark)

            # Use NiceGUI's reactive data binding for initial selection
            # Set initial values directly using NiceGUI's reactive patterns
            
            # Use NiceGUI's on_change events for reactive filtering
            # The filtering will be handled automatically by the refreshable methods

    def _update_selection_badges(self):
        """Update the badge displays for all selection types"""
        # Update pipeline badges
        self.pipeline_selection_badges.clear()
        with self.pipeline_selection_badges:
            selected_pipelines = self.pipeline_selection.value or []
            for pipeline_key in selected_pipelines:
                pipeline_name = {'McodePipeline': 'Direct mCODE', 'NlpMcodePipeline': 'NLP + mCODE'}.get(pipeline_key, pipeline_key)
                ui.badge(pipeline_name, color='positive').classes('cursor-pointer').on('click', lambda _, key=pipeline_key: self._remove_selection('pipeline', key))
        
        # Update prompt badges if prompt selection exists
        if hasattr(self, 'prompt_selection_badges'):
            self.prompt_selection_badges.clear()
            with self.prompt_selection_badges:
                if hasattr(self, 'prompt_selection') and self.prompt_selection:
                    selected_prompts = self.prompt_selection.value or []
                    for prompt_key in selected_prompts:
                        # Extract prompt name from the composite key (pipeline:prompt_name)
                        if ':' in prompt_key:
                            prompt_name = prompt_key.split(':', 1)[1]
                        else:
                            prompt_name = prompt_key
                        ui.badge(prompt_name, color='positive').classes('cursor-pointer').on('click', lambda _, key=prompt_key: self._remove_selection('prompt', key))
        
        # Update model badges
        self.model_selection_badges.clear()
        with self.model_selection_badges:
            selected_models = self.model_selection.value or []
            for model_key in selected_models:
                model_name = self.available_models.get(model_key, {}).get('name', model_key)
                ui.badge(model_name, color='positive').classes('cursor-pointer').on('click', lambda _, key=model_key: self._remove_selection('model', key))
        
        # Update trial badges
        self.trial_selection_badges.clear()
        with self.trial_selection_badges:
            selected_trials = self.trial_selection.value or []
            for trial_key in selected_trials:
                ui.badge(trial_key, color='positive').classes('cursor-pointer').on('click', lambda _, key=trial_key: self._remove_selection('trial', key))
        
        # Update prompt-pipeline association badges in control panel
        self._update_prompt_pipeline_badges()
        
        # Update filtered tasks using NiceGUI's reactive patterns
        # The refreshable methods will handle the UI updates automatically
        self._update_pagination_controls.refresh()
        self._update_validation_display.refresh()

    def _remove_selection(self, selection_type: str, key: str):
        """Remove a selection from the specified select box"""
        if selection_type == 'pipeline':
            current_values = self.pipeline_selection.value or []
            new_values = [v for v in current_values if v != key]
            self.pipeline_selection.set_value(new_values)
            # When pipeline changes, update prompt selection
            self._update_prompt_selection()
        elif selection_type == 'prompt':
            current_values = self.prompt_selection.value or []
            new_values = [v for v in current_values if v != key]
            self.prompt_selection.set_value(new_values)
        elif selection_type == 'model':
            current_values = self.model_selection.value or []
            new_values = [v for v in current_values if v != key]
            self.model_selection.set_value(new_values)
        elif selection_type == 'trial':
            current_values = self.trial_selection.value or []
            new_values = [v for v in current_values if v != key]
            self.trial_selection.set_value(new_values)
        
        # Update badges after removal
        self._update_selection_badges()

    def _clear_all_selections(self):
        """Clear all selections by setting all select boxes to empty lists"""
        self.pipeline_selection.set_value([])
        self.prompt_selection.set_value([])
        self.model_selection.set_value([])
        self.trial_selection.set_value([])
        self._update_selection_badges()
        
    def _select_all_items(self):
        """Select all items by setting all select boxes to their full option lists"""
        pipeline_options = {'McodePipeline': 'Direct mCODE', 'NlpMcodePipeline': 'NLP + mCODE'}
        self.pipeline_selection.set_value(list(pipeline_options.keys()))
        
        # Update prompt selection based on pipeline hierarchy first
        self._update_prompt_selection()
        
        # After updating prompts, set all prompts as selected
        if hasattr(self, 'prompt_selection') and self.prompt_selection and hasattr(self.prompt_selection, 'options'):
            prompt_options = self.prompt_selection.options or {}
            if prompt_options:
                self.prompt_selection.set_value(list(prompt_options.keys()))

        model_options = {k: v.get('name', k) for k, v in self.available_models.items()}
        if hasattr(self, 'model_selection') and self.model_selection:
            self.model_selection.set_value(list(model_options.keys()))
        
        trial_options = {k: k for k in self.trial_data.keys()}
        self.trial_selection.set_value(list(trial_options.keys()))
        
        self._update_selection_badges()
        self._update_validation_display.refresh()
        self._update_pagination_controls.refresh()
        self._update_validation_display.refresh()
        self._update_pagination_controls.refresh()
        self._update_validation_display.refresh()
        self._update_pagination_controls.refresh()
    
    def _initialize_selections(self):
        """Initialize selections after UI setup"""
        self._select_all_items()
        if hasattr(self, 'preloaded_tasks_badge'):
            self.preloaded_tasks_badge.set_text(f'{len(self.active_validations)} Preloaded')
        
    def _update_prompt_pipeline_badges(self):
        """Update prompt-pipeline association badges in the control panel"""
        # Clear any existing prompt-pipeline badges
        if hasattr(self, 'prompt_pipeline_badges_container'):
            self.prompt_pipeline_badges_container.clear()
        else:
            # Create container for prompt-pipeline badges if it doesn't exist
            with ui.card().classes('w-full mt-4'):
                ui.label('Prompt-Pipeline Associations').classes('text-lg font-semibold mb-3')
                self.prompt_pipeline_badges_container = ui.column().classes('gap-2')
        
        # Get selected pipelines
        selected_pipelines = self.pipeline_selection.value or []
        if not selected_pipelines:
            with self.prompt_pipeline_badges_container:
                ui.label('Select pipelines to see prompt associations').classes('text-gray-500 text-sm')
            return
        
        # Load prompt-pipeline associations from prompts_config.json
        try:
            prompts_config_path = Path("prompts/prompts_config.json")
            if prompts_config_path.exists():
                with open(prompts_config_path, 'r') as f:
                    prompts_config = json.load(f)
                
                # Create mapping of prompt types to pipelines
                prompt_pipeline_map = {}
                
                # Process each prompt category
                for category_name, prompts_list in prompts_config.get('prompts', {}).items():
                    for prompt_info in prompts_list:
                        prompt_name = prompt_info.get('name')
                        prompt_type = prompt_info.get('prompt_type')
                        compatible_pipelines = prompt_info.get('compatible_pipelines', [])
                        
                        if isinstance(compatible_pipelines, str):
                            compatible_pipelines = [compatible_pipelines]
                        
                        # Only include prompts for selected pipelines
                        relevant_pipelines = [p for p in compatible_pipelines if p in selected_pipelines]
                        if relevant_pipelines:
                            if prompt_type not in prompt_pipeline_map:
                                prompt_pipeline_map[prompt_type] = {}
                            prompt_pipeline_map[prompt_type][prompt_name] = relevant_pipelines
                
                # Display badges organized by prompt type
                with self.prompt_pipeline_badges_container:
                    for prompt_type, prompts in prompt_pipeline_map.items():
                        with ui.row().classes('items-center gap-2'):
                            ui.badge(prompt_type, color='blue').props('outline')
                            ui.label('→').classes('text-gray-400')
                            for prompt_name, pipelines in prompts.items():
                                pipeline_names = []
                                for pipeline in pipelines:
                                    pipeline_name = {'McodePipeline': 'Direct mCODE', 'NlpMcodePipeline': 'NLP + mCODE'}.get(pipeline, pipeline)
                                    pipeline_names.append(pipeline_name)
                                
                                badge_text = f"{prompt_name} ({', '.join(pipeline_names)})"
                                ui.badge(badge_text, color='positive').props('dense')
            else:
                with self.prompt_pipeline_badges_container:
                    ui.label('prompts_config.json not found').classes('text-gray-500 text-sm')
        except Exception as e:
            with self.prompt_pipeline_badges_container:
                ui.label(f'Error loading prompt associations: {str(e)}').classes('text-red-500 text-sm')
    
    def _update_prompt_selection(self) -> None:
        """Update prompt selection based on pipeline filter - load prompts for selected pipelines"""
        selected_pipelines = self.pipeline_selection.value or []
        
        # Clear prompt container and update prompt options
        self.prompt_container.clear()
        with self.prompt_container:
            if not selected_pipelines:
                ui.label("Select pipelines to see available prompts.").classes('text-gray-500')
                self.prompt_selection.set_visibility(False)
            else:
                # Load prompts for selected pipelines
                prompt_options = {}
                for pipeline in selected_pipelines:
                    try:
                        # Get prompts organized by type for this pipeline
                        prompts_by_type = prompt_loader.get_prompts_by_pipeline(pipeline)
                        
                        # Collect all prompts for this pipeline
                        for prompt_type, prompts in prompts_by_type.items():
                            for prompt_info in prompts:
                                prompt_name = prompt_info.get('name')
                                prompt_key = f"{pipeline}:{prompt_name}"
                                prompt_options[prompt_key] = f"{prompt_name} ({pipeline})"
                    except Exception as e:
                        continue
                
                if prompt_options:
                    # Update prompt selection with available options
                    if hasattr(self, 'prompt_selection') and self.prompt_selection:
                        self.prompt_selection.set_options(prompt_options)
                        self.prompt_selection.set_value(list(prompt_options.keys()))  # Select all by default
                        self.prompt_selection.set_visibility(True)
                    
                    # Show pipeline info
                    ui.label(f"Selected {len(selected_pipelines)} pipeline(s)").classes('text-sm text-gray-600')
                    for pipeline in selected_pipelines:
                        ui.badge(pipeline, color='blue').props('outline').classes('mr-1 mb-1')
                else:
                    ui.label("No prompts available for selected pipelines.").classes('text-gray-500')
                    if hasattr(self, 'prompt_selection') and self.prompt_selection:
                        self.prompt_selection.set_visibility(False)

        # Update filtered tasks using NiceGUI's reactive patterns
        self.total_tasks_label.set_text(f"Total Tasks: {len(self.filtered_tasks)}")
        self._update_pagination_controls.refresh()
        self._update_validation_display.refresh()
    
    def _update_worker_count(self) -> None:
        """Update worker count in real-time based on concurrency slider"""
        worker_count = int(self.concurrency_selection.value)
        if self.worker_count_badge:
            self.worker_count_badge.set_text(f'{worker_count} Workers')
        
        # If benchmark is running, update the task queue
        if self.is_benchmark_running and self.task_queue:
            async def update_workers():
                await self.task_queue.update_worker_count(worker_count)
            background_tasks.create(update_workers)
            ui.notify(f'Updated to {worker_count} workers', type='info')
    
    def _setup_validation_display(self) -> None:
        """Setup unified task dashboard container with pagination and filters"""
        with ui.card().classes('w-full mt-4 shadow-lg'):
            with ui.row().classes('w-full justify-between items-center p-2'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('assignment', size='1.5rem').classes('text-blue-600')
                    ui.label('Task Dashboard').classes('text-xl font-bold')
                with ui.row().classes('items-center gap-2'):
                    ui.icon('update', size='1rem').classes('text-green-500').tooltip('Auto-refreshing')
                    ui.label('Live Updates').classes('text-sm text-gray-500')
            
            # Add filters and pagination controls
            self._setup_filters_and_pagination()
            
            ui.separator()
            # Initialize the validation display content
            self._update_validation_display()
            

    def _setup_filters_and_pagination(self) -> None:
        """Setup filtering and pagination controls using NiceGUI reactive patterns"""
        with ui.row().classes('w-full justify-between items-center p-2 bg-gray-50'):
            # Filters section
            with ui.row().classes('items-center gap-4'):
                # Status filter - store UI component reference
                with ui.row().classes('items-center gap-2'):
                    ui.label('Status:').classes('text-sm font-medium')
                    self.status_filter_component = ui.select(['all', 'pending', 'processing', 'success', 'failed'],
                             value=self.status_filter,
                             on_change=lambda e: self._on_filter_change('status', e.value)).classes('w-32')
                
                # Search filter - store UI component reference
                with ui.row().classes('items-center gap-2'):
                    ui.label('Search:').classes('text-sm font-medium')
                    self.search_filter_component = ui.input(placeholder='Filter tasks...',
                            value=self.search_filter,
                            on_change=lambda e: self._on_filter_change('search', e.value)).classes('w-40')
            
            # Pagination section
            with ui.row().classes('items-center gap-4'):
                # Page size selector
                with ui.row().classes('items-center gap-2'):
                    ui.label('Per page:').classes('text-sm font-medium')
                    self.page_size_selector = ui.select([5, 10, 20, 50],
                                                       value=self.page_size,
                                                       on_change=self._on_page_size_change).classes('w-20')
                
                # Page info - use reactive text binding
                self.page_info_label = ui.label().classes('text-sm text-gray-600')
        
        # Pagination widget - use reactive binding
        with ui.row().classes('w-full justify-center p-2'):
            self.pagination_widget = ui.pagination(1, 1, direction_links=True, on_change=self._on_page_change)
        

    def _on_filter_change(self, filter_type: str, value: str) -> None:
        """Handle filter changes"""
        if filter_type == 'status':
            self.status_filter = value
        elif filter_type == 'search':
            self.search_filter = value
        
        # Reset to first page when filters change
        self.current_page = 1
        self._update_pagination_controls.refresh()
        self._update_validation_display.refresh()

    def _on_page_size_change(self, e) -> None:
        """Handle page size changes using NiceGUI reactive patterns"""
        self.page_size = e.value
        self.current_page = 1  # Reset to first page
        # Use NiceGUI's reactive refresh
        self._update_pagination_controls.refresh()
        self._update_validation_display.refresh()

    def _on_page_change(self, e) -> None:
        """Handle page changes using NiceGUI reactive patterns"""
        self.current_page = e.value
        # Use NiceGUI's reactive refresh
        self._update_validation_display.refresh()

    def _update_filtered_tasks(self) -> None:
        """Update the filtered tasks list based on current filters using NiceGUI's reactive patterns"""
        all_tasks = list(self.active_validations.items())
        
        # Get current selections using NiceGUI's reactive value access
        selected_pipelines = self.pipeline_selection.value or []
        selected_prompts = self.prompt_selection.value or [] if hasattr(self, 'prompt_selection') else []
        selected_models = self.model_selection.value or []
        selected_trials = self.trial_selection.value or []
        worker_filter = self.worker_filter.value or '' if hasattr(self, 'worker_filter') and self.worker_filter else ''
        
        # Get current filter values from UI components
        status_filter = self.status_filter_component.value if hasattr(self, 'status_filter_component') else self.status_filter
        search_filter = self.search_filter_component.value if hasattr(self, 'search_filter_component') else self.search_filter
        
        filtered_tasks = []
        for task_id, validation in all_tasks:
            # DEBUG: Log each task being processed
            
            # Filter by pipeline - use pipeline-driven validation
            if selected_pipelines:
                actual_pipeline_type = validation.get('pipeline_type')
                
                # Check if any selected pipeline matches the actual pipeline type using pipeline-driven validation
                pipeline_matches = False
                for selected_pipeline in selected_pipelines:
                    # Get the actual pipeline type from the pipeline class name
                    pipeline_type = self._get_pipeline_type_from_class(selected_pipeline)
                    
                    # Handle the mismatch: validation stores class names, but filtering expects abbreviated types
                    # Also check if the selected pipeline class name matches the stored pipeline type directly
                    if pipeline_type == actual_pipeline_type or selected_pipeline == actual_pipeline_type:
                        pipeline_matches = True
                        break
                
                if not pipeline_matches:
                    continue
            
            # Filter by prompt - use pipeline-driven validation with the enhanced validation method
            if selected_prompts:
                task_prompt = validation.get('prompt', '')
                task_pipeline = validation.get('pipeline_type', '')
                
                # Check if prompt is valid for the pipeline using the enhanced validation method
                prompt_matches = False
                for prompt_key in selected_prompts:
                    if ':' in prompt_key:
                        pipeline_part, prompt_name = prompt_key.split(':', 1)
                        # Get the actual pipeline type from the pipeline class name
                        pipeline_type = self._get_pipeline_type_from_class(pipeline_part)
                        
                        # Use the enhanced validation method that handles composite prompts
                        if (pipeline_type == task_pipeline or pipeline_part == task_pipeline) and self._is_prompt_valid_for_pipeline(task_prompt, pipeline_type):
                            prompt_matches = True
                            break
                    else:
                        # For composite prompts, check if the composite prompt name is valid for the pipeline
                        if self._is_prompt_valid_for_pipeline(task_prompt, task_pipeline):
                            prompt_matches = True
                            break
                
                if not prompt_matches:
                    continue
            
            # Filter by model - only apply if models are selected
            if selected_models:
                model_key = next((k for k, v in self.available_models.items()
                                if v.get('name') == validation.get('model')), None)
                if model_key not in selected_models:
                    continue
            
            # Filter by trial - only apply if trials are selected
            if selected_trials and validation.get('trial') not in selected_trials:
                continue
            
            # Filter by worker
            if worker_filter and worker_filter.lower() not in str(validation.get('worker_id', '')).lower():
                continue
            
            # Apply status filter
            if status_filter != 'all':
                status_map = {
                    'pending': 'Queued',
                    'processing': 'Processing',
                    'success': 'Success',
                    'failed': 'Failed'
                }
                target_status = status_map.get(status_filter, status_filter)
                if validation.get('status', '').lower() != target_status.lower():
                    continue
            
            # Apply search filter
            if search_filter:
                search_lower = search_filter.lower()
                if not (search_lower in validation.get('prompt', '').lower() or
                        search_lower in validation.get('model', '').lower() or
                        search_lower in validation.get('trial', '').lower() or
                        search_lower in task_id.lower()):
                    continue
            
            filtered_tasks.append((task_id, validation))

        self.filtered_tasks = filtered_tasks
        
        # Update total tasks count in UI
        if hasattr(self, 'total_tasks_label') and self.total_tasks_label:
            self.total_tasks_label.set_text(f"Total Tasks: {len(self.filtered_tasks)}")
    
    def _get_pipeline_type_from_class(self, pipeline_class_name: str) -> str:
        """Get the pipeline type from a pipeline class name using pipeline-driven validation"""
        # Map pipeline class names to their corresponding pipeline types
        pipeline_type_mapping = {
            'McodePipeline': 'DIRECT_MCODE',
            'NlpMcodePipeline': 'NLP_MCODE'
        }
        
        return pipeline_type_mapping.get(pipeline_class_name, pipeline_class_name)
    
    def _is_prompt_valid_for_pipeline(self, prompt_name: str, pipeline_type: str) -> bool:
        """Check if a prompt is valid for a given pipeline type using pipeline-driven validation"""
        try:
            # Load prompts config to check compatibility
            prompts_config_path = Path("prompts/prompts_config.json")
            if not prompts_config_path.exists():
                return True  # Allow all if config not found
            
            with open(prompts_config_path, 'r') as f:
                prompts_config = json.load(f)
            
            # Find the prompt in config
            for category_name, prompts_list in prompts_config.get('prompts', {}).items():
                for prompt_info in prompts_list:
                    if prompt_info.get('name') == prompt_name or prompt_name in prompt_info.get('name', ''):
                        compatible_pipelines = prompt_info.get('compatible_pipelines', [])
                        if isinstance(compatible_pipelines, str):
                            compatible_pipelines = [compatible_pipelines]
                        
                        # Check if pipeline type matches any compatible pipeline
                        pipeline_map = {
                            'DIRECT_MCODE': ['McodePipeline'],
                            'NLP_MCODE': ['NlpMcodePipeline']
                        }
                        
                        target_pipelines = pipeline_map.get(pipeline_type, [])
                        return any(p in compatible_pipelines for p in target_pipelines)
            
            # For composite prompts (containing →), validate both parts
            if '→' in prompt_name:
                parts = prompt_name.split('→')
                if len(parts) == 2:
                    return pipeline_type == 'NLP_MCODE'  # Composite prompts are for NLP pipeline
            
            return True  # Default to allow if not found
        except Exception:
            return True  # Default to allow on error

    @ui.refreshable
    def _update_pagination_controls(self) -> None:
        """Update pagination controls based on filtered tasks"""
        self._update_filtered_tasks()
        
        total_tasks = len(self.filtered_tasks)
        total_pages = max(1, (total_tasks + self.page_size - 1) // self.page_size)
        
        # DEBUG: Log pagination state
        
        # Update pagination widget
        if hasattr(self, 'pagination_widget') and self.pagination_widget:
            self.pagination_widget.min = 1
            self.pagination_widget.max = total_pages
            self.pagination_widget.value = min(self.current_page, total_pages)
        
        # Update page info label with proper pagination info
        if hasattr(self, 'page_info_label') and self.page_info_label:
            start_idx = (self.current_page - 1) * self.page_size + 1
            end_idx = min(self.current_page * self.page_size, total_tasks)
            page_text = f"Page {self.current_page} of {total_pages} ({total_tasks} tasks)"
            if total_tasks > 0:
                page_text += f" - Showing {start_idx}-{end_idx}"
            self.page_info_label.text = page_text
            
        # DEBUG: Log actual pagination widget state
        if hasattr(self, 'pagination_widget') and self.pagination_widget:
    
                pass


    

    

    
    @ui.refreshable
    def _update_validation_display(self) -> None:
        """Update validation display with pagination using NiceGUI's reactive patterns"""
        # Get tasks from active validations (populated during framework initialization)
        if not self.active_validations:
            with ui.row().classes('w-full justify-center p-8'):
                ui.icon('hourglass_empty', size='3rem').classes('text-gray-300')
                ui.label('Loading tasks... Please wait for framework initialization to complete.').classes('text-gray-500 text-lg ml-4')
            return
        
        # Ensure filtered_tasks is populated
        self._update_filtered_tasks()
        
        # Get current page tasks
        start_idx = (self.current_page - 1) * self.page_size
        end_idx = start_idx + self.page_size
        current_page_tasks = self.filtered_tasks[start_idx:end_idx] if self.filtered_tasks else []
        
        if not current_page_tasks:
            with ui.row().classes('w-full justify-center p-8'):
                ui.icon('inbox', size='3rem').classes('text-gray-300')
                ui.label('No tasks match current filters').classes('text-gray-500 text-lg ml-4')
            
            # Notify user when no tasks match filters
            if not self.filtered_tasks:
                ui.notify('No tasks match the current filter criteria.', type='warning')
            return
        
        # Count statuses from all filtered tasks (not just current page)
        all_filtered_validations = {task_id: validation for task_id, validation in self.filtered_tasks}
        active_count = sum(1 for v in all_filtered_validations.values() if v.get('status') in ['Queued', 'Processing'])
        success_count = sum(1 for v in all_filtered_validations.values() if v.get('status') == 'Success')
        failed_count = sum(1 for v in all_filtered_validations.values() if v.get('status') == 'Failed')
        
        # Display stats
        with ui.row().classes('w-full justify-between items-center mb-4 p-3 bg-gray-50 rounded-lg'):
            with ui.row().classes('items-center gap-6'):
                with ui.row().classes('items-center gap-1'):
                    ui.badge(f'{active_count}', color='orange').props('floating')
                    ui.label('Active').classes('text-sm font-medium')
                with ui.row().classes('items-center gap-1'):
                    ui.badge(f'{success_count}', color='green').props('floating')
                    ui.label('Success').classes('text-sm font-medium')
                with ui.row().classes('items-center gap-1'):
                    ui.badge(f'{failed_count}', color='red').props('floating')
                    ui.label('Failed').classes('text-sm font-medium')
            with ui.row().classes('items-center gap-2'):
                # Worker activity from live tasks
                worker_counts = {}
                for validation in all_filtered_validations.values():
                    worker_id = validation.get('worker_id')
                    if worker_id and worker_id != 'N/A' and validation.get('status') == 'Processing':
                        worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
                
                if worker_counts:
                    ui.label('Active Workers:').classes('text-sm font-medium')
                    for worker_id, count in sorted(worker_counts.items()):
                        ui.badge(f'W{worker_id}: {count}', color='orange').props('dense').classes('text-xs')
                
                if active_count > 0:
                    ui.spinner('dots', color='orange', size='sm')
                    ui.label(f'Processing {active_count} tasks...').classes('text-sm text-orange-600 font-medium')
                else:
                    ui.icon('check_circle', color='green', size='sm')
                    ui.label('All tasks completed').classes('text-sm text-green-600 font-medium')
        
        # Color legend for badge types
        with ui.row().classes('w-full items-center gap-6 mb-3 p-2 bg-white rounded border'):
            ui.label('Badge Colors:').classes('text-xs font-medium text-gray-600')
            with ui.row().classes('items-center gap-4'):
                with ui.row().classes('items-center gap-1'):
                    ui.badge('Prompts', color='blue').props('dense outline')
                with ui.row().classes('items-center gap-1'):
                    ui.badge('Models', color='purple').props('dense outline')
                with ui.row().classes('items-center gap-1'):
                    ui.badge('Trials', color='teal').props('dense outline')
                with ui.row().classes('items-center gap-1'):
                    ui.badge('Pipelines', color='amber').props('dense outline')
        
        # Display individual task cards for current page only
        for task_id, validation in current_page_tasks:
            self._create_individual_task_card(task_id, validation)
    
    @ui.refreshable
    def _create_individual_task_card(self, task_id: str, validation: Dict[str, Any]) -> None:
        """Create individual task card with pure NiceGUI refreshable implementation"""
        status = validation.get('status', 'Unknown')
        
        # Card styling based on status
        card_styles = {
            'Success': ('border-l-4 border-green-500 bg-green-50', 'check_circle', 'green'),
            'Failed': ('border-l-4 border-red-500 bg-red-50', 'error', 'red'),
            'Processing': ('border-l-4 border-orange-500 bg-orange-50', 'sync', 'orange'),
            'Queued': ('border-l-4 border-orange-500 bg-orange-50', 'hourglass_empty', 'orange')
        }
        
        card_class, icon, icon_color = card_styles.get(status, ('border-l-4 border-blue-500 bg-blue-50', 'info', 'blue'))
        
        with ui.card().classes(f'w-full {card_class} p-3 mb-2'):
            with ui.row().classes('w-full items-center gap-4'):
                # Status icon with spinner for active tasks
                if status in ['Processing', 'Queued']:
                    ui.spinner('dots', color=icon_color, size='sm')
                else:
                    ui.icon(icon, color=icon_color, size='1.2rem')
                
                # Task info with badges
                with ui.column().classes('flex-grow'):
                    with ui.row().classes('items-center gap-2'):
                        ui.badge(validation.get('prompt', 'N/A'), color='blue').props('outline')
                        ui.label('+').classes('text-gray-400')
                        ui.badge(validation.get('model', 'N/A'), color='purple').props('outline')
                        ui.label('on').classes('text-gray-400')
                        ui.badge(validation.get('trial', 'N/A'), color='teal').props('outline')
                        
                        # Pipeline badge
                        if validation.get('pipeline_type'):
                            ui.badge(validation['pipeline_type'], color='amber').props('dense')
                    
                    # Results or status
                    if status == 'Success':
                        with ui.row().classes('items-center gap-4 mt-1'):
                            ui.label(f"F1: {validation.get('f1_score', 'N/A')}").classes('text-sm font-bold text-green-600')
                            ui.label(f"P: {validation.get('precision', 'N/A')}").classes('text-sm')
                            ui.label(f"R: {validation.get('recall', 'N/A')}").classes('text-sm')
                            if validation.get('duration_ms') != 'N/A':
                                ui.label(f"⏱ {validation['duration_ms']}ms").classes('text-sm text-gray-500')
                    elif status == 'Failed':
                        error_msg = validation.get('error_message', 'Unknown error')
                        ui.label(error_msg[:80] + '...' if len(error_msg) > 80 else error_msg).classes('text-sm text-red-600')
                    else:
                        ui.label(validation.get('details', 'Processing...')).classes('text-sm text-gray-600')
                
                # Worker badge
                worker_id = validation.get('worker_id', 'N/A')
                if worker_id != 'N/A':
                    ui.badge(f'{worker_id}', color='info').props('floating')
            
            # Live logs section
            live_logs = validation.get('live_logs', [])
            if live_logs:
                with ui.column().classes('w-full mt-2'):
                    ui.label('Live Logs').classes('text-sm font-bold text-gray-700')
                    with ui.card().classes('w-full bg-gray-900 text-white p-2'):
                        log_text = '\n'.join(live_logs[-5:])  # Show last 5 entries
                        ui.code(log_text).classes('text-xs font-mono')




    

    

    
    def _setup_results_display(self) -> None:
        """Setup results display area"""
        with ui.card().classes('w-full mt-4'):
            ui.label('Benchmark Results').classes('text-lg font-semibold mb-2')
            # Use refreshable function directly
            self._update_results_display()
    
    def _run_benchmark(self) -> None:
        """Start processing the preloaded benchmark tasks using centralized pipeline"""
        
        if self.is_benchmark_running:
            self.benchmark_status_text = "Benchmark is already running"
            return

        # Get selections from the new select boxes
        selected_models = self.model_selection.value or []
        selected_trials = self.trial_selection.value or []
        selected_pipelines = self.pipeline_selection.value or []
        
        
        if not (selected_models and selected_trials and selected_pipelines):
            self.benchmark_status_text = "Please select at least one model, trial, and pipeline."
            return
        
        # Filter existing tasks to only process selected combinations using new select boxes
        # First update filtered tasks based on current selections
        self._update_filtered_tasks()
        
        # Collect all filtered task IDs
        tasks_to_process = [task_id for task_id, _ in self.filtered_tasks]
        
        
        if not tasks_to_process:
            self.benchmark_status_text = "No tasks match your selection criteria."
            return
        
        # Strict NiceGUI implementation - these should never be None
        self.run_benchmark_button.disable()
        self.stop_benchmark_button.set_visibility(True)
        self.is_benchmark_running = True
        self.benchmark_cancelled = False
        
        
        def execute_benchmark():
            """Execute benchmark using pure NiceGUI background_tasks"""
            try:
                # Start processing in background
                background_tasks.create(self._process_selected_tasks(tasks_to_process))
            except Exception as e:
                logging.error(f"Benchmark failed: {str(e)}")
                self.run_benchmark_button.enable()
                self.stop_benchmark_button.set_visibility(False)
                self.is_benchmark_running = False
                ui.notify(f"Benchmark failed: {str(e)}", type='negative')
        
        execute_benchmark()
        ui.notify(f"Starting benchmark with {len(tasks_to_process)} tasks", type='positive')
        self.benchmark_status_text = "Benchmark running..."
    
    async def _process_selected_tasks(self, task_ids: List[str]):
        """Process selected tasks with pure NiceGUI implementation"""
        try:
            # Initialize task queue
            concurrency = int(self.concurrency_selection.value)
            self.task_queue = await initialize_task_queue(max_workers=concurrency)
            
            self.total_tasks_count = len(task_ids)
            
            # Create and queue tasks
            for i, task_id in enumerate(task_ids):
                if self.benchmark_cancelled:
                    break
                    
                if task_id in self.active_validations:
                    validation = self.active_validations[task_id]
                    validation['status'] = 'Processing'
                    validation['worker_id'] = f'W-{(i % concurrency) + 1}'
                    
                    # Create task for queue
                    task = BenchmarkTask(
                        task_id=task_id,
                        prompt_name=validation['prompt'],
                        model_name=validation['model'],
                        trial_id=validation['trial'],
                        trial_data=self.trial_data.get(validation['trial'], {}),
                        expected_entities=validation.get('expected_entities', []),
                        expected_mappings=validation.get('expected_mappings', []),
                        pipeline_type=validation.get('pipeline_type', 'NLP_MCODE')
                    )
                    
                    await self.task_queue.add_task(task, self._task_completion_callback)
            
            # Start workers and wait for completion
            await self.task_queue.start_workers()
            
            # Monitor progress with @ui.refreshable updates
            while not self.benchmark_cancelled and self.task_queue.completed_tasks < self.total_tasks_count:
                await asyncio.sleep(0.5)
                # UI updates happen automatically via @ui.refreshable
            
            await shutdown_task_queue()
            
        finally:
            self.run_benchmark_button.enable()
            self.stop_benchmark_button.set_visibility(False)
            self.is_benchmark_running = False
            if not self.benchmark_cancelled:
                ui.notify("Benchmark completed", type='positive')
    
    def _task_completion_callback(self, task: BenchmarkTask) -> None:
        """Callback for task completion - pure data updates only"""
        # Update validation data only (no UI calls)
        if task.task_id in self.active_validations:
            validation = self.active_validations[task.task_id]
            status = 'Success' if task.status == TaskStatus.SUCCESS else 'Failed'
            
            validation.update({
                'status': status,
                'details': f"F1={task.f1_score:.3f}" if task.status == TaskStatus.SUCCESS and task.f1_score is not None else task.error_message,
                'status_icon': '✅' if status == 'Success' else '❌',
                'precision': f"{task.precision:.3f}" if task.precision is not None else 'N/A',
                'recall': f"{task.recall:.3f}" if task.recall is not None else 'N/A',
                'f1_score': f"{task.f1_score:.3f}" if task.f1_score is not None else 'N/A',
                'compliance_score': f"{task.compliance_score:.2%}" if task.compliance_score is not None else 'N/A',
                'duration_ms': f"{task.duration_ms:.1f}" if task.duration_ms else 'N/A',
                'token_usage': str(task.token_usage) if task.token_usage else 'N/A',
                'error_message': task.error_message or ''
            })
            
            # Add live log entry
            worker_info = f" (Worker: {task.worker_id})" if task.worker_id else ""
            live_log_entry = f"🏁 {status}: {task.prompt_name} + {task.model_name}{worker_info}"
            validation['live_logs'].append(live_log_entry)
            
            # Keep only last 10 log entries
            if len(validation['live_logs']) > 10:
                validation['live_logs'] = validation['live_logs'][-10:]
        
        # Update completed tasks count for progress tracking
        self.completed_tasks_count = self.task_queue.completed_tasks if self.task_queue else 0
        
        # Store benchmark result
        self.benchmark_results.append(task)
    
    def _stop_benchmark(self) -> None:
        """Stop benchmark execution"""
        if not self.is_benchmark_running:
            ui.notify("No benchmark is currently running", type='warning')
            return
        
        self.benchmark_cancelled = True
        ui.notify("Stopping benchmark execution...", type='info')
        
        # Stop queue in background
        def stop_queue():
            if self.task_queue:
                asyncio.create_task(shutdown_task_queue())
        
        background_tasks.create(stop_queue)
    
    def _reset_interface(self) -> None:
        """Reset the interface to initial state"""
        if self.is_benchmark_running:
            self.benchmark_cancelled = True
            self.is_benchmark_running = False
            self.run_benchmark_button.enable()
            self.stop_benchmark_button.set_visibility(False)
            
            def cleanup():
                if self.task_queue:
                    asyncio.create_task(shutdown_task_queue())
            
            background_tasks.create(cleanup)
        
        # Reset data
        self.benchmark_results = []
        self.validation_results = []
        self.benchmark_cancelled = False
        self.active_validations.clear()
        
        # Reload tasks
        self._preload_all_benchmark_tasks()
        self._select_all_items()
        
        ui.notify("Interface reset completed", type='positive')

    @ui.refreshable
    def _update_results_display(self) -> None:
        """Update the results display with simple aggregated data"""
        if not self.benchmark_results:
            ui.label("No results yet.").classes('text-gray-500 italic')
            return

        # Simple aggregation without pandas
        successful_results = [r for r in self.benchmark_results if r.status == TaskStatus.SUCCESS]
        
        if not successful_results:
            ui.label("No successful results yet.").classes('text-gray-500 italic')
            return

        # Group by prompt and model
        grouped = {}
        for result in successful_results:
            key = (result.prompt_name, result.model_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Calculate averages
        summary = []
        for (prompt, model), results in grouped.items():
            avg_f1 = sum(r.f1_score for r in results if r.f1_score) / len(results)
            avg_precision = sum(r.precision for r in results if r.precision) / len(results)
            avg_recall = sum(r.recall for r in results if r.recall) / len(results)
            avg_duration = sum(r.duration_ms for r in results if r.duration_ms) / len(results)
            summary.append((prompt, model, avg_f1, avg_precision, avg_recall, avg_duration))
        
        # Sort by F1 score
        summary.sort(key=lambda x: x[2], reverse=True)
        
        top_n = min(self.top_n_selection.value if hasattr(self, 'top_n_selection') else 5, len(summary))
        
        ui.label(f"Top {top_n} Benchmark Combinations").classes('text-md font-semibold')
        with ui.grid(columns=6).classes('w-full gap-2'):
            for col in ['Prompt', 'Model', 'F1 Score', 'Precision', 'Recall', 'Avg. Time (ms)']:
                ui.label(col).classes('font-bold')
            for prompt, model, f1, precision, recall, duration in summary[:top_n]:
                ui.label(prompt)
                ui.label(model)
                ui.label(f"{f1:.3f}")
                ui.label(f"{precision:.3f}")
                ui.label(f"{recall:.3f}")
                ui.label(f"{duration:.1f}")

    def display_cache_stats(self) -> None:
        """Display simple cache statistics"""
        content = """
        <div style="font-family: sans-serif;">
            <h3 style="font-weight: bold; border-bottom: 1px solid #ccc; padding-bottom: 5px;">Cache Status</h3>
            <p><strong>API Cache:</strong> Mock data - 150 entries, 2.3 MB</p>
            <p><strong>LLM Cache:</strong> Mock data - 89 entries, 1.7 MB</p>
            <p><strong>Status:</strong> Operational</p>
        </div>
        """
        
        self.cache_stats_content.set_content(content)
        self.cache_stats_dialog.open()





def run_benchmark_task_tracker(port: int = 8089):
    """Run the benchmark task tracker UI"""
    tracker = BenchmarkTaskTrackerUI()
    
    # Initialize selections after UI is ready
    tracker._initialize_selections()
    
    ui.run(title='Benchmark Task Tracker', port=port, reload=True)


if __name__ in {"__main__", "__mp_main__"}:
    import argparse
    parser = argparse.ArgumentParser(description='Run Benchmark Task Tracker')
    parser.add_argument('--port', type=int, default=8089, help='Port to run the UI on')
    args = parser.parse_args()
    run_benchmark_task_tracker(args.port)