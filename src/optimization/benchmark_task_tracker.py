"""
Benchmark Task Tracker - Extended implementation using pure NiceGUI events, binding, and state management.
Integrates with mcode-optimize framework for benchmark validation tasks.
"""

import sys
import os
# Add the parent directory to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import random
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

# Import existing components
from src.optimization.benchmark_task import BenchmarkTask
from src.optimization.strict_prompt_optimization_framework import (
    StrictPromptOptimizationFramework, PromptType,
    APIConfig, PromptVariant
)

# Import utilities
from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader
from src.utils.logging_config import get_logger


class BenchmarkTaskTrackerUI:
    """Extended UI for benchmark task tracking with mcode-optimize integration"""
    
    def __init__(self):
        # Initialize the framework first
        self.framework = StrictPromptOptimizationFramework()
        self.available_prompts = {}
        self.available_models = {}
        self.trial_data = {}
        self.gold_standard_data = {}
        
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
        
        # Benchmark state
        self.is_benchmark_running = False
        self.benchmark_cancelled = False
        self.benchmark_results = []
        self.validation_results = []
        self.preloaded_validations = []  # Store preloaded validations
        
        # Dark mode
        self.dark_mode = ui.dark_mode()
        
        # Load libraries and data
        self._load_libraries()
        self._load_test_data()
        
        # Setup UI components
        self._setup_ui()
    
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
                    
                    # Add test cases to the framework
                    for case_id, case_data in self.trial_data.items():
                        self.framework.add_test_case(case_id, case_data)
            
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
    
    def _setup_benchmark_components(self) -> None:
        """Setup additional benchmark-specific UI components"""
        # These will be added to the UI in the setup_ui method
        pass
    
    def _update_statistics(self):
        """Override parent class method to prevent updating non-existent UI elements"""
        # Intentionally left empty to prevent updating statistics UI elements
        # that were removed from the interface
        pass

    def setup_ui(self):
        """Setup the main UI layout with benchmark components"""
        # Setup the base UI
        with ui.header().classes('bg-primary text-white p-4 items-center'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Benchmark Task Tracker').classes('text-2xl font-bold')
                with ui.row().classes('items-center'):
                    ui.button('Toggle Dark Mode', on_click=self._toggle_dark_mode).props('flat color=white')
                    ui.button('Reset', on_click=self._reset_interface).props('flat color=white')
        
        with ui.column().classes('w-full p-4 gap-4'):
            with ui.card().classes('w-full'):
                self._setup_benchmark_control_panel()
            self._setup_validation_display()
            self._setup_results_display()
            self._setup_live_logger()
    
    def _setup_benchmark_control_panel(self) -> None:
        """Setup the benchmark control panel"""
        with ui.card().classes('w-full'):
            ui.label('Benchmark Control Panel').classes('text-lg font-semibold mb-4')
            
            with ui.row().classes('w-full gap-4'):
                # Prompt selection - select all by default
                prompt_options = {key: f"{info.get('name', key)} ({info.get('prompt_type', 'Unknown')})"
                                 for key, info in self.available_prompts.items()}
                self.prompt_selection = ui.select(
                    prompt_options,
                    label='Select Prompts',
                    multiple=True,
                    value=list(prompt_options.keys()),  # Select all by default
                    on_change=lambda: self._update_validation_display()
                ).classes('w-1/3')
                
                # Model selection - select all by default
                model_options = {key: info.get('name', key) for key, info in self.available_models.items()}
                self.model_selection = ui.select(
                    model_options,
                    label='Select Models',
                    multiple=True,
                    value=list(model_options.keys()),  # Select all by default
                    on_change=lambda: self._update_validation_display()
                ).classes('w-1/3')
                
                # Trial selection - select all by default
                trial_options = {key: key for key in self.trial_data.keys()}
                self.trial_selection = ui.select(
                    trial_options,
                    label='Select Trials',
                    multiple=True,
                    value=list(trial_options.keys()),  # Select all by default
                    on_change=lambda: self._update_validation_display()
                ).classes('w-1/3')
            
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
                self.concurrency_selection = ui.number('Concurrency Level', value=1, min=1, max=10).classes('w-1/4')
            
            with ui.row().classes('w-full gap-2 mt-4'):
                self.run_benchmark_button = ui.button('Run Benchmark', on_click=self._run_benchmark).props('icon=play_arrow color=positive')
                self.stop_benchmark_button = ui.button('Stop Benchmark', on_click=self._stop_benchmark).props('icon=stop color=negative')
                self.stop_benchmark_button.set_visibility(False)
                
                self.benchmark_status = ui.label('Ready to run benchmark').classes('self-center ml-4')
            
            self.benchmark_progress = ui.linear_progress(0).classes('w-full mt-2')
    
    def _setup_live_logger(self) -> None:
        """Setup live logging display"""
        with ui.card().classes('w-full mt-4'):
            ui.label('Live Logging').classes('text-lg font-semibold mb-2')
            self.live_log_display = ui.log().classes('w-full h-48')
    
    def _setup_validation_display(self) -> None:
        """Setup validation results display"""
        with ui.card().classes('w-full mt-4'):
            with ui.row().classes('w-full justify-between items-center'):
                ui.label('Validation Results').classes('text-lg font-semibold')
                ui.label('Updated in real-time as benchmarks run').classes('text-sm text-gray-500')
            self.validation_display = ui.column().classes('w-full')
            # Load initial validation results
            self._load_initial_validations()
    
    def _load_initial_validations(self) -> None:
        """Preload all validations before running benchmarks"""
        # Preload all possible validation combinations
        self.preloaded_validations = []
        
        # Get all possible combinations of prompts, models, and trials
        prompt_keys = list(self.available_prompts.keys())
        model_keys = list(self.available_models.keys())
        trial_ids = list(self.trial_data.keys())
        
        # Create preloaded validation entries for all combinations
        for prompt_key in prompt_keys:
            prompt_info = self.available_prompts.get(prompt_key)
            prompt_name = prompt_info.get('name', prompt_key) if prompt_info else prompt_key
            
            for model_key in model_keys:
                model_info = self.available_models.get(model_key)
                model_name = model_info.get('name', model_key) if model_info else model_key
                
                for trial_id in trial_ids:
                    # Add preloaded validation with "Pending" status
                    self.preloaded_validations.append({
                        'prompt': prompt_name,
                        'model': model_name,
                        'trial': trial_id,
                        'status': 'Pending',
                        'details': 'Waiting to run',
                        'status_icon': '🔵',  # Blue circle for pending
                        'precision': '-',
                        'recall': '-',
                        'f1_score': '-',
                        'duration_ms': '-',
                        'token_usage': '-'
                    })
        
        # Update display with preloaded validations
        self._update_validation_display()
    
    def _update_preloaded_validation_status(self, prompt_name: str, model_name: str, trial_id: str,
                                           status: str, details: str, status_icon: str,
                                           precision: str = '-', recall: str = '-', f1_score: str = '-',
                                           duration_ms: str = '-', token_usage: str = '-') -> None:
        """Update the status of a preloaded validation"""
        for validation in self.preloaded_validations:
            if (validation['prompt'] == prompt_name and
                validation['model'] == model_name and
                validation['trial'] == trial_id):
                validation['status'] = status
                validation['details'] = details
                validation['status_icon'] = status_icon
                validation['precision'] = precision
                validation['recall'] = recall
                validation['f1_score'] = f1_score
                validation['duration_ms'] = duration_ms
                validation['token_usage'] = token_usage
                break
    
    def _auto_run_all_validations(self) -> None:
        """Auto-run all validations with all options selected by default"""
        # Select all options by default
        if self.prompt_selection:
            self.prompt_selection.set_value(list(self.available_prompts.keys()))
        if self.model_selection:
            self.model_selection.set_value(list(self.available_models.keys()))
        if self.trial_selection:
            self.trial_selection.set_value(list(self.trial_data.keys()))
        
        # Display initial message
        with self.validation_display:
            ui.label('Ready to run all validations. Click "Run Benchmark" to start.').classes('text-gray-500 italic')
    
    def _update_validation_display(self) -> None:
        """Update validation display based on current filters"""
        if not self.validation_display:
            return
            
        # Clear current display
        self.validation_display.clear()
        
        # Get current selections
        selected_prompts = self.prompt_selection.value or []
        selected_models = self.model_selection.value or []
        selected_trials = self.trial_selection.value or []
        
        # Use preloaded validations if available, otherwise use validation results
        validation_data = self.preloaded_validations if self.preloaded_validations else self.validation_results
        
        # Filter validation results based on selections
        filtered_results = validation_data
        if selected_prompts:
            filtered_results = [r for r in filtered_results if r.get('prompt') in selected_prompts]
        if selected_models:
            filtered_results = [r for r in filtered_results if r.get('model') in selected_models]
        if selected_trials:
            filtered_results = [r for r in filtered_results if r.get('trial') in selected_trials]
        
        # Display filtered results
        with self.validation_display:
            if not filtered_results:
                ui.label('No validation results match current filters.').classes('text-gray-500 italic')
            else:
                # Display summary
                ui.label(f'Showing {len(filtered_results)} validation results').classes('text-sm text-gray-500 mb-2')
                
                # Display results in a table with status icons and benchmark fields
                columns = [
                    {'name': 'status_icon', 'label': '', 'field': 'status_icon', 'sortable': True},  # Status icon column
                    {'name': 'prompt', 'label': 'Prompt', 'field': 'prompt', 'sortable': True},
                    {'name': 'model', 'label': 'Model', 'field': 'model', 'sortable': True},
                    {'name': 'trial', 'label': 'Trial', 'field': 'trial', 'sortable': True},
                    {'name': 'precision', 'label': 'Precision', 'field': 'precision', 'sortable': True},
                    {'name': 'recall', 'label': 'Recall', 'field': 'recall', 'sortable': True},
                    {'name': 'f1_score', 'label': 'F1 Score', 'field': 'f1_score', 'sortable': True},
                    {'name': 'duration_ms', 'label': 'Time (ms)', 'field': 'duration_ms', 'sortable': True},
                    {'name': 'token_usage', 'label': 'Tokens', 'field': 'token_usage', 'sortable': True},
                    {'name': 'details', 'label': 'Details', 'field': 'details'}
                ]
                
                ui.table(columns=columns, rows=filtered_results).classes('w-full')
    
    def _setup_results_display(self) -> None:
        """Setup results display area"""
        with ui.card().classes('w-full mt-4'):
            ui.label('Benchmark Results').classes('text-lg font-semibold mb-2')
            self.results_display = ui.column().classes('w-full')
    
    def _run_benchmark(self) -> None:
        """Run benchmark tasks"""
        if self.is_benchmark_running:
            ui.notify("Benchmark is already running", type='warning')
            return
        
        # Get selected configurations
        selected_prompts = self.prompt_selection.value or []
        selected_models = self.model_selection.value or []
        selected_trials = self.trial_selection.value or []
        
        if not (selected_prompts and selected_models and selected_trials):
            ui.notify("Please select at least one prompt, model, and trial", type='warning')
            return
        
        # Disable run button and enable stop button
        self.run_benchmark_button.disable()
        self.stop_benchmark_button.set_visibility(True)
        self.is_benchmark_running = True
        self.benchmark_cancelled = False
        
        # Start benchmark execution
        async def execute_benchmark():
            try:
                await self._execute_benchmark_async(selected_prompts, selected_models, selected_trials)
            except Exception as e:
                ui.notify(f"Benchmark failed: {str(e)}", type='negative')
                logging.error(f"Benchmark failed: {str(e)}")
            finally:
                # Re-enable run button and disable stop button
                self.run_benchmark_button.enable()
                self.stop_benchmark_button.set_visibility(False)
                self.is_benchmark_running = False
                self.benchmark_status.set_text("Benchmark completed")
                self.benchmark_progress.set_value(1.0)
        
        background_tasks.create(execute_benchmark())
        ui.notify("Starting benchmark execution", type='positive')
        self.benchmark_status.set_text("Benchmark running...")
    
    async def _execute_benchmark_async(self, prompt_keys: List[str], model_keys: List[str], trial_ids: List[str]) -> None:
        """Execute benchmark asynchronously"""
        total_combinations = len(prompt_keys) * len(model_keys) * len(trial_ids)
        current_index = 0
        start_time = time.time()
        
        self.benchmark_results = []
        self.validation_results = []  # Clear previous validation results
        self._update_validation_display()  # Update display to show empty state
        
        try:
            # Create prompt variants
            prompt_variants = []
            for prompt_key in prompt_keys:
                prompt_info = self.available_prompts.get(prompt_key)
                if not prompt_info:
                    continue
                
                # Determine prompt type
                prompt_type_str = prompt_info.get('prompt_type', 'NLP_EXTRACTION')
                if prompt_type_str == 'NLP_EXTRACTION':
                    prompt_type = PromptType.NLP_EXTRACTION
                elif prompt_type_str == 'MCODE_MAPPING':
                    prompt_type = PromptType.MCODE_MAPPING
                else:
                    prompt_type = PromptType.NLP_EXTRACTION
                
                variant = PromptVariant(
                    name=prompt_info.get('name', prompt_key),
                    prompt_type=prompt_type,
                    prompt_key=prompt_key,
                    description=prompt_info.get('description', ''),
                    version=prompt_info.get('version', '1.0.0'),
                    tags=prompt_info.get('tags', [])
                )
                
                self.framework.add_prompt_variant(variant)
                prompt_variants.append(variant)
            
            # Create API configs
            for model_key in model_keys:
                config_name = f"model_{model_key.replace('-', '_').replace('.', '_')}"
                try:
                    api_config = APIConfig(name=config_name, model=model_key)
                    self.framework.add_api_config(api_config)
                except Exception as e:
                    logging.warning(f"Failed to create API config for {model_key}: {str(e)}")
                    continue
            
            # Execute benchmarks
            for prompt_variant in prompt_variants:
                for model_key in model_keys:
                    config_name = f"model_{model_key.replace('-', '_').replace('.', '_')}"
                    
                    for trial_id in trial_ids:
                        if self.benchmark_cancelled:
                            break
                        
                        current_index += 1
                        progress = current_index / total_combinations
                        self.benchmark_progress.set_value(progress)
                        self.benchmark_status.set_text(f"Running {current_index}/{total_combinations}")
                        
                        # Get gold standard data
                        expected_entities = []
                        expected_mappings = []
                        if trial_id in self.gold_standard_data:
                            gold_data = self.gold_standard_data[trial_id]
                            expected_entities = gold_data.get('expected_extraction', {}).get('entities', [])
                            expected_mappings = gold_data.get('expected_mcode_mappings', {}).get('mapped_elements', [])
                        
                        try:
                            # Update preloaded validation status to "Processing"
                            self._update_preloaded_validation_status(
                                prompt_variant.name, model_key, trial_id,
                                'Processing',
                                'Benchmark in progress...',
                                '🔄'
                            )
                            self._update_validation_display()
                            
                            # Run benchmark
                            result = await run.io_bound(
                                self.framework.run_benchmark,
                                prompt_variant_id=prompt_variant.id,
                                api_config_name=config_name,
                                test_case_id=trial_id,
                                pipeline_callback=self._create_pipeline_callback(),
                                expected_entities=expected_entities,
                                expected_mappings=expected_mappings,
                                current_index=current_index,
                                total_count=total_combinations,
                                benchmark_start_time=start_time
                            )
                            
                            self.benchmark_results.append(result)
                            
                            # Update preloaded validation status with benchmark metrics
                            self._update_preloaded_validation_status(
                                prompt_variant.name, model_key, trial_id,
                                'Success' if result.success else 'Failed',
                                f"F1={result.f1_score:.3f}, Compliance={result.compliance_score:.2%}" if result.success else result.error_message,
                                '✅' if result.success else '❌',
                                f"{result.precision:.3f}" if result.success else '-',
                                f"{result.recall:.3f}" if result.success else '-',
                                f"{result.f1_score:.3f}" if result.success else '-',
                                f"{result.duration_ms:.1f}" if result.success else '-',
                                f"{result.token_usage}" if result.success else '-'
                            )
                            
                            # Log result
                            self.live_log_display.push(
                                f"✅ {prompt_variant.name} + {model_key} + {trial_id}: "
                                f"F1={result.f1_score:.3f}, Compliance={result.compliance_score:.2%}"
                            )
                            
                            # Update validation display
                            self._update_validation_display()
                            
                        except Exception as e:
                            logging.error(f"Benchmark failed for {prompt_variant.name} + {model_key} + {trial_id}: {str(e)}")
                            
                            # Add to validation results
                            self.validation_results.append({
                                'prompt': prompt_variant.name,
                                'model': model_key,
                                'trial': trial_id,
                                'status': 'Failed',
                                'details': str(e)
                            })
                            
                            self.live_log_display.push(
                                f"❌ {prompt_variant.name} + {model_key} + {trial_id}: Failed - {str(e)}"
                            )
                            
                            # Update preloaded validation status for failed benchmark
                            self._update_preloaded_validation_status(
                                prompt_variant.name, model_key, trial_id,
                                'Failed',
                                str(e),
                                '❌'
                            )
                            
                            # Update validation display
                            self._update_validation_display()
                
                if self.benchmark_cancelled:
                    break
            
            # Display results
            await self._display_benchmark_results()
            
        except Exception as e:
            logging.error(f"Benchmark execution failed: {str(e)}")
            ui.notify(f"Benchmark execution failed: {str(e)}", type='negative')
    
    def _create_pipeline_callback(self) -> Callable:
        """Create pipeline callback for benchmark execution"""
        def pipeline_callback(test_data, prompt_content, prompt_variant_id, api_config_name=None):
            from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
            
            # Get the prompt variant to determine prompt type
            variant = self.framework.prompt_variants.get(prompt_variant_id)
            if not variant:
                raise ValueError(f"Prompt variant {prompt_variant_id} not found")
            
            # Get the API configuration if provided
            model_name = None
            temperature = None
            max_tokens = None
            if api_config_name:
                api_config = self.framework.api_configs.get(api_config_name)
                if api_config:
                    model_name = api_config.model
                    temperature = api_config.temperature
                    max_tokens = api_config.max_tokens
            
            # Create a NEW pipeline instance for each benchmark run to ensure cache isolation
            # Pass explicit model configuration if available
            pipeline = StrictDynamicExtractionPipeline()
            
            # If we have explicit model configuration, update the pipeline components
            if model_name:
                # Update NLP engine configuration
                pipeline.nlp_engine.model_name = model_name
                if temperature is not None:
                    pipeline.nlp_engine.temperature = temperature
                if max_tokens is not None:
                    pipeline.nlp_engine.max_tokens = max_tokens
                
                # Update LLM mapper configuration
                pipeline.llm_mapper.model_name = model_name
                if temperature is not None:
                    pipeline.llm_mapper.temperature = temperature
                if max_tokens is not None:
                    pipeline.llm_mapper.max_tokens = max_tokens
            
            # Set the prompt content directly on the NLP engine based on prompt type
            if variant.prompt_type == PromptType.NLP_EXTRACTION:
                pipeline.nlp_engine.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
            elif variant.prompt_type == PromptType.MCODE_MAPPING:
                pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE = prompt_content
            
            return pipeline.process_clinical_trial(test_data)
        
        return pipeline_callback
    
    async def _display_benchmark_results(self) -> None:
        """Display benchmark results"""
        if not self.benchmark_results:
            ui.notify("No benchmark results to display", type='info')
            return
        
        self.results_display.clear()
        
        with self.results_display:
            # Summary statistics
            total_runs = len(self.benchmark_results)
            successful_runs = len([r for r in self.benchmark_results if r.success])
            success_rate = successful_runs / total_runs if total_runs > 0 else 0
            
            avg_f1 = sum(r.f1_score for r in self.benchmark_results) / total_runs if total_runs > 0 else 0
            avg_compliance = sum(r.compliance_score for r in self.benchmark_results) / total_runs if total_runs > 0 else 0
            
            ui.markdown(f"""
            ## Benchmark Summary
            - **Total Runs**: {total_runs}
            - **Successful Runs**: {successful_runs}
            - **Success Rate**: {success_rate:.1%}
            - **Average F1 Score**: {avg_f1:.3f}
            - **Average Compliance Score**: {avg_compliance:.2%}
            """)
            
            # Detailed results table
            ui.label('Detailed Results').classes('text-lg font-semibold mt-4')
            
            columns = [
                {'name': 'prompt', 'label': 'Prompt', 'field': 'prompt', 'sortable': True},
                {'name': 'model', 'label': 'Model', 'field': 'model', 'sortable': True},
                {'name': 'trial', 'label': 'Trial', 'field': 'trial', 'sortable': True},
                {'name': 'f1', 'label': 'F1 Score', 'field': 'f1', 'sortable': True},
                {'name': 'compliance', 'label': 'Compliance', 'field': 'compliance', 'sortable': True},
                {'name': 'duration', 'label': 'Duration (ms)', 'field': 'duration', 'sortable': True},
                {'name': 'tokens', 'label': 'Tokens', 'field': 'tokens', 'sortable': True}
            ]
            
            rows = []
            for result in self.benchmark_results:
                # Get prompt name
                prompt_variant = self.framework.prompt_variants.get(result.prompt_variant_id)
                prompt_name = prompt_variant.name if prompt_variant else "Unknown"
                
                # Get model name
                api_config = self.framework.api_configs.get(result.api_config_name)
                model_name = api_config.model if api_config else "Unknown"
                
                rows.append({
                    'prompt': prompt_name,
                    'model': model_name,
                    'trial': result.test_case_id,
                    'f1': f"{result.f1_score:.3f}",
                    'compliance': f"{result.compliance_score:.2%}",
                    'duration': f"{result.duration_ms:.1f}",
                    'tokens': result.token_usage
                })
            
            ui.table(columns=columns, rows=rows, pagination=10).classes('w-full')
    
    def _stop_benchmark(self) -> None:
        """Stop benchmark execution"""
        if not self.is_benchmark_running:
            ui.notify("No benchmark is currently running", type='info')
            return
        
        self.benchmark_cancelled = True
        self.benchmark_status.set_text("Benchmark stopping...")
        ui.notify("Stopping benchmark execution", type='info')
    
    def _reset_interface(self) -> None:
        """Reset the interface to its initial state"""
        # Cancel any running benchmarks
        if self.is_benchmark_running:
            self.benchmark_cancelled = True
            self.is_benchmark_running = False
            self.run_benchmark_button.enable()
            self.stop_benchmark_button.set_visibility(False)
        
        # Clear benchmark results
        self.benchmark_results = []
        self.validation_results = []
        
        # Reset progress and status
        self.benchmark_progress.set_value(0)
        self.benchmark_status.set_text("Ready to run benchmark")
        
        # Reset benchmark cancellation state to ensure clean state
        self.benchmark_cancelled = False
        
        # Clear log display
        if self.live_log_display:
            self.live_log_display.clear()
        
        # Clear results display
        if self.results_display:
            self.results_display.clear()
        
        # Reset selections to default (all selected)
        if self.prompt_selection:
            self.prompt_selection.set_value(list(self.available_prompts.keys()))
        if self.model_selection:
            self.model_selection.set_value(list(self.available_models.keys()))
        if self.trial_selection:
            self.trial_selection.set_value(list(self.trial_data.keys()))
        
        # Reload initial validations
        self._load_initial_validations()
        
        ui.notify("Interface reset to initial state", type='positive')


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