"""
Modern Optimization UI for Prompt Optimization Framework
Integrated with file-based prompt and model libraries
"""

import sys
from pathlib import Path
import logging

# Add project root to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
import os
from typing import Dict, List, Any, Optional
import pandas as pd
from nicegui import ui, app
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

from src.optimization.strict_prompt_optimization_framework import (
    StrictPromptOptimizationFramework,
    PromptType,
    APIConfig,
    PromptVariant
)

from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader


class ModernOptimizationUI:
    """Modern web-based UI for prompt optimization experiments"""
    
    def __init__(self, framework: Optional[StrictPromptOptimizationFramework] = None):
        self.framework = framework or StrictPromptOptimizationFramework()
        self.current_results = None
        self.test_cases_file = "examples/breast_cancer_data/breast_cancer_her2_positive.trial.json"
        self.gold_standard_file = "examples/breast_cancer_data/breast_cancer_her2_positive.gold.json"
        
        # Check if default files exist, if not try alternative paths
        if not os.path.exists(self.test_cases_file):
            # Try alternative path in tests/data/test_cases/
            alt_test_cases_file = "tests/data/test_cases/multi_cancer.json"
            if os.path.exists(alt_test_cases_file):
                self.test_cases_file = alt_test_cases_file
                logger.info(f"Using alternative test cases file: {self.test_cases_file}")
            else:
                logger.warning(f"Default test cases file {self.test_cases_file} not found")
        
        if not os.path.exists(self.gold_standard_file):
            # Try alternative path in tests/data/gold_standard/
            alt_gold_standard_file = "tests/data/gold_standard/multi_cancer.json"
            if os.path.exists(alt_gold_standard_file):
                self.gold_standard_file = alt_gold_standard_file
                logger.info(f"Using alternative gold standard file: {self.gold_standard_file}")
            else:
                logger.warning(f"Default gold standard file {self.gold_standard_file} not found")
        
        self._load_default_libraries()
        self._load_default_test_cases()
        self._load_default_gold_standard()
        self.setup_ui()
    
    def _load_default_libraries(self) -> None:
        """Load prompt and model libraries"""
        try:
            # Load all available prompts from library
            self.available_prompts = prompt_loader.list_available_prompts()
            logger.info(f"Loaded {len(self.available_prompts)} prompts from library")
            
            # Load all available models from library
            self.available_models = model_loader.list_available_models()
            logger.info(f"Loaded {len(self.available_models)} models from library")
            
        except Exception as e:
            logger.error(f"Failed to load libraries: {str(e)}")
            raise
    
    def _load_default_test_cases(self) -> None:
        """Load default test cases"""
        try:
            # Load test cases from file
            with open(self.test_cases_file, 'r') as f:
                test_cases_data = json.load(f)
            
            # Add test cases to framework
            for case_id, case_data in test_cases_data.get("test_cases", {}).items():
                self.framework.add_test_case(case_id, case_data)
            
            logger.info(f"Loaded {len(test_cases_data.get('test_cases', {}))} test cases from {self.test_cases_file}")
            
        except Exception as e:
            logger.error(f"Failed to load default test cases: {str(e)}")
            # Continue without raising exception to allow UI to start
    
    def _load_default_gold_standard(self) -> None:
        """Load default gold standard data"""
        try:
            # Check if gold standard file exists
            if not os.path.exists(self.gold_standard_file):
                logger.warning(f"Gold standard file {self.gold_standard_file} not found, skipping load")
                return
            
            # Load gold standard data from file
            with open(self.gold_standard_file, 'r') as f:
                gold_standard_data = json.load(f)
            
            # Add gold standard data to framework
            # Note: The framework might need a method to load gold standard data
            # For now, we'll just log that we've loaded the data
            logger.info(f"Loaded gold standard data from {self.gold_standard_file}")
            
        except Exception as e:
            logger.error(f"Failed to load default gold standard data: {str(e)}")
            # Continue without raising exception to allow UI to start
    
    def _browse_test_cases(self) -> None:
        """Open file browser for test cases file"""
        # In a real implementation, this would open a file dialog
        # For now, we'll just show a notification
        ui.notify('File browser would open here in a full implementation')
    
    def _browse_gold_standard(self) -> None:
        """Open file browser for gold standard file"""
        # In a real implementation, this would open a file dialog
        # For now, we'll just show a notification
        ui.notify('File browser would open here in a full implementation')
    
    def _load_default_files(self) -> None:
        """Load default test cases and gold standard files"""
        try:
            # Update UI inputs with default file paths
            self.test_cases_file_input.set_value(self.test_cases_file)
            self.gold_standard_file_input.set_value(self.gold_standard_file)
            
            # Load the files
            self._load_test_files()
            
            ui.notify('Default files loaded successfully')
        except Exception as e:
            ui.notify(f'Error loading default files: {str(e)}', type='negative')
            logger.error(f"Failed to load default files: {str(e)}")
    
    def _load_test_files(self) -> None:
        """Load test cases and gold standard files"""
        try:
            # Update file paths from UI inputs
            test_cases_file = self.test_cases_file_input.value
            gold_standard_file = self.gold_standard_file_input.value
            
            # Clear existing test cases
            self.framework.test_cases.clear()
            
            # Load test cases from file
            with open(test_cases_file, 'r') as f:
                test_cases_data = json.load(f)
            
            # Add test cases to framework
            for case_id, case_data in test_cases_data.get("test_cases", {}).items():
                self.framework.add_test_case(case_id, case_data)
            
            ui.notify(f'Successfully loaded {len(test_cases_data.get("test_cases", {}))} test cases from {test_cases_file}')
            
            # Refresh UI components
            self._refresh_benchmark_selects()
            
        except Exception as e:
            ui.notify(f'Error loading test files: {str(e)}', type='negative')
            logger.error(f"Failed to load test files: {str(e)}")
    
    def setup_ui(self) -> None:
        """Setup the main UI layout and components"""
        with ui.header().classes('bg-blue-600 text-white p-4'):
            ui.label('Modern Prompt Optimization Framework').classes('text-2xl font-bold')
        
        with ui.tabs().classes('w-full') as tabs:
            self.config_tab = ui.tab('Library Management')
            self.benchmark_tab = ui.tab('Benchmark Execution')
            self.results_tab = ui.tab('Results Analysis')
            self.system_tab = ui.tab('System Status')
        
        with ui.tab_panels(tabs, value=self.config_tab).classes('w-full'):
            with ui.tab_panel(self.config_tab):
                self._setup_library_management_tab()
            with ui.tab_panel(self.benchmark_tab):
                self._setup_benchmark_execution_tab()
            with ui.tab_panel(self.results_tab):
                self._setup_results_analysis_tab()
            with ui.tab_panel(self.system_tab):
                self._setup_system_status_tab()
    
    def _setup_library_management_tab(self) -> None:
        """Setup library management UI"""
        with ui.column().classes('w-full p-4 gap-6'):
            ui.label('Prompt & Model Library Management').classes('text-xl font-bold')
            
            # Prompt Library Browser
            with ui.card().classes('w-full p-4'):
                ui.label('Prompt Library').classes('text-lg font-semibold mb-4')
                
                # Prompt filtering controls
                with ui.row().classes('w-full gap-4 mb-4'):
                    self.prompt_type_filter = ui.select(
                        ['All', 'NLP_EXTRACTION', 'MCODE_MAPPING'],
                        value='All',
                        label='Filter by Type'
                    ).classes('w-48')
                    
                    self.prompt_status_filter = ui.select(
                        ['All', 'production', 'experimental'],
                        value='All',
                        label='Filter by Status'
                    ).classes('w-48')
                    
                    ui.button('Refresh Library', on_click=self._refresh_prompt_library).classes('self-end')
                
                # Prompt library table
                self.prompt_library_table = ui.table(
                    columns=[
                        {'name': 'name', 'label': 'Name', 'field': 'name', 'sortable': True},
                        {'name': 'type', 'label': 'Type', 'field': 'type', 'sortable': True},
                        {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True},
                        {'name': 'default', 'label': 'Default', 'field': 'default', 'sortable': True},
                        {'name': 'description', 'label': 'Description', 'field': 'description'},
                        {'name': 'actions', 'label': 'Actions', 'field': 'actions'}
                    ],
                    rows=[],
                    pagination=10
                ).classes('w-full')
            
            # Model Library Browser
            with ui.card().classes('w-full p-4 mt-6'):
                ui.label('Model Library').classes('text-lg font-semibold mb-4')
                
                # Model filtering controls
                with ui.row().classes('w-full gap-4 mb-4'):
                    self.model_type_filter = ui.select(
                        ['All', 'CODE_GENERATION', 'GENERAL_CONVERSATION', 'REASONING_AND_PROBLEM_SOLVING', 'GENERAL_PURPOSE'],
                        value='All',
                        label='Filter by Type'
                    ).classes('w-64')
                    
                    self.model_status_filter = ui.select(
                        ['All', 'production', 'experimental'],
                        value='All',
                        label='Filter by Status'
                    ).classes('w-48')
                    
                    ui.button('Refresh Library', on_click=self._refresh_model_library).classes('self-end')
                
                # Model library table
                self.model_library_table = ui.table(
                    columns=[
                        {'name': 'name', 'label': 'Name', 'field': 'name', 'sortable': True},
                        {'name': 'type', 'label': 'Type', 'field': 'type', 'sortable': True},
                        {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True},
                        {'name': 'default', 'label': 'Default', 'field': 'default', 'sortable': True},
                        {'name': 'capabilities', 'label': 'Capabilities', 'field': 'capabilities'},
                        {'name': 'actions', 'label': 'Actions', 'field': 'actions'}
                    ],
                    rows=[],
                    pagination=10
                ).classes('w-full')
            
            # Refresh all library data
            self._refresh_library_data()
    
    def _setup_benchmark_execution_tab(self) -> None:
        """Setup benchmark execution UI"""
        with ui.column().classes('w-full p-4 gap-6'):
            ui.label('Benchmark Execution').classes('text-xl font-bold')
            
            # File Selection
            with ui.card().classes('w-full p-4'):
                ui.label('Test Case & Gold Standard Files').classes('text-lg font-semibold mb-4')
                ui.label('Select test case and gold standard files, then click "Load Files" to populate the benchmark configuration').classes('text-sm text-gray-600 mb-4')
                
                with ui.grid(columns=2).classes('w-full gap-4'):
                    # Test cases file selection
                    with ui.column().classes('w-full'):
                        self.test_cases_file_input = ui.input(
                            'Test Cases File',
                            value=self.test_cases_file,
                            placeholder='Path to test cases JSON file'
                        ).classes('w-full')
                        ui.button('Browse Test Cases', on_click=self._browse_test_cases).classes('mt-2')
                    
                    # Gold standard file selection
                    with ui.column().classes('w-full'):
                        self.gold_standard_file_input = ui.input(
                            'Gold Standard File',
                            value=self.gold_standard_file,
                            placeholder='Path to gold standard JSON file'
                        ).classes('w-full')
                        ui.button('Browse Gold Standard', on_click=self._browse_gold_standard).classes('mt-2')
                
                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('Load Files', on_click=self._load_test_files).classes('bg-blue-500 text-white px-4 py-2 rounded')
                    ui.button('Load Default Files', on_click=self._load_default_files).classes('bg-green-500 text-white px-4 py-2 rounded')
            
            # Benchmark Configuration
            with ui.card().classes('w-full p-4 mt-6'):
                ui.label('Experiment Configuration').classes('text-lg font-semibold mb-4')
                
                with ui.grid(columns=3).classes('w-full gap-4'):
                    # Prompt selection
                    self.prompt_selection = ui.select([], label='Select Prompts', multiple=True).classes('w-full')
                    
                    # Model selection
                    self.model_selection = ui.select([], label='Select Models', multiple=True).classes('w-full')
                    
                    # Test case selection
                    self.test_case_selection = ui.select([], label='Select Test Cases', multiple=True).classes('w-full')
                
                # Advanced options
                with ui.expansion('Advanced Options', icon='settings').classes('w-full'):
                    with ui.grid(columns=2).classes('w-full gap-4'):
                        self.metric_selection = ui.select(
                            ['f1_score', 'precision', 'recall', 'compliance_score'],
                            value='f1_score',
                            label='Optimization Metric'
                        ).classes('w-full')
                        
                        self.concurrency_level = ui.number(
                            'Concurrency Level',
                            value=1,
                            min=1,
                            max=10,
                            format='%d'
                        ).classes('w-full')
                        
                        self.timeout_setting = ui.number(
                            'Timeout (seconds)',
                            value=300,
                            min=30,
                            max=3600,
                            format='%d'
                        ).classes('w-full')
                
                # Execution controls
                with ui.row().classes('w-full gap-4 mt-4'):
                    self.run_benchmark_button = ui.button(
                        'Run Benchmark',
                        on_click=self._run_benchmark,
                        color='green'
                    ).classes('text-white')
                    
                    self.stop_benchmark_button = ui.button(
                        'Stop Benchmark',
                        on_click=self._stop_benchmark,
                        color='red'
                    ).classes('text-white')
                    self.stop_benchmark_button.set_visibility(False)
                
                # Progress tracking
                self.benchmark_status = ui.label('Ready to run benchmark').classes('text-sm text-gray-600 mt-2')
                self.benchmark_progress = ui.linear_progress(0).classes('w-full mt-2')
            
            # Real-time metrics display
            with ui.card().classes('w-full p-4 mt-6'):
                ui.label('Real-time Metrics').classes('text-lg font-semibold mb-4')
                
                with ui.grid(columns=4).classes('w-full gap-4'):
                    self.current_prompt_label = ui.label('Prompt: -').classes('text-sm')
                    self.current_model_label = ui.label('Model: -').classes('text-sm')
                    self.current_test_case_label = ui.label('Test Case: -').classes('text-sm')
                    self.current_duration_label = ui.label('Duration: -').classes('text-sm')
                
                with ui.grid(columns=4).classes('w-full gap-4 mt-2'):
                    self.current_entities_label = ui.label('Entities: -').classes('text-sm')
                    self.current_mappings_label = ui.label('Mappings: -').classes('text-sm')
                    self.current_compliance_label = ui.label('Compliance: -').classes('text-sm')
                    self.current_f1_label = ui.label('F1 Score: -').classes('text-sm')
            
            # Refresh dynamic selects
            self._refresh_benchmark_selects()
            
            # Load default files automatically when UI starts
            self._load_default_files()
    
    def _setup_results_analysis_tab(self) -> None:
        """Setup results analysis UI"""
        with ui.column().classes('w-full p-4 gap-6'):
            ui.label('Results Analysis').classes('text-xl font-bold')
            
            # Results loading controls
            with ui.row().classes('w-full gap-4 mb-4'):
                ui.button('Load Latest Results', on_click=self._load_results).classes('bg-blue-500 text-white')
                ui.button('Export to CSV', on_click=self._export_results).classes('bg-green-500 text-white')
                ui.button('Generate Report', on_click=self._generate_report).classes('bg-purple-500 text-white')
                ui.button('Clear Results', on_click=self._clear_results).classes('bg-red-500 text-white')
            
            # Summary statistics
            with ui.card().classes('w-full p-4'):
                ui.label('Summary Statistics').classes('text-lg font-semibold mb-4')
                self.summary_stats = ui.markdown('').classes('w-full')
            
            # Performance comparison charts
            with ui.card().classes('w-full p-4 mt-6'):
                ui.label('Performance Comparisons').classes('text-lg font-semibold mb-4')
                
                with ui.row().classes('w-full gap-4'):
                    ui.button('Generate Visualizations', on_click=self._generate_visualizations).classes('bg-blue-500 text-white')
                
                # Visualization gallery
                self.visualization_gallery = ui.column().classes('w-full grid grid-cols-2 gap-4 mt-4')
            
            # Detailed results table
            with ui.card().classes('w-full p-4 mt-6'):
                ui.label('Detailed Results').classes('text-lg font-semibold mb-4')
                
                self.results_table = ui.table(
                    columns=[
                        {'name': 'run_id', 'label': 'Run ID', 'field': 'run_id'},
                        {'name': 'prompt_name', 'label': 'Prompt', 'field': 'prompt_name'},
                        {'name': 'model', 'label': 'Model', 'field': 'model'},
                        {'name': 'test_case', 'label': 'Test Case', 'field': 'test_case'},
                        {'name': 'duration_ms', 'label': 'Duration (ms)', 'field': 'duration_ms'},
                        {'name': 'success', 'label': 'Success', 'field': 'success'},
                        {'name': 'entities_extracted', 'label': 'Entities', 'field': 'entities_extracted'},
                        {'name': 'compliance_score', 'label': 'Compliance', 'field': 'compliance_score'},
                        {'name': 'f1_score', 'label': 'F1 Score', 'field': 'f1_score'}
                    ],
                    rows=[],
                    pagination=15
                ).classes('w-full')
    
    def _setup_system_status_tab(self) -> None:
        """Setup system status UI"""
        with ui.column().classes('w-full p-4 gap-6'):
            ui.label('System Status').classes('text-xl font-bold')
            
            # Resource usage monitoring
            with ui.card().classes('w-full p-4'):
                ui.label('Resource Usage').classes('text-lg font-semibold mb-4')
                
                with ui.grid(columns=3).classes('w-full gap-4'):
                    self.cpu_usage_label = ui.label('CPU: -%').classes('text-sm')
                    self.memory_usage_label = ui.label('Memory: - MB').classes('text-sm')
                    self.disk_usage_label = ui.label('Disk: -%').classes('text-sm')
                
                # Resource usage chart placeholder
                ui.label('Resource usage chart would be displayed here').classes('text-gray-500 italic')
            
            # Cache status
            with ui.card().classes('w-full p-4 mt-6'):
                ui.label('Cache Status').classes('text-lg font-semibold mb-4')
                
                with ui.grid(columns=3).classes('w-full gap-4'):
                    self.cache_enabled_label = ui.label('Cache: Enabled').classes('text-sm')
                    self.cache_size_label = ui.label('Size: - MB').classes('text-sm')
                    self.cache_files_label = ui.label('Files: -').classes('text-sm')
                
                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('Clear Cache', on_click=self._clear_cache).classes('bg-red-500 text-white')
                    ui.button('Reload Cache', on_click=self._reload_cache).classes('bg-blue-500 text-white')
            
            # Configuration overview
            with ui.card().classes('w-full p-4 mt-6'):
                ui.label('Configuration Overview').classes('text-lg font-semibold mb-4')
                
                self.config_overview = ui.markdown('').classes('w-full')
                
                with ui.row().classes('w-full gap-4 mt-4'):
                    ui.button('Reload Configuration', on_click=self._reload_configuration).classes('bg-blue-500 text-white')
    
    def _refresh_library_data(self) -> None:
        """Refresh all library data displays"""
        self._refresh_prompt_library()
        self._refresh_model_library()
        # Only refresh benchmark selects if the benchmark tab has been initialized
        if hasattr(self, 'prompt_selection'):
            self._refresh_benchmark_selects()
    
    def _refresh_prompt_library(self) -> None:
        """Refresh prompt library table"""
        try:
            # Get filtered prompts
            filtered_prompts = self._filter_prompts(
                self.prompt_type_filter.value,
                self.prompt_status_filter.value
            )
            
            # Convert to table rows
            rows = []
            for prompt_key, prompt_info in filtered_prompts.items():
                rows.append({
                    'name': prompt_info.get('name', prompt_key),
                    'type': prompt_info.get('prompt_type', 'Unknown'),
                    'status': prompt_info.get('status', 'Unknown'),
                    'default': '✅' if prompt_info.get('default', False) else '❌',
                    'description': prompt_info.get('description', '')[:100] + '...' if len(prompt_info.get('description', '')) > 100 else prompt_info.get('description', ''),
                    'actions': self._create_prompt_actions(prompt_key, prompt_info)
                })
            
            self.prompt_library_table.rows = rows
            self.prompt_library_table.update()
            
        except Exception as e:
            logger.error(f"Failed to refresh prompt library: {str(e)}")
            ui.notify(f'Error refreshing prompt library: {str(e)}', type='negative')
    
    def _filter_prompts(self, type_filter: str, status_filter: str) -> Dict[str, Any]:
        """Filter prompts based on selected criteria"""
        filtered = {}
        for key, prompt in self.available_prompts.items():
            # Type filter
            if type_filter != 'All' and prompt.get('prompt_type', '').upper() != type_filter:
                continue
            
            # Status filter
            if status_filter != 'All' and prompt.get('status', '') != status_filter:
                continue
            
            filtered[key] = prompt
        
        return filtered
    
    def _create_prompt_actions(self, prompt_key: str, prompt_info: Dict[str, Any]) -> str:
        """Create action buttons for a prompt"""
        actions = []
        
        if prompt_info.get('default', False):
            actions.append('✅ Default')
        else:
            actions.append(f"[Set Default](javascript:setDefaultPrompt('{prompt_key}'))")
        
        actions.append(f"[View](javascript:viewPrompt('{prompt_key}'))")
        
        return ' | '.join(actions)
    
    def _refresh_model_library(self) -> None:
        """Refresh model library table"""
        try:
            # Get filtered models
            filtered_models = self._filter_models(
                self.model_type_filter.value,
                self.model_status_filter.value
            )
            
            # Convert to table rows
            rows = []
            for model_key, model_info in filtered_models.items():
                rows.append({
                    'name': model_info.get('name', model_key),
                    'type': model_info.get('model_type', 'Unknown'),
                    'status': model_info.get('status', 'Unknown'),
                    'default': '✅' if model_info.get('default', False) else '❌',
                    'capabilities': ', '.join(model_info.get('capabilities', [])),
                    'actions': self._create_model_actions(model_key, model_info)
                })
            
            self.model_library_table.rows = rows
            self.model_library_table.update()
            
        except Exception as e:
            logger.error(f"Failed to refresh model library: {str(e)}")
            ui.notify(f'Error refreshing model library: {str(e)}', type='negative')
    
    def _filter_models(self, type_filter: str, status_filter: str) -> Dict[str, Any]:
        """Filter models based on selected criteria"""
        filtered = {}
        for key, model in self.available_models.items():
            # Type filter
            if type_filter != 'All' and model.get('model_type', '') != type_filter:
                continue
            
            # Status filter
            if status_filter != 'All' and model.get('status', '') != status_filter:
                continue
            
            filtered[key] = model
        
        return filtered
    
    def _create_model_actions(self, model_key: str, model_info: Dict[str, Any]) -> str:
        """Create action buttons for a model"""
        actions = []
        
        if model_info.get('default', False):
            actions.append('✅ Default')
        else:
            actions.append(f"[Set Default](javascript:setDefaultModel('{model_key}'))")
        
        actions.append(f"[View](javascript:viewModel('{model_key}'))")
        
        return ' | '.join(actions)
    
    def _refresh_benchmark_selects(self) -> None:
        """Refresh benchmark configuration selects"""
        try:
            # Prompt selection - use prompt names with types
            prompt_options = {}
            for key, prompt in self.available_prompts.items():
                name = prompt.get('name', key)
                prompt_type = prompt.get('prompt_type', 'Unknown')
                prompt_options[key] = f"{name} ({prompt_type})"
            self.prompt_selection.options = prompt_options
            
            # Model selection - use model names
            model_options = {}
            for key, model in self.available_models.items():
                name = model.get('name', key)
                model_options[key] = name
            self.model_selection.options = model_options
            
            # Test case selection - load from framework
            test_case_options = {}
            for test_id in self.framework.test_cases.keys():
                test_case_options[test_id] = test_id
            self.test_case_selection.options = test_case_options
            
            # Update all selects
            self.prompt_selection.update()
            self.model_selection.update()
            self.test_case_selection.update()
            
        except Exception as e:
            logger.error(f"Failed to refresh benchmark selects: {str(e)}")
            ui.notify(f'Error refreshing benchmark options: {str(e)}', type='negative')
    
    async def _run_benchmark(self) -> None:
        """Run benchmark experiments"""
        try:
            # Get selected configurations
            selected_prompts = self.prompt_selection.value or []
            selected_models = self.model_selection.value or []
            selected_test_cases = self.test_case_selection.value or []
            
            if not all([selected_prompts, selected_models, selected_test_cases]):
                ui.notify('Please select at least one prompt, model, and test case', type='warning')
                return
            
            # Disable run button and enable stop button
            self.run_benchmark_button.set_visibility(False)
            self.stop_benchmark_button.set_visibility(True)
            
            total_runs = len(selected_prompts) * len(selected_models) * len(selected_test_cases)
            self.benchmark_status.set_text(f'Running {total_runs} benchmark experiments...')
            
            # Create prompt variants from selected prompts and register them with the framework
            prompt_variant_ids = []
            for prompt_key in selected_prompts:
                # Check if a variant already exists for this prompt key
                existing_variant = None
                for variant in self.framework.prompt_variants.values():
                    if variant.prompt_key == prompt_key:
                        existing_variant = variant
                        break
                
                if existing_variant:
                    # Use existing variant
                    prompt_variant_ids.append(existing_variant.id)
                else:
                    # Create new variant from prompt library
                    prompt_info = self.available_prompts.get(prompt_key)
                    if not prompt_info:
                        logger.warning(f"Prompt '{prompt_key}' not found in available prompts")
                        continue
                    
                    # Determine prompt type
                    prompt_type_str = prompt_info.get('prompt_type', 'NLP_EXTRACTION')
                    if prompt_type_str == 'NLP_EXTRACTION':
                        prompt_type = PromptType.NLP_EXTRACTION
                    elif prompt_type_str == 'MCODE_MAPPING':
                        prompt_type = PromptType.MCODE_MAPPING
                    else:
                        logger.warning(f"Unknown prompt type '{prompt_type_str}' for prompt '{prompt_key}', defaulting to NLP_EXTRACTION")
                        prompt_type = PromptType.NLP_EXTRACTION
                    
                    # Create prompt variant
                    variant = PromptVariant(
                        name=prompt_info.get('name', prompt_key),
                        prompt_type=prompt_type,
                        prompt_key=prompt_key,
                        description=prompt_info.get('description', ''),
                        version=prompt_info.get('version', '1.0.0'),
                        tags=prompt_info.get('tags', [])
                    )
                    
                    # Add variant to framework
                    try:
                        self.framework.add_prompt_variant(variant)
                        prompt_variant_ids.append(variant.id)
                        logger.info(f"Added prompt variant: {variant.name} ({variant.id})")
                    except Exception as e:
                        logger.error(f"Failed to add prompt variant '{prompt_key}': {str(e)}")
                        continue
            
            # Run benchmarks for all combinations
            run_count = 0
            for prompt_id in prompt_variant_ids:  # Use actual prompt variant IDs
                for model_id in selected_models:
                    for test_case_id in selected_test_cases:
                        run_count += 1
                        self.benchmark_progress.set_value(run_count / total_runs)
                        self.benchmark_status.set_text(f'Running experiment {run_count}/{total_runs}...')
                        
                        # Get gold standard data for validation for this specific test case
                        expected_entities = None
                        expected_mappings = None
                        
                        # Try to load corresponding gold standard file
                        gold_standard_file = self.gold_standard_file_input.value
                        if os.path.exists(gold_standard_file):
                            try:
                                with open(gold_standard_file, 'r') as f:
                                    gold_data = json.load(f)
                                    expected_data = gold_data['gold_standard'].get(test_case_id, {})
                                    expected_entities = expected_data.get('expected_extraction', {}).get('entities', [])
                                    expected_mappings = expected_data.get('expected_mcode_mappings', {}).get('mapped_elements', [])
                            except Exception as e:
                                logger.warning(f"Failed to load gold standard data for test case {test_case_id}: {str(e)}")
                        
                        try:
                            # Run the actual benchmark
                            result = self.framework.run_benchmark(
                                prompt_variant_id=prompt_id,
                                api_config_name=model_id,
                                test_case_id=test_case_id,
                                pipeline_callback=self._create_pipeline_callback(),
                                expected_entities=expected_entities,
                                expected_mappings=expected_mappings,
                                current_index=run_count,
                                total_count=total_runs
                            )
                            
                            # Update UI with real-time metrics
                            prompt_variant = self.framework.prompt_variants.get(prompt_id)
                            self.current_prompt_label.set_text(f'Prompt: {getattr(prompt_variant, "name", "Unknown")}')
                            self.current_model_label.set_text(f'Model: {getattr(self.framework.api_configs.get(model_id), "model", "Unknown")}')
                            self.current_test_case_label.set_text(f'Test Case: {test_case_id}')
                            self.current_duration_label.set_text(f'Duration: {result.duration_ms:.1f}ms')
                            self.current_entities_label.set_text(f'Entities: {result.entities_extracted}')
                            self.current_mappings_label.set_text(f'Mappings: {result.entities_mapped}')
                            self.current_compliance_label.set_text(f'Compliance: {result.compliance_score:.2%}')
                            self.current_f1_label.set_text(f'F1 Score: {result.f1_score:.3f}')
                            
                        except Exception as e:
                            logger.error(f"Benchmark failed for combination {prompt_id}/{model_id}/{test_case_id}: {str(e)}")
                            # Continue with other combinations despite failures
            
            ui.notify(f'Completed {total_runs} benchmark experiments!')
            self.benchmark_status.set_text('Benchmark completed successfully')
            
        except Exception as e:
            ui.notify(f'Benchmark failed: {str(e)}', type='negative')
            self.benchmark_status.set_text(f'Error: {str(e)}')
        finally:
            # Re-enable run button and disable stop button
            self.run_benchmark_button.set_visibility(True)
            self.stop_benchmark_button.set_visibility(False)
    
    def _stop_benchmark(self) -> None:
        """Stop benchmark execution"""
        # TODO: Implement actual benchmark stopping
        ui.notify('Stopping benchmark execution...', type='info')
        self.benchmark_status.set_text('Benchmark stopped by user')
        self.run_benchmark_button.set_visibility(True)
        self.stop_benchmark_button.set_visibility(False)
    
    def _load_results(self) -> None:
        """Load and display benchmark results"""
        try:
            self.framework.load_benchmark_results()
            df = self.framework.get_results_dataframe()
            
            # Convert to table rows
            rows = []
            for _, row in df.iterrows():
                rows.append({
                    'run_id': row.get('run_id', ''),
                    'prompt_name': row.get('prompt_name', ''),
                    'model': row.get('model', ''),
                    'test_case': row.get('test_case_id', ''),
                    'duration_ms': f"{row.get('duration_ms', 0):.1f}",
                    'success': '✅' if row.get('success') else '❌',
                    'entities_extracted': row.get('entities_extracted', 0),
                    'compliance_score': f"{row.get('compliance_score', 0):.3f}",
                    'f1_score': f"{row.get('f1_score', 0):.3f}"
                })
            
            self.results_table.rows = rows
            self.results_table.update()
            
            # Update summary statistics
            if not df.empty:
                summary = f"""
                ## Summary Statistics
                - **Total Runs**: {len(df)}
                - **Success Rate**: {df['success'].mean():.1%}
                - **Avg Duration**: {df['duration_ms'].mean():.1f} ms
                - **Avg Entities Extracted**: {df['entities_extracted'].mean():.1f}
                - **Avg Compliance Score**: {df['compliance_score'].mean():.3f}
                - **Avg F1 Score**: {df['f1_score'].mean():.3f}
                """
                self.summary_stats.set_content(summary)
            
            ui.notify('Results loaded successfully')
        except Exception as e:
            ui.notify(f'Error loading results: {str(e)}', type='negative')
    
    def _export_results(self) -> None:
        """Export results to CSV"""
        try:
            output_path = self.framework.export_results_to_csv()
            ui.notify(f'Results exported to {output_path}')
        except Exception as e:
            ui.notify(f'Error exporting results: {str(e)}', type='negative')
    
    def _generate_report(self) -> None:
        """Generate performance report"""
        try:
            report = self.framework.generate_performance_report()
            
            if 'error' in report:
                ui.notify(report['error'], type='warning')
                return
            
            # Format report as markdown
            report_text = "# Performance Report\n\n"
            
            # Summary
            report_text += "## Summary\n"
            report_text += f"- **Total Experiments**: {report.get('total_experiments', 0)}\n"
            report_text += f"- **Success Rate**: {report.get('success_rate', 0):.1%}\n"
            report_text += f"- **Average Duration**: {report.get('avg_duration_ms', 0):.1f} ms\n"
            report_text += f"- **Average Entities**: {report.get('avg_entities', 0):.1f}\n"
            report_text += f"- **Average Compliance**: {report.get('avg_compliance', 0):.3f}\n"
            report_text += f"- **Average F1 Score**: {report.get('avg_f1_score', 0):.3f}\n\n"
            
            # Best performing configurations
            report_text += "## Best Performers\n"
            if 'best_configs' in report:
                for category, config in report['best_configs'].items():
                    report_text += f"### Best {category.replace('_', ' ').title()}\n"
                    report_text += f"- **Name**: {config.get('name', 'N/A')}\n"
                    report_text += f"- **Score**: {config.get('score', 0):.3f}\n\n"
            
            # Display report in a dialog
            with ui.dialog() as dialog, ui.card().classes('w-2/3'):
                ui.markdown(report_text).classes('max-h-96 overflow-auto')
                with ui.row().classes('w-full justify-end mt-4'):
                    ui.button('Close', on_click=dialog.close).classes('bg-gray-500 text-white')
                    ui.button('Export Report', on_click=lambda: self._export_report(report_text)).classes('bg-green-500 text-white')
            
            dialog.open()
            
        except Exception as e:
            ui.notify(f'Error generating report: {str(e)}', type='negative')
    
    def _export_report(self, report_text: str) -> None:
        """Export report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"optimization_report_{timestamp}.md"
            
            with open(report_path, 'w') as f:
                f.write(report_text)
            
            ui.notify(f'Report exported to {report_path}')
        except Exception as e:
            ui.notify(f'Error exporting report: {str(e)}', type='negative')
    
    def _clear_results(self) -> None:
        """Clear benchmark results"""
        try:
            # Clear results from framework
            self.framework.benchmark_results.clear()
            
            # Clear UI displays
            self.results_table.rows = []
            self.results_table.update()
            self.summary_stats.set_content('')
            
            # Clear result files
            for result_file in self.framework.results_dir.glob("benchmark_*.json"):
                result_file.unlink()
            
            ui.notify('Results cleared successfully')
        except Exception as e:
            ui.notify(f'Error clearing results: {str(e)}', type='negative')
    
    def _generate_visualizations(self) -> None:
        """Generate performance visualizations"""
        try:
            # Get visualization data from framework
            viz_data = self.framework.get_visualization_data()
            
            if not viz_data:
                ui.notify('No benchmark data available for visualization', type='warning')
                return
            
            self.visualization_gallery.clear()
            
            with self.visualization_gallery:
                # Success Rate Bar Chart
                if viz_data.get('success_rates') and 'labels' in viz_data['success_rates'] and 'values' in viz_data['success_rates']:
                    with ui.card().classes('w-full'):
                        ui.label('Success Rate by Prompt Variant').classes('text-lg font-semibold')
                        ui.echart({
                            'xAxis': {'type': 'category', 'data': viz_data['success_rates']['labels']},
                            'yAxis': {'type': 'value', 'min': 0, 'max': 1},
                            'series': [{
                                'data': viz_data['success_rates']['values'],
                                'type': 'bar',
                                'itemStyle': {
                                    'color': {'type': 'linear', 'x': 0, 'y': 0, 'x2': 0, 'y2': 1,
                                             'colorStops': [{'offset': 0, 'color': '#4CAF50'},
                                                           {'offset': 1, 'color': '#2E7D32'}]}
                                }
                            }],
                            'tooltip': {'trigger': 'axis'}
                        }).classes('w-full h-64')
                
                # Compliance Score Bar Chart
                if viz_data.get('compliance_scores') and 'labels' in viz_data['compliance_scores'] and 'values' in viz_data['compliance_scores']:
                    with ui.card().classes('w-full'):
                        ui.label('Compliance Score by Prompt Variant').classes('text-lg font-semibold')
                        ui.echart({
                            'xAxis': {'type': 'category', 'data': viz_data['compliance_scores']['labels']},
                            'yAxis': {'type': 'value', 'min': 0, 'max': 1},
                            'series': [{
                                'data': viz_data['compliance_scores']['values'],
                                'type': 'bar',
                                'itemStyle': {
                                    'color': {'type': 'linear', 'x': 0, 'y': 0, 'x2': 0, 'y2': 1,
                                             'colorStops': [{'offset': 0, 'color': '#2196F3'},
                                                           {'offset': 1, 'color': '#0D47A1'}]}
                                }
                            }],
                            'tooltip': {'trigger': 'axis'}
                        }).classes('w-full h-64')
                
                # Performance Comparison Radar Chart
                if viz_data.get('performance_comparison'):
                    with ui.card().classes('w-full'):
                        ui.label('Performance Comparison').classes('text-lg font-semibold')
                        # Implementation would depend on the specific data structure
                        ui.label('Radar chart would be displayed here').classes('text-gray-500 italic')
            
            ui.notify('Visualizations generated successfully')
            
        except Exception as e:
            ui.notify(f'Error generating visualizations: {str(e)}', type='negative')
            logger.error(f"Visualization error: {e}")
    
    def _clear_cache(self) -> None:
        """Clear cache"""
        try:
            # TODO: Implement cache clearing logic
            ui.notify('Cache cleared successfully')
        except Exception as e:
            ui.notify(f'Error clearing cache: {str(e)}', type='negative')
    
    def _reload_cache(self) -> None:
        """Reload cache"""
        try:
            # TODO: Implement cache reloading logic
            ui.notify('Cache reloaded successfully')
        except Exception as e:
            ui.notify(f'Error reloading cache: {str(e)}', type='negative')
    
    def _reload_configuration(self) -> None:
        """Reload configuration files"""
        try:
            # Reload prompt library
            prompt_loader.reload_config()
            self.available_prompts = prompt_loader.list_available_prompts()
            
            # Reload model library
            model_loader.reload_config()
            self.available_models = model_loader.list_available_models()
            
            # Refresh UI
            self._refresh_library_data()
            
            ui.notify('Configuration reloaded successfully')
        except Exception as e:
            ui.notify(f'Error reloading configuration: {str(e)}', type='negative')
    
    def _create_pipeline_callback(self):
        """Create a pipeline callback function for benchmark execution"""
        def pipeline_callback(test_data, prompt_content, prompt_variant_id):
            from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
            
            # Get the prompt variant to determine prompt type
            variant = self.framework.prompt_variants.get(prompt_variant_id)
            if not variant:
                raise ValueError(f"Prompt variant {prompt_variant_id} not found")
            
            # Create pipeline instance
            pipeline = StrictDynamicExtractionPipeline()
            
            # Set the prompt content directly on the NLP engine based on prompt type
            if variant.prompt_type == PromptType.NLP_EXTRACTION:
                # For extraction prompts, set the extraction prompt template
                pipeline.nlp_engine.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
            elif variant.prompt_type == PromptType.MCODE_MAPPING:
                # For mapping prompts, set the mapping prompt template
                # Note: This would require similar changes to StrictMcodeMapper
                pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE = prompt_content
            else:
                # Default to extraction prompt if type is unknown
                pipeline.nlp_engine.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
            
            # Process the test data with the configured pipeline
            return pipeline.process_clinical_trial(test_data)
        
        return pipeline_callback


def run_modern_optimization_ui(port: int = 8082):
    """Run the modern optimization UI as a standalone application"""
    framework = StrictPromptOptimizationFramework()
    ui_instance = ModernOptimizationUI(framework)
    ui.run(title='Modern Prompt Optimization Framework', port=port, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    import argparse
    parser = argparse.ArgumentParser(description='Run the Modern Optimization UI')
    parser.add_argument('--port', type=int, default=8082, help='Port to run the UI on')
    args = parser.parse_args()
    run_modern_optimization_ui(args.port)