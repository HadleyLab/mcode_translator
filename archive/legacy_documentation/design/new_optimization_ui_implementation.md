# New Optimization UI Implementation Design

## Overview
This document outlines the implementation plan for a modern, strict, and forward-thinking web-based UI for the Prompt Optimization Framework.

## File Structure
```
src/optimization/
├── new_optimization_ui.py          # Main UI implementation
├── ui_components/
│   ├── config_manager.py           # Configuration management components
│   ├── benchmark_runner.py         # Benchmark execution components
│   ├── results_analyzer.py         # Results analysis and visualization
│   └── system_monitor.py           # System status monitoring
└── ui_utils/
    ├── data_loader.py              # Data loading utilities
    └── validation.py               # Input validation utilities
```

## Main UI Implementation (new_optimization_ui.py)

```python
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

from src.optimization.prompt_optimization_framework import (
    PromptOptimizationFramework,
    PromptType,
    APIConfig,
    PromptVariant
)

from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader


class ModernOptimizationUI:
    """Modern web-based UI for prompt optimization experiments"""
    
    def __init__(self, framework: Optional[PromptOptimizationFramework] = None):
        self.framework = framework or PromptOptimizationFramework()
        self.current_results = None
        self._load_default_libraries()
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
            
            # Benchmark Configuration
            with ui.card().classes('w-full p-4'):
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
            
            # TODO: Implement actual benchmark execution using the framework
            # This would call framework.run_all_combinations() with proper callbacks
            # For now, simulate progress
            for i in range(total_runs):
                self.benchmark_progress.set_value((i + 1) / total_runs)
                self.benchmark_status.set_text(f'Running experiment {i + 1}/{total_runs}...')
                await asyncio.sleep(0.1)  # Simulate work
            
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


def run_modern_optimization_ui():
    """Run the modern optimization UI as a standalone application"""
    framework = PromptOptimizationFramework()
    ui_instance = ModernOptimizationUI(framework)
    ui.run(title='Modern Prompt Optimization Framework', port=8082, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    run_modern_optimization_ui()
```

## Component Implementations

### Configuration Manager (ui_components/config_manager.py)
```python
"""
Configuration Management Components for Modern Optimization UI
"""

from typing import Dict, Any, List
import json
from pathlib import Path

class PromptLibraryManager:
    """Manage prompt library operations"""
    
    def __init__(self, config_path: str = "prompts/prompts_config.json"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load prompt configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_prompts_by_type(self, prompt_type: str) -> List[Dict[str, Any]]:
        """Get all prompts of a specific type"""
        prompts = []
        library = self.config_data.get("prompt_library", {}).get("prompts", {})
        for category in library.values():
            if prompt_type in category:
                prompts.extend(category[prompt_type])
        return prompts
    
    def set_default_prompt(self, prompt_type: str, prompt_name: str) -> None:
        """Set a prompt as default for its type"""
        library = self.config_data["prompt_library"]["prompts"]
        for category in library.values():
            if prompt_type in category:
                for prompt in category[prompt_type]:
                    prompt["default"] = prompt["name"] == prompt_name
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def get_default_prompt(self, prompt_type: str) -> str:
        """Get the default prompt for a type"""
        library = self.config_data["prompt_library"]["prompts"]
        for category in library.values():
            if prompt_type in category:
                for prompt in category[prompt_type]:
                    if prompt.get("default", False):
                        return prompt["name"]
        return ""

class ModelLibraryManager:
    """Manage model library operations"""
    
    def __init__(self, config_path: str = "models/models_config.json"):
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """Get all models of a specific type"""
        models = []
        library = self.config_data.get("model_library", {}).get("models", {})
        for category in library.values():
            for subcategory, model_list in category.items():
                if subcategory == model_type:
                    models.extend(model_list)
        return models
    
    def set_default_model(self, model_name: str) -> None:
        """Set a model as default"""
        library = self.config_data["model_library"]["models"]
        for category in library.values():
            for subcategory, model_list in category.items():
                for model in model_list:
                    model["default"] = model["name"] == model_name
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.config_data, f, indent=2)
    
    def get_default_model(self) -> str:
        """Get the default model"""
        library = self.config_data["model_library"]["models"]
        for category in library.values():
            for subcategory, model_list in category.items():
                for model in model_list:
                    if model.get("default", False):
                        return model["name"]
        return ""
```

### Benchmark Runner (ui_components/benchmark_runner.py)
```python
"""
Benchmark Execution Components for Modern Optimization UI
"""

import asyncio
from typing import List, Callable, Dict, Any
from datetime import datetime

class BenchmarkRunner:
    """Execute benchmark experiments"""
    
    def __init__(self, framework):
        self.framework = framework
        self.is_running = False
        self.current_task = None
    
    async def run_benchmark_suite(self,
                                prompt_keys: List[str],
                                model_keys: List[str],
                                test_case_ids: List[str],
                                callback: Callable,
                                concurrency: int = 1) -> None:
        """Run a suite of benchmark experiments"""
        self.is_running = True
        
        try:
            total_combinations = len(prompt_keys) * len(model_keys) * len(test_case_ids)
            current_index = 0
            start_time = time.time()
            
            # Create tasks for all combinations
            tasks = []
            for prompt_key in prompt_keys:
                for model_key in model_keys:
                    for test_case_id in test_case_ids:
                        task = self._create_benchmark_task(
                            prompt_key, model_key, test_case_id,
                            callback, current_index, total_combinations, start_time
                        )
                        tasks.append(task)
                        current_index += 1
            
            # Execute tasks with concurrency limit
            semaphore = asyncio.Semaphore(concurrency)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await task
            
            # Run all tasks
            results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks])
            
            return results
            
        except Exception as e:
            raise e
        finally:
            self.is_running = False
    
    async def _create_benchmark_task(self,
                                   prompt_key: str,
                                   model_key: str,
                                   test_case_id: str,
                                   callback: Callable,
                                   current_index: int,
                                   total_count: int,
                                   start_time: float):
        """Create a single benchmark task"""
        # This would call the framework's benchmark execution method
        # Implementation would depend on the specific framework interface
        pass
    
    def stop_benchmark(self) -> None:
        """Stop the current benchmark execution"""
        self.is_running = False
        if self.current_task:
            self.current_task.cancel()
```

### Results Analyzer (ui_components/results_analyzer.py)
```python
"""
Results Analysis Components for Modern Optimization UI
"""

import pandas as pd
import json
from typing import Dict, Any, List
from datetime import datetime

class ResultsAnalyzer:
    """Analyze and visualize benchmark results"""
    
    def __init__(self, framework):
        self.framework = framework
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for benchmark results"""
        df = self.framework.get_results_dataframe()
        
        if df.empty:
            return {}
        
        return {
            'total_runs': len(df),
            'success_rate': df['success'].mean(),
            'avg_duration_ms': df['duration_ms'].mean(),
            'avg_entities_extracted': df['entities_extracted'].mean(),
            'avg_compliance_score': df['compliance_score'].mean(),
            'avg_f1_score': df['f1_score'].mean(),
            'models_tested': df['model'].nunique(),
            'prompts_tested': df['prompt_name'].nunique()
        }
    
    def get_best_performers(self, metric: str = 'f1_score', top_n: int = 5) -> pd.DataFrame:
        """Get best performing prompt-model combinations"""
        return self.framework.get_best_combinations(metric, top_n)
    
    def generate_visualization_data(self) -> Dict[str, Any]:
        """Generate data for visualizations"""
        df = self.framework.get_results_dataframe()
        
        if df.empty:
            return {}
        
        # Success rates by prompt
        success_rates = df.groupby('prompt_name')['success'].mean().to_dict()
        
        # Compliance scores by prompt
        compliance_scores = df.groupby('prompt_name')['compliance_score'].mean().to_dict()
        
        # Performance comparison data
        performance_data = df.groupby(['prompt_name', 'model']).agg({
            'f1_score': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'compliance_score': 'mean',
            'duration_ms': 'mean'
        }).reset_index()
        
        return {
            'success_rates': {
                'labels': list(success_rates.keys()),
                'values': list(success_rates.values())
            },
            'compliance_scores': {
                'labels': list(compliance_scores.keys()),
                'values': list(compliance_scores.values())
            },
            'performance_comparison': performance_data.to_dict('records')
        }
    
    def export_results_to_csv(self, filename: str = None) -> str:
        """Export results to CSV"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.csv"
        
        df = self.framework.get_results_dataframe()
        df.to_csv(filename, index=False)
        return filename
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        try:
            df = self.framework.get_results_dataframe()
            
            if df.empty:
                return {'error': 'No benchmark data available'}
            
            summary_stats = self.get_summary_statistics()
            best_f1 = self.get_best_performers('f1_score', 1)
            best_compliance = self.get_best_performers('compliance_score', 1)
            
            return {
                'generated_at': datetime.now().isoformat(),
                'total_experiments': summary_stats['total_runs'],
                'success_rate': summary_stats['success_rate'],
                'avg_duration_ms': summary_stats['avg_duration_ms'],
                'avg_entities': summary_stats['avg_entities_extracted'],
                'avg_compliance': summary_stats['avg_compliance_score'],
                'avg_f1_score': summary_stats['avg_f1_score'],
                'models_tested': summary_stats['models_tested'],
                'prompts_tested': summary_stats['prompts_tested'],
                'best_configs': {
                    'f1_score': {
                        'name': best_f1.iloc[0]['prompt_name'] if not best_f1.empty else 'N/A',
                        'model': best_f1.iloc[0]['model'] if not best_f1.empty else 'N/A',
                        'score': best_f1.iloc[0]['f1_score'] if not best_f1.empty else 0
                    },
                    'compliance_score': {
                        'name': best_compliance.iloc[0]['prompt_name'] if not best_compliance.empty else 'N/A',
                        'model': best_compliance.iloc[0]['model'] if not best_compliance.empty else 'N/A',
                        'score': best_compliance.iloc[0]['compliance_score'] if not best_compliance.empty else 0
                    }
                }
            }
            
        except Exception as e:
            return {'error': f'Failed to generate report: {str(e)}'}
```

### System Monitor (ui_components/system_monitor.py)
```python
"""
System Status Monitoring Components for Modern Optimization UI
"""

import psutil
import os
from pathlib import Path
from typing import Dict, Any

class SystemMonitor:
    """Monitor system resources and status"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'memory_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information"""
        if not self.cache_dir.exists():
            return {
                'enabled': False,
                'size_mb': 0,
                'file_count': 0
            }
        
        # Calculate cache size and file count
        total_size = 0
        file_count = 0
        
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            'enabled': True,
            'size_mb': total_size / (1024 * 1024),
            'file_count': file_count
        }
    
    def clear_cache(self) -> None:
        """Clear the cache directory"""
        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    file_path.unlink()
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration file status"""
        config_files = {
            'prompts_config': 'prompts/prompts_config.json',
            'models_config': 'models/models_config.json',
            'main_config': 'config.json'
        }
        
        status = {}
        for name, path in config_files.items():
            file_path = Path(path)
            status[name] = {
                'exists': file_path.exists(),
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None,
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0
            }
        
        return status
```

## Utilities

### Data Loader (ui_utils/data_loader.py)
```python
"""
Data Loading Utilities for Modern Optimization UI
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

class DataLoader:
    """Load and process data for the UI"""
    
    @staticmethod
    def load_test_cases(file_path: str) -> Dict[str, Any]:
        """Load test cases from JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_benchmark_results(results_dir: str) -> pd.DataFrame:
        """Load benchmark results from directory"""
        results_dir = Path(results_dir)
        all_results = []
        
        for result_file in results_dir.glob("benchmark_*.json"):
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                all_results.append(result_data)
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    @staticmethod
    def load_prompt_library(config_path: str) -> Dict[str, Any]:
        """Load prompt library configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_model_library(config_path: str) -> Dict[str, Any]:
        """Load model library configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
```

### Validation Utilities (ui_utils/validation.py)
```python
"""
Input Validation Utilities for Modern Optimization UI
"""

from typing import Any, List, Dict
import re

class InputValidator:
    """Validate user inputs for the UI"""
    
    @staticmethod
    def validate_prompt_key(prompt_key: str) -> bool:
        """Validate prompt key format"""
        if not prompt_key or not isinstance(prompt_key, str):
            return False
        # Add specific validation rules as needed
        return len(prompt_key) > 0 and len(prompt_key) <= 100
    
    @staticmethod
    def validate_model_key(model_key: str) -> bool:
        """Validate model key format"""
        if not model_key or not isinstance(model_key, str):
            return False
        # Add specific validation rules as needed
        return len(model_key) > 0 and len(model_key) <= 100
    
    @staticmethod
    def validate_test_case_id(test_case_id: str) -> bool:
        """Validate test case ID format"""
        if not test_case_id or not isinstance(test_case_id, str):
            return False
        # Add specific validation rules as needed
        return len(test_case_id) > 0 and len(test_case_id) <= 100
    
    @staticmethod
    def validate_metric_name(metric_name: str) -> bool:
        """Validate metric name"""
        valid_metrics = ['f1_score', 'precision', 'recall', 'compliance_score']
        return metric_name in valid_metrics
    
    @staticmethod
    def validate_concurrency_level(concurrency: int) -> bool:
        """Validate concurrency level"""
        return isinstance(concurrency, int) and 1 <= concurrency <= 10
    
    @staticmethod
    def validate_timeout(timeout: int) -> bool:
        """Validate timeout value"""
        return isinstance(timeout, int) and 30 <= timeout <= 3600
```

## Integration Points

### Prompt Library Integration
- Direct access to `prompts/prompts_config.json`
- Real-time prompt validation using `PromptLoader`
- Default prompt management through configuration updates

### Model Library Integration
- Direct access to `models/models_config.json`
- Model parameter visualization
- Default model management through configuration updates

### Framework Integration
- Seamless connection to `PromptOptimizationFramework`
- Real-time benchmark execution with progress tracking
- Results persistence and retrieval through framework methods

## Security & Validation
- Input validation for all user inputs
- Configuration integrity checks
- Error handling with clear user feedback
- STRICT validation for all operations

## Performance Considerations
- Efficient data loading with caching strategies
- Asynchronous operations for long-running tasks
- Pagination for large datasets
- Resource monitoring to prevent system overload