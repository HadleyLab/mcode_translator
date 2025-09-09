"""mCODE Translation Optimizer - Proper Reactive Implementation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from nicegui import ui, run, background_tasks, app

from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader
from src.pipeline.mcode_pipeline import McodePipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class McodeOptimizerProper:
    """Proper mCODE optimizer using NiceGUI reactive state"""
    
    def __init__(self):
        """Initialize optimizer"""
        self.tasks = {}
        self.is_running = False
        self.max_concurrent_tasks = 5
        self.pipeline_cache = {}
        
        # Reactive state for UI updates
        self.task_status_counts = {'Queued': 0, 'Processing': 0, 'Success': 0, 'Failed': 0}
        self.performance_stats = {'avg_f1': 0, 'best_f1': 0}
        
        # Load data and setup
        self._load_data()
        self._create_tasks()
        self._update_stats()
    
    def _load_data(self):
        """Load configuration data"""
        self.pipelines = prompt_loader.list_available_pipelines()
        self.models = model_loader.get_all_models()
        
        # Get direct mCODE prompts only
        pipeline_prompts = prompt_loader.get_prompts_by_pipeline('McodePipeline')
        self.prompts = pipeline_prompts.get('DIRECT_MCODE', [])
        
        # Load trial data
        trial_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.trial.json")
        with open(trial_file, 'r') as f:
            trial_data = json.load(f)
        self.trials = trial_data.get('test_cases', {})
    
    def _create_tasks(self):
        """Create tasks for all combinations"""
        task_id = 0
        for model_key, model_config in self.models.items():
            for prompt_info in self.prompts:
                for trial_id in self.trials.keys():
                    task = {
                        'id': str(task_id),
                        'model': model_config.name,
                        'model_key': model_key,
                        'prompt': prompt_info['name'],
                        'prompt_key': prompt_info['key'] if 'key' in prompt_info else prompt_info['name'],
                        'trial': trial_id,
                        'trial_data': self.trials[trial_id],
                        'status': 'Queued',
                        'f1_score': None,
                        'precision': None,
                        'recall': None,
                        'duration_ms': None,
                        'mcode_extraction': None,
                        'extracted_elements': [],
                        'validation_results': None,
                        'token_usage': None,
                        'error_message': None,
                        'logs': []
                    }
                    self.tasks[str(task_id)] = task
                    task_id += 1
    
    def _update_stats(self):
        """Update reactive statistics"""
        # Count by status
        self.task_status_counts = {'Queued': 0, 'Processing': 0, 'Success': 0, 'Failed': 0}
        for task in self.tasks.values():
            status = task['status']
            if status in self.task_status_counts:
                self.task_status_counts[status] += 1
        
        # Performance stats
        success_tasks = [t for t in self.tasks.values() if t['status'] == 'Success']
        if success_tasks:
            self.performance_stats['avg_f1'] = sum(t['f1_score'] or 0 for t in success_tasks) / len(success_tasks)
            self.performance_stats['best_f1'] = max(t['f1_score'] or 0 for t in success_tasks)
        else:
            self.performance_stats = {'avg_f1': 0, 'best_f1': 0}
    
    def setup_ui(self):
        """Setup main UI with proper reactive bindings"""
        ui.label('🎯 mCODE Optimization - Proper Reactive').classes('text-3xl font-bold text-center mb-6')
        
        # Controls section
        with ui.card().classes('w-full mb-4'):
            ui.label('⚙️ Controls').classes('text-xl font-bold mb-4')
            
            with ui.row().classes('gap-4 items-center'):
                ui.label('Concurrent Tasks:')
                self.concurrency_slider = ui.slider(
                    min=1, max=10, value=self.max_concurrent_tasks,
                    on_change=self._update_concurrency
                ).classes('w-48')
                ui.label().bind_text_from(self.concurrency_slider, 'value')
                
                ui.separator().props('vertical')
                
                self.start_button = ui.button(
                    '🚀 Start Optimization', 
                    on_click=self._start_optimization
                ).classes('bg-green-500 text-white')
                
                self.stop_button = ui.button(
                    '⏹️ Stop', 
                    on_click=self._stop_optimization
                ).classes('bg-red-500 text-white')
                self.stop_button.set_enabled(False)
        
        # Stats section with reactive binding
        with ui.card().classes('w-full mb-4'):
            ui.label('📊 Statistics').classes('text-xl font-bold mb-4')
            
            with ui.row().classes('gap-4') as self.stats_row:
                # Status counts - bound to reactive state
                for status, color in [('Queued', 'gray'), ('Processing', 'orange'), ('Success', 'green'), ('Failed', 'red')]:
                    ui.chip().bind_text_from(self, 'task_status_counts', 
                                           backward=lambda d, s=status: f'{s}: {d.get(s, 0)}').props(f'color={color}')
                
                # Performance stats - bound to reactive state  
                ui.chip().bind_text_from(self, 'performance_stats',
                                       backward=lambda d: f'Avg F1: {d.get("avg_f1", 0):.3f}').props('color=blue')
                ui.chip().bind_text_from(self, 'performance_stats', 
                                       backward=lambda d: f'Best F1: {d.get("best_f1", 0):.3f}').props('color=blue')
        
        # Results table with reactive content
        with ui.card().classes('w-full'):
            ui.label('📋 Results Table').classes('text-xl font-bold mb-4')
            
            with ui.scroll_area().classes('w-full h-[600px]'):
                self.results_container = ui.column().classes('w-full')
                self._render_results_table()
    
    def _render_results_table(self):
        """Render results table"""
        self.results_container.clear()
        
        with self.results_container:
            with ui.card().classes('w-full overflow-x-auto'):
                # Header
                with ui.row().classes('w-full bg-gray-100 p-2 font-bold text-xs border-b items-center'):
                    ui.label('Status').classes('w-20')
                    ui.label('Model').classes('w-24')
                    ui.label('Prompt').classes('w-32')
                    ui.label('Trial').classes('w-16')
                    ui.label('F1').classes('w-16 text-right')
                    ui.label('Prec').classes('w-16 text-right')
                    ui.label('Recall').classes('w-16 text-right')
                    ui.label('Time (ms)').classes('w-20 text-right')
                    ui.label('Elements').classes('w-16 text-right')
                    ui.label('Details').classes('w-16 text-center')
                
                # Data rows
                for task in self.tasks.values():
                    self._render_task_row(task)
    
    def _render_task_row(self, task):
        """Render a single task row"""
        status_colors = {
            'Queued': 'gray',
            'Processing': 'orange',
            'Success': 'green', 
            'Failed': 'red'
        }
        
        with ui.row().classes('w-full p-2 text-xs border-b items-center hover:bg-gray-50'):
            # Status
            ui.chip(task['status'], color=status_colors.get(task['status'], 'gray')).classes('w-20')
            
            # Model
            ui.label(task['model'][:20]).classes('w-24 truncate')
            
            # Prompt  
            ui.label(task['prompt'][:28]).classes('w-32 truncate')
            
            # Trial
            ui.label(task['trial']).classes('w-16')
            
            # F1 Score
            f1_text = f"{task['f1_score']:.3f}" if task['f1_score'] is not None else "-"
            ui.label(f1_text).classes('w-16 text-right')
            
            # Precision
            prec_text = f"{task['precision']:.3f}" if task['precision'] is not None else "-"
            ui.label(prec_text).classes('w-16 text-right')
            
            # Recall
            recall_text = f"{task['recall']:.3f}" if task['recall'] is not None else "-"
            ui.label(recall_text).classes('w-16 text-right')
            
            # Duration
            dur_text = str(task['duration_ms']) if task['duration_ms'] is not None else "-"
            ui.label(dur_text).classes('w-20 text-right')
            
            # Elements count
            elem_count = len(task['extracted_elements']) if task['extracted_elements'] else 0
            ui.label(str(elem_count)).classes('w-16 text-right')
            
            # Details button
            ui.button('📋', on_click=lambda t=task: self._show_task_details(t)).classes('w-16 text-xs')
    
    def _show_task_details(self, task):
        """Show detailed task information"""
        with ui.dialog() as dialog, ui.card().classes('w-[800px]'):
            ui.label(f"Task Details - {task['id']}").classes('text-xl font-bold mb-4')
            
            with ui.row().classes('gap-4 mb-4'):
                ui.label(f"Model: {task['model']}")
                ui.label(f"Prompt: {task['prompt']}")
                ui.label(f"Trial: {task['trial']}")
                ui.label(f"Status: {task['status']}")
            
            if task['status'] == 'Success':
                with ui.column().classes('gap-2'):
                    ui.label(f"F1 Score: {task['f1_score']:.3f}")
                    ui.label(f"Precision: {task['precision']:.3f}")
                    ui.label(f"Recall: {task['recall']:.3f}")
                    ui.label(f"Duration: {task['duration_ms']}ms")
                    ui.label(f"Extracted Elements: {len(task['extracted_elements'])}")
                    
                    if task['extracted_elements']:
                        ui.label("mCODE Elements:").classes('font-bold mt-2')
                        with ui.scroll_area().classes('h-48 w-full'):
                            for elem in task['extracted_elements']:
                                ui.label(f"• {elem}").classes('text-sm')
            
            elif task['status'] == 'Failed' and task['error_message']:
                ui.label("Error:").classes('font-bold text-red-500')
                ui.label(task['error_message']).classes('text-red-500')
            
            # Logs
            if task['logs']:
                ui.label("Logs:").classes('font-bold mt-4')
                with ui.scroll_area().classes('h-32 w-full'):
                    for log in task['logs']:
                        ui.label(log).classes('text-sm text-gray-600')
            
            ui.button('Close', on_click=dialog.close)
        
        dialog.open()
    
    def _update_concurrency(self, e):
        """Update concurrency limit"""
        self.max_concurrent_tasks = int(e.value)
    
    def _update_ui_state(self):
        """Update UI state and trigger reactive updates"""
        self._update_stats()
        self._render_results_table()
    
    async def _start_optimization(self):
        """Start the optimization process"""
        self.is_running = True
        self.start_button.set_enabled(False)
        self.stop_button.set_enabled(True)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Create tasks for processing
        async def process_task_with_semaphore(task):
            async with semaphore:
                await self._process_single_task(task)
        
        # Start all tasks
        queued_tasks = [t for t in self.tasks.values() if t['status'] == 'Queued']
        await asyncio.gather(*[process_task_with_semaphore(task) for task in queued_tasks])
        
        self.is_running = False
        self.start_button.set_enabled(True)
        self.stop_button.set_enabled(False)
        
        # Final update
        self._update_ui_state()
    
    def _stop_optimization(self):
        """Stop the optimization process"""
        self.is_running = False
        self.start_button.set_enabled(True)
        self.stop_button.set_enabled(False)
    
    async def _process_single_task(self, task):
        """Process a single optimization task"""
        if not self.is_running:
            return
        
        try:
            # Update status to processing
            task['status'] = 'Processing'
            task['logs'].append("🔄 Starting task processing...")
            self._update_ui_state()  # Update UI immediately
            
            # Get or create pipeline
            pipeline_key = task['model_key']
            if pipeline_key not in self.pipeline_cache:
                self.pipeline_cache[pipeline_key] = McodePipeline(
                    model_name=task['model_key']
                )
            pipeline = self.pipeline_cache[pipeline_key]
            
            # Process the task
            start_time = time.time()
            
            result = pipeline.process_direct_mcode(
                text=task['trial_data']['text'],
                prompt_key=task['prompt_key']
            )
            
            end_time = time.time()
            task['duration_ms'] = int((end_time - start_time) * 1000)
            
            # Extract results
            if result and result.get('status') == 'success':
                mcode_data = result.get('mcode_extraction', {})
                task['mcode_extraction'] = mcode_data
                task['extracted_elements'] = list(mcode_data.keys()) if mcode_data else []
                
                # Calculate validation metrics
                if 'expected_elements' in task['trial_data']:
                    expected = set(task['trial_data']['expected_elements'])
                    extracted = set(task['extracted_elements'])
                    
                    true_positives = len(expected & extracted)
                    false_positives = len(extracted - expected)
                    false_negatives = len(expected - extracted)
                    
                    task['precision'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    task['recall'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    task['f1_score'] = 2 * (task['precision'] * task['recall']) / (task['precision'] + task['recall']) if (task['precision'] + task['recall']) > 0 else 0
                
                # Token usage
                task['token_usage'] = result.get('token_usage', {})
                
                task['status'] = 'Success'
                task['logs'].append(f"✅ Success! F1: {task['f1_score']:.3f}, Elements: {len(task['extracted_elements'])}")
            
            else:
                task['status'] = 'Failed'
                task['error_message'] = result.get('error', 'Unknown error') if result else 'No result returned'
                task['logs'].append(f"❌ Failed: {task['error_message']}")
        
        except Exception as e:
            task['status'] = 'Failed'
            task['error_message'] = str(e)
            task['logs'].append(f"❌ Exception: {str(e)}")
        
        finally:
            # Update UI after task completion
            self._update_ui_state()


# Main execution
def main():
    """Main function to run the optimizer"""
    optimizer = McodeOptimizerProper()
    optimizer.setup_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()
    ui.run(title="mCODE Optimizer - Proper", port=8082)