"""mCODE Translation Optimizer - Reactive UI Implementation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from nicegui import ui, run, background_tasks

from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader
from src.pipeline.mcode_pipeline import McodePipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class McodeOptimizerReactive:
    """Reactive mCODE optimizer using @ui.refreshable for automatic UI updates"""
    
    def __init__(self):
        """Initialize with reactive task state"""
        self.tasks = {}  # Dict for quick lookup
        self.pipeline_cache = {}
        
        # Load data and setup
        self._load_data()
        self._create_tasks()
    
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
                    self.tasks[str(task_id)] = {
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
                    task_id += 1
    
    def setup_ui(self):
        """Setup main UI with reactive state management"""
        # Initialize reactive state
        self.is_running, self.set_is_running = ui.state(False)
        self.max_concurrent_tasks, self.set_max_concurrent_tasks = ui.state(5)
        self.task_refresh_counter, self.set_task_refresh_counter = ui.state(0)
        
        ui.label('🎯 mCODE Optimization - Reactive Processing').classes('text-3xl font-bold text-center mb-6')
        
        # Controls section
        self._render_controls()
        
        # Results section
        self._render_results()
    
    @ui.refreshable
    def _render_controls(self):
        """Render controls section with reactive updates"""
        with ui.card().classes('w-full mb-4'):
            ui.label('⚙️ Controls').classes('text-xl font-bold mb-4')
            
            with ui.row().classes('gap-4 items-center'):
                ui.label('Concurrent Tasks:')
                ui.slider(
                    min=1, max=10, value=self.max_concurrent_tasks,
                    on_change=lambda e: self.set_max_concurrent_tasks(int(e.value))
                ).classes('w-48')
                ui.label().bind_text_from(self.max_concurrent_tasks, lambda x: str(x))
                
                ui.separator().props('vertical')
                
                ui.button(
                    '🚀 Start Optimization', 
                    on_click=self._start_optimization
                ).classes('bg-green-500 text-white').bind_enabled_from(self.is_running, lambda x: not x)
                
                ui.button(
                    '⏹️ Stop', 
                    on_click=self._stop_optimization
                ).classes('bg-red-500 text-white').bind_enabled_from(self.is_running, lambda x: x)
    
    @ui.refreshable 
    def _render_results(self):
        """Render results section with reactive updates"""
        with ui.card().classes('w-full'):
            ui.label('📊 Results').classes('text-xl font-bold mb-4')
            
            # Summary stats
            self._render_summary_stats()
            
            # Results table
            with ui.scroll_area().classes('w-full h-[600px]'):
                self._render_results_table()
    
    def _render_summary_stats(self):
        """Render summary statistics"""
        with ui.row().classes('gap-4 mb-4'):
            # Count by status
            status_counts = {}
            for task in self.tasks.values():
                status = task['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Create chips for each status
            colors = {
                'Queued': 'gray',
                'Processing': 'orange', 
                'Success': 'green',
                'Failed': 'red'
            }
            
            for status, count in status_counts.items():
                if count > 0:
                    ui.chip(f'{status}: {count}', color=colors.get(status, 'gray'))
            
            # Performance stats
            success_tasks = [t for t in self.tasks.values() if t['status'] == 'Success']
            if success_tasks:
                avg_f1 = sum(t['f1_score'] or 0 for t in success_tasks) / len(success_tasks)
                best_f1 = max(t['f1_score'] or 0 for t in success_tasks)
                ui.chip(f'Avg F1: {avg_f1:.3f}', color='blue')
                ui.chip(f'Best F1: {best_f1:.3f}', color='blue')
    
    def _render_results_table(self):
        """Render the complete results table"""
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
    
    async def _start_optimization(self):
        """Start the optimization process with reactive updates"""
        self.set_is_running(True)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Create tasks for processing
        async def process_task_with_semaphore(task):
            async with semaphore:
                await self._process_single_task(task)
        
        # Start all tasks
        queued_tasks = [t for t in self.tasks.values() if t['status'] == 'Queued']
        await asyncio.gather(*[process_task_with_semaphore(task) for task in queued_tasks])
        
        self.set_is_running(False)
    
    def _stop_optimization(self):
        """Stop the optimization process"""
        self.set_is_running(False)
    
    async def _process_single_task(self, task):
        """Process a single optimization task with reactive updates"""
        if not self.is_running:
            return
        
        try:
            # Update status to processing
            task['status'] = 'Processing'
            task['logs'].append("🔄 Starting task processing...")
            self._trigger_ui_refresh()  # Trigger UI update
            
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
            # Trigger UI update
            self._trigger_ui_refresh()
    
    def _trigger_ui_refresh(self):
        """Trigger UI refresh by updating the refresh counter"""
        self.set_task_refresh_counter(self.task_refresh_counter + 1)
        # Force refresh of results section
        self._render_results.refresh()


# Main execution
def main():
    """Main function to run the optimizer"""
    optimizer = McodeOptimizerReactive()
    optimizer.setup_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()
    ui.run(title="mCODE Optimizer - Reactive", port=8080)