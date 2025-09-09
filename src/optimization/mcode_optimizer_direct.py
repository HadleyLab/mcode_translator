"""mCODE Translation Optimizer - Direct UI Updates"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any

from nicegui import ui, run, app

from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader
from src.pipeline.mcode_pipeline import McodePipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class McodeOptimizerDirect:
    """Direct UI updates without complex reactivity"""
    
    def __init__(self):
        """Initialize optimizer"""
        self.tasks = {}
        self.is_running = False
        self.max_concurrent_tasks = 5
        self.pipeline_cache = {}
        
        # UI elements for direct updates
        self.status_labels = {}
        self.result_rows = {}
        
        # Load data and setup
        self._load_data()
        self._create_tasks()
    
    def _load_data(self):
        """Load configuration data"""
        self.models = model_loader.get_all_models()
        
        # Get direct mCODE prompts only
        pipeline_prompts = prompt_loader.get_prompts_by_pipeline('McodePipeline')
        self.prompts = pipeline_prompts.get('DIRECT_MCODE', [])
        
        # Load trial data
        trial_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.trial.json")
        with open(trial_file, 'r') as f:
            trial_data = json.load(f)
        self.trials = trial_data.get('test_cases', {})
        
        # Load gold standard data for expected elements
        gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
        with open(gold_file, 'r') as f:
            gold_data = json.load(f)
        
        # Add expected elements to trial data
        for trial_id in self.trials.keys():
            if trial_id in gold_data.get('gold_standard', {}):
                gold_trial = gold_data['gold_standard'][trial_id]
                expected_mappings = gold_trial.get('expected_mcode_mappings', {}).get('mapped_elements', [])
                # Extract the mCODE element names for comparison
                expected_elements = [mapping['Mcode_element'] for mapping in expected_mappings]
                self.trials[trial_id]['expected_elements'] = expected_elements
        
        # Load gold standard data for expected elements
        gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
        with open(gold_file, 'r') as f:
            gold_data = json.load(f)
        
        # Add expected elements to trial data
        for trial_id in self.trials.keys():
            if trial_id in gold_data.get('gold_standard', {}):
                gold_trial = gold_data['gold_standard'][trial_id]
                expected_mappings = gold_trial.get('expected_mcode_mappings', {}).get('mapped_elements', [])
                # Extract the element names for comparison
                expected_elements = [mapping['Mcode_element'] for mapping in expected_mappings]
                self.trials[trial_id]['expected_elements'] = expected_elements
    
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
    
    def setup_ui(self):
        """Setup main UI with direct element references"""
        ui.label('🎯 mCODE Optimization - Direct Updates').classes('text-3xl font-bold text-center mb-6')
        
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
        
        # Stats section with direct label references
        with ui.card().classes('w-full mb-4'):
            ui.label('📊 Statistics').classes('text-xl font-bold mb-4')
            
            with ui.row().classes('gap-4'):
                self.status_labels['queued'] = ui.chip('Queued: 0', color='gray')
                self.status_labels['processing'] = ui.chip('Processing: 0', color='orange')
                self.status_labels['success'] = ui.chip('Success: 0', color='green')
                self.status_labels['failed'] = ui.chip('Failed: 0', color='red')
                self.status_labels['avg_f1'] = ui.chip('Avg F1: 0.000', color='blue')
                self.status_labels['best_f1'] = ui.chip('Best F1: 0.000', color='blue')
        
        # Results table with direct row references
        with ui.card().classes('w-full'):
            ui.label('📋 Results Table').classes('text-xl font-bold mb-4')
            
            with ui.scroll_area().classes('w-full h-[600px]'):
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
                    
                    # Data rows with direct element references
                    for task_id, task in self.tasks.items():
                        self._create_task_row(task)
        
        # Initial stats update
        self._update_stats_display()
    
    def _create_task_row(self, task):
        """Create a task row with direct element references"""
        status_colors = {'Queued': 'gray', 'Processing': 'orange', 'Success': 'green', 'Failed': 'red'}
        
        with ui.row().classes('w-full p-2 text-xs border-b items-center hover:bg-gray-50'):
            # Store references for direct updates
            row_elements = {}
            
            # Status chip
            row_elements['status'] = ui.chip(task['status'], color=status_colors.get(task['status'], 'gray')).classes('w-20')
            
            # Static info
            ui.label(task['model'][:20]).classes('w-24 truncate')
            ui.label(task['prompt'][:28]).classes('w-32 truncate')
            ui.label(task['trial']).classes('w-16')
            
            # Dynamic metrics
            row_elements['f1'] = ui.label('-').classes('w-16 text-right')
            row_elements['precision'] = ui.label('-').classes('w-16 text-right')
            row_elements['recall'] = ui.label('-').classes('w-16 text-right')
            row_elements['duration'] = ui.label('-').classes('w-20 text-right')
            row_elements['elements'] = ui.label('0').classes('w-16 text-right')
            
            # Details button
            ui.button('📋', on_click=lambda t=task: self._show_task_details(t)).classes('w-16 text-xs')
            
            # Store row elements for direct updates
            self.result_rows[task['id']] = row_elements
    
    def _update_stats_display(self):
        """Update statistics display directly"""
        # Count by status
        status_counts = {'Queued': 0, 'Processing': 0, 'Success': 0, 'Failed': 0}
        for task in self.tasks.values():
            status = task['status']
            if status in status_counts:
                status_counts[status] += 1
        
        # Update status chips directly
        self.status_labels['queued'].text = f'Queued: {status_counts["Queued"]}'
        self.status_labels['processing'].text = f'Processing: {status_counts["Processing"]}'
        self.status_labels['success'].text = f'Success: {status_counts["Success"]}'
        self.status_labels['failed'].text = f'Failed: {status_counts["Failed"]}'
        
        # Performance stats
        success_tasks = [t for t in self.tasks.values() if t['status'] == 'Success' and t['f1_score'] is not None]
        if success_tasks:
            avg_f1 = sum(t['f1_score'] for t in success_tasks) / len(success_tasks)
            best_f1 = max(t['f1_score'] for t in success_tasks)
            self.status_labels['avg_f1'].text = f'Avg F1: {avg_f1:.3f}'
            self.status_labels['best_f1'].text = f'Best F1: {best_f1:.3f}'
        else:
            self.status_labels['avg_f1'].text = 'Avg F1: 0.000'
            self.status_labels['best_f1'].text = 'Best F1: 0.000'
    
    def _update_task_row(self, task):
        """Update a specific task row directly"""
        if task['id'] not in self.result_rows:
            return
        
        row_elements = self.result_rows[task['id']]
        status_colors = {'Queued': 'gray', 'Processing': 'orange', 'Success': 'green', 'Failed': 'red'}
        
        # Update status chip
        row_elements['status'].text = task['status']
        row_elements['status'].props(f'color={status_colors.get(task["status"], "gray")}')
        
        # Update metrics
        row_elements['f1'].text = f"{task['f1_score']:.3f}" if task['f1_score'] is not None else "-"
        row_elements['precision'].text = f"{task['precision']:.3f}" if task['precision'] is not None else "-"
        row_elements['recall'].text = f"{task['recall']:.3f}" if task['recall'] is not None else "-"
        row_elements['duration'].text = str(task['duration_ms']) if task['duration_ms'] is not None else "-"
        row_elements['elements'].text = str(len(task['extracted_elements']))
    
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
            
            elif task['status'] == 'Failed' and task['error_message']:
                ui.label("Error:").classes('font-bold text-red-500')
                ui.label(task['error_message']).classes('text-red-500')
            
            if task['logs']:
                ui.label("Logs:").classes('font-bold mt-4')
                with ui.scroll_area().classes('h-32 w-full'):
                    for log in task['logs']:
                        ui.label(log).classes('text-sm text-gray-600')
            
            ui.button('Close', on_click=dialog.close)
        
        dialog.open()
    
    def _refresh_ui_for_task(self, task_id):
        """Refresh UI immediately when a task completes (called from main thread)"""
        if task_id in self.tasks:
            # Update the specific task row
            self._update_task_row(self.tasks[task_id])
            
            # Update overall statistics
            self._update_stats_display()

    async def _async_refresh_ui(self, task_id):
        """Async wrapper to safely update UI from background tasks"""
        try:
            # Use NiceGUI's safe way to update UI from async context
            self._refresh_ui_for_task(task_id)
        except Exception as e:
            print(f"DEBUG: Error updating UI for task {task_id}: {e}")
    
    def _update_concurrency(self, e):
        """Update concurrency limit"""
        self.max_concurrent_tasks = int(e.value)
    
    def _start_optimization(self):
        """Start the optimization process"""
        if self.is_running:
            return
            
        self.is_running = True
        self.start_button.set_enabled(False)
        self.stop_button.set_enabled(True)
        
        ui.notify(f'Starting optimization with {self.max_concurrent_tasks} concurrent workers...')
        
        # Start the actual processing
        asyncio.create_task(self._run_optimization())
    
    def _stop_optimization(self):
        """Stop the optimization process"""
        self.is_running = False
        self.start_button.set_enabled(True)
        self.stop_button.set_enabled(False)
        ui.notify('Optimization stopped')
    
    async def _run_optimization(self):
        """Run optimization with callback-based UI updates"""
        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
            
            # Create tasks for processing
            async def process_task_with_semaphore(task):
                async with semaphore:
                    await self._process_single_task(task)
            
            # Start all queued tasks
            queued_tasks = [t for t in self.tasks.values() if t['status'] == 'Queued']
            await asyncio.gather(*[process_task_with_semaphore(task) for task in queued_tasks])
            
        except Exception as e:
            logger.error(f'Optimization error: {str(e)}')
        
        finally:
            if self.is_running:  # Only update if not manually stopped
                self.is_running = False
                # Use simple flag to signal completion
                self.optimization_complete = True
    
    async def _process_single_task(self, task):
        """Process a single optimization task with callback-based UI updates"""
        if not self.is_running:
            return
        
        try:
            # Update status to processing
            task['status'] = 'Processing'
            task['logs'].append("🔄 Starting task processing...")
            print(f"DEBUG: Starting task {task['id']} - Model: {task['model_key']}, Prompt: {task['prompt_key']}, Trial: {task['trial']}")
            
            # Update task status to Processing for UI\n            task['status'] = 'Processing'
            
            # Get or create pipeline
            pipeline_key = task['model_key']
            print(f"DEBUG: Getting pipeline for {pipeline_key}")
            if pipeline_key not in self.pipeline_cache:
                print(f"DEBUG: Creating new pipeline for {pipeline_key}")
                
                # Set model configuration through environment
                import os
                original_model = os.environ.get('MODEL_NAME')
                os.environ['MODEL_NAME'] = task['model_key']
                print(f"DEBUG: Set MODEL_NAME environment variable to {task['model_key']}")
                
                try:
                    self.pipeline_cache[pipeline_key] = McodePipeline(
                        prompt_name=task['prompt_key']
                    )
                    print(f"DEBUG: Pipeline created successfully")
                finally:
                    # Restore original model if it existed
                    if original_model:
                        os.environ['MODEL_NAME'] = original_model
                    else:
                        os.environ.pop('MODEL_NAME', None)
            pipeline = self.pipeline_cache[pipeline_key]
            
            # Debug trial data
            print(f"DEBUG: Available trial data keys: {list(task['trial_data'].keys())}")
            
            # Get trial identifier for debugging
            trial_id = "unknown"
            if 'protocolSection' in task['trial_data']:
                protocol = task['trial_data']['protocolSection']
                if 'identificationModule' in protocol:
                    trial_id = protocol['identificationModule'].get('nctId', 'unknown')
            print(f"DEBUG: Processing trial ID: {trial_id}")
            
            # Process the task
            start_time = time.time()
            print(f"DEBUG: Calling pipeline.process_clinical_trial...")
            
            # Set model environment variable for processing
            import os
            original_model = os.environ.get('MODEL_NAME')
            os.environ['MODEL_NAME'] = task['model_key']
            print(f"DEBUG: Set MODEL_NAME to {task['model_key']} for processing")
            
            try:
                result = pipeline.process_clinical_trial(
                    trial_data=task['trial_data'],
                    task_id=str(task['id'])
                )
            finally:
                # Restore original model environment
                if original_model:
                    os.environ['MODEL_NAME'] = original_model
                else:
                    os.environ.pop('MODEL_NAME', None)
            
            end_time = time.time()
            task['duration_ms'] = int((end_time - start_time) * 1000)
            print(f"DEBUG: Pipeline call completed in {task['duration_ms']}ms")
            print(f"DEBUG: Result type: {type(result)}")
            
            # Extract results from PipelineResult object
            if result and not result.error:
                print(f"DEBUG: Task {task['id']} - Success result received")
                print(f"DEBUG: mCODE mappings count: {len(result.mcode_mappings)}")
                print(f"DEBUG: Validation results: {result.validation_results}")
                
                # Store extracted mCODE elements 
                task['extracted_elements'] = [
                    mapping.get('Mcode_element', 'Unknown') 
                    for mapping in result.mcode_mappings
                ]
                
                # Calculate validation metrics
                if 'expected_elements' in task['trial_data']:
                    expected = set(task['trial_data']['expected_elements'])
                    extracted = set(task['extracted_elements'])
                    print(f"DEBUG: Expected: {expected}")
                    print(f"DEBUG: Extracted: {extracted}")
                    
                    true_positives = len(expected & extracted)
                    false_positives = len(extracted - expected)
                    false_negatives = len(expected - extracted)
                    
                    task['precision'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    task['recall'] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    task['f1_score'] = 2 * (task['precision'] * task['recall']) / (task['precision'] + task['recall']) if (task['precision'] + task['recall']) > 0 else 0
                    print(f"DEBUG: Calculated F1: {task['f1_score']:.3f}")
                else:
                    print(f"DEBUG: No expected_elements in trial data - using default metrics")
                    task['precision'] = 0.0
                    task['recall'] = 0.0
                    task['f1_score'] = 0.0
                
                task['status'] = 'Success'
                task['logs'].append(f"✅ Success! F1: {task['f1_score']:.3f}, Elements: {len(task['extracted_elements'])}")
            
            else:
                print(f"DEBUG: Task {task['id']} - Failed or no result")
                if result:
                    print(f"DEBUG: Result error: {result.error}")
                    print(f"DEBUG: Result validation: {result.validation_results}")
                else:
                    print(f"DEBUG: No result returned from pipeline")
                    
                task['status'] = 'Failed'
                task['error_message'] = result.error if result and result.error else 'Unknown error or no result returned'
                task['logs'].append(f"❌ Failed: {task['error_message']}")
        
        except Exception as e:
            print(f"DEBUG: Exception in task {task['id']}: {str(e)}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            
            task['status'] = 'Failed'
            task['error_message'] = str(e)
            task['logs'].append(f"❌ Exception: {str(e)}")
        
        finally:
            # Trigger immediate UI refresh using async callback
            asyncio.create_task(self._async_refresh_ui(task['id']))


# Main execution
def main():
    """Main function to run the optimizer"""
    optimizer = McodeOptimizerDirect()
    optimizer.setup_ui()


if __name__ in {"__main__", "__mp_main__"}:
    main()
    ui.run(title="mCODE Optimizer - Direct Updates", port=8085)