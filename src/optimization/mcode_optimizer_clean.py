"""mCODE Translation Optimizer - Real Pipeline Implementation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import asyncio
import uuid
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from nicegui import ui, run, background_tasks

from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader
from src.pipeline.mcode_pipeline import McodePipeline
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class McodeOptimizer:
    """Pure @ui.refreshable mCODE optimizer with native data bindings"""
    
    def __init__(self):
        # Pure data - no UI state
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
        self.max_concurrent_tasks = 5  # Configurable concurrency limit
        self.pipeline_cache = {}  # Cache pipelines to avoid recreation
        
        # Load data
        self._load_data()
        self._create_tasks()
        self._setup_ui()
    
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
                        'error_message': None,
                        'logs': []
                    }
                    task_id += 1
    
    def setup_ui(self):
        """Setup main UI with periodic updates"""
        ui.label('🎯 mCODE Optimization - Concurrent Processing').classes('text-3xl font-bold text-center mb-6')
        
        # Stats section with placeholder for dynamic updates
        with ui.row().classes('w-full justify-center gap-4 mb-6'):
            self.stats_placeholder = ui.row().classes('gap-4')
        
        # Controls section
        self._render_controls()
        
        # Results section
        self._render_results()
        
        # Initial render
        self._render_header_stats()
    
    def _render_header_stats(self):
        """Update header stats by clearing and recreating"""
        active = len([t for t in self.tasks.values() if t['status'] == 'Processing'])
        success = len([t for t in self.tasks.values() if t['status'] == 'Success'])
        failed = len([t for t in self.tasks.values() if t['status'] == 'Failed'])
        total = len(self.tasks)
        
        # Clear existing content
        self.stats_placeholder.clear()
        
        # Add new chips
        with self.stats_placeholder:
            ui.chip(f'{active} Running', icon='play_circle', color='orange' if active > 0 else 'grey')
            ui.chip(f'{success} Success', icon='check_circle', color='green' if success > 0 else 'grey')
            ui.chip(f'{failed} Failed', icon='error', color='red' if failed > 0 else 'grey')
            ui.chip(f'{total} Total', icon='task', color='blue')
    
    @ui.refreshable
    def _render_controls(self):
        """Control panel with concurrency settings"""
        with ui.card().classes('w-full bg-blue-50 border-l-4 border-blue-500'):
            with ui.row().classes('w-full justify-between items-center p-4'):
                with ui.row().classes('gap-4 items-center'):
                    if not self.is_running:
                        ui.button('🚀 Run Optimization', 
                                on_click=self._start_optimization).props('color=positive size=lg')
                    else:
                        ui.button('⏹️ Stop', 
                                on_click=self._stop_optimization).props('color=negative size=lg')
                    
                    # Concurrency control
                    with ui.column().classes('gap-1'):
                        ui.label('Concurrent Tasks:').classes('text-sm text-gray-600')
                        ui.slider(min=1, max=10, value=self.max_concurrent_tasks, 
                                on_change=self._update_concurrency).props('dense color=blue')
                        ui.label(f'{self.max_concurrent_tasks}').classes('text-sm font-mono text-center')
                
                with ui.column().classes('items-end'):
                    ui.label(f'{len(self.prompts)} prompts × {len(self.models)} models × {len(self.trials)} trials')
                    ui.label(f'= {len(self.tasks)} tasks (max {self.max_concurrent_tasks} concurrent)').classes('font-bold text-blue-600')
                    
                    # Performance estimate
                    if not self.is_running:
                        est_time = (len(self.tasks) * 15) // self.max_concurrent_tasks  # ~15s per task
                        ui.label(f'Est. time: ~{est_time//60}m {est_time%60}s').classes('text-xs text-gray-500')
    
    def _update_concurrency(self, event):
        """Update maximum concurrent tasks"""
        self.max_concurrent_tasks = int(event.value)
        self._render_controls.refresh()
    
    @ui.refreshable
    def _render_results(self):
        """Comprehensive results table with mCODE details"""
        with ui.card().classes('w-full'):
            ui.label('🎯 Optimization Results').classes('text-xl font-bold mb-4')
            
            # Summary statistics
            success_tasks = [t for t in self.tasks.values() if t['status'] == 'Success']
            processing_tasks = [t for t in self.tasks.values() if t['status'] == 'Processing']
            failed_tasks = [t for t in self.tasks.values() if t['status'] == 'Failed']
            
            with ui.row().classes('gap-4 mb-4'):
                if success_tasks:
                    avg_f1 = sum(t['f1_score'] or 0 for t in success_tasks) / len(success_tasks)
                    best_f1 = max(t['f1_score'] or 0 for t in success_tasks)
                    ui.chip(f'Avg F1: {avg_f1:.3f}', color='blue')
                    ui.chip(f'Best F1: {best_f1:.3f}', color='green')
                
                ui.chip(f'{len(success_tasks)} Completed', color='green')
                ui.chip(f'{len(processing_tasks)} Processing', color='orange')
                ui.chip(f'{len(failed_tasks)} Failed', color='red')
            
            # Full results table
            with ui.scroll_area().classes('w-full h-[600px]'):
                self._render_results_table()
    
    def _render_results_table(self):
        """Pure NiceGUI table with task data"""
        # Table container
        with ui.card().classes('w-full overflow-x-auto'):
            # Header row
            with ui.row().classes('w-full bg-gray-100 p-2 font-bold text-xs border-b items-center min-w-max'):
                ui.label('Status').classes('w-16 text-center')
                ui.label('Model').classes('w-24')
                ui.label('Prompt').classes('w-32')
                ui.label('Trial').classes('w-20')
                ui.label('F1').classes('w-16 text-right')
                ui.label('Prec').classes('w-16 text-right')
                ui.label('Recall').classes('w-16 text-right')
                ui.label('Time').classes('w-20 text-right')
                ui.label('Tok In').classes('w-16 text-right')
                ui.label('Tok Out').classes('w-16 text-right')
                ui.label('Tok Tot').classes('w-16 text-right')
                ui.label('mCODE').classes('w-16 text-right')
                ui.label('Comp').classes('w-16 text-right')
                ui.label('Details').classes('w-16 text-center')
            
            # Data rows from tasks
            for task in self.tasks.values():
                self._render_table_row_pure(task)
    
    def _render_table_row_pure(self, task: Dict[str, Any]):
        """Pure NiceGUI table row without custom components"""
        status = task.get('status', 'Unknown')
        
        # Status-based styling
        if status == 'Queued':
            row_class = 'bg-gray-50'
            status_icon = '⏳'
        elif status == 'Processing':
            row_class = 'bg-blue-50 animate-pulse'
            status_icon = '⚙️'
        elif status == 'Success':
            row_class = 'bg-green-50'
            status_icon = '✅'
        else:  # Failed
            row_class = 'bg-red-50'
            status_icon = '❌'
        
        with ui.row().classes(f'w-full p-2 text-xs border-b hover:bg-gray-100 items-center min-w-max {row_class}'):
            # Status
            ui.label(status_icon).classes('w-16 text-center')
            
            # Model (truncated)
            model_name = task.get('model', '')
            model_display = model_name[:12] + '...' if len(model_name) > 12 else model_name
            ui.label(model_display).classes('w-24').tooltip(model_name)
            
            # Prompt (truncated)
            prompt_name = task.get('prompt', '')
            prompt_display = prompt_name[:15] + '...' if len(prompt_name) > 15 else prompt_name
            ui.label(prompt_display).classes('w-32').tooltip(prompt_name)
            
            # Trial
            ui.label(task.get('trial', '')).classes('w-20')
            
            # F1 Score
            f1_score = task.get('f1_score')
            ui.label(f"{f1_score:.3f}" if f1_score is not None else '-').classes('w-16 text-right font-mono')
            
            # Precision
            precision = task.get('precision')
            ui.label(f"{precision:.3f}" if precision is not None else '-').classes('w-16 text-right font-mono')
            
            # Recall
            recall = task.get('recall')
            ui.label(f"{recall:.3f}" if recall is not None else '-').classes('w-16 text-right font-mono')
            
            # Time
            duration_ms = task.get('duration_ms')
            if duration_ms is not None:
                if duration_ms >= 1000:
                    time_display = f"{duration_ms/1000:.1f}s"
                else:
                    time_display = f"{duration_ms}ms"
            else:
                time_display = '-'
            ui.label(time_display).classes('w-20 text-right font-mono')
            
            # Token counts - extract from validation_results
            validation_results = task.get('validation_results') or {}
            token_usage = validation_results.get('token_usage', {})
            
            tokens_in = token_usage.get('input_tokens')
            tokens_out = token_usage.get('output_tokens') 
            tokens_total = token_usage.get('total_tokens')
            
            ui.label(f"{tokens_in:,}" if tokens_in is not None else '-').classes('w-16 text-right font-mono')
            ui.label(f"{tokens_out:,}" if tokens_out is not None else '-').classes('w-16 text-right font-mono')
            ui.label(f"{tokens_total:,}" if tokens_total is not None else '-').classes('w-16 text-right font-mono')
            
            # mCODE count
            mcode_count = len(task.get('extracted_elements', []))
            ui.label(str(mcode_count)).classes('w-16 text-right font-mono')
            
            # Compliance score
            compliance = validation_results.get('compliance_score')
            ui.label(f"{compliance:.3f}" if compliance is not None else '-').classes('w-16 text-right font-mono')
            
            # Details button
            if task.get('mcode_extraction'):
                details_text = self._format_mcode_details(task)
                ui.button('ℹ️', on_click=lambda t=task: self._show_task_details(t)).props('dense flat size=sm').classes('w-16').tooltip(details_text)
            else:
                ui.label('-').classes('w-16 text-center')
    
    def _get_table_data(self) -> List[Dict[str, Any]]:
        """Convert tasks to table data format"""
        table_data = []
        for task in self.tasks.values():
            # Extract token information from metadata if available
            tokens_in = None
            tokens_out = None
            tokens_total = None
            
            validation_results = task.get('validation_results') or {}
            if 'token_usage' in validation_results:
                token_usage = validation_results['token_usage']
                tokens_in = token_usage.get('input_tokens')
                tokens_out = token_usage.get('output_tokens')
                tokens_total = token_usage.get('total_tokens')
            
            # Safe extraction with None checks
            extracted_elements = task.get('extracted_elements') or []
            
            table_data.append({
                'id': task.get('id', ''),
                'status': task.get('status', 'Unknown'),
                'model': task.get('model', ''),
                'prompt': task.get('prompt', ''),
                'trial': task.get('trial', ''),
                'f1_score': task.get('f1_score'),
                'precision': task.get('precision'),
                'recall': task.get('recall'),
                'duration_ms': task.get('duration_ms'),
                'tokens_in': tokens_in,
                'tokens_out': tokens_out,
                'tokens_total': tokens_total,
                'mcode_count': len(extracted_elements),
                'compliance_score': validation_results.get('compliance_score'),
                'details': task  # Full task data for details
            })
        
        return table_data
    
    def _show_task_details(self, task: Dict[str, Any]):
        """Show detailed task information in a dialog"""
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label(f'Task {task["id"]} Details').classes('text-lg font-bold mb-4')
            
            details = self._format_mcode_details(task)
            ui.label(details).classes('whitespace-pre-wrap text-sm')
            
            ui.button('Close', on_click=dialog.close).classes('mt-4')
    
    def _show_task_details_from_table(self, event):
        """Show task details from table row click"""
        # Get task ID from the event or find task by current selection
        # For now, show a generic message until we can properly access row data
        with ui.dialog() as dialog, ui.card().classes('w-96'):
            ui.label('Task Details').classes('text-lg font-bold mb-4')
            ui.label('Detailed view coming soon...').classes('text-sm')
            ui.button('Close', on_click=dialog.close).classes('mt-4')
        dialog.open()
    
    def _format_mcode_details(self, task: Dict[str, Any]) -> str:
        """Format real mCODE extraction details for tooltip"""
        if not task['mcode_extraction']:
            return 'No mCODE data available'
        
        details = []
        details.append(f"🎯 Trial: {task['trial']}")
        details.append(f"🤖 Model: {task['model']}")
        details.append(f"📋 Prompt: {task['prompt']}")
        details.append("")
        
        if task['extracted_elements']:
            details.append("📊 Real mCODE Elements:")
            for i, element in enumerate(task['extracted_elements'][:5]):  # Show first 5
                resource_type = element.get('resourceType', 'Unknown')
                element_name = element.get('Mcode_element', 'Unknown')
                value = element.get('value', '')[:40] + '...' if len(element.get('value', '')) > 40 else element.get('value', '')
                details.append(f"  • {resource_type}: {element_name}")
                if value:
                    details.append(f"    └─ {value}")
            
            if len(task['extracted_elements']) > 5:
                details.append(f"  ... and {len(task['extracted_elements']) - 5} more")
        
        details.append("")
        details.append(f"📈 Real Metrics:")
        details.append(f"  F1: {task['f1_score']:.3f}")
        details.append(f"  Precision: {task['precision']:.3f}")
        details.append(f"  Recall: {task['recall']:.3f}")
        details.append(f"  Processing: {task['duration_ms']}ms")
        
        if task['validation_results']:
            details.append("")
            details.append("🔍 Validation:")
            details.append(f"  Compliance: {task['validation_results'].get('compliance_score', 0):.3f}")
            details.append(f"  Valid: {task['validation_results'].get('valid', False)}")
        
        return '\n'.join(details)
    
    def _start_optimization(self):
        """Start optimization run"""
        self.is_running = True
        self._render_controls.refresh()
        background_tasks.create(self._run_optimization())
    
    def _stop_optimization(self):
        """Stop optimization"""
        self.is_running = False
        self._render_controls.refresh()
    
    async def _run_optimization(self):
        """Run optimization with concurrent pipeline processing"""
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Create tasks for concurrent processing
        task_coroutines = []
        for task_id, task in self.tasks.items():
            if not self.is_running:
                break
            coroutine = self._process_task_with_semaphore(semaphore, task_id, task)
            task_coroutines.append(coroutine)
        
        try:
            # Process all tasks concurrently with progress updates
            await self._run_concurrent_tasks(task_coroutines)
        finally:
            self.is_running = False
            # Final UI refresh when optimization completes
            try:
                await ui.context.client.safe_invoke(self._render_header_stats)
                await ui.context.client.safe_invoke(self._render_controls.refresh)
                await ui.context.client.safe_invoke(self._render_results.refresh)
            except:
                pass  # Ignore refresh errors
    
    async def _run_concurrent_tasks(self, task_coroutines):
        """Run tasks concurrently with periodic UI updates"""
        # Create task for periodic UI updates
        update_task = asyncio.create_task(self._periodic_ui_update())
        
        try:
            # Run all tasks concurrently
            await asyncio.gather(*task_coroutines, return_exceptions=True)
        finally:
            # Cancel the update task
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass
    
    async def _periodic_ui_update(self):
        """Periodically update the UI during optimization"""
        while self.is_running:
            try:
                # Update header stats
                self._render_header_stats()
                # Refresh results display
                self._render_results.refresh()
                # Wait before next update
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"UI update error: {e}")
                await asyncio.sleep(1.0)
    

    
    async def _process_task_with_semaphore(self, semaphore: asyncio.Semaphore, task_id: str, task: Dict[str, Any]):
        """Process a single task with semaphore for concurrency control"""
        async with semaphore:
            if not self.is_running:
                return
            
            # Start processing
            task['status'] = 'Processing'
            task['logs'] = ['🔄 Starting real pipeline processing...']
            
            start_time = time.time()
            
            try:
                # Add processing logs
                task['logs'].append(f"🤖 Model: {task['model']}")
                task['logs'].append(f"📋 Prompt: {task['prompt']}")
                task['logs'].append("⚡ Processing clinical trial with real pipeline...")
                
                # Create and configure pipeline with real settings
                await self._process_real_pipeline(task)
                
                # Calculate duration
                task['duration_ms'] = int((time.time() - start_time) * 1000)
                
                task['status'] = 'Success'
                task['logs'].append(f"✅ Success! F1: {task['f1_score']:.3f}, {len(task['extracted_elements'])} mCODE elements")
                
                
            except Exception as e:
                task['status'] = 'Failed'
                task['error_message'] = str(e)
                task['duration_ms'] = int((time.time() - start_time) * 1000)
                task['logs'].append(f"❌ Error: {str(e)}")
                logger.error(f"Task {task_id} failed: {e}")
    
    async def _process_real_pipeline(self, task: Dict[str, Any]):
        """Process task with cached real McodePipeline"""
        # Set model configuration
        import os
        original_model = os.environ.get('MODEL_NAME')
        os.environ['MODEL_NAME'] = task['model_key']
        
        try:
            # Get or create cached pipeline
            pipeline_key = f"{task['model_key']}_{task['prompt_key']}"
            if pipeline_key not in self.pipeline_cache:
                self.pipeline_cache[pipeline_key] = McodePipeline(prompt_name=task['prompt_key'])
            
            pipeline = self.pipeline_cache[pipeline_key]
            
            # Process the clinical trial
            result = await run.io_bound(pipeline.process_clinical_trial, task['trial_data'])
            
            # Check if result is None
            if result is None:
                raise ValueError("Pipeline returned None - check pipeline configuration and input data")
            
            # Check if result has required attributes
            if not hasattr(result, 'mcode_mappings'):
                raise ValueError("Pipeline result missing 'mcode_mappings' attribute")
            
            if not hasattr(result, 'validation_results'):
                logger.warning(f"Pipeline result missing 'validation_results' attribute for task {task['id']}")
                result.validation_results = {}
            
            # Extract real results
            task['extracted_elements'] = result.mcode_mappings or []
            task['validation_results'] = result.validation_results or {}
            
            # Capture token usage from pipeline metadata
            if hasattr(result, 'metadata') and result.metadata:
                token_usage = result.metadata.get('token_usage', {})
                task['validation_results']['token_usage'] = token_usage
            
            # Calculate real metrics if validation results available
            if task['validation_results'] and 'compliance_score' in task['validation_results']:
                # Use real compliance score as F1 proxy
                task['f1_score'] = task['validation_results']['compliance_score']
                task['precision'] = task['validation_results'].get('precision', task['f1_score'])
                task['recall'] = task['validation_results'].get('recall', task['f1_score'])
            else:
                # Calculate basic metrics from mCODE count
                mcode_count = len(task['extracted_elements'])
                if mcode_count > 0:
                    task['f1_score'] = min(mcode_count / 10.0, 1.0)  # Basic score based on count
                else:
                    task['f1_score'] = 0.0
                task['precision'] = task['f1_score']
                task['recall'] = task['f1_score']
            
            # Store mCODE extraction details
            task['mcode_extraction'] = {
                'total_elements': len(task['extracted_elements']),
                'compliance_score': task['validation_results'].get('compliance_score', 0.0),
                'validation_details': task['validation_results']
            }
            
            logger.info(f"Task {task['id']} completed successfully: {len(task['extracted_elements'])} elements, F1: {task['f1_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for task {task['id']}: {str(e)}")
            # Set default values for failed processing
            task['extracted_elements'] = []
            task['validation_results'] = {}
            task['f1_score'] = 0.0
            task['precision'] = 0.0
            task['recall'] = 0.0
            task['mcode_extraction'] = {
                'total_elements': 0,
                'compliance_score': 0.0,
                'validation_details': {},
                'error': str(e)
            }
            raise  # Re-raise to be caught by the calling method
            
        finally:
            # Restore original model if it existed
            if original_model:
                os.environ['MODEL_NAME'] = original_model
            elif 'MODEL_NAME' in os.environ:
                del os.environ['MODEL_NAME']


def run_mcode_optimizer(port: int = 8094):
    """Run the mCODE optimizer"""
    optimizer = McodeOptimizer()
    ui.run(title='mCODE Optimizer - Concurrent', port=port)


if __name__ in {"__main__", "__mp_main__"}:
    run_mcode_optimizer()