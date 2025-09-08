#!/usr/bin/env python3
"""
STRICT Benchmark Task Tracker - Pure NiceGUI with Real Data Only
NO MOCKS, NO FALLBACKS - Exception-based debugging for real-time fixes
"""

import asyncio
import uuid
import json
import sys
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import nicegui
from nicegui import ui, background_tasks

# STRICT: Import real validation and execution modules 
from src.utils.prompt_loader import prompt_loader
from src.utils.model_loader import model_loader, ModelConfig
from src.optimization.prompt_optimization_framework import PromptOptimizationFramework


@dataclass
class ValidationTask:
    """Strict validation task structure"""
    id: str
    prompt: str
    model: str  
    trial: str
    pipeline_type: str
    pipeline_name: str
    status: str = 'Queued'
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    duration_ms: Optional[int] = None
    worker_id: str = 'N/A'
    error_message: Optional[str] = None


class StrictBenchmarkTracker:
    """STRICT benchmark tracker with real data only"""
    
    def __init__(self):
        # STRICT: Real data sources only
        self.available_models: Dict[str, ModelConfig] = {}
        self.prompt_data: Dict[str, Any] = {}
        self.trial_data: Dict[str, Any] = {}
        self.validation_tasks: Dict[str, ValidationTask] = {}
        self.is_benchmark_running = False
        self.benchmark_cancelled = False
        self.completed_tasks_count = 0
        self.optimization_framework = None
        
        # Initialize real data
        self._load_real_data()
        
    def _load_real_data(self):
        """STRICT: Load real data or fail immediately"""
        try:
            # Load real models
            self.available_models = model_loader.get_all_models()
            if not self.available_models:
                raise Exception("STRICT: No models loaded from configuration")
            
            # Load real prompts using correct API
            self.available_pipelines = prompt_loader.list_available_pipelines()
            if not self.available_pipelines:
                raise Exception("STRICT: No pipelines loaded from prompt_loader")
            
            self.all_prompts = prompt_loader.list_available_prompts()
            if not self.all_prompts:
                raise Exception("STRICT: No prompts loaded from prompt_loader")
            
            # Load real trial data
            import os
            trial_files = []
            examples_path = "examples/breast_cancer_data"
            if os.path.exists(examples_path):
                trial_files = [f for f in os.listdir(examples_path) if f.endswith('.json')]
            
            if not trial_files:
                raise Exception("STRICT: No trial data files found in examples/breast_cancer_data")
            
            for trial_file in trial_files:
                trial_path = os.path.join(examples_path, trial_file)
                with open(trial_path, 'r') as f:
                    trial_id = trial_file.replace('.json', '')
                    self.trial_data[trial_id] = json.load(f)
            
            # Initialize optimization framework
            self.optimization_framework = PromptOptimizationFramework()
            
            print(f"STRICT: Loaded {len(self.available_models)} models, {len(self.available_pipelines)} pipelines, {len(self.trial_data)} trials")
            
        except Exception as e:
            error_msg = f"STRICT DATA LOADING FAILED: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _generate_validation_tasks(self):
        """STRICT: Generate real validation tasks"""
        try:
            task_count = 0
            
            # Generate tasks for each pipeline using real API
            for pipeline_key in self.available_pipelines:
                pipeline_prompts = prompt_loader.get_prompts_by_pipeline(pipeline_key)
                if not pipeline_prompts:
                    raise Exception(f"STRICT: No prompts found for pipeline {pipeline_key}")
                
                for prompt_type, prompts in pipeline_prompts.items():
                    for prompt_info in prompts:
                        prompt_name = prompt_info.get('name')
                        if not prompt_name:
                            raise Exception(f"STRICT: Prompt missing name in {pipeline_key}/{prompt_type}")
                        
                        for model_key, model_config in self.available_models.items():
                            for trial_id in self.trial_data.keys():
                                task_id = str(uuid.uuid4())
                                
                                task = ValidationTask(
                                    id=task_id,
                                    prompt=prompt_name,
                                    model=model_config.name,
                                    trial=trial_id,
                                    pipeline_type=pipeline_key,
                                    pipeline_name=prompt_info.get('display_name', pipeline_key)
                                )
                                
                                self.validation_tasks[task_id] = task
                                task_count += 1
            
            if task_count == 0:
                raise Exception("STRICT: No validation tasks generated")
            
            print(f"STRICT: Generated {task_count} validation tasks")
            
        except Exception as e:
            error_msg = f"STRICT TASK GENERATION FAILED: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    async def _execute_real_validation(self, task: ValidationTask):
        """STRICT: Execute real validation with no mocks"""
        try:
            print(f"STRICT: Executing validation for task {task.id}")
            
            # STRICT: Use real optimization framework
            result = await self.optimization_framework.run_benchmark_async(
                prompt_variant_id=task.prompt,
                api_config_name=task.model,
                test_case_id=task.trial,
                pipeline_type=task.pipeline_type
            )
            
            if not result:
                raise Exception(f"STRICT: No result returned for task {task.id}")
            
            # STRICT: Extract real metrics or fail
            if not hasattr(result, 'f1_score') or result.f1_score is None:
                raise Exception(f"STRICT: Missing f1_score in result for task {task.id}")
            
            task.status = 'Success'
            task.f1_score = result.f1_score
            task.precision = getattr(result, 'precision', None)
            task.recall = getattr(result, 'recall', None)
            task.duration_ms = getattr(result, 'execution_time_ms', None)
            
            print(f"STRICT: Task {task.id} completed with F1: {task.f1_score}")
            
        except Exception as e:
            error_msg = f"STRICT VALIDATION EXECUTION FAILED - Task {task.id}: {str(e)}"
            print(error_msg)
            task.status = 'Failed'
            task.error_message = error_msg
            raise Exception(error_msg)
    
    @ui.refreshable
    def task_table_display(self):
        """STRICT: Task table with real data"""
        with ui.column().classes('w-full'):
            ui.label(f'Validation Tasks ({len(self.validation_tasks)} total)').classes('text-lg font-bold')
            
            with ui.scroll_area().classes('h-96 w-full border'):
                with ui.column().classes('w-full gap-1'):
                    for task in list(self.validation_tasks.values())[:50]:  # Show first 50
                        with ui.row().classes('w-full items-center gap-2 p-2 border-b'):
                            # Status icon
                            if task.status == 'Success':
                                ui.icon('check_circle', color='green')
                            elif task.status == 'Failed':
                                ui.icon('error', color='red')
                            elif task.status == 'Processing':
                                ui.icon('sync', color='blue').classes('animate-spin')
                            else:
                                ui.icon('schedule', color='gray')
                            
                            # Task details
                            ui.label(f"{task.prompt[:30]}...").classes('text-sm flex-1')
                            ui.label(task.model).classes('text-sm w-24')
                            ui.label(task.trial).classes('text-sm w-20')
                            
                            # Metrics
                            if task.f1_score is not None:
                                ui.label(f"F1: {task.f1_score:.3f}").classes('text-sm w-20')
                            else:
                                ui.label("-").classes('text-sm w-20')
                            
                            ui.label(task.worker_id).classes('text-sm w-16')
    
    @ui.refreshable  
    def benchmark_controls(self):
        """STRICT: Benchmark controls"""
        with ui.row().classes('gap-4 items-center'):
            if not self.is_benchmark_running:
                ui.button('Run All Tasks', 
                         on_click=self._start_benchmark,
                         icon='play_arrow').props('color=primary')
            else:
                ui.button('Stop Benchmark',
                         on_click=self._stop_benchmark, 
                         icon='stop').props('color=negative')
            
            ui.button('Reset All',
                     on_click=self._reset_benchmark,
                     icon='refresh').props('flat')
    
    async def _start_benchmark(self):
        """STRICT: Start real benchmark execution"""
        try:
            print("STRICT: Starting benchmark execution")
            self.is_benchmark_running = True
            self.benchmark_cancelled = False
            
            # Refresh UI
            self.benchmark_controls.refresh()
            
            # Execute tasks
            background_tasks.create(self._execute_all_tasks())
            
        except Exception as e:
            error_msg = f"STRICT BENCHMARK START FAILED: {str(e)}"
            print(error_msg)
            self.is_benchmark_running = False
            raise Exception(error_msg)
    
    async def _execute_all_tasks(self):
        """STRICT: Execute all validation tasks"""
        try:
            tasks_to_run = [task for task in self.validation_tasks.values() 
                           if task.status == 'Queued']
            
            print(f"STRICT: Executing {len(tasks_to_run)} tasks")
            
            # Process tasks with limited concurrency
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent
            
            async def process_task(task):
                async with semaphore:
                    if self.benchmark_cancelled:
                        return
                    
                    task.status = 'Processing'
                    task.worker_id = f'W-{hash(task.id) % 10}'
                    
                    # Refresh UI
                    self.task_table_display.refresh()
                    
                    # Execute real validation
                    await self._execute_real_validation(task)
                    
                    # Refresh UI again
                    self.task_table_display.refresh()
            
            # Run all tasks
            await asyncio.gather(*[process_task(task) for task in tasks_to_run])
            
            print("STRICT: All tasks completed")
            self.is_benchmark_running = False
            self.benchmark_controls.refresh()
            
        except Exception as e:
            error_msg = f"STRICT TASK EXECUTION FAILED: {str(e)}"
            print(error_msg)
            self.is_benchmark_running = False
            self.benchmark_controls.refresh()
            raise Exception(error_msg)
    
    def _stop_benchmark(self):
        """STRICT: Stop benchmark execution"""
        print("STRICT: Stopping benchmark")
        self.benchmark_cancelled = True
        self.is_benchmark_running = False
        self.benchmark_controls.refresh()
    
    def _reset_benchmark(self):
        """STRICT: Reset all tasks"""
        try:
            print("STRICT: Resetting benchmark")
            
            # Reset all task statuses
            for task in self.validation_tasks.values():
                task.status = 'Queued'
                task.f1_score = None
                task.precision = None
                task.recall = None
                task.duration_ms = None
                task.worker_id = 'N/A'
                task.error_message = None
            
            self.is_benchmark_running = False
            self.benchmark_cancelled = False
            self.completed_tasks_count = 0
            
            # Refresh UI
            self.benchmark_controls.refresh()
            self.task_table_display.refresh()
            
            ui.notify("Benchmark reset", type='positive')
            
        except Exception as e:
            error_msg = f"STRICT RESET FAILED: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    def build_ui(self):
        """STRICT: Build main UI"""
        try:
            ui.colors(primary='#1976d2', secondary='#424242', accent='#82b1ff', 
                     positive='#21ba45', negative='#c10015', info='#31ccec', 
                     warning='#f2c037')
            
            # Create the main page content directly
            ui.label('STRICT Benchmark Task Tracker').classes('text-2xl font-bold mb-4')
            
            # Controls
            self.benchmark_controls()
            
            ui.separator().classes('my-4')
            
            # Task table
            self.task_table_display()
                
        except Exception as e:
            error_msg = f"STRICT UI BUILD FAILED: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)


def run_strict_tracker(port: int = 8091):
    """Run the strict benchmark tracker"""
    try:
        print("STRICT: Starting benchmark tracker")
        
        # Create tracker instance
        tracker = StrictBenchmarkTracker()
        
        # Generate validation tasks
        tracker._generate_validation_tasks()
        
        print("STRICT: Building UI...")
        
        # Build UI
        tracker.build_ui()
        
        print("STRICT: Starting NiceGUI server...")
        
    except Exception as e:
        error_msg = f"STRICT TRACKER STARTUP FAILED: {str(e)}"
        print(error_msg)
        print(f"STRICT TRACEBACK: {traceback.format_exc()}")
        raise Exception(error_msg)
    
    # Run NiceGUI outside try/except to ensure it starts
    ui.run(port=port, title="STRICT Benchmark Tracker", show=False)


if __name__ in {"__main__", "__mp_main__"}:
    run_strict_tracker()