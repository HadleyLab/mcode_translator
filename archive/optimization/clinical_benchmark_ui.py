
"""
Clinical Trial Benchmark UI - Modern interface for clinical trial validation optimization
Features uber list of validations with filters for prompt, model, and trial
"""

import sys
from pathlib import Path
import logging
import json
import asyncio
import os
from typing import Dict, List, Any, Optional, Set
import pandas as pd
from nicegui import ui, app, run
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add project root to path for proper imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logger
logger = logging.getLogger(__name__)

from src.optimization.prompt_optimization_framework import (
    PromptOptimizationFramework,
    PromptType,
    APIConfig,
    PromptVariant
)

from src.utils.prompt_loader import prompt_loader
from src.utils import model_loader


class ValidationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    OPTIMAL = "optimal"


@dataclass
class ValidationResult:
    """Result of a single validation run"""
    validation_id: str
    prompt_key: str
    model_key: str
    trial_id: str
    prompt_type: str
    duration_ms: float
    success: bool
    entities_extracted: int
    compliance_score: float
    f1_score: float
    status: ValidationStatus
    timestamp: datetime
    token_usage: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert ValidationResult to a dictionary for UI display."""
        return {
            "validation_id": self.validation_id,
            "prompt_key": self.prompt_key,
            "model_key": self.model_key,
            "trial_id": self.trial_id,
            "prompt_type": self.prompt_type,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "entities_extracted": self.entities_extracted,
            "compliance_score": self.compliance_score,
            "f1_score": self.f1_score,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "token_usage": self.token_usage
        }


class ClinicalBenchmarkUI:
    """Modern clinical trial benchmark UI with uber list concept"""
    
    def __init__(self, framework: Optional[PromptOptimizationFramework] = None):
        self.framework = framework or PromptOptimizationFramework()
        self.validations: List[Dict[str, Any]] = []
        self.validation_results: Dict[str, ValidationResult] = {}
        self.filtered_validations: List[Dict[str, Any]] = []
        self.selected_validations: Set[str] = set()
        self.selected_validation: Optional[str] = None
        self.gold_standard_data: Dict[str, Any] = {}
        self.ui_update_event = asyncio.Event()
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.main_loop)
        
        # Default file paths
        self.test_cases_file = "examples/breast_cancer_data/breast_cancer_her2_positive.trial.json"
        self.gold_standard_file = "examples/breast_cancer_data/breast_cancer_her2_positive.gold.json"
        
        # UI components
        self.validation_table = None
        self.validation_count = None
        self.optimal_results = None
        self.log_display = None
        self.optimize_button = None
        self.optimize_status = None
        self.nlp_prompt_filter = None
        self.mcode_prompt_filter = None
        self.model_filter = None
        self.trial_filter = None
        self.run_selected_button = None
        self.expand_selected_button = None
        
        # Load libraries and setup
        self._load_libraries()
        self._load_test_cases()
        self._load_gold_standard()
        
        self.setup_ui()
        
        # Auto-generate all validations on startup (after UI is created)
        self._generate_validations()
        
        ui.timer(0.1, self._update_ui_on_event)
    
    def _load_libraries(self) -> None:
        """Load prompt and model libraries with strict validation"""
        try:
            self.available_prompts = prompt_loader.list_available_prompts()
            self.available_models = model_loader.list_available_models()
            logger.info(f"Loaded {len(self.available_prompts)} prompts and {len(self.available_models)} models")
            
            # Validate that we have unique keys
            prompt_keys = list(self.available_prompts.keys())
            model_keys = list(self.available_models.keys())
            
            if len(prompt_keys) != len(set(prompt_keys)):
                raise ValueError("Duplicate prompt keys detected in prompt library")
                
            if len(model_keys) != len(set(model_keys)):
                raise ValueError("Duplicate model keys detected in model library")
                
        except Exception as e:
            logger.error(f"Failed to load libraries: {str(e)}")
            raise
    
    def _load_test_cases(self) -> None:
        """Load test cases from file - strict implementation with no fallbacks"""
        if not os.path.exists(self.test_cases_file):
            raise FileNotFoundError(
                f"Test cases file not found: {self.test_cases_file}\n"
                f"Expected path: {os.path.abspath(self.test_cases_file)}\n"
                "Please ensure the breast cancer trial data exists in examples/breast_cancer_data/"
            )
        
        try:
            with open(self.test_cases_file, 'r') as f:
                test_cases_data = json.load(f)
            
            self.trial_ids = list(test_cases_data.get("test_cases", {}).keys())
            
            if not self.trial_ids:
                raise ValueError("No test cases found in the trial data file")
            
            # Load test cases into the framework
            for trial_id, trial_data in test_cases_data.get("test_cases", {}).items():
                self.framework.add_test_case(trial_id, trial_data)
                
            logger.info(f"Loaded {len(self.trial_ids)} test cases: {self.trial_ids}")
            
        except Exception as e:
            logger.error(f"Failed to load test cases: {str(e)}")
            raise RuntimeError(f"Failed to load test cases from {self.test_cases_file}: {str(e)}")
            
    def _load_gold_standard(self) -> None:
        """Load gold standard data from file"""
        if not os.path.exists(self.gold_standard_file):
            raise FileNotFoundError(f"Gold standard file not found: {self.gold_standard_file}")

        try:
            with open(self.gold_standard_file, 'r') as f:
                self.gold_standard_data = json.load(f).get("gold_standard", {})
            logger.info(f"Loaded gold standard data for {len(self.gold_standard_data)} test cases.")
        except Exception as e:
            logger.error(f"Failed to load gold standard data: {str(e)}")
            raise RuntimeError(f"Failed to load gold standard data from {self.gold_standard_file}: {str(e)}")

    def _generate_validations(self) -> None:
        """Generate all possible validation combinations with strict JSON alignment"""
        self.validations = []
        
        # Add prompt variants to the framework
        for prompt_key, prompt_info in self.available_prompts.items():
            # Convert prompt_type to lowercase to match PromptType enum values
            prompt_type_value = prompt_info.get('prompt_type', 'nlp_extraction').lower()
            prompt_variant = PromptVariant(
                id=prompt_key,
                name=prompt_info.get('name', prompt_key),
                prompt_type=PromptType(prompt_type_value),
                prompt_key=prompt_key,
                description=prompt_info.get('description', ''),
                version=prompt_info.get('version', '1.0.0')
            )
            try:
                self.framework.add_prompt_variant(prompt_variant)
            except Exception as e:
                logger.warning(f"Failed to add prompt variant {prompt_key}: {str(e)}")
        
        # Generate validations with improved performance and strict JSON alignment
        validations_to_create = []
        for prompt_key, prompt_info in self.available_prompts.items():
            prompt_type_value = prompt_info.get('prompt_type', 'nlp_extraction').lower()
            prompt_name = prompt_info.get('name', prompt_key)
            
            for model_key, model_info in self.available_models.items():
                model_name = model_info.get('name', model_key)
                
                for trial_id in self.trial_ids:
                    validation = {
                        'id': f"{prompt_key}_{model_key}_{trial_id}",
                        'prompt_key': prompt_key,
                        'model_key': model_key,
                        'trial_id': trial_id,
                        'prompt_type': prompt_type_value,
                        'prompt_name': prompt_name,
                        'model_name': model_name,
                        'status': ValidationStatus.PENDING,
                        'last_run': None,
                        'score': 0.0,
                        'selected': False  # For checkbox selection
                    }
                    validations_to_create.append(validation)
        
        # Batch add validations for better performance
        self.validations.extend(validations_to_create)
        
        logger.info(f"Generated {len(self.validations)} validation combinations")
        self.filtered_validations = self.validations.copy()
        logger.info(f"Filtered validations: {len(self.filtered_validations)}")
        self._update_validation_count()
        self._update_validation_list()  # Force update the table
    
    def _apply_filters(self) -> None:
        """Apply current filters to validation list"""
        self.filtered_validations = []
        
        for validation in self.validations:
            matches = True
            
            # Apply NLP prompt filter - if filter has values, only include selected prompts
            if (hasattr(self, 'nlp_prompt_filter') and self.nlp_prompt_filter.value and
                validation['prompt_type'] == 'NLP_EXTRACTION' and
                validation['prompt_key'] not in self.nlp_prompt_filter.value):
                matches = False
                
            # Apply mCODE prompt filter - if filter has values, only include selected prompts
            if matches and (hasattr(self, 'Mcode_prompt_filter') and self.mcode_prompt_filter.value and
                validation['prompt_type'] == 'MCODE_MAPPING' and
                validation['prompt_key'] not in self.mcode_prompt_filter.value):
                matches = False
                
            # Apply model filter - if filter has values, only include selected models
            if matches and (hasattr(self, 'model_filter') and self.model_filter.value and
                validation['model_key'] not in self.model_filter.value):
                matches = False
                
            # Apply trial filter - if filter has values, only include selected trials
            if matches and (hasattr(self, 'trial_filter') and self.trial_filter.value and
                validation['trial_id'] not in self.trial_filter.value):
                matches = False
                
            if matches:
                self.filtered_validations.append(validation)
        
        self._update_validation_list()
        self._update_validation_count()
    
    def _update_validation_count(self) -> None:
        """Update validation count display with detailed statistics"""
        total_validations = len(self.validations)
        filtered_count = len(self.filtered_validations)
        selected_count = sum(1 for v in self.validations if v.get('selected', False))
        filtered_selected = sum(1 for v in self.filtered_validations if v.get('selected', False))
        
        count_text = f"""
        📊 Validation Statistics:
        • Total Validations: {total_validations}
        • Filtered: {filtered_count}
        • Selected for Execution: {selected_count} ({filtered_selected} in current view)
        """
        
        if hasattr(self, 'validation_count'):
            self.validation_count.set_content(count_text)
    
    def _update_validation_list(self) -> None:
        """Update the validation list display"""
        rows = []
        for validation in self.filtered_validations:
            result = self.validation_results.get(validation['id'])
            status_icon = self._get_status_icon(validation['status'])
            score = result.f1_score if result else 0.0
            token_usage = result.token_usage if result else 0
            
            rows.append({
                'id': validation['id'],
                'prompt': f"{validation['prompt_name']} ({validation['prompt_type']})",
                'model': validation['model_name'],
                'trial': validation['trial_id'],
                'status': status_icon,
                'score': f"{score:.3f}" if result else "-",
                'tokens': f"{token_usage:,}" if token_usage > 0 else "-",
                'details': result.to_dict() if result else {}
            })
        
        self.validation_table.rows = rows
        self.validation_table.update()
        self._update_validation_count()
        # Update button states after table update
        self._update_button_states()
    
    def _get_status_icon(self, status: ValidationStatus) -> str:
        """Get status icon for display"""
        icons = {
            ValidationStatus.PENDING: "⏳",
            ValidationStatus.RUNNING: "🔄",
            ValidationStatus.COMPLETED: "✅",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.OPTIMAL: "⭐"
        }
        return icons.get(status, "❓")

    async def _update_ui_on_event(self) -> None:
        """Update the UI when the event is set."""
        if self.ui_update_event.is_set():
            self.ui_update_event.clear()
            self._update_validation_list()
    
    def _update_validation_state(self, validation_id: str, status: ValidationStatus) -> None:
        """Update the state of a specific validation and trigger UI update."""
        validation = next((v for v in self.validations if v['id'] == validation_id), None)
        if validation:
            validation['status'] = status
            self.ui_update_event.set()
            
    
    
    def _run_benchmark_sync(self, prompt_variant_id: str, api_config_name: str, test_case_id: str,
                           expected_entities: List[Any], expected_mappings: List[Any]) -> Any:
        """Synchronous wrapper for running benchmark - to be used with run.io_bound"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                self.framework.run_benchmark_async(
                    prompt_variant_id=prompt_variant_id,
                    api_config_name=api_config_name,
                    test_case_id=test_case_id,
                    expected_entities=expected_entities,
                    expected_mappings=expected_mappings
                )
            )
            return result
        finally:
            loop.close()

    async def _run_validation(self, validation_id: str) -> None:
        """Run a single validation using run.io_bound for async operations."""
        validation = next((v for v in self.validations if v['id'] == validation_id), None)
        if not validation:
            return

        # Store original status to restore if needed
        original_status = validation['status']
        self._update_validation_state(validation_id, ValidationStatus.RUNNING)

        try:
            model_key = validation['model_key']
            # Create the correct API config name that matches what the framework expects
            api_config_name = f"model_{model_key.replace('-', '_').replace('.', '_')}"
            
            gold_standard = self.gold_standard_data.get(validation['trial_id'], {})
            expected_entities = gold_standard.get('expected_extraction', {}).get('entities', [])
            expected_mappings = gold_standard.get('expected_mcode_mappings', {}).get('mapped_elements', [])

            # Update button states to reflect running process
            if hasattr(self, 'optimize_button') and self.optimize_button:
                self.optimize_button.disable()
                self.optimize_button.props('icon=hourglass_empty color=warning')
            if hasattr(self, 'optimize_status') and self.optimize_status:
                self.optimize_status.set_text(f"Running validation {validation_id}...")
            if hasattr(self, 'log_display') and self.log_display:
                self.log_display.push(f"🚀 Starting validation {validation_id}")

            # Use run.io_bound to execute the synchronous wrapper in a thread pool
            result = await run.io_bound(
                self._run_benchmark_sync,
                prompt_variant_id=validation['prompt_key'],
                api_config_name=api_config_name,
                test_case_id=validation['trial_id'],
                expected_entities=expected_entities,
                expected_mappings=expected_mappings
            )

            # Capture token usage from result
            token_usage = getattr(result, 'token_usage', 0)
            
            validation_result = ValidationResult(
                validation_id=validation_id,
                prompt_key=validation['prompt_key'],
                model_key=validation['model_key'],
                trial_id=validation['trial_id'],
                prompt_type=validation['prompt_type'],
                duration_ms=result.duration_ms,
                success=result.success,
                entities_extracted=result.entities_extracted,
                compliance_score=result.compliance_score,
                f1_score=result.f1_score,
                status=ValidationStatus.COMPLETED,
                timestamp=datetime.now(),
                token_usage=token_usage
            )
            
            self.validation_results[validation_id] = validation_result
            self._update_validation_state(validation_id, ValidationStatus.COMPLETED)
            validation['score'] = result.f1_score
            validation['last_run'] = datetime.now()
            
            # Include token economics in the notification
            ui.notify(f"Validation completed: {result.f1_score:.3f} F1 score, {token_usage} tokens used", type='positive')
            self.log_display.push(f"✅ Validation {validation_id} completed with F1: {result.f1_score:.3f}, tokens: {token_usage}")
            self.log_display.push(f"📊 Results - Entities: {result.entities_extracted}, Compliance: {result.compliance_score:.2f}, Duration: {result.duration_ms:.2f}ms")

        except Exception as e:
            logger.error(f"Validation failed for {validation_id}: {str(e)}")
            self._update_validation_state(validation_id, ValidationStatus.FAILED)
            ui.notify(f"Validation failed: {str(e)}", type='negative')
            self.log_display.push(f"❌ Validation {validation_id} failed: {str(e)}")
        
        finally:
            # Restore button states
            if hasattr(self, 'optimize_button') and self.optimize_button:
                self.optimize_button.enable()
                self.optimize_button.props('icon=play_arrow color=primary')
            if hasattr(self, 'optimize_status') and self.optimize_status:
                self.optimize_status.set_text("Ready")
            self.ui_update_event.set()
    
    async def _run_single_validation(self, validation_id: str) -> None:
        """Run a single validation from UI action"""
        # Disable the optimize button during single validation run
        if hasattr(self, 'optimize_button'):
            self.optimize_button.disable()
        
        # Run the validation
        await self._run_validation(validation_id)
        
        # Re-enable the optimize button after validation completes
        if hasattr(self, 'optimize_button'):
            self.optimize_button.enable()

    async def _run_optimization(self) -> None:
        """Run optimization on selected validations"""
        # Get selected validations from the table
        selected_ids = self.validation_table.selected if hasattr(self.validation_table, 'selected') else []
        
        if not selected_ids:
            ui.notify('No validations selected for optimization', type='warning')
            return
        
        # Find the validation objects for the selected IDs
        selected_validations = [v for v in self.filtered_validations if v['id'] in selected_ids]
        
        if not selected_validations:
            ui.notify('No validations selected for optimization', type='warning')
            return
        
        # Update button states to reflect running process
        self.optimize_button.disable()
        self.optimize_button.props('icon=hourglass_empty color=warning')
        self.optimize_status.set_text(f"Running {len(selected_validations)} selected validations...")
        self.log_display.push(f"🚀 Starting optimization of {len(selected_validations)} validations")
        
        # Track progress
        completed_count = 0
        total_count = len(selected_validations)
        total_tokens = 0
        
        # Run validations sequentially to maintain better control and state updates
        for i, validation in enumerate(selected_validations):
            if validation['status'] != ValidationStatus.COMPLETED:
                self.optimize_status.set_text(f"Running validation {i+1}/{total_count}: {validation['prompt_name']} on {validation['model_name']}")
                self.log_display.push(f"🔄 Running validation {i+1}/{total_count}: {validation['prompt_name']} on {validation['model_name']}")
                await self._run_validation(validation['id'])
                completed_count += 1
                
                # Update progress
                self.optimize_status.set_text(f"Completed {completed_count}/{total_count} validations...")
                self.log_display.push(f"✅ Completed {completed_count}/{total_count} validations...")
                
                # Accumulate token usage
                result = self.validation_results.get(validation['id'])
                if result:
                    total_tokens += getattr(result, 'token_usage', 0)
        
        # Find best validation
        best_score = 0.0
        best_validation = None
        
        for validation in selected_validations:
            result = self.validation_results.get(validation['id'])
            if result and result.f1_score > best_score:
                best_score = result.f1_score
                best_validation = validation
        
        # Mark optimal validation
        if best_validation:
            self._update_validation_state(best_validation['id'], ValidationStatus.OPTIMAL)
            if best_validation['id'] in self.validation_results:
                self.validation_results[best_validation['id']].status = ValidationStatus.OPTIMAL
            
            # Update results display
            self._update_optimal_results(best_validation, best_score)
            
            ui.notify(f"Optimization complete! Best F1 score: {best_score:.3f}, Total tokens used: {total_tokens}", type='positive')
            self.log_display.push(f"⭐ Optimization complete! Best F1 score: {best_score:.3f}, Total tokens used: {total_tokens}")
        else:
            self.log_display.push("⚠️ Optimization completed but no valid results found")
        
        # Restore button states
        self.optimize_button.enable()
        self.optimize_button.set_visibility(True)
        self.optimize_button.props('icon=play_arrow color=primary')
        self.optimize_status.set_text(f"Completed {len(selected_validations)} validations, {total_tokens} tokens used")
    
    def _update_optimal_results(self, validation: Dict[str, Any], score: float) -> None:
        """Update optimal results display"""
        # Get token usage from validation result
        result = self.validation_results.get(validation['id'])
        token_usage = result.token_usage if result else 0
        
        result_text = f"""
        ## Optimal Configuration
        - **Prompt**: {validation['prompt_name']} ({validation['prompt_type']})
        - **Model**: {validation['model_name']}
        - **Trial**: {validation['trial_id']}
        - **F1 Score**: {score:.3f}
        - **Tokens Used**: {token_usage:,}
        - **Status**: ✅ Optimal
        """
        self.optimal_results.set_content(result_text)
    
    def _stop_optimization(self) -> None:
        """Stop optimization process"""
        # This method is no longer needed as we are not using a separate stop button.
        # The optimization process will run to completion.
        # We can add cancellation logic here in the future if needed.
        self.optimize_button.props('icon=play_arrow color=primary')
        self.optimize_status.set_text("Optimization stopped by user")
        ui.notify("Optimization stopped", type='warning')
    
    def setup_ui(self) -> None:
        """Setup the main UI layout"""
        with ui.header().classes('bg-primary text-white p-2'):
            ui.label('Clinical Trial Benchmark Optimizer').classes('text-lg font-bold')

        with ui.column().classes('w-full p-4 gap-4'):
            with ui.card().classes('w-full p-4'):
                self._setup_control_panel()
            with ui.card().classes('w-full p-4'):
                self._setup_validation_table()
    
    def _setup_control_panel(self) -> None:
        """Setup combined control panel with filters and log display"""
        ui.label('Control Panel').classes('text-lg font-semibold')
        
        # Filters section
        ui.label('Filters').classes('text-md font-semibold mt-2')
        
        # Use prompt loader to get prompt information
        nlp_prompts = {}
        Mcode_prompts = {}
        
        for key, info in self.available_prompts.items():
            prompt_name = info.get('name', key)
            prompt_type = info.get('prompt_type', '').upper()
            
            if prompt_type == 'NLP_EXTRACTION':
                nlp_prompts[key] = prompt_name
            elif prompt_type == 'MCODE_MAPPING':
                Mcode_prompts[key] = prompt_name
        
        self.nlp_prompt_filter = ui.select(options=nlp_prompts, label='NLP Prompts', multiple=True, on_change=self._apply_filters).classes('w-full')
        self.mcode_prompt_filter = ui.select(options=Mcode_prompts, label='mCODE Prompts', multiple=True, on_change=self._apply_filters).classes('w-full')
        
        # Use model loader to get model information
        model_options = {key: info.get('name', key) for key, info in self.available_models.items()}
        self.model_filter = ui.select(options=model_options, label='Models', multiple=True, on_change=self._apply_filters).classes('w-full')
        
        # Trial options
        trial_options = {trial: trial for trial in self.trial_ids}
        self.trial_filter = ui.select(options=trial_options, label='Trials', multiple=True, on_change=self._apply_filters).classes('w-full')
        
        # Filter control buttons (keep only filter-related buttons)
        with ui.row().classes('w-full gap-2 mt-4'):
            ui.button(on_click=self._clear_filters).props('icon=clear flat')
            ui.button(on_click=self._select_all).props('icon=select_all flat')
            ui.button(on_click=self._deselect_all).props('icon=deselect flat')
        
        # Bulk operation controls
        with ui.row().classes('w-full gap-2 mt-4'):
            self.optimize_button = ui.button('Run Selected Validations', on_click=self._run_optimization).props('icon=play_arrow color=primary')
            self.optimize_status = ui.label('Ready').classes('self-center ml-4')
        
        # Real-time log display
        ui.label('Real-time Validation Log').classes('text-md font-semibold mt-4')
        self.log_display = ui.log().classes('w-full h-32')

    def _setup_validation_table(self) -> None:
        """Setup the validation table."""
        self.validation_count = ui.markdown().classes('text-sm mb-2')
        
        # Initialize results display components
        self.optimal_results = ui.markdown('No optimization results yet').classes('w-full')
        
        # Create action buttons area
        with ui.row().classes('w-full justify-end gap-2 mb-4'):
            self.run_selected_button = ui.button('Run Selected', on_click=self._run_selected_validation).props('icon=play_arrow color=primary')
            self.expand_selected_button = ui.button('View Details', on_click=self._show_selected_details).props('icon=visibility color=secondary')
        
        with ui.table(
            columns=[
                {'name': 'status', 'label': 'Status', 'field': 'status', 'sortable': True, 'align': 'center'},
                {'name': 'action', 'label': 'Action', 'field': 'action', 'sortable': False, 'align': 'center'},
                {'name': 'prompt', 'label': 'Prompt', 'field': 'prompt', 'sortable': True},
                {'name': 'model', 'label': 'Model', 'field': 'model', 'sortable': True},
                {'name': 'trial', 'label': 'Trial', 'field': 'trial', 'sortable': True},
                {'name': 'score', 'label': 'F1', 'field': 'score', 'sortable': True},
                {'name': 'tokens', 'label': 'Tokens', 'field': 'tokens', 'sortable': True, 'align': 'right'}
            ],
            rows=[],
            pagination=15,
            selection='multiple',
            on_select=self._handle_row_selection
        ).classes('w-full') as table:
            self.validation_table = table
            table.add_slot('body-cell-action', self._render_action_cell)
    
    def _clear_filters(self) -> None:
        """Clear all filters"""
        self.nlp_prompt_filter.set_value(None)
        self.mcode_prompt_filter.set_value(None)
        self.model_filter.set_value(None)
        self.trial_filter.set_value(None)
        self._apply_filters()
    
    def _select_all(self) -> None:
        """Select all validations in current view"""
        # Select all rows in the table
        self.validation_table.selected = [row['id'] for row in self.validation_table.rows]
        ui.notify("All visible validations selected")
    
    def _deselect_all(self) -> None:
        """Deselect all validations in current view"""
        # Clear all selections
        self.validation_table.selected = []
        ui.notify("All visible validations deselected")
    
    def _handle_row_selection(self, event: Any) -> None:
        """Handle row selection in the validation table."""
        if hasattr(self.validation_table, 'selected') and self.validation_table.selected:
            # For multiple selections, we'll use the first selected item for details
            selected_ids = self.validation_table.selected
            if selected_ids:
                self.selected_validation = selected_ids[0]
                logger.info(f"Selected validation: {self.selected_validation}")
            else:
                self.selected_validation = None
                logger.info("No validation selected")
        else:
            self.selected_validation = None
            logger.info("No validation selected")
        
        # Update button states based on selection
        self._update_button_states()
    
    def _run_selected_validation(self) -> None:
        """Run the selected validation."""
        if not self.selected_validation:
            ui.notify("No validation selected to run", type='warning')
            return
        
        logger.info(f"Running selected validation: {self.selected_validation}")
        asyncio.create_task(self._run_validation(self.selected_validation))
    
    def _run_single_validation(self, validation_id: str) -> None:
        """Run a single validation from the action button."""
        logger.info(f"Running single validation: {validation_id}")
        asyncio.create_task(self._run_validation(validation_id))

    def _show_selected_details(self) -> None:
        """Show details for the selected validation."""
        if not self.selected_validation:
            ui.notify("No validation selected to view details", type='warning')
            return
        
        # Find the selected row data
        selected_row = None
        for row in self.validation_table.rows:
            if row['id'] == self.selected_validation:
                selected_row = row
                break
        
        if selected_row:
            logger.info(f"Showing details for: {selected_row['id']}")
            # Show details in a dialog
            with ui.dialog() as dialog, ui.card():
                ui.markdown(f"""
                ### Validation Details: {selected_row['prompt']} - {selected_row['model']}
                - **Trial**: {selected_row['trial']}
                - **Status**: {selected_row['status']}
                - **F1 Score**: {selected_row['score']}
                - **Tokens Used**: {selected_row['tokens']}
                """)
                with ui.card_actions():
                    ui.button('Close', on_click=dialog.close)
            dialog.open()
        else:
            logger.warning(f"Selected validation {self.selected_validation} not found in table")

    def _update_button_states(self) -> None:
        """Update button states based on selected validation status."""
        if not self.selected_validation:
            self.run_selected_button.disable()
            self.expand_selected_button.disable()
            return
        
        # Find the selected validation in the main list
        selected_validation = None
        for validation in self.validations:
            if validation['id'] == self.selected_validation:
                selected_validation = validation
                break
        
        if selected_validation:
            status = selected_validation['status']
            # Disable run button if validation is running, completed, or optimal
            if status in [ValidationStatus.RUNNING, ValidationStatus.COMPLETED, ValidationStatus.OPTIMAL]:
                self.run_selected_button.disable()
            else:
                self.run_selected_button.enable()
            
            # Always enable details button if selected
            self.expand_selected_button.enable()
        else:
            self.run_selected_button.disable()
            self.expand_selected_button.disable()

    def _export_results(self) -> None:
        """Export results to CSV"""
        try:
            if not self.validation_results:
                ui.notify("No results to export", type='warning')
                return
            
            # Create DataFrame from results
            data = []
            for result in self.validation_results.values():
                data.append({
                    'validation_id': result.validation_id,
                    'prompt_key': result.prompt_key,
                    'model_key': result.model_key,
                    'trial_id': result.trial_id,
                    'prompt_type': result.prompt_type,
                    'duration_ms': result.duration_ms,
                    'success': result.success,
                    'entities_extracted': result.entities_extracted,
                    'compliance_score': result.compliance_score,
                    'f1_score': result.f1_score,
                    'status': result.status.value,
                    'timestamp': result.timestamp.isoformat()
                })
            
            df = pd.DataFrame(data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.csv"
            df.to_csv(output_path, index=False)
            
            ui.notify(f"Results exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            ui.notify(f"Error exporting results: {str(e)}", type='negative')
    
    def _clear_results(self) -> None:
        """Clear all results"""
        self.validation_results.clear()
        for validation in self.validations:
            self._update_validation_state(validation['id'], ValidationStatus.PENDING)
            validation['score'] = 0.0
            validation['last_run'] = None
        
        self._update_validation_list()
        self.optimal_results.set_content('No optimization results yet')
        ui.notify("All results cleared")
    
    async def _show_run_details(self, event: any) -> None:
        """Show run details in a dialog."""
        row = event.args
        details = row.get('details')

        if not details:
            ui.notify("No details available for this run.", type='warning')
            return

        markdown_content = f"""
        ### Run Details: {details.get('validation_id')}
        - **Prompt**: `{details.get('prompt_key')}`
        - **Model**: `{details.get('model_key')}`
        - **Trial**: `{details.get('trial_id')}`
        - **Timestamp**: `{details.get('timestamp')}`
        - **Duration**: `{details.get('duration_ms', 0):.2f} ms`
        - **F1 Score**: `{details.get('f1_score', 0):.3f}`
        - **Success**: `{details.get('success')}`
        - **Entities Extracted**: `{details.get('entities_extracted')}`
        - **Compliance Score**: `{details.get('compliance_score', 0):.2f}`
        """
        
        with ui.dialog() as dialog, ui.card():
            ui.markdown(markdown_content)
            with ui.card_actions():
                ui.button('Close', on_click=dialog.close)
        
        await dialog

    def _create_pipeline_callback(self, prompt_content: str, prompt_type: str):
        """Create pipeline callback for benchmark execution"""
        def pipeline_callback(test_data):
            from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
            
            pipeline = StrictDynamicExtractionPipeline()
            
            # Set appropriate prompt based on type
            if prompt_type == 'NLP_EXTRACTION':
                pipeline.nlp_extractor.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
            elif prompt_type == 'MCODE_MAPPING':
                # This would require changes to McodeMapper
                pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE = prompt_content
            
            return pipeline.process_clinical_trial(test_data)
        
        return pipeline_callback


def run_clinical_benchmark_ui(port: int = 8084):
    """Run the clinical benchmark UI"""
    ui_instance = ClinicalBenchmarkUI()
    ui.run(title='Clinical Trial Benchmark Optimizer', port=port, reload=True)


if __name__ in {"__main__", "__mp_main__"}:
    import argparse
    parser = argparse.ArgumentParser(description='Run Clinical Trial Benchmark UI')
    parser.add_argument('--port', type=int, default=8084, help='Port to run the UI on')
    args = parser.parse_args()
    run_clinical_benchmark_ui(args.port)