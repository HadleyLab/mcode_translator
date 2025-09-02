# Gold Standard Loading Implementation Plan

## Overview
This document outlines the implementation plan for adding gold standard loading functionality to the pipeline task tracker. This will enable the system to load and use gold standard data for validation and benchmarking purposes.

## Implementation Steps

### 1. Add Gold Standard Loading Method
Add a method to the `PipelineTaskTrackerUI` class to load gold standard data from JSON files:

```python
def _load_gold_standard_data(self) -> Optional[Dict[str, Any]]:
    """Load gold standard data from JSON file"""
    try:
        gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
        if gold_file.exists():
            with open(gold_file, 'r') as f:
                data = json.load(f)
                logger.info("Successfully loaded gold standard data")
                return data.get("gold_standard", {})
        else:
            logger.warning("Gold standard data file not found")
            return None
    except Exception as e:
        logger.error(f"Failed to load gold standard data: {e}")
        return None
```

### 2. Update Class Initialization
Modify the `__init__` method to load gold standard data during initialization:

```python
def __init__(self):
    # Existing initialization code...
    
    # Gold standard data
    self.gold_standard_data = self._load_gold_standard_data()
    
    # Setup UI
    self._setup_ui()
```

### 3. Add Gold Standard Data to Task Processing
Update the `_process_pipeline_task` method to include gold standard data when available:

```python
async def _process_pipeline_task(self, task: PipelineTask):
    # Existing code...
    
    try:
        # Get gold standard data for validation
        expected_entities = []
        expected_mappings = []
        
        if self.gold_standard_data and task.test_case_name in self.gold_standard_data:
            gold_data = self.gold_standard_data[task.test_case_name]
            expected_entities = gold_data.get('expected_extraction', {}).get('entities', [])
            expected_mappings = gold_data.get('expected_mcode_mappings', {}).get('mapped_elements', [])
        
        # Create pipeline instance with selected prompts
        if self.pipeline_selector.value == 'Direct to mCODE':
            prompt_name = self.direct_mcode_prompt_selector.value
            pipeline = McodePipeline(prompt_name=prompt_name)
        else:
            extraction_prompt = self.nlp_prompt_selector.value
            mapping_prompt = self.mcode_prompt_selector.value
            pipeline = NlpExtractionToMcodeMappingPipeline(
                extraction_prompt_name=extraction_prompt,
                mapping_prompt_name=mapping_prompt
            )
        
        # Process the clinical trial with validation
        result = await self._run_pipeline_with_tracking(pipeline, task, expected_entities, expected_mappings)
```

### 4. Update Pipeline Execution Method
Modify the `_run_pipeline_with_tracking` method to accept and use gold standard data:

```python
async def _run_pipeline_with_tracking(self, pipeline: 'ProcessingPipeline', task: PipelineTask, 
                                  expected_entities: List[Dict[str, Any]] = None,
                                  expected_mappings: List[Dict[str, Any]] = None):
    # Existing code...
    
    try:
        # Run the pipeline process
        result = await run.io_bound(
            pipeline.process_clinical_trial,
            task.trial_data
        )
        
        # Perform validation if gold standard data is available
        if result and (expected_entities or expected_mappings):
            self._perform_validation(task, result, expected_entities, expected_mappings)
        
        # Existing code for updating task status...
        
        return result
```

### 5. Add Validation Method
Implement a method to perform validation against gold standard data:

```python
def _perform_validation(self, task: PipelineTask, result: Any, 
                       expected_entities: List[Dict[str, Any]], 
                       expected_mappings: List[Dict[str, Any]]) -> None:
    """Perform validation against gold standard data"""
    try:
        # Calculate extraction metrics
        if expected_entities and hasattr(result, 'extracted_entities'):
            self._calculate_extraction_metrics(task.nlp_extraction, result.extracted_entities, expected_entities)
        
        # Calculate mapping metrics
        if expected_mappings and hasattr(result, 'mcode_mappings'):
            self._calculate_mapping_metrics(task.mcode_mapping, result.mcode_mappings, expected_mappings)
            
    except Exception as e:
        logger.error(f"Validation failed for task {task.id}: {e}")
        task.error_message = f"Validation failed: {str(e)}"

def _calculate_extraction_metrics(self, subtask: LLMCallTask, actual_entities: List[Dict[str, Any]], 
                                 expected_entities: List[Dict[str, Any]]) -> None:
    """Calculate extraction metrics using fuzzy text matching"""
    try:
        # Extract texts for comparison
        actual_texts = set(entity.get('text', '') for entity in actual_entities)
        expected_texts = set(entity.get('text', '') for entity in expected_entities)
        
        # Calculate precision, recall, F1-score
        true_positives = len(actual_texts & expected_texts)
        false_positives = len(actual_texts - expected_texts)
        false_negatives = len(expected_texts - actual_texts)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update subtask with metrics
        subtask.precision = precision
        subtask.recall = recall
        subtask.f1_score = f1_score
        
        logger.info(f"Extraction validation - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
        
    except Exception as e:
        logger.error(f"Extraction metrics calculation failed: {e}")

def _calculate_mapping_metrics(self, subtask: LLMCallTask, actual_mappings: List[Dict[str, Any]], 
                              expected_mappings: List[Dict[str, Any]]) -> None:
    """Calculate mapping metrics using mCODE element matching"""
    try:
        # Create sets of (mcode_element, value) tuples for comparison
        actual_tuples = set((m.get('mcode_element', ''), m.get('value', '')) for m in actual_mappings)
        expected_tuples = set((m.get('mcode_element', ''), m.get('value', '')) for m in expected_mappings)
        
        # Calculate precision, recall, F1-score
        true_positives = len(actual_tuples & expected_tuples)
        false_positives = len(actual_tuples - expected_tuples)
        false_negatives = len(expected_tuples - actual_tuples)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Update subtask with metrics
        subtask.precision = precision
        subtask.recall = recall
        subtask.f1_score = f1_score
        
        logger.info(f"Mapping validation - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
        
    except Exception as e:
        logger.error(f"Mapping metrics calculation failed: {e}")
```

### 6. Extend Data Classes
Add validation metrics fields to the `LLMCallTask` data class:

```python
@dataclass
class LLMCallTask:
    """Represents a single LLM call task (extraction or mapping)"""
    name: str  # "NLP Extraction" or "mCODE Mapping"
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    details: Optional[str] = None
    
    # Validation metrics
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate duration of the task"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
```

### 7. Update UI Display
Modify the `_create_subtask_row` method to display validation metrics:

```python
def _create_subtask_row(self, subtask: LLMCallTask):
    """Create a row for a sub-task"""
    with ui.card().classes('w-full bg-gray-50 dark:bg-gray-80 mt-2'):
        with ui.row().classes('w-full justify-between items-center'):
            ui.label(subtask.name).classes('font-medium')
            
            # Status indicator
            status_color = {
                TaskStatus.PENDING: 'blue',
                TaskStatus.RUNNING: 'orange',
                TaskStatus.SUCCESS: 'green',
                TaskStatus.FAILED: 'red'
            }.get(subtask.status, 'gray')
            
            status_text = subtask.status.value.capitalize()
            if subtask.duration:
                status_text += f" ({subtask.duration:.2f}s)"
                
            ui.label(status_text).classes(f'text-{status_color}-600')
        
        # Validation metrics
        if subtask.precision is not None and subtask.recall is not None and subtask.f1_score is not None:
            with ui.row().classes('w-full text-sm mt-1'):
                ui.label(f"Precision: {subtask.precision:.3f}").classes('text-gray-600 dark:text-gray-400 mr-4')
                ui.label(f"Recall: {subtask.recall:.3f}").classes('text-gray-600 dark:text-gray-400 mr-4')
                ui.label(f"F1-Score: {subtask.f1_score:.3f}").classes('text-gray-600 dark:text-gray-400')
        
        # Additional details
        if subtask.details or subtask.error_message or subtask.token_usage:
            with ui.column().classes('w-full text-sm mt-1'):
                if subtask.details:
                    ui.label(subtask.details).classes('text-gray-600 dark:text-gray-400')
                
                if subtask.error_message:
                    ui.label(f"Error: {subtask.error_message}").classes('text-red-600')
                
                if subtask.token_usage:
                    token_text = f"Tokens: {subtask.token_usage.get('total_tokens', 'N/A')} "
                    token_text += f"(Prompt: {subtask.token_usage.get('prompt_tokens', 'N/A')}, "
                    token_text += f"Completion: {subtask.token_usage.get('completion_tokens', 'N/A')})"
                    ui.label(token_text).classes('text-gray-600 dark:text-gray-400')
```

## Testing Plan
1. Verify gold standard data loads correctly
2. Test validation metrics calculation with sample data
3. Confirm UI displays validation metrics properly
4. Validate error handling for missing or malformed gold standard data
5. Test performance impact of validation calculations

## Dependencies
- Existing JSON loading functionality
- Logging infrastructure
- UI components for displaying metrics
- Pipeline processing methods

## Rollout Strategy
1. Implement gold standard loading functionality
2. Add validation metrics calculation
3. Extend data classes with validation fields
4. Update UI to display validation results
5. Test with sample data
6. Document new features