# Validation Logic Implementation Plan

## Overview
This document outlines the implementation plan for adding validation logic to the pipeline task tracker. This will enable the system to validate pipeline results against gold standard data and calculate relevant metrics.

## Implementation Steps

### 1. Extend LLMCallTask Data Class
Add validation metrics fields to the `LLMCallTask` data class:

```python
@dataclass
class LLMCallTask:
    """Represents a single LLM call task (extraction or mapping)"""
    name: str  # "NLP Extraction" or "Mcode Mapping"
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
    compliance_score: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate duration of the task"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
```

### 2. Add Validation Methods
Implement methods to perform validation against gold standard data:

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
        if expected_mappings and hasattr(result, 'Mcode_mappings'):
            self._calculate_mapping_metrics(task.mcode_mapping, result.mcode_mappings, expected_mappings)
            
        # Calculate compliance score from validation results
        if hasattr(result, 'validation_results'):
            task.mcode_mapping.compliance_score = result.validation_results.get('compliance_score', 0.0)
            if task.pipeline_type == 'Direct to Mcode':
                task.nlp_extraction.compliance_score = result.validation_results.get('compliance_score', 0.0)
            
    except Exception as e:
        logger.error(f"Validation failed for task {task.id}: {e}")
        task.error_message = f"Validation failed: {str(e)}"

def _calculate_extraction_metrics(self, subtask: LLMCallTask, actual_entities: List[Dict[str, Any]], 
                                 expected_entities: List[Dict[str, Any]]) -> None:
    """Calculate extraction metrics using fuzzy text matching"""
    try:
        # Use the existing fuzzy matching logic from the benchmark framework
        true_positives, false_positives, false_negatives = self._calculate_fuzzy_text_matches(
            actual_entities, expected_entities
        )
        
        # Calculate precision, recall, F1-score
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
    """Calculate mapping metrics using Mcode element matching"""
    try:
        # Create sets of (Mcode_element, value) tuples for comparison
        actual_tuples = set((m.get('Mcode_element', '').lower(), m.get('value', '').lower()) for m in actual_mappings)
        expected_tuples = set((m.get('Mcode_element', '').lower(), m.get('value', '').lower()) for m in expected_mappings)
        
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

def _calculate_fuzzy_text_matches(self, extracted_entities: List[Dict[str, Any]],
                                 expected_entities: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """
    Calculate text matches using fuzzy matching to handle different text representations
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    extracted_texts = [entity.get('text', '') for entity in extracted_entities]
    expected_texts = [entity.get('text', '') for entity in expected_entities]
    
    # Track matches
    matched_extracted = set()
    matched_expected = set()
    
    # First pass: exact matches
    for i, extracted_text in enumerate(extracted_texts):
        for j, expected_text in enumerate(expected_texts):
            if extracted_text == expected_text:
                matched_extracted.add(i)
                matched_expected.add(j)
    
    # Second pass: fuzzy matches for remaining entities
    for i, extracted_text in enumerate(extracted_texts):
        if i in matched_extracted:
            continue
            
        for j, expected_text in enumerate(expected_texts):
            if j in matched_expected:
                continue
            
            # Check if extracted text contains expected text (partial match)
            if expected_text.lower() in extracted_text.lower():
                matched_extracted.add(i)
                matched_expected.add(j)
                continue
            
            # Check if expected text contains extracted text (partial match)
            if extracted_text.lower() in expected_text.lower():
                matched_extracted.add(i)
                matched_expected.add(j)
                continue
            
            # Check for combined entities (e.g., "Pregnancy or breastfeeding" should match both)
            combined_match = False
            if " or " in extracted_text.lower():
                parts = [part.strip() for part in extracted_text.lower().split(" or ")]
                if expected_text.lower() in parts:
                    matched_extracted.add(i)
                    matched_expected.add(j)
                    combined_match = True
            
            if not combined_match and " or " in expected_text.lower():
                parts = [part.strip() for part in expected_text.lower().split(" or ")]
                if extracted_text.lower() in parts:
                    matched_extracted.add(i)
                    matched_expected.add(j)
    
    # Calculate metrics
    true_positives = len(matched_expected)
    false_positives = len(extracted_entities) - len(matched_extracted)
    false_negatives = len(expected_entities) - len(matched_expected)
    
    return true_positives, false_negatives
```

### 3. Update Pipeline Processing
Modify the `_process_pipeline_task` method to include validation:

```python
async def _process_pipeline_task(self, task: PipelineTask):
    """Process a single pipeline task"""
    logger.info(f"Processing pipeline task {task.id}")
    
    # Update task status
    task.status = TaskStatus.RUNNING
    task.start_time = asyncio.get_event_loop().time()
    if self.status_label:
        self.status_label.set_text(f"Running task {task.id}")
    
    try:
        # Get gold standard data for validation
        expected_entities = []
        expected_mappings = []
        
        if self.gold_standard_data and task.test_case_name in self.gold_standard_data:
            gold_data = self.gold_standard_data[task.test_case_name]
            expected_entities = gold_data.get('expected_extraction', {}).get('entities', [])
            expected_mappings = gold_data.get('expected_mcode_mappings', {}).get('mapped_elements', [])
        
        # Create pipeline instance with selected prompts
        if self.pipeline_selector.value == 'Direct to Mcode':
            prompt_name = self.direct_mcode_prompt_selector.value
            pipeline = McodePipeline(prompt_name=prompt_name)
        else:
            extraction_prompt = self.nlp_prompt_selector.value
            mapping_prompt = self.mcode_prompt_selector.value
            pipeline = NlpMcodePipeline(
                extraction_prompt_name=extraction_prompt,
                mapping_prompt_name=mapping_prompt
            )
        
        # Process the clinical trial with validation
        result = await self._run_pipeline_with_tracking(pipeline, task, expected_entities, expected_mappings)
        
        # Perform validation if gold standard data is available
        if result and (expected_entities or expected_mappings):
            self._perform_validation(task, result, expected_entities, expected_mappings)
        
        # Update task with result
        task.status = TaskStatus.SUCCESS
        task.end_time = asyncio.get_event_loop().time()
        self._add_notification(f"Task {task.id} completed", 'positive')
        logger.info(f"Pipeline task {task.id} completed successfully")
        
    except Exception as e:
        # Handle task failure
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.end_time = asyncio.get_event_loop().time()
        self._add_notification(f"Task {task.id} failed: {str(e)}", 'negative')
        logger.error(f"Pipeline task {task.id} failed: {e}")
```

### 4. Update Pipeline Execution Method
Modify the `_run_pipeline_with_tracking` method to accept and use gold standard data:

```python
async def _run_pipeline_with_tracking(self, pipeline: 'ProcessingPipeline', task: PipelineTask, 
                                  expected_entities: List[Dict[str, Any]] = None,
                                  expected_mappings: List[Dict[str, Any]] = None):
    """Run pipeline with tracking of individual LLM calls"""
    # Existing code for getting model and prompt information...
    
    # Update task details based on pipeline type
    if isinstance(pipeline, McodePipeline):
        task.mcode_mapping.name = "Direct to Mcode"
        task.mcode_mapping.status = TaskStatus.RUNNING
        task.mcode_mapping.start_time = asyncio.get_event_loop().time()
        task.mcode_mapping.details = f"Mapping text to Mcode using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={mapping_prompt})..."
        task.nlp_extraction.details = "Not applicable for this pipeline"
        task.nlp_extraction.status = TaskStatus.SUCCESS
    else:
        task.nlp_extraction.status = TaskStatus.RUNNING
        task.nlp_extraction.start_time = asyncio.get_event_loop().time()
        task.nlp_extraction.details = f"Extracting entities using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={extraction_prompt})..."
    
    self._update_task_list()
    
    try:
        # Run the pipeline process
        result = await run.io_bound(
            pipeline.process_clinical_trial,
            task.trial_data
        )
        
        if result:
            if isinstance(pipeline, NlpMcodePipeline):
                task.nlp_extraction.status = TaskStatus.SUCCESS
                task.nlp_extraction.end_time = asyncio.get_event_loop().time()
                task.nlp_extraction.details = f"Extracted {len(result.extracted_entities)} entities using {model_name} with prompt '{extraction_prompt}'"
                if result.metadata and 'token_usage' in result.metadata:
                    task.nlp_extraction.token_usage = result.metadata['token_usage']

                task.mcode_mapping.status = TaskStatus.RUNNING
                task.mcode_mapping.start_time = asyncio.get_event_loop().time()
                task.mcode_mapping.details = f"Mapping entities to Mcode using {model_name} (temp={temperature}, max_tokens={max_tokens}, prompt={mapping_prompt})..."
                self._update_task_list()

            task.mcode_mapping.status = TaskStatus.SUCCESS
            task.mcode_mapping.end_time = asyncio.get_event_loop().time()
            task.mcode_mapping.details = f"Mapped {len(result.mcode_mappings)} Mcode elements using {model_name} with prompt '{mapping_prompt}'"
            if result.metadata and 'token_usage' in result.metadata:
                task.mcode_mapping.token_usage = result.metadata['token_usage']
        
        return result
        
    except Exception as e:
        # Determine which step failed based on what was running
        if task.nlp_extraction.status == TaskStatus.RUNNING:
            task.nlp_extraction.status = TaskStatus.FAILED
            task.nlp_extraction.error_message = str(e)
            task.nlp_extraction.end_time = asyncio.get_event_loop().time()
        elif task.mcode_mapping.status == TaskStatus.RUNNING:
            task.mcode_mapping.status = TaskStatus.FAILED
            task.mcode_mapping.error_message = str(e)
            task.mcode_mapping.end_time = asyncio.get_event_loop().time()
        
        raise e
```

## Testing Plan
1. Verify validation logic correctly calculates metrics with sample data
2. Test fuzzy text matching with various text variations
3. Confirm Mcode element matching works with case-insensitive comparisons
4. Validate error handling for missing or malformed validation data
5. Test performance impact of validation calculations

## Dependencies
- Gold standard loading functionality
- Existing pipeline processing methods
- Logging infrastructure
- UI components for displaying metrics

## Rollout Strategy
1. Extend LLMCallTask data class with validation fields
2. Implement validation methods
3. Update pipeline processing to include validation
4. Test with sample data
5. Document new features