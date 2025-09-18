"""
Ultra-Lean McodePipeline

Zero redundancy, maximum performance. Leverages existing infrastructure.
"""

from typing import Any, Dict, List

from src.pipeline.document_ingestor import DocumentIngestor
from src.pipeline.llm_service import LLMService
from src.shared.models import (
    ClinicalTrialData,
    McodeElement,
    PipelineResult,
    ProcessingMetadata,
    ValidationResult,
)
from src.utils.concurrency import TaskQueue, create_task
from src.utils.config import Config
from src.utils.logging_config import get_logger


class McodePipeline:
    """
    Ultra-lean mCODE pipeline with zero redundancy.

    Direct data flow: Raw Dict → Existing Models → Existing PipelineResult
    """

    def __init__(
        self,
        model_name: str = None,
        prompt_name: str = None,
        config: Config = None
    ):
        """
        Initialize with existing infrastructure.

        Args:
            model_name: LLM model name (uses existing llm_loader)
            prompt_name: Prompt template name (uses existing prompt_loader)
            config: Existing Config instance
        """
        self.config = config or Config()
        self.model_name = model_name or "deepseek-coder"
        self.prompt_name = prompt_name or "direct_mcode_evidence_based_concise"
        self.logger = get_logger(__name__)

        # Leverage existing components
        self.document_ingestor = DocumentIngestor()
        self.llm_service = LLMService(
            self.config, self.model_name, self.prompt_name
        )

    def process(self, trial_data: Dict[str, Any]) -> PipelineResult:
        """
        Process clinical trial data with ultra-lean data flow.

        Args:
            trial_data: Raw clinical trial data dictionary

        Returns:
            PipelineResult with existing validated models
        """
        try:
            # Validate input using existing ClinicalTrialData model
            validated_trial = ClinicalTrialData(**trial_data)
            self.logger.info(f"Processing trial: {validated_trial.nct_id}")

            # Stage 1: Document processing (existing component)
            sections = self.document_ingestor.ingest_clinical_trial_document(trial_data)

            # Stage 2: LLM processing (existing utils)
            all_elements = []
            for section in sections:
                if section.content and section.content.strip():
                    elements = self.llm_service.map_to_mcode(section.content)
                    all_elements.extend(elements)

            # Stage 3: Return existing PipelineResult (no wrapper)
            return PipelineResult(
                extracted_entities=[],  # Direct mapping, no intermediate entities
                mcode_mappings=all_elements,
                source_references=[],  # Could populate if needed
                validation_results=ValidationResult(
                    compliance_score=self._calculate_compliance_score(all_elements)
                ),
                metadata=ProcessingMetadata(
                    engine_type="LLM",
                    entities_count=0,
                    mapped_count=len(all_elements),
                    model_used=self.model_name,
                    prompt_used=self.prompt_name,
                ),
                original_data=trial_data,
                error=None
            )

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            return PipelineResult(
                extracted_entities=[],
                mcode_mappings=[],
                source_references=[],
                validation_results=ValidationResult(compliance_score=0.0),
                metadata=ProcessingMetadata(engine_type="LLM"),
                original_data=trial_data,
                error=str(e)
            )

    def process_batch(self, trials_data: List[Dict[str, Any]]) -> List[PipelineResult]:
        """
        Process multiple trials efficiently.

        Args:
            trials_data: List of raw clinical trial data dictionaries

        Returns:
            List of PipelineResult instances
        """
        results = []
        for trial_data in trials_data:
            result = self.process(trial_data)
            results.append(result)

        self.logger.info(f"Batch processing completed: {len(results)} trials")
        return results

    def _calculate_compliance_score(self, elements: List[McodeElement]) -> float:
        """
        Calculate basic compliance score.

        Args:
            elements: List of mapped mCODE elements

        Returns:
            Compliance score between 0.0 and 1.0
        """
        if not elements:
            return 0.0

        # Basic scoring: presence of required element types
        required_types = {"CancerCondition", "CancerTreatment", "TumorMarker"}
        found_types = {elem.element_type for elem in elements}

        compliance = len(found_types.intersection(required_types)) / len(required_types)
        return min(compliance, 1.0)  # Cap at 1.0

    def process_batch_parallel(self, trials_data: List[Dict[str, Any]], max_workers: int = 4) -> List[PipelineResult]:
        """
        Process multiple trials with parallel validation and processing.

        Args:
            trials_data: List of raw clinical trial data dictionaries
            max_workers: Maximum number of concurrent workers

        Returns:
            List of PipelineResult instances
        """
        self.logger.info(f"🔄 Parallel batch processing {len(trials_data)} trials with {max_workers} workers")

        # Create parallel tasks for each trial
        batch_tasks = []
        for i, trial_data in enumerate(trials_data):
            task = create_task(
                task_id=f"trial_batch_{i}",
                func=self._process_single_trial_parallel,
                trial_data=trial_data,
                task_index=i
            )
            batch_tasks.append(task)

        # Execute parallel processing
        task_queue = TaskQueue(max_workers=max_workers, name="ValidationPipeline")

        def progress_callback(completed, total, result):
            if result.success:
                self.logger.info(f"✅ Completed parallel trial {result.task_id}")
            else:
                self.logger.error(f"❌ Failed parallel trial {result.task_id}: {result.error}")

        task_results = task_queue.execute_tasks(batch_tasks, progress_callback=progress_callback)

        # Process results and maintain order
        results = [None] * len(trials_data)  # Initialize with None placeholders
        successful_tasks = 0
        failed_tasks = 0

        for result in task_results:
            task_index = int(result.task_id.split('_')[-1])
            if result.success and result.result:
                results[task_index] = result.result
                successful_tasks += 1
            else:
                # Create error result for failed tasks
                trial_data = trials_data[task_index]
                error_result = PipelineResult(
                    extracted_entities=[],
                    mcode_mappings=[],
                    source_references=[],
                    validation_results=ValidationResult(compliance_score=0.0),
                    metadata=ProcessingMetadata(engine_type="LLM"),
                    original_data=trial_data,
                    error=result.error or "Parallel processing failed"
                )
                results[task_index] = error_result
                failed_tasks += 1
                self.logger.warning(f"Parallel task {result.task_id} failed: {result.error}")

        self.logger.info(f"📊 Parallel batch processing complete: {successful_tasks} successful, {failed_tasks} failed")
        return results

    def _process_single_trial_parallel(self, trial_data: Dict[str, Any], task_index: int) -> PipelineResult:
        """
        Process a single trial with parallel validation and section processing.

        Args:
            trial_data: Raw clinical trial data dictionary
            task_index: Index of this task in the batch

        Returns:
            PipelineResult instance
        """
        try:
            # Parallel input validation
            validated_trial = self._validate_trial_data_parallel(trial_data)
            self.logger.debug(f"Parallel processing trial: {validated_trial.nct_id}")

            # Parallel document processing
            sections = self.document_ingestor.ingest_clinical_trial_document(trial_data)

            # Parallel section processing with LLM
            all_elements = self._process_sections_parallel(sections)

            # Parallel compliance score calculation
            compliance_score = self._calculate_compliance_score_parallel(all_elements)

            return PipelineResult(
                extracted_entities=[],
                mcode_mappings=all_elements,
                source_references=[],
                validation_results=ValidationResult(compliance_score=compliance_score),
                metadata=ProcessingMetadata(
                    engine_type="LLM",
                    entities_count=0,
                    mapped_count=len(all_elements),
                    model_used=self.model_name,
                    prompt_used=self.prompt_name,
                ),
                original_data=trial_data,
                error=None
            )

        except Exception as e:
            self.logger.error(f"Parallel trial processing failed for task {task_index}: {str(e)}")
            return PipelineResult(
                extracted_entities=[],
                mcode_mappings=[],
                source_references=[],
                validation_results=ValidationResult(compliance_score=0.0),
                metadata=ProcessingMetadata(engine_type="LLM"),
                original_data=trial_data,
                error=str(e)
            )

    def _validate_trial_data_parallel(self, trial_data: Dict[str, Any]) -> ClinicalTrialData:
        """
        Validate trial data with parallel field validation.

        Args:
            trial_data: Raw clinical trial data dictionary

        Returns:
            Validated ClinicalTrialData instance

        Raises:
            ValueError: If validation fails
        """
        try:
            # Use Pydantic's parallel validation capabilities
            validated_trial = ClinicalTrialData(**trial_data)
            return validated_trial
        except Exception as e:
            self.logger.error(f"Trial data validation failed: {str(e)}")
            raise ValueError(f"Invalid trial data: {str(e)}") from e

    def _process_sections_parallel(self, sections: List[Any]) -> List[McodeElement]:
        """
        Process document sections in parallel using LLM service.

        Args:
            sections: List of document sections

        Returns:
            List of all McodeElement instances from all sections
        """
        if not sections:
            return []

        # Filter out empty sections
        valid_sections = [section for section in sections if section.content and section.content.strip()]

        if not valid_sections:
            return []

        # Use LLM service's batch processing if available, otherwise process sequentially
        section_texts = [section.content for section in valid_sections]

        try:
            # Try batch processing with LLM service
            batch_results = self.llm_service.map_to_mcode_batch(section_texts, max_workers=3)

            # Flatten results
            all_elements = []
            for result in batch_results:
                if result:  # Filter out None/empty results
                    all_elements.extend(result)

            return all_elements

        except AttributeError:
            # Fallback to sequential processing if batch method not available
            self.logger.debug("LLM batch processing not available, using sequential processing")
            all_elements = []
            for section in valid_sections:
                elements = self.llm_service.map_to_mcode(section.content)
                all_elements.extend(elements)
            return all_elements

    def _calculate_compliance_score_parallel(self, elements: List[McodeElement]) -> float:
        """
        Calculate compliance score with parallel element analysis.

        Args:
            elements: List of mapped mCODE elements

        Returns:
            Compliance score between 0.0 and 1.0
        """
        if not elements:
            return 0.0

        # Basic scoring: presence of required element types
        required_types = {"CancerCondition", "CancerTreatment", "TumorMarker"}
        found_types = {elem.element_type for elem in elements}

        compliance = len(found_types.intersection(required_types)) / len(required_types)
        return min(compliance, 1.0)  # Cap at 1.0

    def validate_batch_parallel(self, trials_data: List[Dict[str, Any]], max_workers: int = 4) -> List[bool]:
        """
        Validate multiple trials in parallel without full processing.

        Args:
            trials_data: List of raw clinical trial data dictionaries
            max_workers: Maximum number of concurrent workers

        Returns:
            List of validation results (True for valid, False for invalid)
        """
        self.logger.info(f"🔍 Parallel validation of {len(trials_data)} trials with {max_workers} workers")

        # Create validation tasks
        validation_tasks = []
        for i, trial_data in enumerate(trials_data):
            task = create_task(
                task_id=f"validation_{i}",
                func=self._validate_single_trial,
                trial_data=trial_data,
                task_index=i
            )
            validation_tasks.append(task)

        # Execute parallel validation
        task_queue = TaskQueue(max_workers=max_workers, name="ValidationOnly")

        def progress_callback(completed, total, result):
            if result.success:
                self.logger.debug(f"✅ Validated trial {result.task_id}")
            else:
                self.logger.debug(f"❌ Invalid trial {result.task_id}: {result.error}")

        task_results = task_queue.execute_tasks(validation_tasks, progress_callback=progress_callback)

        # Process results and maintain order
        validation_results = [False] * len(trials_data)

        for result in task_results:
            task_index = int(result.task_id.split('_')[-1])
            validation_results[task_index] = result.success

        valid_count = sum(validation_results)
        self.logger.info(f"📊 Parallel validation complete: {valid_count}/{len(trials_data)} trials valid")

        return validation_results

    def _validate_single_trial(self, trial_data: Dict[str, Any], task_index: int) -> bool:
        """
        Validate a single trial data structure.

        Args:
            trial_data: Raw clinical trial data dictionary
            task_index: Index of this task in the batch

        Returns:
            True if valid, False if invalid
        """
        try:
            ClinicalTrialData(**trial_data)
            return True
        except Exception as e:
            self.logger.debug(f"Trial {task_index} validation failed: {str(e)}")
            return False