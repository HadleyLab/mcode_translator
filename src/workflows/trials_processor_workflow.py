"""
Trials Processor Workflow - Process clinical trials with mCODE mapping.

This workflow handles processing clinical trial data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline import McodePipeline
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.logging_config import get_logger

from .base_workflow import ProcessorWorkflow, WorkflowResult
from src.shared.models import enhance_trial_with_mcode_results


class TrialsProcessorWorkflow(ProcessorWorkflow):
    """
    Workflow for processing clinical trials with mCODE mapping.

    Processes trial data and stores mCODE summaries to CORE Memory.
    """

    def __init__(self, config, memory_storage: Optional[McodeMemoryStorage] = None):
        """
        Initialize the trials processor workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.pipeline = None

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the trials processing workflow.

        Args:
            **kwargs: Workflow parameters including:
                - trials_data: List of trial data to process
                - model: LLM model to use
                - prompt: Prompt template to use
                - store_in_memory: Whether to store results in core memory

        Returns:
            WorkflowResult: Processing results
        """
        try:
            # Extract parameters
            trials_data = kwargs.get("trials_data", [])
            model = kwargs.get("model")
            prompt = kwargs.get("prompt", "direct_mcode_evidence_based_concise")
            store_in_memory = kwargs.get("store_in_memory", True)

            if not trials_data:
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for processing.",
                )

            # Initialize pipeline if needed
            if not self.pipeline:
                self.pipeline = McodePipeline(prompt_name=prompt, model_name=model)

            # Process trials
            processed_trials = []
            successful_count = 0
            failed_count = 0

            self.logger.info(
                f"🔬 Processing {len(trials_data)} trials with mCODE pipeline"
            )

            for i, trial in enumerate(trials_data):
                try:
                    self.logger.info(f"Processing trial {i+1}/{len(trials_data)}")

                    # Process with mCODE pipeline
                    result = self.pipeline.process_clinical_trial(trial)

                    # Add mCODE results to trial using standardized utility
                    enhanced_trial = enhance_trial_with_mcode_results(trial, result)

                    processed_trials.append(enhanced_trial)
                    successful_count += 1

                    # Store to core memory if requested
                    if store_in_memory and self.memory_storage:
                        trial_id = self._extract_trial_id(trial)
                        mcode_data = enhanced_trial["McodeResults"]

                        success = self.memory_storage.store_trial_mcode_summary(
                            trial_id, mcode_data
                        )
                        if success:
                            self.logger.info(
                                f"✅ Stored trial {trial_id} mCODE summary"
                            )
                        else:
                            self.logger.warning(
                                f"❌ Failed to store trial {trial_id} mCODE summary"
                            )

                except Exception as e:
                    self.logger.error(f"Failed to process trial {i+1}: {e}")
                    failed_count += 1

                    # Add error information to trial
                    error_trial = trial.copy()
                    error_trial["McodeProcessingError"] = str(e)
                    processed_trials.append(error_trial)

            # Calculate success rate
            total_count = len(trials_data)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Processing complete: {successful_count}/{total_count} trials successful"
            )

            return self._create_result(
                success=successful_count > 0,
                data=processed_trials,
                metadata={
                    "total_trials": total_count,
                    "successful": successful_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "model_used": model,
                    "prompt_used": prompt,
                    "stored_in_memory": store_in_memory
                    and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials processing")

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        try:
            return trial["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"unknown_trial_{hash(str(trial)) % 10000}"

    def process_single_trial(self, trial: Dict[str, Any], **kwargs) -> WorkflowResult:
        """
        Process a single clinical trial.

        Args:
            trial: Trial data to process
            **kwargs: Additional processing parameters

        Returns:
            WorkflowResult: Processing result
        """
        result = self.execute(trials_data=[trial], **kwargs)

        # Return single trial result
        if result.success and result.data:
            return self._create_result(
                success=True, data=result.data[0], metadata=result.metadata
            )
        else:
            return result

    def validate_trial_data(self, trial: Dict[str, Any]) -> bool:
        """
        Validate that trial data has required fields for processing.

        Args:
            trial: Trial data to validate

        Returns:
            bool: True if valid for processing
        """
        try:
            # Check for required fields
            protocol_section = trial.get("protocolSection", {})
            identification = protocol_section.get("identificationModule", {})
            nct_id = identification.get("nctId")

            if not nct_id:
                self.logger.warning("Trial missing NCT ID")
                return False

            eligibility = protocol_section.get("eligibilityModule", {})
            criteria = eligibility.get("eligibilityCriteria")

            if not criteria:
                self.logger.warning(f"Trial {nct_id} missing eligibility criteria")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating trial data: {e}")
            return False

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processing capabilities.

        Returns:
            Dict with processing statistics
        """
        if not self.pipeline:
            return {"status": "pipeline_not_initialized"}

        return {
            "status": "ready",
            "model": (
                getattr(self.pipeline.llm_mapper, "model_name", "unknown")
                if hasattr(self.pipeline, "llm_mapper")
                else "unknown"
            ),
            "prompt_template": getattr(self.pipeline, "prompt_name", "unknown"),
            "memory_storage_available": self.memory_storage is not None,
        }
