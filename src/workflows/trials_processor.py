"""
Trials Processor Workflow - Process clinical trials with mCODE mapping.

This workflow handles processing clinical trial data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

import asyncio
from typing import Any, Dict, List, Optional

from src.pipeline import McodePipeline
from src.shared.models import enhance_trial_with_mcode_results
from src.storage.mcode_memory_storage import OncoCoreMemory
from src.utils.concurrency import AsyncTaskQueue, create_task

from .base_workflow import TrialsProcessorWorkflow as BaseTrialsProcessorWorkflow
from .base_workflow import WorkflowResult
from .trial_extractor import TrialExtractor
from .trial_summarizer import TrialSummarizer


class TrialsProcessor(BaseTrialsProcessorWorkflow):
    """
    Workflow for processing clinical trials with mCODE mapping.

    Processes trial data and stores mCODE summaries to CORE Memory.
    """

    def __init__(self, config: Any, memory_storage: Optional[OncoCoreMemory] = None):
        """
        Initialize the trials processor workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.pipeline: Optional[Any] = None

        # Initialize component classes
        self.extractor = TrialExtractor()
        self.summarizer = TrialSummarizer()

    async def execute_async(self, **kwargs: Any) -> WorkflowResult:
        """
        Execute the trials processing workflow asynchronously.

        By default, does NOT store results to CORE memory. Use store_in_memory=True to enable.

        Args:
            **kwargs: Workflow parameters including:
                 - trials_data: List of trial data to process
                 - engine: Processing engine ('regex' or 'llm')
                 - model: LLM model to use (for llm engine)
                 - prompt: Prompt template to use (for llm engine)
                 - workers: Number of concurrent workers
                 - store_in_memory: Whether to store results in CORE memory (default: False)

        Returns:
            WorkflowResult: Processing results
        """
        try:
            self.logger.info("Starting trials processor workflow execution")
            # Extract parameters
            trials_data = kwargs.get("trials_data", [])
            engine = kwargs.get("engine", "llm")
            model = kwargs.get("model") or "gpt-4o"
            prompt = kwargs.get("prompt", "direct_mcode_evidence_based_concise")
            workers = kwargs.get("workers", 0)

            # Default to NOT store in CORE memory - use --ingest to enable
            store_in_memory = False

            self.logger.info(
                f"Extracted parameters: trials_data="
                f"{len(trials_data) if trials_data else 0}, engine={engine}, "
                f"model={model}, prompt={prompt}, store_in_memory={store_in_memory}"
            )

            if not trials_data:
                self.logger.warning("No trial data provided for processing")
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for processing.",
                )

            # Initialize processing engine
            self.logger.info(f"Initializing McodePipeline with {engine} engine...")
            if not self.pipeline:
                self.logger.info(
                    f"Creating McodePipeline with engine={engine}, prompt_name={prompt}, "
                    f"model_name={model}"
                )
                try:
                    self.pipeline = McodePipeline(
                        prompt_name=prompt, model_name=model, engine=engine
                    )
                    self.logger.info("McodePipeline initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize McodePipeline: {e}")
                    raise
            else:
                self.logger.info("McodePipeline already initialized")

            # Process trials
            processed_trials: List[Any] = []
            successful_count = 0
            failed_count = 0

            self.logger.info(f"🔬 Processing {len(trials_data)} trials with {engine} engine")

            # Use AsyncTaskQueue for controlled concurrency
            effective_workers = max(1, workers)  # Ensure at least 1 worker

            self.logger.info(
                f"⚡ Using AsyncTaskQueue with {effective_workers} concurrent task{'s' if effective_workers > 1 else ''}"
            )

            # Create tasks for AsyncTaskQueue
            tasks = []
            for i, trial in enumerate(trials_data):
                task = create_task(
                    task_id=f"process_trial_{i+1}",
                    func=self._process_single_trial_async,
                    trial=trial,
                    model=model,
                    prompt=prompt,
                    index=i + 1,
                    store_in_memory=store_in_memory,
                )
                tasks.append(task)

            # Create and execute AsyncTaskQueue
            task_queue = AsyncTaskQueue(max_concurrent=effective_workers, name="TrialsProcessorQueue")

            def progress_callback(completed: int, total: int, result: Any) -> None:
                if result.success:
                    self.logger.debug(f"✅ Completed trial {completed}/{total}")
                else:
                    self.logger.warning(f"❌ Failed trial {completed}/{total}: {result.error}")

            task_results = await task_queue.execute_tasks(tasks, progress_callback=progress_callback)

            # Process results
            processed_trials = []
            successful_count = 0
            failed_count = 0

            for task_result in task_results:
                if task_result.success and task_result.result:
                    trial_result, success = task_result.result
                    processed_trials.append(trial_result)
                    if success:
                        successful_count += 1
                    else:
                        failed_count += 1
                else:
                    self.logger.error(f"Task failed: {task_result.error}")
                    failed_count += 1
                    # Add error trial
                    error_trial = {"McodeProcessingError": str(task_result.error)}
                    processed_trials.append(error_trial)

            # Calculate success rate
            total_count = len(trials_data)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Processing complete: {successful_count}/{total_count} trials successful"
            )

            # Log final summary
            if successful_count > 0:
                self.logger.info("✅ Trials processing completed successfully!")
            else:
                self.logger.error(f"❌ All {total_count} trials failed processing")

            return self._create_result(
                success=successful_count > 0,
                data=processed_trials,
                metadata={
                    "total_trials": total_count,
                    "successful": successful_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "engine_used": engine,
                    "model_used": model if engine == "llm" else None,
                    "prompt_used": prompt if engine == "llm" else None,
                    "stored_in_memory": store_in_memory and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials processing")

    def execute(self, **kwargs: Any) -> WorkflowResult:
        """
        Execute the trials processing workflow.

        By default, does NOT store results to CORE memory. Use store_in_memory=True to enable.

        Args:
            **kwargs: Workflow parameters including:
                 - trials_data: List of trial data to process
                 - engine: Processing engine ('regex' or 'llm')
                 - model: LLM model to use (for llm engine)
                 - prompt: Prompt template to use (for llm engine)
                 - workers: Number of concurrent workers
                 - store_in_memory: Whether to store results in CORE memory (default: False)

        Returns:
            WorkflowResult: Processing results
        """
        return asyncio.run(self.execute_async(**kwargs))

    # Removed caching methods - only API calls should be cached

    # Removed _make_trial_serializable - now in TrialCacheManager

    def _extract_trial_mcode_elements_cached(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial mCODE elements (no caching - pure computation)."""
        # mCODE extraction is pure computation, not API calls, so no caching
        return self.extractor.extract_trial_mcode_elements(trial)

    # Removed caching methods - only API calls should be cached

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        try:
            return str(trial["protocolSection"]["identificationModule"]["nctId"])
        except (KeyError, TypeError):
            import hashlib

            return f"unknown_trial_{hashlib.md5(str(trial).encode('utf-8')).hexdigest()[:8]}"

    # Removed old _extract_trial_mcode_elements method - now using TrialExtractor class

    # Removed all old extraction methods - now using TrialExtractor class

    def _extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive trial metadata for storage."""
        metadata: Dict[str, Any] = {}

        try:
            # Ensure trial is a dict
            if not isinstance(trial, dict):
                self.logger.error(f"Trial data is not a dict in metadata extraction: {type(trial)}")
                return metadata

            protocol_section = trial.get("protocolSection", {})
            if not isinstance(protocol_section, dict):
                self.logger.error(
                    f"Protocol section is not a dict in metadata extraction: {type(protocol_section)}"
                )
                return metadata

            # Basic trial info
            identification = protocol_section.get("identificationModule", {})
            if isinstance(identification, dict):
                metadata["nct_id"] = identification.get("nctId")
                metadata["brief_title"] = identification.get("briefTitle")
                metadata["official_title"] = identification.get("officialTitle")

            # Status and dates
            status = protocol_section.get("statusModule", {})
            if isinstance(status, dict):
                metadata["overall_status"] = status.get("overallStatus")
                start_struct = status.get("startDateStruct", {})
                if isinstance(start_struct, dict):
                    metadata["start_date"] = start_struct.get("date")
                completion_struct = status.get("completionDateStruct", {})
                if isinstance(completion_struct, dict):
                    metadata["completion_date"] = completion_struct.get("date")

            # Design information
            design = protocol_section.get("designModule", {})
            if isinstance(design, dict):
                metadata["study_type"] = design.get("studyType")
                metadata["phase"] = design.get("phases", [])
                metadata["primary_purpose"] = design.get("primaryPurpose")

            # Eligibility
            eligibility = protocol_section.get("eligibilityModule", {})
            if isinstance(eligibility, dict):
                metadata["minimum_age"] = eligibility.get("minimumAge")
                metadata["maximum_age"] = eligibility.get("maximumAge")
                metadata["sex"] = eligibility.get("sex")
                metadata["healthy_volunteers"] = eligibility.get("healthyVolunteers")

            # Conditions and interventions
            conditions_module = protocol_section.get("conditionsModule", {})
            if isinstance(conditions_module, dict):
                conditions = conditions_module.get("conditions", [])
                if isinstance(conditions, list):
                    metadata["conditions"] = [
                        c.get("name") for c in conditions if isinstance(c, dict)
                    ]

            interventions_module = protocol_section.get("armsInterventionsModule", {})
            if isinstance(interventions_module, dict):
                interventions = interventions_module.get("interventions", [])
                if isinstance(interventions, list):
                    metadata["interventions"] = [
                        i.get("name") for i in interventions if isinstance(i, dict)
                    ]

        except Exception as e:
            self.logger.error(f"Error extracting trial metadata: {e}")

        return metadata

    def _generate_trial_natural_language_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
        """Generate comprehensive natural language summary for clinical trial."""
        return self.summarizer.generate_trial_natural_language_summary(
            trial_id, mcode_elements, trial_data
        )

    def _generate_trial_regex_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
        """Generate a simple text-based summary for regex-processed trials."""
        try:
            # Extract basic trial information
            protocol_section = trial_data.get("protocolSection", {})
            identification = protocol_section.get("identificationModule", {})
            title = identification.get("briefTitle", "Unknown Title")

            # Extract key mCODE elements
            conditions = []
            if "TrialCancerConditions" in mcode_elements:
                cancer_conditions = mcode_elements["TrialCancerConditions"]
                if isinstance(cancer_conditions, list):
                    conditions = [
                        c.get("display", str(c)) for c in cancer_conditions if isinstance(c, dict)
                    ]
                elif isinstance(cancer_conditions, dict):
                    conditions = [cancer_conditions.get("display", str(cancer_conditions))]

            # Build simple summary
            summary_parts = [f"Clinical Trial {trial_id}: {title}"]

            if conditions:
                summary_parts.append(f"Conditions: {', '.join(conditions[:3])}")  # Limit to first 3

            # Add enrollment and status if available
            design = protocol_section.get("designModule", {})
            enrollment = design.get("enrollmentInfo", {}).get("count", "Unknown")
            if enrollment != "Unknown":
                summary_parts.append(f"Enrollment: {enrollment} participants")

            status = protocol_section.get("statusModule", {}).get("overallStatus", "Unknown")
            if status != "Unknown":
                summary_parts.append(f"Status: {status}")

            return ". ".join(summary_parts) + "."

        except Exception as e:
            self.logger.error(f"Error generating regex summary for {trial_id}: {e}")
            return f"Clinical Trial {trial_id}: Basic summary generation failed."

    def _convert_trial_mcode_to_mappings_format(
        self, mcode_elements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert trial mCODE elements to standardized mappings format for storage."""
        mappings = []

        try:
            for element_name, element_data in mcode_elements.items():
                self.logger.debug(
                    f"Converting trial element {element_name}: type={type(element_data)}, value={element_data}"
                )

                if isinstance(element_data, list):
                    # Handle multiple values (e.g., multiple conditions, interventions)
                    for item in element_data:
                        if isinstance(item, dict):
                            mapping = {
                                "mcode_element": element_name,
                                "value": item.get("display", str(item)),
                                "system": item.get("system"),
                                "code": item.get("code"),
                                "interpretation": item.get("interpretation"),
                            }
                        else:
                            mapping = {
                                "mcode_element": element_name,
                                "value": str(item),
                                "system": None,
                                "code": None,
                                "interpretation": None,
                            }
                        mappings.append(mapping)
                else:
                    # Handle single values
                    if isinstance(element_data, dict):
                        mapping = {
                            "mcode_element": element_name,
                            "value": element_data.get("display", str(element_data)),
                            "system": element_data.get("system"),
                            "code": element_data.get("code"),
                            "interpretation": element_data.get("interpretation"),
                        }
                    else:
                        mapping = {
                            "mcode_element": element_name,
                            "value": str(element_data),
                            "system": None,
                            "code": None,
                            "interpretation": None,
                        }
                    mappings.append(mapping)
        except Exception as e:
            self.logger.error(f"Error converting trial mCODE elements to mappings format: {e}")
            self.logger.debug(f"mcode_elements: {mcode_elements}")
            raise

        return mappings

    def _format_trial_mcode_element(self, element_name: str, system: str, code: str) -> str:
        """Centralized function to format trial mCODE elements consistently."""
        # Clean up system URLs to standard names
        if "snomed" in system.lower():
            clean_system = "SNOMED"
        elif "loinc" in system.lower():
            clean_system = "LOINC"
        elif "cvx" in system.lower():
            clean_system = "CVX"
        elif "rxnorm" in system.lower():
            clean_system = "RxNorm"
        elif "icd" in system.lower():
            clean_system = "ICD"
        elif "clinicaltrials.gov" in system.lower():
            clean_system = "ClinicalTrials.gov"
        else:
            # Remove URLs and keep only the system identifier
            clean_system = system.split("/")[-1].split(":")[-1].upper()

        return f"(mCODE: {element_name}; {clean_system}:{code})"

    def _process_single_trial_wrapper(
        self,
        trial: Dict[str, Any],
        model: str,
        prompt: str,
        index: int,
        store_in_memory: bool = False,
    ) -> tuple[Any, bool]:
        """
        Synchronous wrapper for async trial processing.

        Args:
            trial: Trial data to process
            model: LLM model to use
            prompt: Prompt template to use
            index: Trial index for logging

        Returns:
            tuple: (trial_result, success)
        """
        import asyncio

        async def process_async() -> tuple[Any, bool]:
            return await self._process_single_trial_async(
                trial, model, prompt, index, store_in_memory
            )

        return asyncio.run(process_async())

    async def _process_single_trial_async(
        self,
        trial: Dict[str, Any],
        model: str,
        prompt: str,
        index: int,
        store_in_memory: bool = False,
    ) -> tuple[Any, bool]:
        """
        Process a single trial asynchronously using unified pipeline.

        Args:
            trial: Trial data to process
            model: Model name (ignored for regex)
            prompt: Prompt name (ignored for regex)
            index: Trial index for logging

        Returns:
            tuple: (trial_result, success)
        """
        try:
            self.logger.info(f"🔬 Processing trial {index}")

            # Process trial (no caching for workflow results)

            # Process with unified pipeline (handles both LLM and regex)
            enhanced_trial = None
            mcode_success = False
            try:
                if self.pipeline is not None:
                    result = await self.pipeline.process(trial)
                    # Add mCODE results to trial using standardized utility
                    enhanced_trial = enhance_trial_with_mcode_results(trial, result)
                    mcode_success = True
                else:
                    # Create basic enhanced trial without mCODE results
                    enhanced_trial = trial.copy()
                    enhanced_trial["McodeProcessingError"] = "Pipeline not initialized"
            except Exception as mcode_error:
                self.logger.warning(
                    f"mCODE pipeline failed for trial {self._extract_trial_id(trial)}: {mcode_error}"
                )
                # Create basic enhanced trial without mCODE results
                enhanced_trial = trial.copy()
                enhanced_trial["McodeProcessingError"] = str(mcode_error)

            # Generate natural language summary using the summarizer
            trial_id = self._extract_trial_id(trial)
            mcode_elements = enhanced_trial.get("McodeResults", {}).get("mcode_mappings", [])
            summary = self.summarizer.generate_trial_natural_language_summary(
                trial_id, mcode_elements, trial
            )

            # Add summary to results
            if "McodeResults" not in enhanced_trial:
                enhanced_trial["McodeResults"] = {}
            enhanced_trial["McodeResults"]["natural_language_summary"] = summary

            # No caching for workflow results

            return enhanced_trial, mcode_success

        except Exception as e:
            self.logger.error(f"Failed to process trial {index}: {e}")
            # Add error information to trial
            error_trial = trial.copy()
            error_trial["ProcessingError"] = str(e)
            return error_trial, False

    def process_single_trial(self, trial: Dict[str, Any], **kwargs: Any) -> WorkflowResult:
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
            from typing import List, cast

            return self._create_result(
                success=True,
                data=cast(List[Any], result.data)[0],
                metadata=result.metadata,
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

    # Removed cache management methods - only API calls should be cached

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

    def _check_trial_has_full_data(self, trial: Dict[str, Any]) -> bool:
        """Check if trial data appears to be complete (from full study API) vs partial (from search API)."""
        if not trial or not isinstance(trial, dict):
            return False

        protocol_section = trial.get("protocolSection", {})
        if not isinstance(protocol_section, dict):
            return False

        # Check for fields that indicate full study data
        indicators = []

        # 1. Detailed eligibility criteria (longer than search results)
        eligibility = protocol_section.get("eligibilityModule", {})
        if isinstance(eligibility, dict):
            criteria = eligibility.get("eligibilityCriteria", "")
            if criteria and len(criteria) > 100:  # Search results are usually truncated
                indicators.append(True)

        # 2. Interventions with detailed information
        arms = protocol_section.get("armsInterventionsModule", {})
        if isinstance(arms, dict):
            interventions = arms.get("interventions", [])
            if interventions and len(interventions) > 0:
                # Check if interventions have detailed descriptions
                detailed_interventions = any(
                    isinstance(i, dict) and i.get("description", "") for i in interventions
                )
                if detailed_interventions:
                    indicators.append(True)

        # 3. Outcomes module (rarely present in search results)
        outcomes = protocol_section.get("outcomesModule", {})
        if isinstance(outcomes, dict) and outcomes.get("primaryOutcomes"):
            indicators.append(True)

        # 4. Derived section (only in full study data)
        derived_section = trial.get("derivedSection")
        if derived_section and isinstance(derived_section, dict):
            indicators.append(True)

        # 5. Detailed sponsor/collaborator information
        sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
        if isinstance(sponsor, dict):
            collaborators = sponsor.get("collaborators", [])
            if collaborators and len(collaborators) > 0:
                indicators.append(True)

        # Consider data complete if we have at least 3 indicators
        return len(indicators) >= 3
