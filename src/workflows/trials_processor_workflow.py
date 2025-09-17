"""
Trials Processor Workflow - Process clinical trials with mCODE mapping.

This workflow handles processing clinical trial data with mCODE mapping
and stores the resulting summaries to CORE Memory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline import McodePipeline
from src.services.summarizer import McodeSummarizer
from src.shared.models import enhance_trial_with_mcode_results
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.api_manager import APIManager
from src.utils.logging_config import get_logger

from .base_workflow import TrialsProcessorWorkflow, WorkflowResult


class ClinicalTrialsProcessorWorkflow(TrialsProcessorWorkflow):
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

        # Initialize caching for workflow-level results
        api_manager = APIManager()
        self.workflow_cache = api_manager.get_cache("trials_processor")
        self.mcode_cache = api_manager.get_cache("mcode_extraction")
        self.summary_cache = api_manager.get_cache("trial_summaries")

    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the trials processing workflow.

        By default, does NOT store results to CORE memory. Use store_in_memory=True to enable.

        Args:
            **kwargs: Workflow parameters including:
                - trials_data: List of trial data to process
                - model: LLM model to use
                - prompt: Prompt template to use
                - workers: Number of concurrent workers
                - store_in_memory: Whether to store results in CORE memory (default: False)

        Returns:
            WorkflowResult: Processing results
        """
        try:
            self.logger.info("Starting trials processor workflow execution")
            # Extract parameters
            trials_data = kwargs.get("trials_data", [])
            model = kwargs.get("model")
            prompt = kwargs.get("prompt", "direct_mcode_evidence_based_concise")
            workers = kwargs.get("workers", 0)

            # Default to NOT store in CORE memory - use --ingest to enable
            store_in_memory = False

            self.logger.info(
                f"Extracted parameters: trials_data={len(trials_data) if trials_data else 0}, model={model}, prompt={prompt}, store_in_memory={store_in_memory}"
            )

            if not trials_data:
                self.logger.warning("No trial data provided for processing")
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for processing.",
                )

            # Initialize pipeline if needed
            self.logger.info("Initializing McodePipeline...")
            if not self.pipeline:
                self.logger.info(
                    f"Creating McodePipeline with prompt_name={prompt}, model_name={model}"
                )
                try:
                    self.pipeline = McodePipeline(prompt_name=prompt, model_name=model)
                    self.logger.info("McodePipeline initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize McodePipeline: {e}")
                    raise
            else:
                self.logger.info("McodePipeline already initialized")

            # Process trials
            processed_trials = []
            successful_count = 0
            failed_count = 0

            self.logger.info(
                f"🔬 Processing {len(trials_data)} trials with mCODE pipeline"
            )

            if workers > 0:
                # Use concurrent processing
                self.logger.info(
                    f"⚡ Using concurrent processing with {workers} workers"
                )
                import asyncio
                import concurrent.futures
                from functools import partial

                async def process_concurrent():
                    # Create a thread pool for CPU-bound LLM processing
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=workers
                    ) as executor:
                        loop = asyncio.get_event_loop()

                        # Create tasks for concurrent processing
                        tasks = []
                        for i, trial in enumerate(trials_data):
                            task = loop.run_in_executor(
                                executor,
                                partial(
                                    self._process_single_trial_sync,
                                    trial,
                                    model,
                                    prompt,
                                    i + 1,
                                ),
                            )
                            tasks.append(task)

                        # Wait for all tasks to complete
                        results = await asyncio.gather(*tasks, return_exceptions=True)

                        # Process results
                        processed_trials = []
                        successful_count = 0
                        failed_count = 0

                        for result in results:
                            if isinstance(result, Exception):
                                self.logger.error(
                                    f"Task failed with exception: {result}"
                                )
                                failed_count += 1
                                # Add error trial
                                error_trial = {"McodeProcessingError": str(result)}
                                processed_trials.append(error_trial)
                            else:
                                trial_result, success = result
                                processed_trials.append(trial_result)
                                if success:
                                    successful_count += 1
                                else:
                                    failed_count += 1

                        return processed_trials, successful_count, failed_count

                processed_trials, successful_count, failed_count = asyncio.run(
                    process_concurrent()
                )

            else:
                # Use sequential processing
                self.logger.info("📝 Using sequential processing")
                processed_trials = []
                successful_count = 0
                failed_count = 0

                for i, trial in enumerate(trials_data):
                    try:
                        self.logger.info(
                            f"🔬 Processing trial {i+1}/{len(trials_data)}"
                        )
                        self.logger.debug(
                            f"Trial type: {type(trial)}, keys: {trial.keys() if isinstance(trial, dict) else 'Not a dict'}"
                        )

                        # Debug: Check trial data integrity before processing
                        protocol_section = trial.get("protocolSection", {})
                        identification = protocol_section.get(
                            "identificationModule", {}
                        )
                        trial_id = identification.get("nctId", "Unknown")
                        self.logger.debug(f"Trial {i+1} NCT ID: {trial_id}")
                        self.logger.debug(
                            f"Trial {i+1} brief title: {identification.get('briefTitle', 'Unknown')}"
                        )

                        # Check for cached processed trial result
                        cached_result = self._get_cached_trial_result(
                            trial, model, prompt
                        )
                        if cached_result is not None:
                            self.logger.info(
                                f"✅ Cache HIT for trial {trial_id} - using cached result"
                            )
                            processed_trials.append(cached_result)
                            successful_count += 1
                            continue

                        # Try to process with mCODE pipeline
                        enhanced_trial = None
                        mcode_success = False
                        try:
                            self.logger.debug(
                                f"Starting mCODE pipeline processing for trial {i+1}"
                            )
                            result = self.pipeline.process_clinical_trial(trial)
                            self.logger.debug(f"mCODE pipeline result: {result}")

                            # Add mCODE results to trial using standardized utility
                            enhanced_trial = enhance_trial_with_mcode_results(
                                trial, result
                            )
                            mcode_success = True
                        except Exception as mcode_error:
                            self.logger.warning(
                                f"mCODE pipeline failed for trial {trial_id}: {mcode_error}"
                            )
                            # Create basic enhanced trial without mCODE results
                            enhanced_trial = trial.copy()
                            enhanced_trial["McodeProcessingError"] = str(mcode_error)

                        # Cache the processed trial result
                        self._cache_trial_result(enhanced_trial, model, prompt)

                        # Always generate natural language summary, regardless of mCODE success
                        self.logger.debug(
                            f"Generating natural language summary for trial {trial_id}"
                        )
                        try:
                            # Extract comprehensive mCODE elements from trial data (with caching)
                            comprehensive_mcode = (
                                self._extract_trial_mcode_elements_cached(trial)
                            )
                            self.logger.debug(
                                f"Extracted {len(comprehensive_mcode)} mCODE elements: {list(comprehensive_mcode.keys())}"
                            )

                            # Create natural language summary for CORE knowledge graph (with caching)
                            natural_language_summary = (
                                self._generate_trial_natural_language_summary_cached(
                                    trial_id, comprehensive_mcode, trial
                                )
                            )

                            # Check if we have full trial data for better summarization
                            has_full_data = self._check_trial_has_full_data(trial)
                            if has_full_data:
                                self.logger.info(
                                    f"✅ Generated comprehensive trial summary for {trial_id} using full clinical data: {natural_language_summary[:100]}..."
                                )
                            else:
                                self.logger.info(
                                    f"⚠️  Generated trial summary for {trial_id} using partial data (consider using NCT ID for complete data): {natural_language_summary[:100]}..."
                                )

                            # Add natural language summary to the enhanced trial data
                            if "McodeResults" not in enhanced_trial:
                                enhanced_trial["McodeResults"] = {}
                            enhanced_trial["McodeResults"][
                                "natural_language_summary"
                            ] = natural_language_summary
                            self.logger.debug(
                                f"Added natural language summary to enhanced_trial McodeResults: {len(natural_language_summary)} chars"
                            )

                            # Store to CORE memory if requested
                            if store_in_memory and self.memory_storage:
                                # Prepare comprehensive mCODE data for storage
                                mcode_data = {
                                    "mcode_mappings": self._convert_trial_mcode_to_mappings_format(
                                        comprehensive_mcode
                                    ),
                                    "natural_language_summary": natural_language_summary,
                                    "comprehensive_mcode_elements": comprehensive_mcode,
                                    "trial_metadata": self._extract_trial_metadata(
                                        trial
                                    ),
                                    "pipeline_results": enhanced_trial.get(
                                        "McodeResults", {}
                                    ),
                                    "original_trial_data": trial,  # Include original trial data for summarizer
                                }

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
                        except Exception as summary_error:
                            self.logger.error(
                                f"Failed to generate summary for trial {trial_id}: {summary_error}"
                            )

                        processed_trials.append(enhanced_trial)
                        if mcode_success:
                            successful_count += 1
                        # Still count as processed even if mCODE failed but summary succeeded

                    except Exception as e:
                        self.logger.error(f"Failed to process trial {i+1}: {e}")
                        failed_count += 1

                        # Add error information to trial
                        error_trial = trial.copy()
                        error_trial["ProcessingError"] = str(e)
                        processed_trials.append(error_trial)

            # Calculate success rate
            total_count = len(trials_data)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Processing complete: {successful_count}/{total_count} trials successful"
            )

            # Log final summary
            if successful_count > 0:
                self.logger.info(f"✅ Trials processing completed successfully!")
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
                    "model_used": model,
                    "prompt_used": prompt,
                    "stored_in_memory": store_in_memory and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials processing")

    def _get_cached_trial_result(
        self, trial: Dict[str, Any], model: str, prompt: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached processed trial result if available."""
        trial_id = self._extract_trial_id(trial)
        cache_key_data = {
            "function": "processed_trial",
            "trial_id": trial_id,
            "model": model,
            "prompt": prompt,
            "trial_hash": hash(str(trial))
            % 1000000,  # Include trial content hash for cache invalidation
        }

        cached_result = self.workflow_cache.get_by_key(cache_key_data)
        if cached_result is not None:
            self.logger.debug(f"Cache HIT for processed trial {trial_id}")
            return cached_result
        return None

    def _cache_trial_result(
        self, processed_trial: Dict[str, Any], model: str, prompt: str
    ) -> None:
        """Cache processed trial result."""
        trial_id = self._extract_trial_id(processed_trial)
        cache_key_data = {
            "function": "processed_trial",
            "trial_id": trial_id,
            "model": model,
            "prompt": prompt,
            "trial_hash": hash(str(processed_trial)) % 1000000,
        }

        # Convert McodeElement objects to dicts for JSON serialization
        serializable_trial = self._make_trial_serializable(processed_trial)
        self.workflow_cache.set_by_key(serializable_trial, cache_key_data)
        self.logger.debug(f"Cached processed trial {trial_id}")

    def _make_trial_serializable(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Convert McodeElement objects to dictionaries for JSON serialization."""
        import copy

        serializable_trial = copy.deepcopy(trial)

        # Convert McodeResults if present
        if "McodeResults" in serializable_trial:
            mcode_results = serializable_trial["McodeResults"]
            if "mcode_mappings" in mcode_results:
                # Convert McodeElement objects to dicts
                mappings = []
                for mapping in mcode_results["mcode_mappings"]:
                    if hasattr(mapping, "model_dump"):  # Pydantic model
                        mappings.append(mapping.model_dump())
                    else:  # Already a dict
                        mappings.append(mapping)
                mcode_results["mcode_mappings"] = mappings

        return serializable_trial

    def _extract_trial_mcode_elements_cached(
        self, trial: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract trial mCODE elements with caching."""
        trial_id = self._extract_trial_id(trial)
        cache_key_data = {
            "function": "mcode_extraction",
            "trial_id": trial_id,
            "trial_hash": hash(str(trial)) % 1000000,
        }

        cached_result = self.mcode_cache.get_by_key(cache_key_data)
        if cached_result is not None:
            self.logger.debug(f"Cache HIT for mCODE extraction {trial_id}")
            return cached_result

        # Extract mCODE elements
        mcode_elements = self._extract_trial_mcode_elements(trial)

        # Cache the result
        self.mcode_cache.set_by_key(mcode_elements, cache_key_data)
        self.logger.debug(f"Cached mCODE extraction for {trial_id}")

        return mcode_elements

    def _generate_trial_natural_language_summary_cached(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
        """Generate natural language summary with caching."""
        cache_key_data = {
            "function": "natural_language_summary",
            "trial_id": trial_id,
            "mcode_hash": hash(str(mcode_elements)) % 1000000,
            "trial_hash": hash(str(trial_data)) % 1000000,
        }

        cached_result = self.summary_cache.get_by_key(cache_key_data)
        if cached_result is not None:
            self.logger.debug(f"Cache HIT for natural language summary {trial_id}")
            return cached_result

        # Generate summary
        summary = self._generate_trial_natural_language_summary(
            trial_id, mcode_elements, trial_data
        )

        # Cache the result
        self.summary_cache.set_by_key(summary, cache_key_data)
        self.logger.debug(f"Cached natural language summary for {trial_id}")

        return summary

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        try:
            return trial["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"unknown_trial_{hash(str(trial)) % 10000}"

    def _extract_trial_mcode_elements(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive mCODE elements from clinical trial data.

        This mirrors the patient mCODE extraction but focuses on trial-specific elements
        that align with patient profiles for optimal matching.
        """
        mcode_elements = {}

        try:
            # Ensure trial is a dict
            if not isinstance(trial, dict):
                self.logger.error(
                    f"Trial data is not a dict: {type(trial)}, value: {trial}"
                )
                return mcode_elements

            protocol_section = trial.get("protocolSection", {})
            if not isinstance(protocol_section, dict):
                self.logger.error(
                    f"Protocol section is not a dict: {type(protocol_section)}, value: {protocol_section}"
                )
                return mcode_elements

            self.logger.debug(
                f"Processing protocol section with keys: {list(protocol_section.keys())}"
            )

            # Extract trial identification and basic info
            identification = protocol_section.get("identificationModule", {})
            if isinstance(identification, dict):
                mcode_elements.update(
                    self._extract_trial_identification(identification)
                )

            # Extract eligibility criteria in mCODE space
            eligibility = protocol_section.get("eligibilityModule", {})
            if isinstance(eligibility, dict):
                mcode_elements.update(
                    self._extract_trial_eligibility_mcode(eligibility)
                )

            # Extract conditions as mCODE CancerCondition
            conditions = protocol_section.get("conditionsModule", {})
            if isinstance(conditions, dict):
                mcode_elements.update(self._extract_trial_conditions_mcode(conditions))

            # Extract interventions as mCODE CancerRelatedMedicationStatement
            interventions = protocol_section.get("armsInterventionsModule", {})
            if isinstance(interventions, dict):
                mcode_elements.update(
                    self._extract_trial_interventions_mcode(interventions)
                )

            # Extract design and outcomes information
            design = protocol_section.get("designModule", {})
            if isinstance(design, dict):
                mcode_elements.update(self._extract_trial_design_mcode(design))

            # Extract temporal information
            status = protocol_section.get("statusModule", {})
            if isinstance(status, dict):
                mcode_elements.update(self._extract_trial_temporal_mcode(status))

            # Extract sponsor and organization information
            sponsor = protocol_section.get("sponsorCollaboratorsModule", {})
            if isinstance(sponsor, dict):
                sponsor_elements = self._extract_trial_sponsor_mcode(sponsor)
                mcode_elements.update(sponsor_elements)
                self.logger.debug(
                    f"Extracted sponsor elements: {list(sponsor_elements.keys())}"
                )

        except Exception as e:
            self.logger.error(
                f"Error extracting comprehensive trial mCODE elements: {e}"
            )
            self.logger.debug(f"Trial data type: {type(trial)}")
            if isinstance(trial, dict):
                self.logger.debug(f"Trial keys: {list(trial.keys())}")

        self.logger.debug(
            f"Final mCODE elements extracted: {len(mcode_elements)} total elements"
        )
        return mcode_elements

    def _extract_trial_identification(
        self, identification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract trial identification information."""
        elements = {}

        nct_id = identification.get("nctId")
        if nct_id:
            elements["TrialIdentifier"] = {
                "system": "https://clinicaltrials.gov",
                "code": nct_id,
                "display": f"Clinical Trial {nct_id}",
            }

        brief_title = identification.get("briefTitle")
        if brief_title:
            elements["TrialTitle"] = {
                "display": brief_title,
            }

        official_title = identification.get("officialTitle")
        if official_title:
            elements["TrialOfficialTitle"] = {
                "display": official_title,
            }

        return elements

    def _extract_trial_eligibility_mcode(
        self, eligibility: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract eligibility criteria in mCODE space for patient matching."""
        elements = {}

        # Type guard: ensure eligibility is a dict
        if not isinstance(eligibility, dict):
            self.logger.warning(f"Expected dict for eligibility, got {type(eligibility)}")
            return elements

        # Age criteria mapping to mCODE
        min_age = eligibility.get("minimumAge")
        max_age = eligibility.get("maximumAge")
        if min_age or max_age:
            elements["TrialAgeCriteria"] = {
                "minimumAge": min_age,
                "maximumAge": max_age,
                "ageUnit": "Years",  # Standardize to years
            }

        # Sex criteria mapping to mCODE AdministrativeGender
        sex = eligibility.get("sex")
        if sex:
            elements["TrialSexCriteria"] = {
                "system": "http://hl7.org/fhir/administrative-gender",
                "code": sex.lower(),
                "display": sex.capitalize(),
            }

        # Healthy volunteers criteria
        healthy_volunteers = eligibility.get("healthyVolunteers")
        if healthy_volunteers is not None:
            elements["TrialHealthyVolunteers"] = {
                "allowed": healthy_volunteers,
                "display": (
                    "Accepts healthy volunteers"
                    if healthy_volunteers
                    else "Does not accept healthy volunteers"
                ),
            }

        # Eligibility criteria text for detailed matching
        criteria_text = eligibility.get("eligibilityCriteria")
        if criteria_text:
            elements["TrialEligibilityCriteria"] = {
                "text": criteria_text,
                "display": "Detailed eligibility criteria for patient matching",
            }

        return elements

    def _extract_trial_conditions_mcode(
        self, conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract trial conditions as mCODE CancerCondition for matching."""
        elements = {}

        # Type guard: ensure conditions is a dict
        if not isinstance(conditions, dict):
            self.logger.warning(f"Expected dict for conditions, got {type(conditions)}")
            return elements

        condition_list = conditions.get("conditions", [])
        if condition_list:
            cancer_conditions = []
            comorbid_conditions = []

            for condition in condition_list:
                # Handle both dict and string formats for conditions
                if isinstance(condition, dict):
                    condition_name = condition.get("name", "").lower()
                    condition_code = condition.get("code", "Unknown")
                elif isinstance(condition, str):
                    # Handle case where condition is just a string
                    condition_name = condition.lower()
                    condition_code = "Unknown"
                    self.logger.debug(f"Condition is string format: {condition}")
                else:
                    self.logger.warning(f"Unexpected condition type {type(condition)}: {condition}")
                    continue

                # Check if it's a cancer condition
                if any(
                    cancer_term in condition_name
                    for cancer_term in [
                        "cancer",
                        "carcinoma",
                        "neoplasm",
                        "tumor",
                        "malignant",
                        "leukemia",
                        "lymphoma",
                        "sarcoma",
                        "glioma",
                        "melanoma",
                        "breast cancer",
                    ]
                ):
                    cancer_conditions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": condition_code,
                            "display": condition_name if condition_name else "Unknown cancer condition",
                            "interpretation": "Confirmed",
                        }
                    )
                else:
                    comorbid_conditions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": condition_code,
                            "display": condition_name if condition_name else "Unknown condition",
                        }
                    )

            if cancer_conditions:
                elements["TrialCancerConditions"] = cancer_conditions
            if comorbid_conditions:
                elements["TrialComorbidConditions"] = comorbid_conditions

        return elements

    def _extract_trial_interventions_mcode(
        self, interventions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract trial interventions as mCODE CancerRelatedMedicationStatement."""
        elements = {}

        # Type guard: ensure interventions is a dict
        if not isinstance(interventions, dict):
            self.logger.warning(f"Expected dict for interventions, got {type(interventions)}")
            return elements

        intervention_list = interventions.get("interventions", [])
        if intervention_list:
            medication_interventions = []
            other_interventions = []

            for intervention in intervention_list:
                # Type guard: ensure intervention is a dict
                if not isinstance(intervention, dict):
                    self.logger.warning(f"Expected dict for intervention, got {type(intervention)}: {intervention}")
                    continue

                intervention_type = intervention.get("type", "").lower()
                intervention_name = intervention.get("name", "")
                description = intervention.get("description", "")

                if intervention_type in ["drug", "biological", "device"]:
                    medication_interventions.append(
                        {
                            "system": "http://snomed.info/sct",
                            "code": "Unknown",  # Would need RxNorm mapping
                            "display": intervention_name,
                            "description": description,
                            "interventionType": intervention_type,
                        }
                    )
                else:
                    other_interventions.append(
                        {
                            "display": intervention_name,
                            "description": description,
                            "interventionType": intervention_type,
                        }
                    )

            if medication_interventions:
                elements["TrialMedicationInterventions"] = medication_interventions
            if other_interventions:
                elements["TrialOtherInterventions"] = other_interventions

        return elements

    def _extract_trial_design_mcode(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial design elements for mCODE mapping."""
        elements = {}

        # Type guard: ensure design is a dict
        if not isinstance(design, dict):
            self.logger.warning(f"Expected dict for design, got {type(design)}")
            return elements

        # Study type
        study_type = design.get("studyType")
        if study_type:
            elements["TrialStudyType"] = {
                "display": study_type,
                "code": study_type.lower().replace(" ", "_"),
            }

        # Phase information
        phase = design.get("phases", [])
        if phase:
            elements["TrialPhase"] = {
                "display": ", ".join(phase),
                "phases": phase,
            }

        # Primary purpose
        primary_purpose = design.get("primaryPurpose")
        if primary_purpose:
            elements["TrialPrimaryPurpose"] = {
                "display": primary_purpose,
                "code": primary_purpose.lower().replace(" ", "_"),
            }

        # Enrollment information
        enrollment_info = design.get("enrollmentInfo", {})
        if enrollment_info:
            elements["TrialEnrollment"] = {
                "count": enrollment_info.get("count"),
                "type": enrollment_info.get("type"),
            }

        return elements

    def _extract_trial_temporal_mcode(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal information for trial phases and timelines."""
        elements = {}

        # Type guard: ensure status is a dict
        if not isinstance(status, dict):
            self.logger.warning(f"Expected dict for status, got {type(status)}")
            return elements

        # Overall status
        overall_status = status.get("overallStatus")
        if overall_status:
            elements["TrialStatus"] = {
                "display": overall_status,
                "code": overall_status.lower().replace(" ", "_"),
            }

        # Start date
        start_date = status.get("startDateStruct", {}).get("date")
        if start_date:
            elements["TrialStartDate"] = {
                "date": start_date,
                "display": f"Trial started on {start_date}",
            }

        # Completion date
        completion_date = status.get("completionDateStruct", {}).get("date")
        if completion_date:
            elements["TrialCompletionDate"] = {
                "date": completion_date,
                "display": f"Trial completed on {completion_date}",
            }

        # Primary completion date
        primary_completion_date = status.get("primaryCompletionDateStruct", {}).get(
            "date"
        )
        if primary_completion_date:
            elements["TrialPrimaryCompletionDate"] = {
                "date": primary_completion_date,
                "display": f"Primary completion on {primary_completion_date}",
            }

        return elements

    def _extract_trial_sponsor_mcode(self, sponsor: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sponsor and organization information."""
        elements = {}

        # Type guard: ensure sponsor is a dict
        if not isinstance(sponsor, dict):
            self.logger.warning(f"Expected dict for sponsor, got {type(sponsor)}")
            return elements

        # Lead sponsor
        lead_sponsor = sponsor.get("leadSponsor", {})
        if lead_sponsor:
            elements["TrialLeadSponsor"] = {
                "name": lead_sponsor.get("name"),
                "class": lead_sponsor.get("class"),
            }

        # Responsible party
        responsible_party = sponsor.get("responsibleParty", {})
        if responsible_party:
            elements["TrialResponsibleParty"] = {
                "name": responsible_party.get("name"),
                "type": responsible_party.get("type"),
                "affiliation": responsible_party.get("affiliation"),
            }

        return elements

    def _extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive trial metadata for storage."""
        metadata = {}

        try:
            # Ensure trial is a dict
            if not isinstance(trial, dict):
                self.logger.error(
                    f"Trial data is not a dict in metadata extraction: {type(trial)}"
                )
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
        """Generate comprehensive natural language summary for clinical trial using McodeSummarizer."""
        try:
            # Use the McodeSummarizer service
            summarizer = McodeSummarizer(include_dates=True)
            summary = summarizer.create_trial_summary(trial_data)
            self.logger.info(
                f"Generated comprehensive trial summary for {trial_id}: {summary[:200]}..."
            )
            self.logger.debug(f"Full trial summary length: {len(summary)} characters")
            return summary

        except Exception as e:
            self.logger.error(f"Error generating trial natural language summary: {e}")
            self.logger.debug(f"Trial data for error: {trial_data}")
            return f"Clinical Trial {trial_id}: Error generating comprehensive summary - {str(e)}"

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
            self.logger.error(
                f"Error converting trial mCODE elements to mappings format: {e}"
            )
            self.logger.debug(f"mcode_elements: {mcode_elements}")
            raise

        return mappings

    def _format_trial_mcode_element(
        self, element_name: str, system: str, code: str
    ) -> str:
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

    def _process_single_trial_sync(
        self,
        trial: Dict[str, Any],
        model: str,
        prompt: str,
        index: int,
    ) -> tuple:
        """
        Process a single trial synchronously for concurrent processing.

        Args:
            trial: Trial data to process
            model: LLM model to use
            prompt: Prompt template to use
            index: Trial index for logging

        Returns:
            tuple: (trial_result, success)
        """
        try:
            self.logger.info(f"🔬 Processing trial {index}")

            # Check for cached processed trial result
            cached_result = self._get_cached_trial_result(trial, model, prompt)
            if cached_result is not None:
                self.logger.info(
                    f"✅ Cache HIT for trial {self._extract_trial_id(trial)}"
                )
                return cached_result, True

            # Try to process with mCODE pipeline
            enhanced_trial = None
            mcode_success = False
            try:
                result = self.pipeline.process_clinical_trial(trial)
                # Add mCODE results to trial using standardized utility
                enhanced_trial = enhance_trial_with_mcode_results(trial, result)
                mcode_success = True
            except Exception as mcode_error:
                self.logger.warning(
                    f"mCODE pipeline failed for trial {self._extract_trial_id(trial)}: {mcode_error}"
                )
                # Create basic enhanced trial without mCODE results
                enhanced_trial = trial.copy()
                enhanced_trial["McodeProcessingError"] = str(mcode_error)

            # Cache the processed trial result
            self._cache_trial_result(enhanced_trial, model, prompt)

            # Always generate natural language summary, regardless of mCODE success
            trial_id = self._extract_trial_id(trial)
            try:
                # Extract comprehensive mCODE elements from trial data (with caching)
                comprehensive_mcode = self._extract_trial_mcode_elements_cached(trial)

                # Create natural language summary for CORE knowledge graph (with caching)
                natural_language_summary = (
                    self._generate_trial_natural_language_summary_cached(
                        trial_id, comprehensive_mcode, trial
                    )
                )
                self.logger.info(
                    f"Generated comprehensive trial summary for {trial_id}: {natural_language_summary[:100]}..."
                )

                # Add natural language summary to the enhanced trial data
                if "McodeResults" not in enhanced_trial:
                    enhanced_trial["McodeResults"] = {}
                enhanced_trial["McodeResults"][
                    "natural_language_summary"
                ] = natural_language_summary
                self.logger.debug(
                    f"Added natural language summary to enhanced_trial McodeResults: {len(natural_language_summary)} chars"
                )

                # Store to core memory if requested
                if store_in_memory and self.memory_storage:
                    mcode_data = {
                        "mcode_mappings": self._convert_trial_mcode_to_mappings_format(
                            comprehensive_mcode
                        ),
                        "natural_language_summary": natural_language_summary,
                        "comprehensive_mcode_elements": comprehensive_mcode,
                        "trial_metadata": self._extract_trial_metadata(trial),
                        "pipeline_results": enhanced_trial.get("McodeResults", {}),
                    }

                    success = self.memory_storage.store_trial_mcode_summary(
                        trial_id, mcode_data
                    )
                    if success:
                        self.logger.info(f"✅ Stored trial {trial_id} mCODE summary")
                    else:
                        self.logger.warning(
                            f"❌ Failed to store trial {trial_id} mCODE summary"
                        )
            except Exception as summary_error:
                self.logger.error(
                    f"Failed to generate summary for trial {trial_id}: {summary_error}"
                )

            return enhanced_trial, mcode_success

        except Exception as e:
            self.logger.error(f"Failed to process trial {index}: {e}")
            # Add error information to trial
            error_trial = trial.copy()
            error_trial["ProcessingError"] = str(e)
            return error_trial, False

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

    def clear_workflow_caches(self) -> None:
        """Clear all workflow-related caches."""
        self.workflow_cache.clear_cache()
        self.mcode_cache.clear_cache()
        self.summary_cache.clear_cache()
        self.logger.info("Cleared all workflow caches")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics for all workflow caches."""
        return {
            "workflow_cache": self.workflow_cache.get_stats(),
            "mcode_cache": self.mcode_cache.get_stats(),
            "summary_cache": self.summary_cache.get_stats(),
        }

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
            "cache_stats": self.get_cache_stats(),
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
                    isinstance(i, dict) and i.get("description", "")
                    for i in interventions
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
