"""
Trials Summarizer Workflow - Generate natural language summaries from mCODE trial data.

This workflow handles generating comprehensive natural language summaries
from processed mCODE trial data and stores them in CORE Memory.
"""

from typing import Any, Dict, Optional, cast

from src.services.summarizer import McodeSummarizer
from src.storage.mcode_memory_storage import OncoCoreMemory

from .base_summarizer import BaseSummarizerWorkflow
from .base_workflow import WorkflowResult


class TrialsSummarizerWorkflow(BaseSummarizerWorkflow):
    """
    Workflow for generating natural language summaries from mCODE trial data.

    Takes processed mCODE trial data and generates comprehensive summaries
    for storage in CORE Memory.
    """

    @property
    def memory_space(self) -> str:
        """Trials summarizers use 'trials_summaries' space."""
        return "trials_summaries"

    def __init__(self, config: Any, memory_storage: Optional[OncoCoreMemory] = None):
        """
        Initialize the trials summarizer workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config, memory_storage)
        self.summarizer = McodeSummarizer()

    def execute(self, **kwargs: Any) -> WorkflowResult:
        """
        Execute the trials summarization workflow.

        Args:
            **kwargs: Workflow parameters including:
                - trials_data: List of trial data to summarize
                - store_in_memory: Whether to store results in CORE memory (default: False)

        Returns:
            WorkflowResult: Summarization results
        """
        try:
            self.logger.info("Starting trials summarizer workflow execution")

            # Extract parameters
            trials_data = kwargs.get("trials_data", [])
            store_in_memory = kwargs.get("store_in_memory", False)

            if not trials_data:
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for summarization.",
                )

            # Generate summaries
            processed_trials = []
            successful_count = 0
            failed_count = 0

            self.logger.info(f"📝 Generating summaries for {len(trials_data)} trials")

            for trial in trials_data:
                try:
                    trial_id = self._extract_trial_id(trial)

                    # Generate natural language summary using McodeSummarizer with processed mCODE elements
                    mcode_elements = trial.get("McodeResults", {}).get("mcode_mappings", [])
                    self.logger.info(
                        f"Found {len(mcode_elements)} mCODE elements for summarization"
                    )
                    if mcode_elements:
                        # Convert processed mCODE elements to summarizer format
                        formatted_elements = []
                        for element in mcode_elements:
                            # element is an McodeElement object, access attributes directly
                            element_type = element.element_type
                            # Only strip "Trial" prefix if it's followed by a capital letter (e.g., "TrialTitle" -> "Title")
                            if (
                                element_type.startswith("Trial")
                                and len(element_type) > 5
                                and element_type[5].isupper()
                            ):
                                element_name = element_type[5:]  # Remove "Trial" prefix
                            else:
                                element_name = element_type

                            codes = (
                                f"{element.system}:{element.code}"
                                if element.system and element.code
                                else ""
                            )
                            formatted_element = {
                                "element_name": element_name,
                                "value": element.display or "",
                                "codes": codes,
                                "date_qualifier": "",
                            }
                            formatted_elements.append(formatted_element)
                            self.logger.debug(
                                f"Converted element: {element_type} -> {element_name} = '{element.display}'"
                            )

                        self.logger.info(
                            f"Converted {len(formatted_elements)} elements for summarization"
                        )

                        # Use processed elements to generate summary
                        trial_id = self._extract_trial_id(trial)
                        prioritized = self.summarizer._group_elements_by_priority(
                            formatted_elements, "Trial"
                        )
                        self.logger.info(f"Prioritized {len(prioritized)} elements")
                        sentences = self.summarizer._generate_sentences_from_elements(
                            prioritized, trial_id
                        )
                        self.logger.info(f"Generated {len(sentences)} sentences")
                        summary = " ".join(sentences)
                        self.logger.info(f"Final summary: {summary[:200]}...")
                    else:
                        # Fallback to basic extraction if no processed elements
                        self.logger.warning(
                            "No processed mCODE elements found, using basic extraction"
                        )
                        summary = self.summarizer.create_trial_summary(trial)
                    self.logger.debug(f"Generated summary for trial {trial_id}: {summary[:100]}...")

                    # Create processed trial with summary
                    processed_trial = trial.copy()
                    if "McodeResults" not in processed_trial:
                        processed_trial["McodeResults"] = {}
                    processed_trial["McodeResults"]["natural_language_summary"] = summary

                    # Store in CORE Memory if requested
                    if store_in_memory and self.memory_storage:
                        # Extract mCODE elements if available
                        mcode_elements = trial.get("McodeResults", {}).get("mcode_mappings", [])
                        trial_metadata = self._extract_trial_metadata(trial)

                        mcode_data = {
                            "mcode_mappings": mcode_elements,
                            "natural_language_summary": summary,
                            "trial_metadata": trial_metadata,
                            "pipeline_results": trial.get("McodeResults", {}),
                        }

                        success = self.memory_storage.store_trial_mcode_summary(
                            trial_id, mcode_data
                        )
                        if success:
                            self.logger.info(f"✅ Stored trial {trial_id} summary")
                        else:
                            self.logger.warning(f"❌ Failed to store trial {trial_id} summary")

                    processed_trials.append(processed_trial)
                    successful_count += 1

                except Exception as e:
                    self.logger.error(f"Failed to summarize trial: {e}")
                    # Add error information to trial
                    error_trial = trial.copy()
                    error_trial["SummaryError"] = str(e)
                    processed_trials.append(error_trial)
                    failed_count += 1

            # Calculate success rate
            total_count = len(trials_data)
            success_rate = successful_count / total_count if total_count > 0 else 0

            self.logger.info(
                f"📊 Summarization complete: {successful_count}/{total_count} trials successful"
            )

            return self._create_result(
                success=successful_count > 0,
                data=processed_trials,
                metadata={
                    "total_trials": total_count,
                    "successful": successful_count,
                    "failed": failed_count,
                    "success_rate": success_rate,
                    "stored_in_memory": store_in_memory and self.memory_storage is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials summarization")

    def process_single_trial(self, trial: Dict[str, Any], **kwargs: Any) -> WorkflowResult:
        """
        Process a single trial for summarization.

        Args:
            trial: Trial data to summarize
            **kwargs: Additional processing parameters

        Returns:
            WorkflowResult: Summarization result
        """
        result = self.execute(trials_data=[trial], **kwargs)

        # Return single trial result
        if (
            result.success
            and result.data
            and isinstance(result.data, list)
            and len(result.data) > 0
        ):
            return self._create_result(success=True, data=result.data[0], metadata=result.metadata)
        else:
            return result

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        """Extract trial ID from trial data."""
        # Try different possible locations for trial ID
        if "protocolSection" in trial:
            protocol = trial["protocolSection"]
            if "identificationModule" in protocol:
                return cast(str, protocol["identificationModule"].get("nctId", "unknown"))

        # Check for direct trial_id field
        if "trial_id" in trial:
            return cast(str, trial["trial_id"])

        # Check for nctId field
        if "nctId" in trial:
            return cast(str, trial["nctId"])

        return "unknown"

    def _extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial metadata for CORE Memory storage."""
        metadata = {}

        # Extract basic trial information
        if "protocolSection" in trial:
            protocol = trial["protocolSection"]

            # Identification
            if "identificationModule" in protocol:
                ident = protocol["identificationModule"]
                metadata["nct_id"] = ident.get("nctId")
                metadata["brief_title"] = ident.get("briefTitle")
                metadata["official_title"] = ident.get("officialTitle")

            # Status
            if "statusModule" in protocol:
                status = protocol["statusModule"]
                metadata["overall_status"] = status.get("overallStatus")
                metadata["start_date"] = status.get("startDateStruct", {}).get("date")
                metadata["completion_date"] = status.get("completionDateStruct", {}).get("date")

            # Conditions
            if "conditionsModule" in protocol:
                conditions = protocol["conditionsModule"]
                metadata["conditions"] = conditions.get("conditions", [])

            # Eligibility
            if "eligibilityModule" in protocol:
                eligibility = protocol["eligibilityModule"]
                metadata["minimum_age"] = eligibility.get("minimumAge")
                metadata["maximum_age"] = eligibility.get("maximumAge")
                metadata["gender"] = eligibility.get("sex")
                metadata["healthy_volunteers"] = eligibility.get("healthyVolunteers")

        # Add processing timestamp
        from datetime import datetime, timezone

        metadata["processed_at"] = datetime.now(timezone.utc).isoformat()

        return metadata
