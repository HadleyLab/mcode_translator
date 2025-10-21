import asyncio
from typing import Any, Dict, List, Optional, Union

from src.pipeline import McodePipeline
from src.services.data_enrichment import DataEnrichmentService
from src.services.llm.service import LLMService
from src.shared.models import ClinicalTrialData, IdentificationModule, PipelineResult
from src.storage.mcode_memory_storage import OncoCoreMemory
from src.utils.concurrency import AsyncTaskQueue, create_task
from src.shared.data_quality_validator import DataQualityValidator

from .base_workflow import TrialsProcessorWorkflow as BaseTrialsProcessorWorkflow
from .base_workflow import WorkflowResult
from .trial_extractor import TrialExtractor
from .trial_summarizer import TrialSummarizer


class TrialsProcessor(BaseTrialsProcessorWorkflow):
    def __init__(self, config: Any, memory_storage: Optional[OncoCoreMemory] = None):
        super().__init__(config, memory_storage)
        self.pipeline: Optional[Any] = None
        self.extractor = TrialExtractor()
        self.summarizer = TrialSummarizer()
        self.quality_validator = DataQualityValidator()  # Add data quality validation

        # Initialize data enrichment service
        llm_service = LLMService(config, "deepseek-coder", "direct_mcode_evidence_based_concise")
        self.data_enrichment = DataEnrichmentService(config, llm_service)

        # Initialize logger
        from src.utils.logging_config import get_logger
        self.logger = get_logger(__name__)

    async def execute_async(self, **kwargs: Any) -> WorkflowResult:
        if "trials_data" not in kwargs:
            raise ValueError("trials_data is required")
        if "engine" not in kwargs:
            raise ValueError("engine is required")
        if "model" not in kwargs:
            raise ValueError("model is required")
        if "prompt" not in kwargs:
            raise ValueError("prompt is required")
        if "workers" not in kwargs:
            raise ValueError("workers is required")
        if "store_in_memory" not in kwargs:
            raise ValueError("store_in_memory is required")

        trials_data = kwargs["trials_data"]
        if not isinstance(trials_data, list):
            raise TypeError("trials_data must be a list")
        if not trials_data:
            raise ValueError("trials_data cannot be empty")

        # Validate that each trial is a dict
        for i, trial in enumerate(trials_data):
            if not isinstance(trial, dict):
                raise TypeError(f"trials_data[{i}] must be a dict, got {type(trial)}")

        engine = kwargs["engine"]
        model = kwargs["model"]
        prompt = kwargs["prompt"]
        workers = kwargs["workers"]
        store_in_memory = kwargs["store_in_memory"]

        if not self.pipeline:
            self.pipeline = McodePipeline(prompt_name=prompt, model_name=model, engine=engine)

        effective_workers = max(1, workers)
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

        task_queue = AsyncTaskQueue(max_concurrent=effective_workers, name="TrialsProcessorQueue")

        def progress_callback(completed: int, total: int, result: Any) -> None:
            pass

        task_results = await task_queue.execute_tasks(tasks, progress_callback=progress_callback)

        processed_trials = []
        successful_count = 0
        failed_count = 0
        quality_reports = []

        for task_result in task_results:
            if task_result.success and task_result.result:
                trial_result, success = task_result.result

                # Extract trial ID for validation
                trial_id = self._extract_trial_id(trial_result)

                # Validate data quality
                mcode_elements = trial_result.get("McodeResults", {}).get("mcode_mappings", [])
                # Convert mcode_elements list to dict for validation
                mcode_elements_dict = {}
                if isinstance(mcode_elements, list):
                    for elem in mcode_elements:
                        if isinstance(elem, dict) and "mcode_element" in elem:
                            mcode_elements_dict[elem["mcode_element"]] = elem
                else:
                    mcode_elements_dict = mcode_elements if isinstance(mcode_elements, dict) else {}

                # Convert enhanced trial to ClinicalTrialData for validation
                from src.shared.models import ClinicalTrialData
                clinical_trial_data = ClinicalTrialData(**trial_result)
                quality_report = self.quality_validator.validate_trial_data(
                    clinical_trial_data, mcode_elements
                )
                quality_reports.append({
                    "trial_id": trial_id,
                    "report": quality_report
                })

                # Check if processing can proceed based on quality validation
                if not quality_report.can_proceed:
                    self.logger.warning(
                        f"❌ Trial {trial_id} failed quality validation. "
                        f"Critical issues: {quality_report.critical_issues}"
                    )
                    failed_count += 1
                    continue

                processed_trials.append(trial_result)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1

                # Log quality metrics
                self.logger.info(
                    f"✅ Trial {trial_id} quality validated: "
                    f"{quality_report.coverage_percentage:.1f}% coverage, "
                    f"score: {quality_report.completeness_score:.2f}"
                )
            else:
                failed_count += 1
                error_trial = {"McodeProcessingError": str(task_result.error)}
                processed_trials.append(error_trial)

        total_count = len(trials_data)
        success_rate = successful_count / total_count if total_count > 0 else 0

        # Generate aggregate quality report
        if quality_reports:
            total_critical = sum(r["report"].critical_issues for r in quality_reports)
            total_warnings = sum(r["report"].warning_issues for r in quality_reports)
            avg_coverage = sum(r["report"].coverage_percentage for r in quality_reports) / len(quality_reports)
            avg_completeness = sum(r["report"].completeness_score for r in quality_reports) / len(quality_reports)

            self.logger.info(
                f"📊 Quality Summary: {successful_count}/{total_count} trials passed validation. "
                f"Avg coverage: {avg_coverage:.1f}%, Avg completeness: {avg_completeness:.2f}. "
                f"Total issues: {total_critical} critical, {total_warnings} warnings"
            )

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
                "quality_reports": quality_reports,
                "quality_summary": {
                    "total_critical_issues": sum(r["report"].critical_issues for r in quality_reports),
                    "total_warning_issues": sum(r["report"].warning_issues for r in quality_reports),
                    "average_coverage": sum(r["report"].coverage_percentage for r in quality_reports) / len(quality_reports) if quality_reports else 0,
                    "average_completeness": sum(r["report"].completeness_score for r in quality_reports) / len(quality_reports) if quality_reports else 0,
                } if quality_reports else None,
            },
        )

    def execute(self, **kwargs: Any) -> WorkflowResult:
        return asyncio.run(self.execute_async(**kwargs))

    def _extract_trial_mcode_elements_cached(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> Dict[str, Any]:
        """Extract mCODE elements handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            # Convert ClinicalTrialData to dictionary format for the extractor
            trial_dict = {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": trial.nct_id,
                        "briefTitle": trial.brief_title,
                        "officialTitle": trial.official_title,
                    }
                }
            }

            # Add eligibility module if available
            if trial.protocol_section.eligibility_module:
                trial_dict["protocolSection"]["eligibilityModule"] = {
                    "minimumAge": trial.protocol_section.eligibility_module.minimum_age,
                    "maximumAge": trial.protocol_section.eligibility_module.maximum_age,
                    "sex": trial.protocol_section.eligibility_module.sex,
                    "healthyVolunteers": trial.protocol_section.eligibility_module.healthy_volunteers,
                    "eligibilityCriteria": trial.protocol_section.eligibility_module.eligibility_criteria,
                }

            # Add conditions module if available
            if trial.protocol_section.conditions_module and trial.protocol_section.conditions_module.conditions:
                trial_dict["protocolSection"]["conditionsModule"] = {
                    "conditions": [{"name": c.name, "code": c.code, "codeSystem": c.code_system}
                                 for c in trial.protocol_section.conditions_module.conditions]
                }

            # Add arms interventions module if available
            if trial.protocol_section.arms_interventions_module:
                arms_dict = {}
                if trial.protocol_section.arms_interventions_module.arm_groups:
                    arms_dict["armGroups"] = [{"label": ag.label, "type": ag.type, "description": ag.description,
                                             "interventionNames": ag.intervention_names}
                                            for ag in trial.protocol_section.arms_interventions_module.arm_groups]
                if trial.protocol_section.arms_interventions_module.interventions:
                    arms_dict["interventions"] = [{"type": i.type, "name": i.name, "description": i.description,
                                                 "armGroupLabels": i.arm_group_labels, "otherNames": i.other_names}
                                                for i in trial.protocol_section.arms_interventions_module.interventions]
                if arms_dict:
                    trial_dict["protocolSection"]["armsInterventionsModule"] = arms_dict

            # Add status module if available
            if trial.protocol_section.status_module:
                trial_dict["protocolSection"]["statusModule"] = trial.protocol_section.status_module

            # Add design module if available
            if trial.protocol_section.design_module:
                trial_dict["protocolSection"]["designModule"] = trial.protocol_section.design_module

            # Add outcomes module if available
            if trial.protocol_section.outcomes_module:
                outcomes_dict = {}
                if trial.protocol_section.outcomes_module.primary_outcomes:
                    outcomes_dict["primaryOutcomes"] = [{"measure": o.measure, "description": o.description,
                                                       "timeFrame": o.time_frame}
                                                      for o in trial.protocol_section.outcomes_module.primary_outcomes]
                if trial.protocol_section.outcomes_module.secondary_outcomes:
                    outcomes_dict["secondaryOutcomes"] = [{"measure": o.measure, "description": o.description,
                                                         "timeFrame": o.time_frame}
                                                        for o in trial.protocol_section.outcomes_module.secondary_outcomes]
                if trial.protocol_section.outcomes_module.other_outcomes:
                    outcomes_dict["otherOutcomes"] = [{"measure": o.measure, "description": o.description,
                                                     "timeFrame": o.time_frame}
                                                    for o in trial.protocol_section.outcomes_module.other_outcomes]
                if outcomes_dict:
                    trial_dict["protocolSection"]["outcomesModule"] = outcomes_dict

            return self.extractor.extract_trial_mcode_elements(trial_dict)

        # Handle old dictionary format
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict or ClinicalTrialData instance")

        return self.extractor.extract_trial_mcode_elements(trial)

    def _extract_trial_id(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> str:
        """Extract trial ID handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.nct_id

        # Handle old dictionary format
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict or ClinicalTrialData instance")

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if not protocol_section:
            raise ValueError("trial must have protocolSection")

        # Try new identification_module format first
        identification = protocol_section.get("identificationModule") or protocol_section.get("identification_module")
        if not identification:
            raise ValueError("trial must have identificationModule")

        # Try new nct_id format first
        nct_id = identification.get("nctId") or identification.get("nct_id")
        if not nct_id:
            raise ValueError("trial must have nctId")

        return str(nct_id)

    def _extract_trial_metadata(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> Dict[str, Any]:
        """Extract trial metadata handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            metadata = {}

            # Extract from identification module
            identification = trial.protocol_section.identification_module
            metadata["nct_id"] = identification.nct_id
            metadata["brief_title"] = identification.brief_title
            metadata["official_title"] = identification.official_title

            # Extract from status module if available
            if trial.protocol_section.status_module:
                status = trial.protocol_section.status_module
                metadata["overall_status"] = status.get("overallStatus")
                if "startDateStruct" in status:
                    metadata["start_date"] = status["startDateStruct"].get("date")
                if "completionDateStruct" in status:
                    metadata["completion_date"] = status["completionDateStruct"].get("date")

            # Extract from design module if available
            if trial.protocol_section.design_module:
                design = trial.protocol_section.design_module
                metadata["study_type"] = design.get("studyType")
                metadata["phase"] = design.get("phases")
                metadata["primary_purpose"] = design.get("primaryPurpose")

            # Extract from eligibility module if available
            if trial.protocol_section.eligibility_module:
                eligibility = trial.protocol_section.eligibility_module
                metadata["minimum_age"] = eligibility.minimum_age
                metadata["maximum_age"] = eligibility.maximum_age
                metadata["sex"] = eligibility.sex
                metadata["healthy_volunteers"] = eligibility.healthy_volunteers

            # Extract conditions
            if trial.conditions:
                metadata["conditions"] = [c.name for c in trial.conditions if c.name]

            # Extract interventions
            if trial.interventions:
                metadata["interventions"] = [i.name for i in trial.interventions if i.name]

            return metadata

        # Handle old dictionary format
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict or ClinicalTrialData instance")

        # Extract metadata directly from dictionary format
        metadata = {}

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if not protocol_section:
            return metadata

        # Try new identification_module format first
        identification = protocol_section.get("identificationModule") or protocol_section.get("identification_module")
        if identification:
            metadata["nct_id"] = identification.get("nctId") or identification.get("nct_id")
            metadata["brief_title"] = identification.get("briefTitle") or identification.get("brief_title")
            metadata["official_title"] = identification.get("officialTitle") or identification.get("official_title")

        # Try new status_module format first
        status_module = protocol_section.get("statusModule") or protocol_section.get("status_module")
        if status_module:
            metadata["overall_status"] = status_module.get("overallStatus")

        # Try new design_module format first
        design_module = protocol_section.get("designModule") or protocol_section.get("design_module")
        if design_module:
            metadata["study_type"] = design_module.get("studyType")
            metadata["phase"] = design_module.get("phases")

        # Try new eligibility_module format first
        eligibility_module = protocol_section.get("eligibilityModule") or protocol_section.get("eligibility_module")
        if eligibility_module:
            metadata["minimum_age"] = eligibility_module.get("minimumAge") or eligibility_module.get("minimum_age")
            metadata["maximum_age"] = eligibility_module.get("maximumAge") or eligibility_module.get("maximum_age")
            metadata["sex"] = eligibility_module.get("sex")
            metadata["healthy_volunteers"] = eligibility_module.get("healthyVolunteers") or eligibility_module.get("healthy_volunteers")

        return metadata

    def _generate_trial_natural_language_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
        return self.summarizer.generate_trial_natural_language_summary(
            trial_id, mcode_elements, trial_data
        )

    def _generate_trial_regex_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Union[Dict[str, Any], ClinicalTrialData]
    ) -> str:
        """Generate regex summary handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial_data, ClinicalTrialData):
            title = trial_data.brief_title or trial_id

            conditions = []
            if "TrialCancerConditions" in mcode_elements:
                cancer_conditions = mcode_elements["TrialCancerConditions"]
                if isinstance(cancer_conditions, list):
                    conditions = [c.get("display", "") for c in cancer_conditions if c.get("display")]
                else:
                    conditions = [cancer_conditions.get("display", "")] if cancer_conditions.get("display") else []

            summary_parts = [f"Clinical Trial {trial_id}: {title}"]

            if conditions:
                summary_parts.append(f"Conditions: {', '.join(conditions[:3])}")

            # Add enrollment info if available
            if trial_data.protocol_section.design_module and "enrollmentInfo" in trial_data.protocol_section.design_module:
                enrollment = trial_data.protocol_section.design_module["enrollmentInfo"].get("count")
                if enrollment:
                    summary_parts.append(f"Enrollment: {enrollment} participants")

            # Add status if available
            if trial_data.protocol_section.status_module:
                status = trial_data.protocol_section.status_module.get("overallStatus")
                if status:
                    summary_parts.append(f"Status: {status}")

            return ". ".join(summary_parts) + "."

        # Handle old dictionary format
        if not isinstance(trial_data, dict):
            raise ValueError("trial_data must be a dict or ClinicalTrialData instance")

        protocol_section = trial_data["protocolSection"]
        identification = protocol_section["identificationModule"]
        title = identification["briefTitle"]

        conditions = []
        if "TrialCancerConditions" in mcode_elements:
            cancer_conditions = mcode_elements["TrialCancerConditions"]
            if isinstance(cancer_conditions, list):
                conditions = [c["display"] for c in cancer_conditions]
            else:
                conditions = [cancer_conditions["display"]]

        summary_parts = [f"Clinical Trial {trial_id}: {title}"]

        if conditions:
            summary_parts.append(f"Conditions: {', '.join(conditions[:3])}")

        design = protocol_section["designModule"]
        enrollment = design["enrollmentInfo"]["count"]
        summary_parts.append(f"Enrollment: {enrollment} participants")

        status = protocol_section["statusModule"]["overallStatus"]
        summary_parts.append(f"Status: {status}")

        return ". ".join(summary_parts) + "."

    def _convert_trial_mcode_to_mappings_format(
        self, mcode_elements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        mappings = []

        for element_name, element_data in mcode_elements.items():
            if isinstance(element_data, list):
                for item in element_data:
                    if isinstance(item, dict):
                        mapping = {
                            "mcode_element": element_name,
                            "value": item["display"],
                            "system": item["system"],
                            "code": item["code"],
                            "interpretation": item["interpretation"],
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
                if isinstance(element_data, dict):
                    mapping = {
                        "mcode_element": element_name,
                        "value": element_data["display"],
                        "system": element_data["system"],
                        "code": element_data["code"],
                        "interpretation": element_data["interpretation"],
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

        return mappings

    def _format_trial_mcode_element(self, element_name: str, system: str, code: str) -> str:
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
        # Validate trial data structure
        if not isinstance(trial, dict):
            raise ValueError(f"Trial data must be a dict, got {type(trial)}")

        if self.pipeline is not None:
            # Convert dictionary to ClinicalTrialData object for pipeline processing
            from src.shared.models import ClinicalTrialData
            clinical_trial_data = ClinicalTrialData(**trial)
            result = await self.pipeline.process(clinical_trial_data)
            # Create enhanced trial data directly from PipelineResult
            enhanced_trial = trial.copy()
            enhanced_trial["McodeResults"] = {
                "extracted_entities": result.extracted_entities,
                "mcode_mappings": result.mcode_mappings,
                "source_references": result.source_references,
                "validation_results": result.validation_results,
                "metadata": result.metadata,
                "token_usage": result.metadata.token_usage if result.metadata else None,
                "error": result.error,
            }
            mcode_success = True
        else:
            enhanced_trial = trial.copy()
            enhanced_trial["McodeProcessingError"] = "Pipeline not initialized"
            mcode_success = False

        trial_id = self._extract_trial_id(trial)
        mcode_elements = enhanced_trial.get("McodeResults", {}).get("mcode_mappings", [])

        # Apply data enrichment to improve completeness
        # Convert mcode_elements list to dict for enrichment service
        mcode_elements_dict = {}
        if isinstance(mcode_elements, list):
            for elem in mcode_elements:
                if isinstance(elem, dict) and "mcode_element" in elem:
                    mcode_elements_dict[elem["mcode_element"]] = elem
        else:
            mcode_elements_dict = mcode_elements if isinstance(mcode_elements, dict) else {}

        enriched_trial_data = await self.data_enrichment.enrich_trial_data(
            enhanced_trial, mcode_elements_dict
        )

        # Generate summary using enriched data
        summary = self.summarizer.generate_trial_natural_language_summary(
            trial_id, mcode_elements, enriched_trial_data
        )

        if "McodeResults" not in enriched_trial_data:
            enriched_trial_data["McodeResults"] = {}
        enriched_trial_data["McodeResults"]["natural_language_summary"] = summary

        return enriched_trial_data, mcode_success

    def process_single_trial(self, trial: Dict[str, Any], **kwargs: Any) -> WorkflowResult:
        result = self.execute(trials_data=[trial], **kwargs)

        if result.success and isinstance(result.data, list) and result.data:
            return self._create_result(
                success=True,
                data=result.data[0],
                metadata=result.metadata,
            )
        else:
            return result

    def validate_trial_data(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> bool:
        """Validate trial data handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            # Check if NCT ID exists
            if not trial.nct_id:
                return False

            # Check if eligibility criteria exists
            if not trial.protocol_section.eligibility_module:
                return False

            criteria = trial.protocol_section.eligibility_module.eligibility_criteria
            if not criteria:
                return False

            return True

        # Handle old dictionary format
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict or ClinicalTrialData instance")

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if not protocol_section:
            raise ValueError("trial must have protocolSection")

        # Try new identification_module format first
        identification = protocol_section.get("identificationModule") or protocol_section.get("identification_module")
        if not identification:
            raise ValueError("trial must have identificationModule")

        # Try new nct_id format first
        nct_id = identification.get("nctId") or identification.get("nct_id")
        if not nct_id:
            return False

        # Try new eligibility_module format first
        eligibility = protocol_section.get("eligibilityModule") or protocol_section.get("eligibility_module")
        if not eligibility:
            raise ValueError("trial must have eligibilityModule")

        criteria = eligibility.get("eligibilityCriteria")
        if not criteria:
            return False

        return True

    # Removed cache management methods - only API calls should be cached

    def get_processing_stats(self) -> Dict[str, Any]:
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

    def _check_trial_has_full_data(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> bool:
        """Check if trial has full data handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            indicators = []

            # Check eligibility criteria length
            if trial.protocol_section.eligibility_module and trial.protocol_section.eligibility_module.eligibility_criteria:
                criteria = trial.protocol_section.eligibility_module.eligibility_criteria
                if len(criteria) > 100:
                    indicators.append(True)

            # Check interventions detail
            if trial.interventions:
                detailed_interventions = any(
                    i.description for i in trial.interventions if i.description
                )
                if detailed_interventions:
                    indicators.append(True)

            # Check outcomes
            if trial.primary_outcomes:
                indicators.append(True)

            # Check derived section equivalent (we can use conditions as proxy)
            if trial.conditions:
                indicators.append(True)

            # Check sponsor collaborators (use sponsor module if available)
            if trial.protocol_section.sponsor_collaborators_module:
                sponsor = trial.protocol_section.sponsor_collaborators_module
                if isinstance(sponsor, dict) and sponsor.get("collaborators"):
                    collaborators = sponsor["collaborators"]
                    if collaborators and len(collaborators) > 0:
                        indicators.append(True)

            return len(indicators) >= 3

        # Handle old dictionary format
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict or ClinicalTrialData instance")

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if not protocol_section:
            return False

        indicators = []

        # Try new eligibility_module format first
        eligibility = protocol_section.get("eligibilityModule") or protocol_section.get("eligibility_module")
        if eligibility:
            criteria = eligibility.get("eligibilityCriteria", "")
            if criteria and len(criteria) > 100:
                indicators.append(True)

        # Try new arms_interventions_module format first
        arms = protocol_section.get("armsInterventionsModule") or protocol_section.get("arms_interventions_module")
        if arms:
            interventions = arms.get("interventions", [])
            if interventions and len(interventions) > 0:
                detailed_interventions = any(
                    isinstance(i, dict) and i.get("description", "") for i in interventions
                )
                if detailed_interventions:
                    indicators.append(True)

        # Try new outcomes_module format first
        outcomes = protocol_section.get("outcomesModule") or protocol_section.get("outcomes_module")
        if isinstance(outcomes, dict) and outcomes.get("primaryOutcomes"):
            indicators.append(True)

        # Check derived section
        derived_section = trial.get("derivedSection")
        if derived_section and isinstance(derived_section, dict):
            indicators.append(True)

        # Check sponsor collaborators
        sponsor = protocol_section.get("sponsorCollaboratorsModule")
        if isinstance(sponsor, dict):
            collaborators = sponsor.get("collaborators", [])
            if collaborators and len(collaborators) > 0:
                indicators.append(True)

        return len(indicators) >= 3
