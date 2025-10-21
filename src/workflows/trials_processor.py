import asyncio
from typing import Any, Dict, List, Optional, Union

from ensemble.trials_ensemble_engine import TrialsEnsembleEngine
from pipeline import McodePipeline
from services.data_enrichment import DataEnrichmentService
from services.llm.service import LLMService
from shared.data_quality_validator import DataQualityValidator
from shared.models import ClinicalTrialData
from storage.mcode_memory_storage import OncoCoreMemory
from utils.concurrency import AsyncTaskQueue, create_task

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

        # Initialize ensemble engine for trials processing
        self.ensemble_engine: Optional[TrialsEnsembleEngine] = None

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

        # Initialize ensemble engine if needed
        if engine == "ensemble" and not self.ensemble_engine:
            self.ensemble_engine = TrialsEnsembleEngine(
                model_name=model,
                config=self.config
            )
            self.logger.info("✅ Initialized TrialsEnsembleEngine for ensemble processing")

        # Initialize pipeline for LLM engine (existing behavior)
        if engine == "llm" and not self.pipeline:
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
                engine=engine,
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

                # Convert dictionaries to McodeElement objects for validation
                mcode_element_objects = []
                if isinstance(mcode_elements, list):
                    for elem in mcode_elements:
                        if isinstance(elem, dict):
                            try:
                                from src.shared.models import McodeElement
                                mcode_obj = McodeElement(
                                    element_type=elem.get("mcode_element", ""),
                                    code=elem.get("code", ""),
                                    display=elem.get("value", ""),
                                    system=elem.get("system", ""),
                                    confidence_score=0.8,  # Default confidence
                                    evidence_text=elem.get("interpretation", "")
                                )
                                mcode_element_objects.append(mcode_obj)
                            except Exception as e:
                                self.logger.warning(f"Failed to convert mcode element to object: {e}")

                # Convert enhanced trial to ClinicalTrialData for validation
                from src.shared.models import ClinicalTrialData

                # Transform trial data to ClinicalTrialData format
                clinical_trial_dict = {
                    "trial_id": self._extract_trial_id(trial_result),
                    "title": self._extract_trial_title(trial_result),
                    "eligibility_criteria": self._extract_eligibility_criteria(trial_result),
                    "conditions": self._extract_conditions(trial_result),
                    "interventions": self._extract_interventions(trial_result),
                    "phase": self._extract_phase(trial_result),
                    "protocol_section": trial_result.get("protocolSection", {}),
                    "has_results": trial_result.get("hasResults", False),
                    "study_type": self._extract_study_type(trial_result),
                    "overall_status": self._extract_overall_status(trial_result),
                }

                clinical_trial_data = ClinicalTrialData(**clinical_trial_dict)
                quality_report = self.quality_validator.validate_trial_data(
                    clinical_trial_data, mcode_element_objects
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
            data=processed_trials if processed_trials else [],
            metadata={
                "total_trials": total_count,
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": success_rate,
                "engine_used": engine,
                "model_used": model if engine in ["llm", "ensemble"] else None,
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

    def _extract_trial_title(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> str:
        """Extract trial title handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.title or trial.trial_id

        # Handle old dictionary format
        if not isinstance(trial, dict):
            return "Unknown Title"

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if not protocol_section:
            return "Unknown Title"

        # Try new identification_module format first
        identification = protocol_section.get("identificationModule") or protocol_section.get("identification_module")
        if identification:
            title = identification.get("briefTitle") or identification.get("officialTitle")
            if title:
                return str(title)

        return "Unknown Title"

    def _extract_eligibility_criteria(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> str:
        """Extract eligibility criteria handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.eligibility_criteria or ""

        # Handle old dictionary format
        if not isinstance(trial, dict):
            return ""

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if protocol_section:
            eligibility_module = protocol_section.get("eligibilityModule") or protocol_section.get("eligibility_module")
            if eligibility_module:
                criteria = eligibility_module.get("eligibilityCriteria")
                if criteria:
                    return str(criteria)

        return ""

    def _extract_conditions(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> List[str]:
        """Extract conditions handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.conditions or []

        # Handle old dictionary format
        if not isinstance(trial, dict):
            return []

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if protocol_section:
            conditions_module = protocol_section.get("conditionsModule") or protocol_section.get("conditions_module")
            if conditions_module and "conditions" in conditions_module:
                conditions = conditions_module["conditions"]
                if isinstance(conditions, list):
                    result = []
                    for c in conditions:
                        if isinstance(c, dict):
                            name = c.get("name", "")
                            if name:
                                result.append(str(name))
                        elif isinstance(c, str):
                            result.append(c)
                    return result
                else:
                    return [str(conditions)]

        return []

    def _extract_interventions(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> List[str]:
        """Extract interventions handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.interventions or []

        # Handle old dictionary format
        if not isinstance(trial, dict):
            return []

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if protocol_section:
            arms_module = protocol_section.get("armsInterventionsModule") or protocol_section.get("arms_interventions_module")
            if arms_module and "interventions" in arms_module:
                interventions = arms_module["interventions"]
                if isinstance(interventions, list):
                    return [str(i.get("name", "")) for i in interventions if i.get("name")]
                else:
                    return [str(interventions)]

        return []

    def _extract_phase(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> str:
        """Extract phase handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.phase or ""

        # Handle old dictionary format
        if not isinstance(trial, dict):
            return ""

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if protocol_section:
            design_module = protocol_section.get("designModule") or protocol_section.get("design_module")
            if design_module:
                phases = design_module.get("phases")
                if phases:
                    if isinstance(phases, list):
                        return ", ".join(str(p) for p in phases)
                    else:
                        return str(phases)

        return ""

    def _extract_study_type(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> str:
        """Extract study type handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.study_type or ""

        # Handle old dictionary format
        if not isinstance(trial, dict):
            return ""

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if protocol_section:
            design_module = protocol_section.get("designModule") or protocol_section.get("design_module")
            if design_module:
                study_type = design_module.get("studyType")
                if study_type:
                    return str(study_type)

        return ""

    def _extract_overall_status(self, trial: Union[Dict[str, Any], ClinicalTrialData]) -> str:
        """Extract overall status handling both old and new model formats."""
        # Handle new ClinicalTrialData model format
        if isinstance(trial, ClinicalTrialData):
            return trial.overall_status or ""

        # Handle old dictionary format
        if not isinstance(trial, dict):
            return ""

        # Try new protocol_section format first
        protocol_section = trial.get("protocolSection") or trial.get("protocol_section")
        if protocol_section:
            status_module = protocol_section.get("statusModule") or protocol_section.get("status_module")
            if status_module:
                status = status_module.get("overallStatus")
                if status:
                    return str(status)

        return ""

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
        engine: str = "llm",
    ) -> tuple[Any, bool]:
        # Validate trial data structure
        if not isinstance(trial, dict):
            raise ValueError(f"Trial data must be a dict, got {type(trial)}")

        if engine == "ensemble" and self.ensemble_engine is not None:
            # Use ensemble processing
            ensemble_result = await self._process_trial_with_ensemble(trial)
            enhanced_trial = trial.copy()
            enhanced_trial["McodeResults"] = self._convert_ensemble_result_to_mcode_format(ensemble_result)
            mcode_success = ensemble_result.is_match
        elif engine == "llm" and self.pipeline is not None:
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
            if engine == "ensemble":
                enhanced_trial["McodeProcessingError"] = "Ensemble engine not initialized"
            else:
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

    async def _process_trial_with_ensemble(self, trial: Dict[str, Any]) -> "EnsembleResult":
        """Process a single trial using the ensemble engine."""
        if not self.ensemble_engine:
            raise ValueError("Ensemble engine not initialized")

        # Prepare input data for ensemble processing
        input_data = {
            "trial_id": self._extract_trial_id(trial),
            "eligibility_criteria": trial.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", ""),
            "conditions": trial.get("conditions", []),
            "phase": trial.get("protocolSection", {}).get("designModule", {}).get("phases", ""),
        }

        # Prepare criteria data for mCODE extraction
        criteria_data = {
            "mcode_extraction_rules": "standard",
            "validation_criteria": "clinical_accuracy",
        }

        # Process with ensemble engine
        ensemble_result = await self.ensemble_engine.process_ensemble(input_data, criteria_data)

        return ensemble_result

    def _convert_ensemble_result_to_mcode_format(self, ensemble_result: "EnsembleResult") -> Dict[str, Any]:
        """Convert EnsembleResult to McodeResults format."""
        # Convert expert assessments to mcode_mappings format
        mcode_mappings = []

        for assessment in ensemble_result.expert_assessments:
            expert_assessment = assessment.get("assessment", {})
            mcode_elements = expert_assessment.get("mcode_elements", [])

            for element in mcode_elements:
                if isinstance(element, dict):
                    # Convert from ensemble format to mcode format
                    mapping = {
                        "mcode_element": element.get("element_type", ""),
                        "value": element.get("display", ""),
                        "system": element.get("system", ""),
                        "code": element.get("code", ""),
                        "interpretation": f"Ensemble confidence: {ensemble_result.confidence_score:.3f}",
                    }
                    mcode_mappings.append(mapping)

        # Create McodeResults structure
        return {
            "extracted_entities": mcode_mappings,
            "mcode_mappings": mcode_mappings,
            "source_references": [
                {
                    "source_type": "ensemble_engine",
                    "source_id": "trials_ensemble",
                    "section": "expert_assessments"
                }
            ],
            "validation_results": {
                "compliance_score": ensemble_result.confidence_score,
                "validation_errors": [],
                "warnings": [],
                "required_elements_present": [elem.get("element_type", "") for elem in mcode_mappings],
                "missing_elements": []
            },
            "metadata": {
                "engine_type": "ensemble",
                "consensus_method": ensemble_result.consensus_method,
                "experts_used": len(ensemble_result.expert_assessments),
                "diversity_score": ensemble_result.diversity_score,
                "processing_time_seconds": ensemble_result.processing_metadata.get("processing_time_seconds", 0),
            },
            "token_usage": None,
            "error": None,
        }

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
