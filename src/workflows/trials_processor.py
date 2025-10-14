import asyncio
from typing import Any, Dict, List, Optional

from src.pipeline import McodePipeline
from src.services.data_enrichment import DataEnrichmentService
from src.services.llm.service import LLMService
from src.shared.models import enhance_trial_with_mcode_results
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
                quality_report = self.quality_validator.validate_trial_data(
                    trial_result, mcode_elements
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

    def _extract_trial_mcode_elements_cached(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        return self.extractor.extract_trial_mcode_elements(trial)

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict")
        if "protocolSection" not in trial:
            raise ValueError("trial must have protocolSection")
        if "identificationModule" not in trial["protocolSection"]:
            raise ValueError("trial must have identificationModule")
        if "nctId" not in trial["protocolSection"]["identificationModule"]:
            raise ValueError("trial must have nctId")

        return str(trial["protocolSection"]["identificationModule"]["nctId"])

    def _extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict")
        if "protocolSection" not in trial:
            raise ValueError("trial must have protocolSection")

        metadata = {}
        protocol_section = trial["protocolSection"]

        identification = protocol_section["identificationModule"]
        metadata["nct_id"] = identification["nctId"]
        metadata["brief_title"] = identification["briefTitle"]
        metadata["official_title"] = identification["officialTitle"]

        status = protocol_section["statusModule"]
        metadata["overall_status"] = status["overallStatus"]
        start_struct = status["startDateStruct"]
        metadata["start_date"] = start_struct["date"]
        completion_struct = status["completionDateStruct"]
        metadata["completion_date"] = completion_struct["date"]

        design = protocol_section["designModule"]
        metadata["study_type"] = design["studyType"]
        metadata["phase"] = design["phases"]
        metadata["primary_purpose"] = design["primaryPurpose"]

        eligibility = protocol_section["eligibilityModule"]
        metadata["minimum_age"] = eligibility["minimumAge"]
        metadata["maximum_age"] = eligibility["maximumAge"]
        metadata["sex"] = eligibility["sex"]
        metadata["healthy_volunteers"] = eligibility["healthyVolunteers"]

        conditions_module = protocol_section["conditionsModule"]
        conditions = conditions_module["conditions"]
        metadata["conditions"] = [c["name"] for c in conditions]

        interventions_module = protocol_section["armsInterventionsModule"]
        interventions = interventions_module["interventions"]
        metadata["interventions"] = [i["name"] for i in interventions]

        return metadata

    def _generate_trial_natural_language_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
        return self.summarizer.generate_trial_natural_language_summary(
            trial_id, mcode_elements, trial_data
        )

    def _generate_trial_regex_summary(
        self, trial_id: str, mcode_elements: Dict[str, Any], trial_data: Dict[str, Any]
    ) -> str:
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
        if self.pipeline is not None:
            result = await self.pipeline.process(trial)
            enhanced_trial = enhance_trial_with_mcode_results(trial, result)
            mcode_success = True
        else:
            enhanced_trial = trial.copy()
            enhanced_trial["McodeProcessingError"] = "Pipeline not initialized"
            mcode_success = False

        trial_id = self._extract_trial_id(trial)
        mcode_elements = enhanced_trial.get("McodeResults", {}).get("mcode_mappings", [])

        # Apply data enrichment to improve completeness
        enriched_trial_data = await self.data_enrichment.enrich_trial_data(
            enhanced_trial, mcode_elements
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

    def validate_trial_data(self, trial: Dict[str, Any]) -> bool:
        if not isinstance(trial, dict):
            raise ValueError("trial must be a dict")
        if "protocolSection" not in trial:
            raise ValueError("trial must have protocolSection")

        protocol_section = trial["protocolSection"]
        identification = protocol_section["identificationModule"]
        nct_id = identification["nctId"]

        if not nct_id:
            return False

        eligibility = protocol_section["eligibilityModule"]
        criteria = eligibility["eligibilityCriteria"]

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

    def _check_trial_has_full_data(self, trial: Dict[str, Any]) -> bool:
        protocol_section = trial["protocolSection"]
        indicators = []

        eligibility = protocol_section["eligibilityModule"]
        criteria = eligibility["eligibilityCriteria"]
        if criteria and len(criteria) > 100:
            indicators.append(True)

        arms = protocol_section["armsInterventionsModule"]
        interventions = arms["interventions"]
        if interventions and len(interventions) > 0:
            detailed_interventions = any(
                isinstance(i, dict) and i["description"] for i in interventions
            )
            if detailed_interventions:
                indicators.append(True)

        outcomes = protocol_section["outcomesModule"]
        if isinstance(outcomes, dict) and outcomes["primaryOutcomes"]:
            indicators.append(True)

        derived_section = trial["derivedSection"]
        if derived_section and isinstance(derived_section, dict):
            indicators.append(True)

        sponsor = protocol_section["sponsorCollaboratorsModule"]
        if isinstance(sponsor, dict):
            collaborators = sponsor["collaborators"]
            if collaborators and len(collaborators) > 0:
                indicators.append(True)

        return len(indicators) >= 3
