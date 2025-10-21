
from typing import Any, Dict, cast

from services.summarizer import McodeSummarizer
from storage.mcode_memory_storage import OncoCoreMemory

from .base_summarizer import BaseSummarizerWorkflow
from .base_workflow import WorkflowResult


class TrialsSummarizerWorkflow(BaseSummarizerWorkflow):

    @property
    def memory_space(self) -> str:
        return "trials_summaries"

    def __init__(self, config: Any, memory_storage: OncoCoreMemory):
        super().__init__(config, memory_storage)
        self.summarizer = McodeSummarizer()

    def execute(self, **kwargs: Any) -> WorkflowResult:
        if "trials_data" not in kwargs:
            raise ValueError("trials_data is required")
        if "store_in_memory" not in kwargs:
            raise ValueError("store_in_memory is required")
        trials_data = kwargs["trials_data"]
        store_in_memory = kwargs["store_in_memory"]
        if not isinstance(trials_data, list):
            raise TypeError("trials_data must be a list")
        if not isinstance(store_in_memory, bool):
            raise TypeError("store_in_memory must be a bool")
        if not trials_data:
            raise ValueError("trials_data cannot be empty")

        processed_trials = []

        for trial in trials_data:
            trial_id = self._extract_trial_id(trial)
            mcode_elements = trial["McodeResults"]["mcode_mappings"]

            formatted_elements = []
            for element in mcode_elements:
                element_type = element.get("element_type") if isinstance(element, dict) else element.element_type
                if (
                    element_type.startswith("Trial")
                    and len(element_type) > 5
                    and element_type[5].isupper()
                ):
                    element_name = element_type[5:]
                else:
                    element_name = element_type

                system = element.get('system', '') if isinstance(element, dict) else getattr(element, 'system', '')
                code = element.get('code', '') if isinstance(element, dict) else getattr(element, 'code', '')
                codes = self.summarizer._format_mcode_display('', system, code) if system and code else ""
                formatted_element = {
                    "element_name": element_name,
                    "value": element.get("display", "") if isinstance(element, dict) else element.display or "",
                    "codes": codes,
                    "date_qualifier": "",
                }
                formatted_elements.append(formatted_element)

            prioritized = self.summarizer._group_elements_by_priority(
                formatted_elements, "Trial"
            )
            sentences = self.summarizer._generate_sentences_from_elements(
                prioritized, trial_id
            )
            summary = " ".join(sentences)

            processed_trial = trial.copy()
            if "McodeResults" not in processed_trial:
                processed_trial["McodeResults"] = {}
            processed_trial["McodeResults"]["natural_language_summary"] = summary

            if store_in_memory:
                self.memory_storage.store_trial_summary(trial_id, summary)

            processed_trials.append(processed_trial)

        return self._create_result(
            success=True,
            data=processed_trials,
            metadata={
                "total_trials": len(trials_data),
                "successful": len(processed_trials),
                "stored_in_memory": store_in_memory,
            },
        )

    def process_single_trial(self, trial: Dict[str, Any], **kwargs: Any) -> WorkflowResult:
        if "store_in_memory" not in kwargs:
            raise ValueError("store_in_memory is required")
        if not isinstance(kwargs["store_in_memory"], bool):
            raise TypeError("store_in_memory must be a bool")
        result = self.execute(trials_data=[trial], **kwargs)
        return self._create_result(success=True, data=result.data[0], metadata=result.metadata)

    def _extract_trial_id(self, trial: Dict[str, Any]) -> str:
        if "protocolSection" in trial:
            protocol = trial["protocolSection"]
            if "identificationModule" in protocol:
                return cast(str, protocol["identificationModule"]["nctId"])
        if "trial_id" in trial:
            return cast(str, trial["trial_id"])
        if "nctId" in trial:
            return cast(str, trial["nctId"])
        raise ValueError("No trial ID found")

    def _extract_trial_metadata(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {}
        if "protocolSection" in trial:
            protocol = trial["protocolSection"]
            if "identificationModule" in protocol:
                ident = protocol["identificationModule"]
                metadata["nct_id"] = ident.get("nctId")
                metadata["brief_title"] = ident.get("briefTitle")
                metadata["official_title"] = ident.get("officialTitle")
            if "statusModule" in protocol:
                status = protocol["statusModule"]
                metadata["overall_status"] = status.get("overallStatus")
                metadata["start_date"] = status.get("startDateStruct", {}).get("date")
                metadata["completion_date"] = status.get("completionDateStruct", {}).get("date")
            if "conditionsModule" in protocol:
                conditions = protocol["conditionsModule"]
                metadata["conditions"] = conditions.get("conditions", [])
            if "eligibilityModule" in protocol:
                eligibility = protocol["eligibilityModule"]
                metadata["minimum_age"] = eligibility.get("minimumAge")
                metadata["maximum_age"] = eligibility.get("maximumAge")
                metadata["gender"] = eligibility.get("sex")
                metadata["healthy_volunteers"] = eligibility.get("healthyVolunteers")
        from datetime import datetime, timezone
        metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
        return metadata
