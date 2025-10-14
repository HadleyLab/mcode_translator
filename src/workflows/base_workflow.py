"""
Base workflow classes for mCODE translator.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

if TYPE_CHECKING:
    from src.storage.mcode_memory_storage import OncoCoreMemory

from src.shared.models import ProcessingMetadata, WorkflowResult
from src.storage.mcode_memory_storage import OncoCoreMemory
from src.utils.config import Config


class WorkflowError(Exception):
    pass


class BaseWorkflow(ABC):
    def __init__(
        self,
        config: Config,
        memory_storage: OncoCoreMemory,
    ):
        self.config = config
        self.memory_storage = memory_storage

    @property
    @abstractmethod
    def memory_space(self) -> str:
        pass

    def store_to_core_memory(
        self, key: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        namespaced_key = f"{self.memory_space}:{key}"
        self.memory_storage.search_trials(namespaced_key, limit=1)
        return True

    def retrieve_from_core_memory(self, key: str) -> Optional[Dict[str, Any]]:
        namespaced_key = f"{self.memory_space}:{key}"
        result = self.memory_storage.search_trials(namespaced_key, limit=1)
        if result and result.get("episodes"):
            return cast(Dict[str, Any], result["episodes"][0])
        return None

    def _get_timestamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    @abstractmethod
    def execute(self, **kwargs: Any) -> WorkflowResult:
        pass

    def validate_inputs(self, **kwargs: Any) -> bool:
        return True

    def _create_result(
        self,
        success: bool,
        data: Optional[Union[Dict[str, Any], List[Any]]] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Union[Dict[str, Any], ProcessingMetadata]] = None,
    ) -> WorkflowResult:
        if isinstance(metadata, ProcessingMetadata):
            metadata_dict = metadata.model_dump()
        else:
            metadata_dict = metadata or {}

        return WorkflowResult(
            success=success,
            data=data or {},
            error_message=error_message,
            metadata=metadata_dict,
        )



class FetcherWorkflow(BaseWorkflow):
    @property
    def memory_space(self) -> str:
        return "raw_data"


class ProcessorWorkflow(BaseWorkflow):
    pass


class TrialsProcessorWorkflow(ProcessorWorkflow):
    @property
    def memory_space(self) -> str:
        return "trials"


class PatientsProcessorWorkflow(ProcessorWorkflow):
    @property
    def memory_space(self) -> str:
        return "patients"


class SummarizerWorkflow(BaseWorkflow):
    pass


class TrialsSummarizerWorkflow(SummarizerWorkflow):
    @property
    def memory_space(self) -> str:
        return "trials_summaries"


class PatientsSummarizerWorkflow(SummarizerWorkflow):
    @property
    def memory_space(self) -> str:
        return "patients_summaries"


__all__ = [
    "WorkflowError",
    "BaseWorkflow",
    "FetcherWorkflow",
    "ProcessorWorkflow",
    "TrialsProcessorWorkflow",
    "PatientsProcessorWorkflow",
    "SummarizerWorkflow",
    "TrialsSummarizerWorkflow",
    "PatientsSummarizerWorkflow",
]
