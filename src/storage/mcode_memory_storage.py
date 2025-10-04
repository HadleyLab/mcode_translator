import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.services.summarizer import McodeSummarizer
from src.shared.models import MemoryStats, SearchResult
from src.utils.config import Config

# Inject heysol_api_client path for imports (per coding standards)
heysol_client_path = (
    Path(__file__).parent.parent.parent.parent / "heysol_api_client" / "src"
)
if str(heysol_client_path) not in sys.path:
    sys.path.insert(0, str(heysol_client_path))

from heysol.clients.mcp_client import HeySolMCPClient


class OncoCoreMemory:
    """
    Unified storage interface for mCODE summaries in CORE Memory.

    This is the mCODE Translator's dedicated CORE Memory instance with patients
    and trials spaces. Stores processed mCODE data as natural language summaries
    that preserve the structured mCODE elements and codes for later analysis.

    Uses HeySol MCP client for maximum code reuse and consistency.
    """

    PATIENTS_SPACE = "Patient Data Repository"
    TRIALS_SPACE = "Clinical Trials Database"

    def __init__(self, api_key: Optional[str] = None, source: Optional[str] = None):
        self.config = Config()
        from services import McodeTrialProcessor
        self.processor = McodeTrialProcessor()
        self.api_key = api_key or self.config.get_core_memory_api_key()
        self.source = source or self.config.get_core_memory_source()
        self.mcp_url = self.config.get_core_memory_api_base_url()
        self._mcp_client: Optional[HeySolMCPClient] = None

    @property
    def mcp_client(self) -> HeySolMCPClient:
        if self._mcp_client is None:
            self._mcp_client = HeySolMCPClient(api_key=self.api_key, mcp_url=self.mcp_url)
        return self._mcp_client



    def store_trial_mcode_summary(self, trial_id: str, mcode_data: Dict[str, Any]) -> bool:
        trial_data = mcode_data.get("original_trial_data")
        if not trial_data:
            trial_data = {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": trial_id,
                        "briefTitle": mcode_data.get("trial_metadata", {}).get("brief_title", "Unknown"),
                        "officialTitle": mcode_data.get("trial_metadata", {}).get("official_title", "Unknown"),
                    },
                    "statusModule": {"overallStatus": mcode_data.get("trial_metadata", {}).get("overall_status", "Unknown")},
                    "designModule": {
                        "studyType": mcode_data.get("trial_metadata", {}).get("study_type", "Unknown"),
                        "phases": mcode_data.get("trial_metadata", {}).get("phase", []),
                    },
                }
            }

        # Process trial data to get summary
        import asyncio
        result = asyncio.run(self.processor.process_trial(trial_data))
        if result.success:
            summary = result.data
            self.mcp_client.ingest(message=summary, space_id=self.TRIALS_SPACE, source=self.source)
            return True
        return False

    def store_patient_mcode_summary(self, patient_id: str, mcode_data: Dict[str, Any]) -> bool:
        patient_bundle = mcode_data.get("original_patient_data") or mcode_data
        # For now, just store the raw data as summary - patient processing not fully implemented
        import json
        summary = f"Patient {patient_id}: {json.dumps(patient_bundle, indent=2)}"
        self.mcp_client.ingest(message=summary, space_id=self.PATIENTS_SPACE, source=self.source)
        return True

    def search_similar_trials(self, query: str, limit: int = 10) -> SearchResult:
        result = self.mcp_client.search(query=query, space_ids=[self.TRIALS_SPACE], limit=limit)
        return SearchResult(
            episodes=result.get("episodes", []),
            facts=result.get("facts", []),
            total_count=len(result.get("episodes", []))
        )

    def search_similar_patients(self, query: str, limit: int = 10) -> SearchResult:
        result = self.mcp_client.search(query=query, space_ids=[self.PATIENTS_SPACE], limit=limit)
        return SearchResult(
            episodes=result.get("episodes", []),
            facts=result.get("facts", []),
            total_count=len(result.get("episodes", []))
        )

    def get_memory_stats(self) -> MemoryStats:
        spaces = self.mcp_client.get_spaces()
        patients_space = next((s for s in spaces if s.get("name") == self.PATIENTS_SPACE), {})
        trials_space = next((s for s in spaces if s.get("name") == self.TRIALS_SPACE), {})
        return MemoryStats(
            spaces=spaces,
            total_spaces=len(spaces),
            patients_space=patients_space,
            trials_space=trials_space,
        )

    def close(self) -> None:
        if self._mcp_client:
            self._mcp_client.close()


# Backward compatibility alias
McodeMemoryStorage = OncoCoreMemory
