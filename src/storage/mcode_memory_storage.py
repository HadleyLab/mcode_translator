"""
mCODE Memory Storage - Unified interface for storing mCODE summaries to CORE Memory.

This module provides a unified interface for storing processed mCODE data
(trials and patients) as natural language summaries that preserve mCODE
structure and codes for later retrieval and analysis.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from heysol.config import HeySolConfig

from src.services.summarizer import McodeSummarizer
from src.utils.config import Config
from src.utils.logging_config import get_logger
from src.utils.onco_core_memory import HeySolError, OncoCoreClient

# Add heysol_api_client to path for imports
heysol_client_path = (
    Path(__file__).parent.parent.parent.parent / "heysol_api_client" / "src"
)
if str(heysol_client_path) not in sys.path:
    sys.path.insert(0, str(heysol_client_path))


class DataStorage(Protocol):
    """Protocol for data storage components."""

    def store(self, key: str, data: Dict[str, Any]) -> bool:
        """Store data with given key. Returns success status."""
        ...

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key."""
        ...


class McodeMemoryStorage:
    """
    Unified storage interface for mCODE summaries in CORE Memory.

    Stores processed mCODE data as natural language summaries that preserve
    the structured mCODE elements and codes for later analysis.
    """

    def __init__(self, api_key: Optional[str] = None, source: Optional[str] = None):
        """
        Initialize the storage interface with centralized configuration.

        Args:
            api_key: Optional CORE Memory API key (will use config if not provided)
            source: Optional source identifier (will use config if not provided)
        """
        self.logger = get_logger(__name__)
        self.config = Config()
        self.summarizer = McodeSummarizer()

        # Use provided values or get from centralized config
        self.api_key = api_key or self.config.get_core_memory_api_key()
        self.source = source or self.config.get_core_memory_source()
        self.base_url = self.config.get_core_memory_api_base_url()
        self.timeout = self.config.get_core_memory_timeout()
        self.max_retries = self.config.get_core_memory_max_retries()
        self.batch_size = self.config.get_core_memory_batch_size()
        self.default_spaces = self.config.get_core_memory_default_spaces()

        # Defer client initialization until first use to avoid auth errors during import
        self._client: Optional[OncoCoreClient] = None

    @property
    def client(self) -> OncoCoreClient:
        """Lazy initialization of the CORE Memory client."""
        if self._client is None:
            config = HeySolConfig(
                api_key=self.api_key, base_url=self.base_url, source=self.source
            )
            self._client = OncoCoreClient(
                api_key=self.api_key, base_url=self.base_url, config=config
            )
        return self._client

    def store_trial_mcode_summary(
        self, trial_id: str, mcode_data: Dict[str, Any]
    ) -> bool:
        """
        Store a processed clinical trial's mCODE summary.

        Args:
            trial_id: Clinical trial identifier (NCT ID)
            mcode_data: Processed mCODE data with mappings and metadata

        Returns:
            bool: True if stored successfully
        """
        try:
            # The summarizer expects the original trial data, not processed mCODE data
            # Extract the original trial data from mcode_data if available
            trial_data = mcode_data.get("original_trial_data")
            if not trial_data:
                # If original trial data is not available, try to reconstruct from metadata
                trial_data = {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": trial_id,
                            "briefTitle": mcode_data.get("trial_metadata", {}).get(
                                "brief_title", "Unknown"
                            ),
                            "officialTitle": mcode_data.get("trial_metadata", {}).get(
                                "official_title", "Unknown"
                            ),
                        },
                        "statusModule": {
                            "overallStatus": mcode_data.get("trial_metadata", {}).get(
                                "overall_status", "Unknown"
                            )
                        },
                        "designModule": {
                            "studyType": mcode_data.get("trial_metadata", {}).get(
                                "study_type", "Unknown"
                            ),
                            "phases": mcode_data.get("trial_metadata", {}).get(
                                "phase", []
                            ),
                        },
                    }
                }

            summary = self.summarizer.create_trial_summary(trial_data)
            self.client.ingest(summary)
            self.logger.info(f"✅ Stored trial {trial_id} mCODE summary in CORE Memory")
            return True
        except HeySolError as e:
            self.logger.error(f"❌ Failed to store trial {trial_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error storing trial {trial_id}: {e}")
            return False

    def store_patient_mcode_summary(
        self, patient_id: str, mcode_data: Dict[str, Any]
    ) -> bool:
        """
        Store a processed patient's mCODE summary.

        Args:
            patient_id: Patient identifier
            mcode_data: Processed mCODE data with mappings and metadata

        Returns:
            bool: True if stored successfully
        """
        try:
            # The summarizer expects the original patient FHIR bundle data
            # Extract it from mcode_data if available, otherwise use the processed data
            patient_bundle = mcode_data.get("original_patient_data")
            if not patient_bundle:
                # If original data is not available, try to reconstruct or use processed data
                self.logger.warning(
                    f"Original patient data not available for {patient_id}, using processed data"
                )
                patient_bundle = mcode_data

            summary = self.summarizer.create_patient_summary(patient_bundle)
            self.client.ingest(summary)
            self.logger.info(
                f"✅ Stored patient {patient_id} mCODE summary in CORE Memory"
            )
            return True
        except HeySolError as e:
            self.logger.error(f"❌ Failed to store patient {patient_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error storing patient {patient_id}: {e}")
            return False

    def search_similar_trials(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for similar trials in CORE Memory.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            Dict with search results
        """
        try:
            return self.client.search(query, limit=limit)
        except HeySolError as e:
            self.logger.error(f"❌ Search failed: {e}")
            return {"episodes": [], "facts": []}

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Returns:
            Dict with memory statistics
        """
        try:
            spaces = self.client.get_spaces()
            return {"spaces": spaces, "total_spaces": len(spaces)}
        except HeySolError as e:
            self.logger.error(f"❌ Failed to get memory stats: {e}")
            return {"spaces": [], "total_spaces": 0}
