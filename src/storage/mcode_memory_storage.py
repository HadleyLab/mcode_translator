"""
OncoCore Memory - Unified interface for storing mCODE summaries to CORE Memory.

This module provides a unified interface for storing processed mCODE data
(patients and trials) as natural language summaries that preserve mCODE
structure and codes for later retrieval and analysis. This is the mCODE Translator's
instance of CORE Memory with dedicated patients and trials spaces.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, cast

# Inject heysol_api_client path for imports (per coding standards)
heysol_client_path = (
    Path(__file__).parent.parent.parent.parent / "heysol_api_client" / "src"
)
if str(heysol_client_path) not in sys.path:
    sys.path.insert(0, str(heysol_client_path))

from heysol.clients import HeySolAPIClient
from heysol.config import HeySolConfig
from heysol.exceptions import HeySolError

from src.services.summarizer import McodeSummarizer
from src.utils.config import Config
from src.utils.logging_config import get_logger


class OncoCoreMemory:
    """
    Unified storage interface for mCODE summaries in CORE Memory.

    This is the mCODE Translator's dedicated CORE Memory instance with patients
    and trials spaces. Stores processed mCODE data as natural language summaries
    that preserve the structured mCODE elements and codes for later analysis.

    Uses HeySol API client directly with --store-to-memory functionality.
    """

    PATIENTS_SPACE = "patients"
    TRIALS_SPACE = "trials"

    def __init__(self, api_key: Optional[str] = None, source: Optional[str] = None):
        """
        Initialize the OncoCore Memory interface with centralized configuration.

        Args:
            api_key: Optional HeySol API key (will use config if not provided)
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

        # Initialize HeySol API client directly
        heysol_config = HeySolConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            source=self.source,
            timeout=self.timeout,
        )
        self.api_client = HeySolAPIClient(config=heysol_config)

        # Ensure spaces exist
        self._ensure_spaces()

    def _ensure_spaces(self) -> None:
        """Ensure the required spaces exist in CORE Memory."""
        try:
            spaces = self.api_client.get_spaces()
            existing_spaces = {space.get("name", "") for space in spaces}

            if self.PATIENTS_SPACE not in existing_spaces:
                self.api_client.create_space(
                    self.PATIENTS_SPACE, "Patient mCODE summaries"
                )
                self.logger.info(f"Created {self.PATIENTS_SPACE} space")

            if self.TRIALS_SPACE not in existing_spaces:
                self.api_client.create_space(
                    self.TRIALS_SPACE, "Clinical trial mCODE summaries"
                )
                self.logger.info(f"Created {self.TRIALS_SPACE} space")

        except HeySolError as e:
            self.logger.error(f"Failed to ensure spaces: {e}")
            raise

    def store_trial_mcode_summary(
        self, trial_id: str, mcode_data: Dict[str, Any]
    ) -> bool:
        """
        Store a processed clinical trial's mCODE summary in the trials space.

        Args:
            trial_id: Clinical trial identifier (NCT ID)
            mcode_data: Processed mCODE data with mappings and metadata

        Returns:
            bool: True if stored successfully
        """
        try:
            # Generate summary using summarizer
            trial_data = mcode_data.get("original_trial_data")
            if not trial_data:
                # Reconstruct from metadata if needed
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

            # Store in trials space
            self.api_client.ingest(
                message=summary, space_id=self.TRIALS_SPACE, source=self.source
            )

            self.logger.info(
                f"✅ Stored trial {trial_id} mCODE summary in {self.TRIALS_SPACE} space"
            )
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
        Store a processed patient's mCODE summary in the patients space.

        Args:
            patient_id: Patient identifier
            mcode_data: Processed mCODE data with mappings and metadata

        Returns:
            bool: True if stored successfully
        """
        try:
            # Generate summary using summarizer
            patient_bundle = mcode_data.get("original_patient_data")
            if not patient_bundle:
                self.logger.warning(
                    f"Original patient data not available for {patient_id}, using processed data"
                )
                patient_bundle = mcode_data

            summary = self.summarizer.create_patient_summary(patient_bundle)

            # Store in patients space
            self.api_client.ingest(
                message=summary, space_id=self.PATIENTS_SPACE, source=self.source
            )

            self.logger.info(
                f"✅ Stored patient {patient_id} mCODE summary in {self.PATIENTS_SPACE} space"
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
        Search for similar trials in the trials space.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            Dict with search results
        """
        try:
            result = self.api_client.search(
                query=query, space_ids=[self.TRIALS_SPACE], limit=limit
            )
            return cast(Dict[str, Any], result)
        except HeySolError as e:
            self.logger.error(f"❌ Search failed: {e}")
            return {"episodes": [], "facts": []}

    def search_similar_patients(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for similar patients in the patients space.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            Dict with search results
        """
        try:
            result = self.api_client.search(
                query=query, space_ids=[self.PATIENTS_SPACE], limit=limit
            )
            return cast(Dict[str, Any], result)
        except HeySolError as e:
            self.logger.error(f"❌ Search failed: {e}")
            return {"episodes": [], "facts": []}

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories in both spaces.

        Returns:
            Dict with memory statistics
        """
        try:
            spaces = cast(list[Dict[str, Any]], self.api_client.get_spaces())
            patients_space: Dict[str, Any] = next(
                (s for s in spaces if s.get("name") == self.PATIENTS_SPACE), {}
            )
            trials_space: Dict[str, Any] = next(
                (s for s in spaces if s.get("name") == self.TRIALS_SPACE), {}
            )

            return {
                "spaces": spaces,
                "total_spaces": len(spaces),
                "patients_space": patients_space,
                "trials_space": trials_space,
            }
        except HeySolError as e:
            self.logger.error(f"❌ Failed to get memory stats: {e}")
            return {"spaces": [], "total_spaces": 0}

    def close(self) -> None:
        """Close the API client and clean up resources."""
        if hasattr(self.api_client, "close"):
            self.api_client.close()
