"""
OncoCore Service - Thin wrapper around HeySol OncoCore API client.

Stores patients to the right space and trials to the right space.
"""

from typing import Any, Dict, Optional

from src.services.heysol_client import OncoCoreClient


class OncoCoreService:
    """
    Thin wrapper around HeySol OncoCore API client for mCODE data storage.

    Automatically routes patients to patient space and trials to trial space.
    """

    PATIENTS_SPACE = "Patient Data Repository"
    TRIALS_SPACE = "Clinical Trials Database"

    def __init__(self, api_key: Optional[str] = None, source: Optional[str] = None):
        """
        Initialize the OncoCore service.

        Args:
            api_key: HeySol API key (optional, uses env if not provided)
            source: Source identifier for stored data
        """
        self.client = OncoCoreClient.from_env()
        self.source = source or "mcode-translator"

    def store_trial_summary(self, trial_id: str, summary: str, space_id: Optional[str] = None, session_id: Optional[str] = None) -> bool:
        """
        Store a trial summary in the trials space.

        Args:
            trial_id: Clinical trial NCT ID
            summary: Natural language summary to store
            space_id: Optional space ID to use instead of default
            session_id: Optional session identifier

        Returns:
            bool: True if successful
        """
        try:
            result = self.client.ingest(
                message=summary,
                source=self.source,
                space_id=space_id or self.TRIALS_SPACE,
                session_id=session_id
            )
            return result.get("success", False)
        except Exception:
            return False

    def store_patient_summary(self, patient_id: str, summary: str) -> bool:
        """
        Store a patient summary in the patients space.

        Args:
            patient_id: Patient identifier
            summary: Natural language summary to store

        Returns:
            bool: True if successful
        """
        try:
            result = self.client.ingest(
                message=summary,
                source=self.source,
                space_id=self.PATIENTS_SPACE
            )
            return result.get("success", False)
        except Exception:
            return False

    def search_trials(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for similar trials.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            Dict containing search results
        """
        return self.client.search(
            query=query,
            space_ids=[self.TRIALS_SPACE],
            limit=limit
        )

    def search_patients(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search for similar patients.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            Dict containing search results
        """
        return self.client.search(
            query=query,
            space_ids=[self.PATIENTS_SPACE],
            limit=limit
        )

    def get_spaces(self) -> list[Dict[str, Any]]:
        """Get all available spaces."""
        return self.client.get_spaces()

    def close(self) -> None:
        """Close the underlying client."""
        self.client.close()


# Backward compatibility aliases
OncoCoreMemory = OncoCoreService
McodeMemoryStorage = OncoCoreService
