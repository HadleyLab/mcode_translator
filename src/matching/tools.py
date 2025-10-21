"""
Tools for interacting with CORE Memory and other matching-related services.
"""

from typing import Any, Dict, List, Optional

from services.heysol_client import OncoCoreClient
from shared.models import McodeElement, PatientData, ClinicalTrialData

async def search_memory(
    client: OncoCoreClient, query: str, space_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Search CORE Memory for a given query."""
    return client.search(query=query, space_ids=space_ids)

async def ingest_memory(
    client: OncoCoreClient, message: str, space_id: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest a message into CORE Memory."""
    return client.ingest(message=message, space_id=space_id)

async def find_similar_patients(
    client: OncoCoreClient, patient: PatientData, limit: int = 5
) -> List[PatientData]:
    """Find patients similar to the given patient."""
    # This is a placeholder for a more sophisticated implementation
    # that would use the knowledge graph to find similar patients.
    query = f"similar patients to {patient.patient_id}"
    results = client.search(query=query)
    # In a real implementation, we would parse the results and
    # fetch the full data for the similar patients.
    return []

async def get_trial_recruitment_status(
    client: OncoCoreClient, trial: ClinicalTrialData
) -> str:
    """Get the recruitment status for a given trial."""
    # This is a placeholder. A real implementation would query
    # a clinical trials API or a database.
    return "Recruiting"