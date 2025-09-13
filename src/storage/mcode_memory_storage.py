"""
mCODE Memory Storage - Unified interface for storing mCODE summaries to CORE Memory.

This module provides a unified interface for storing processed mCODE data
(trials and patients) as natural language summaries that preserve mCODE
structure and codes for later retrieval and analysis.
"""

from typing import Any, Dict, List, Optional

from src.utils.config import Config
from src.utils.core_memory_client import CoreMemoryClient, CoreMemoryError
from src.utils.logging_config import get_logger


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

        # Use provided values or get from centralized config
        self.api_key = api_key or self.config.get_core_memory_api_key()
        self.source = source or self.config.get_core_memory_source()
        self.base_url = self.config.get_core_memory_api_base_url()
        self.timeout = self.config.get_core_memory_timeout()
        self.max_retries = self.config.get_core_memory_max_retries()
        self.batch_size = self.config.get_core_memory_batch_size()
        self.default_spaces = self.config.get_core_memory_default_spaces()

        # Initialize client with centralized config
        self.client = CoreMemoryClient(
            api_key=self.api_key, base_url=self.base_url, source=self.source
        )

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
            summary = self._create_trial_summary(trial_id, mcode_data)
            self.client.ingest(summary)
            self.logger.info(f"✅ Stored trial {trial_id} mCODE summary in CORE Memory")
            return True
        except CoreMemoryError as e:
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
            summary = self._create_patient_summary(patient_id, mcode_data)
            self.client.ingest(summary)
            self.logger.info(
                f"✅ Stored patient {patient_id} mCODE summary in CORE Memory"
            )
            return True
        except CoreMemoryError as e:
            self.logger.error(f"❌ Failed to store patient {patient_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error storing patient {patient_id}: {e}")
            return False

    def _create_trial_summary(self, trial_id: str, mcode_data: Dict[str, Any]) -> str:
        """
        Create a natural language summary of trial mCODE data.

        Preserves mCODE structure and codes for later analysis.

        Args:
            trial_id: Clinical trial identifier
            mcode_data: Processed mCODE mappings and metadata

        Returns:
            str: Natural language summary with embedded mCODE codes
        """
        mappings = mcode_data.get("mcode_mappings", [])
        metadata = mcode_data.get("metadata", {})

        # Extract key trial information
        brief_title = metadata.get("brief_title", "Unknown Trial")
        sponsor = metadata.get("sponsor", "Unknown Sponsor")

        summary_parts = [
            f"Clinical Trial {trial_id}: '{brief_title}' sponsored by {sponsor}."
        ]

        # Add mCODE mappings as structured natural language
        if mappings:
            summary_parts.append("mCODE Analysis:")

            # Group mappings by category for better readability
            categories = self._categorize_mappings(mappings)

            for category, category_mappings in categories.items():
                if category_mappings:
                    summary_parts.append(f"{category}:")
                    for mapping in category_mappings:
                        element = mapping.get("mcode_element", "")
                        value = mapping.get("value", "")
                        code_info = self._extract_code_info(mapping)

                        if value and value != "N/A":
                            summary_parts.append(f"  - {element}: {value} {code_info}")

        return " ".join(summary_parts)

    def _create_patient_summary(
        self, patient_id: str, mcode_data: Dict[str, Any]
    ) -> str:
        """
        Create a natural language summary of patient mCODE data.

        Preserves mCODE structure and codes for later analysis.

        Args:
            patient_id: Patient identifier
            mcode_data: Processed mCODE mappings and metadata

        Returns:
            str: Natural language summary with embedded mCODE codes
        """
        demographics = mcode_data.get("demographics", {})
        mappings = mcode_data.get("mcode_mappings", [])

        # Basic patient information
        name = demographics.get("name", f"Patient {patient_id}")
        age = demographics.get("age", "Unknown")
        gender = demographics.get("gender", "Unknown")

        summary_parts = [
            f"Patient {name} (ID: {patient_id}), {age} years old, {gender}."
        ]

        # Add mCODE mappings
        if mappings:
            summary_parts.append("mCODE Profile:")

            categories = self._categorize_mappings(mappings)

            for category, category_mappings in categories.items():
                if category_mappings:
                    summary_parts.append(f"{category}:")
                    for mapping in category_mappings:
                        element = mapping.get("mcode_element", "")
                        value = mapping.get("value", "")
                        code_info = self._extract_code_info(mapping)

                        if value and value != "N/A":
                            summary_parts.append(f"  - {element}: {value} {code_info}")

        return " ".join(summary_parts)

    def _categorize_mappings(
        self, mappings: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize mCODE mappings for better summary organization.

        Args:
            mappings: List of mCODE mappings

        Returns:
            Dict grouped by category
        """
        categories = {
            "Cancer Characteristics": [],
            "Biomarkers": [],
            "Treatments": [],
            "Demographics": [],
            "Other": [],
        }

        for mapping in mappings:
            element = mapping.get("mcode_element", "")

            if element in [
                "CancerCondition",
                "TNMStage",
                "HistologyMorphologyBehavior",
            ]:
                categories["Cancer Characteristics"].append(mapping)
            elif element in [
                "ERReceptorStatus",
                "HER2ReceptorStatus",
                "TumorMarkerTest",
            ]:
                categories["Biomarkers"].append(mapping)
            elif element in ["CancerTreatment", "CancerRelatedMedication"]:
                categories["Treatments"].append(mapping)
            elif element in ["PatientSex", "PatientAge", "Race", "Ethnicity"]:
                categories["Demographics"].append(mapping)
            else:
                categories["Other"].append(mapping)

        return categories

    def _extract_code_info(self, mapping: Dict[str, Any]) -> str:
        """
        Extract code information from mCODE mapping for inclusion in summary.

        Args:
            mapping: Individual mCODE mapping

        Returns:
            str: Formatted code information
        """
        # Look for code information in the mapping
        # This preserves the structured codes for later analysis
        code_parts = []

        # Check for system and code
        system = mapping.get("system")
        code = mapping.get("code")

        if system and code:
            # Format as [SYSTEM:CODE]
            if "snomed" in system.lower():
                code_parts.append(f"[SNOMED:{code}]")
            elif "loinc" in system.lower():
                code_parts.append(f"[LOINC:{code}]")
            elif "icd" in system.lower():
                code_parts.append(f"[ICD:{code}]")
            else:
                code_parts.append(f"[{system}:{code}]")

        # Add interpretation if available
        interpretation = mapping.get("interpretation")
        if interpretation and interpretation != mapping.get("value"):
            code_parts.append(f"({interpretation})")

        return " ".join(code_parts) if code_parts else ""

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
        except CoreMemoryError as e:
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
        except CoreMemoryError as e:
            self.logger.error(f"❌ Failed to get memory stats: {e}")
            return {"spaces": [], "total_spaces": 0}
