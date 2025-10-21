"""
CoreMemoryGraphEngine - A matching engine that leverages the CORE Memory knowledge graph.
"""

from typing import Any, Dict, List

from matching.base import MatchingEngineBase
from services.heysol_client import OncoCoreClient
from shared.models import McodeElement
from utils.logging_config import get_logger


class CoreMemoryGraphEngine(MatchingEngineBase):
    """
    A matching engine that uses the OncoCoreClient to search the knowledge graph.
    """

    def __init__(self, heysol_client: OncoCoreClient, cache_enabled: bool = True, max_retries: int = 3):
        """
        Initializes the engine with a OncoCoreClient instance.

        Args:
            heysol_client: An instance of the OncoCoreClient.
            cache_enabled: Whether to enable caching for API calls.
            max_retries: Maximum number of retries on failure.
        """
        super().__init__(cache_enabled=cache_enabled, max_retries=max_retries)
        self.heysol_client = heysol_client
        self.logger = get_logger(__name__)
        self.logger.info(f"✅ CoreMemoryGraphEngine initialized with cache: {cache_enabled}, retries: {max_retries}.")

    async def match(
        self, patient_data: Dict[str, Any], trial_criteria: Dict[str, Any]
    ) -> List[McodeElement]:
        """
        Matches patient data against trial criteria using the knowledge graph.

        Args:
            patient_data: The patient's data, used for graph search queries.
            trial_criteria: The trial's eligibility criteria, also used for queries.

        Returns:
            A list of McodeElement instances representing matches found in the graph.
        """
        matches: List[McodeElement] = []

        # Extract key information from patient and trial data
        patient_conditions = patient_data.get("conditions", [])
        trial_criteria_text = trial_criteria.get("eligibilityCriteria", "")

        if not patient_conditions or not trial_criteria_text:
            self.logger.debug("Insufficient patient or trial data for memory matching")
            return []

        try:
            # Construct a comprehensive query combining patient conditions and trial criteria
            query_parts = []

            # Add patient conditions to query
            if patient_conditions:
                condition_query = " OR ".join(f'"{condition}"' for condition in patient_conditions)
                query_parts.append(f"({condition_query})")

            # Add key trial criteria terms
            key_terms = ["breast cancer", "chemotherapy", "ECOG", "stage", "metastatic"]
            for term in key_terms:
                if term.lower() in trial_criteria_text.lower():
                    query_parts.append(f'"{term}"')

            # Combine query parts
            if query_parts:
                combined_query = " AND ".join(query_parts)
            else:
                combined_query = "breast cancer"  # fallback query

            self.logger.debug(f"Memory search query: {combined_query}")

            # Search CORE Memory for relevant information
            search_results = self.heysol_client.search(query=combined_query)

            # Process search results to extract relevant matches
            if search_results and "episodes" in search_results:
                for episode in search_results["episodes"]:
                    episode_content = episode.get("content", "").lower()

                    # Check if episode content relates to patient conditions and trial criteria
                    relevant_matches = []
                    for condition in patient_conditions:
                        if condition.lower() in episode_content:
                            relevant_matches.append(condition)

                    # Check for trial criteria matches
                    if "breast cancer" in episode_content and any(term in episode_content for term in ["chemotherapy", "treatment", "trial"]):
                        relevant_matches.append("breast cancer treatment")

                    # Create McodeElement for each relevant match
                    for match in relevant_matches:
                        matches.append(
                            McodeElement(
                                element_type="PatientTrialRelationship",
                                code=f"MEMORY_{len(matches)}",
                                display=f"Memory-based match: {match}",
                                system="CORE_Memory",
                                confidence_score=0.7,  # Memory-based matches are moderately confident
                                evidence_text=f"Found in CORE Memory: {episode_content[:200]}...",
                            )
                        )

            # Also try knowledge graph search if available
            try:
                graph_query = f"patient trial matching {' '.join(patient_conditions)}"
                graph_results = self.heysol_client.search_knowledge_graph(graph_query)

                if graph_results and "nodes" in graph_results:
                    for node in graph_results["nodes"]:
                        node_type = node.get("type", "").lower()
                        if any(keyword in node_type for keyword in ["patient", "trial", "treatment", "condition"]):
                            matches.append(
                                McodeElement(
                                    element_type="KnowledgeGraphMatch",
                                    code=node.get("id", f"KG_{len(matches)}"),
                                    display=node.get("label", "Knowledge graph entity"),
                                    system="CORE_KnowledgeGraph",
                                    confidence_score=node.get("score", 0.6),
                                    evidence_text=f"Knowledge graph relationship: {node.get('properties', {})}",
                                )
                            )
            except Exception as graph_error:
                self.logger.debug(f"Knowledge graph search failed: {graph_error}")

            self.logger.debug(f"CoreMemoryGraphEngine found {len(matches)} matches.")
            return matches

        except Exception as e:
            self.logger.error(f"❌ CoreMemoryGraphEngine failed: {e}")
            return []