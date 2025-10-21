import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from utils.logging_config import Loggable, logging


@dataclass
class DocumentSection:
    """Represents a section of a clinical trial document with source tracking"""

    name: str
    content: str
    source_type: str  # 'protocol', 'eligibility', 'conditions', 'design', etc.
    position: int
    metadata: Optional[Dict[str, Any]] = None


class DocumentIngestor(Loggable):
    """
    Dynamic document ingestion system for clinical trial documents
    Extracts and processes relevant sections from clinical trial data
    """

    def __init__(self) -> None:
        super().__init__()
        # Set logger to DEBUG level for troubleshooting
        self.logger.setLevel(logging.DEBUG)
        self.supported_sections = {
            "protocolSection": {
                "designModule": "design",
                "conditionsModule": "conditions",
                "eligibilityModule": "eligibility",
                "descriptionModule": "description",
                "identificationModule": "identification",
            },
            "resultsSection": {
                "baselineCharacteristicsModule": "baseline",
                "outcomeMeasuresModule": "outcomes",
            },
        }

    def ingest_clinical_trial_document(self, trial_data: Dict[str, Any]) -> List[DocumentSection]:
        """
        Ingest clinical trial document and extract relevant sections

        Args:
            trial_data: Raw clinical trial data from ClinicalTrials.gov API

        Returns:
            List of DocumentSection objects with source tracking
        """
        sections = []
        position = 0

        # Extract protocol section (main content)
        self.logger.debug(f"Trial data keys: {list(trial_data.keys())}")
        protocol_section = trial_data.get("protocolSection", {})
        self.logger.debug(f"Protocol section keys: {list(protocol_section.keys())}")

        # Process each module in protocol section
        for module_name, section_type in self.supported_sections["protocolSection"].items():
            module_data = protocol_section.get(module_name, {})
            self.logger.debug(f"Processing module {module_name}: {bool(module_data)}")
            if module_data:
                section_content = self._extract_module_content(module_data, module_name)
                self.logger.debug(
                    f"Section content length for {module_name}: {len(section_content)}"
                )
                if section_content:
                    sections.append(
                        DocumentSection(
                            name=module_name,
                            content=section_content,
                            source_type=section_type,
                            position=position,
                            metadata={"module_type": module_name},
                        )
                    )
                    position += 1

        # Process results section if available
        results_section = trial_data.get("resultsSection", {})
        for module_name, section_type in self.supported_sections["resultsSection"].items():
            module_data = results_section.get(module_name, {})
            if module_data:
                section_content = self._extract_module_content(module_data, module_name)
                if section_content:
                    sections.append(
                        DocumentSection(
                            name=module_name,
                            content=section_content,
                            source_type=section_type,
                            position=position,
                            metadata={"module_type": module_name},
                        )
                    )
                    position += 1

        return sections

    def _extract_module_content(self, module_data: Dict[str, Any], module_name: str) -> str:
        """
        Extract textual content from a module based on its structure

        Args:
            module_data: Module data dictionary
            module_name: Name of the module for context-aware extraction

        Returns:
            Extracted textual content as string
        """
        content_parts = []

        if module_name == "eligibilityModule":
            # Extract eligibility criteria
            criteria = module_data.get("eligibilityCriteria")
            if criteria:
                content_parts.append(f"Eligibility Criteria: {criteria}")

            # Extract demographic information
            for key in ["gender", "minimumAge", "maximumAge", "healthyVolunteers"]:
                if key in module_data:
                    content_parts.append(f"{key}: {module_data[key]}")

        elif module_name == "conditionsModule":
            # Extract conditions
            conditions = module_data.get("conditions", [])
            if conditions:
                # Handle both string and dict formats for conditions
                condition_names = []
                for condition in conditions:
                    if isinstance(condition, dict):
                        condition_names.append(condition.get("name", str(condition)))
                    else:
                        condition_names.append(str(condition))
                content_parts.append(f"Conditions: {', '.join(condition_names)}")

        elif module_name == "designModule":
            # Extract design information
            design_info = module_data.get("designInfo", {})
            if design_info:
                content_parts.append(f"Design: {json.dumps(design_info, indent=2)}")

            # Extract interventions
            interventions = module_data.get("interventions", [])
            for intervention in interventions:
                content_parts.append(
                    f"Intervention: {intervention.get('name', '')} - {intervention.get('type', '')}"
                )

        elif module_name == "descriptionModule":
            # Extract descriptions
            for key in ["briefSummary", "detailedDescription"]:
                if key in module_data:
                    content_parts.append(f"{key}: {module_data[key]}")

        elif module_name == "identificationModule":
            # Extract identification info
            for key in ["briefTitle", "officialTitle", "nctId"]:
                if key in module_data:
                    content_parts.append(f"{key}: {module_data[key]}")

        else:
            # Generic extraction for other modules
            content_parts.append(json.dumps(module_data, indent=2))

        return "\n\n".join(content_parts) if content_parts else ""

    def extract_structured_sections(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured sections from clinical trial data with comprehensive metadata

        Args:
            trial_data: Raw clinical trial data

        Returns:
            Dictionary with structured sections and metadata
        """
        structured_data = {
            "trial_metadata": self._extract_trial_metadata(trial_data),
            "sections": {},
            "source_tracking": {
                "trial_id": trial_data.get("protocolSection", {})
                .get("identificationModule", {})
                .get("nctId", "unknown"),
                "extraction_timestamp": "2025-08-23T22:31:49Z",  # Would use actual timestamp in production
                "extraction_method": "dynamic_document_ingestion",
            },
        }

        # Extract all sections
        sections = self.ingest_clinical_trial_document(trial_data)
        for section in sections:
            structured_data["sections"][section.name] = {
                "content": section.content,
                "source_type": section.source_type,
                "position": section.position,
                "metadata": section.metadata,
            }

        return structured_data

    def _extract_trial_metadata(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive trial metadata

        Args:
            trial_data: Raw trial data

        Returns:
            Dictionary with trial metadata
        """
        protocol_section = trial_data.get("protocolSection", {})
        identification = protocol_section.get("identificationModule", {})
        design = protocol_section.get("designModule", {})
        conditions = protocol_section.get("conditionsModule", {})

        return {
            "nct_id": identification.get("nctId"),
            "brief_title": identification.get("briefTitle"),
            "official_title": identification.get("officialTitle"),
            "phase": design.get("phase"),
            "study_type": design.get("studyType"),
            "conditions": conditions.get("conditions", []),
            "interventions": design.get("interventions", []),
            "sponsor": protocol_section.get("sponsorCollaboratorsModule", {})
            .get("leadSponsor", {})
            .get("name"),
        }

    def create_source_reference(
        self, section: DocumentSection, text_fragment: str, start_pos: int, end_pos: int
    ) -> Dict[str, Any]:
        """
        Create a source reference for tracking provenance

        Args:
            section: The document section containing the text
            text_fragment: The specific text fragment being referenced
            start_pos: Start position in the section content
            end_pos: End position in the section content

        Returns:
            Source reference dictionary with comprehensive tracking
        """
        return {
            "section_name": section.name,
            "section_type": section.source_type,
            "section_position": section.position,
            "text_fragment": text_fragment,
            "position_range": {"start": start_pos, "end": end_pos},
            "extraction_context": {
                "module_type": (section.metadata.get("module_type") if section.metadata else None),
                "source_system": "ClinicalTrials.gov",
            },
            "provenance_chain": [
                {
                    "step": "document_ingestion",
                    "timestamp": "2025-08-23T22:31:49Z",  # Would use actual timestamp
                    "method": "dynamic_section_extraction",
                }
            ],
        }

    # Sample clinical trial data structure (commented out for production)
    # Uncomment the following lines for testing:
    #
    # sample_trial_data = {
    #     'protocolSection': {
    #         'identificationModule': {
    #             'nctId': 'NCT12345678',
    #             'briefTitle': 'Sample Clinical Trial',
    #             'officialTitle': 'A Phase III Study of Sample Treatment'
    #         },
    #         'designModule': {
    #             'phase': 'PHASE3',
    #             'studyType': 'INTERVENTIONAL',
    #             'designInfo': {
    #                 'allocation': 'RANDOMIZED',
    #                 'interventionModel': 'PARALLEL_ASSIGNMENT'
    #             },
    #             'interventions': [
    #                 {'name': 'Sample Drug', 'type': 'DRUG'}
    #             ]
    #         },
    #         'conditionsModule': {
    #             'conditions': ['Breast Cancer', 'Metastatic Cancer']
    #         },
    #         'eligibilityModule': {
    #             'eligibilityCriteria': 'INCLUSION CRITERIA:\n- Age ≥18 years\n- Histologically confirmed cancer',
    #             'gender': 'ALL',
    #             'minimumAge': '18 Years',
    #             'maximumAge': '100 Years'
    #         },
    #         'descriptionModule': {
    #             'briefSummary': 'This is a sample clinical trial description.',
    #             'detailedDescription': 'Detailed description of the trial methodology and objectives.'
    #         }
    #     }
    # }
    #
    # if __name__ == "__main__":
    #     ingestor = DocumentIngestor()
    #
    #     # Test section extraction
    #     sections = ingestor.ingest_clinical_trial_document(sample_trial_data)
    #     print(f"Extracted {len(sections)} sections:")
    #     for section in sections:
    #         print(f"  - {section.name} ({section.source_type}): {len(section.content)} chars")
    #
    #     # Test structured extraction
    #     structured_data = ingestor.extract_structured_sections(sample_trial_data)
    #     print(f"\nStructured data keys: {list(structured_data.keys())}")
    #     print(f"Trial NCT ID: {structured_data['trial_metadata']['nct_id']}")
